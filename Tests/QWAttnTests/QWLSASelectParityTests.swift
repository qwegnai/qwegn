import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

/// The multi-threadgroup selector (`qwg_lsa_select_score/_hist2/_expand/_finalize`)
/// must emit exactly the block lists of the single-threadgroup `qwg_lsa_select_blocks`.
final class QWLSASelectParityTests: XCTestCase {
    private func runParity(distinctChunks: Int, T: Int, budget: Int) throws {
        let hal = QWLSATests.hal
        let geometry = QWLSATests.geometry
        let cfg = QWLSAConfig(mode: .svd, budget: budget, clusterBlock: 512)
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
            inDim: geometry.lsaInputDim, indexDim: 64, seed: 101)
        let op = try QWLSAOperator(
            hal: hal, geometry: geometry, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 3, wqIdxF16: basis, wkIdxF16: basis)],
            maxContext: T + 1024, maxPrefillChunk: 512)

        let chunk = 512
        var qBufs: [QWTensorValue] = []
        var kBufs: [QWTensorValue] = []
        for s in 0..<distinctChunks {
            let q = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: UInt64(0x1000 + s))
            let k = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: UInt64(0x2000 + s))
            let qb = hal.device.makeBuffer(bytes: q, length: q.count * 2, options: .storageModeShared)!
            let kb = hal.device.makeBuffer(bytes: k, length: k.count * 2, options: .storageModeShared)!
            qBufs.append(QWTensorValue(region: QWMRegion(buffer: qb, base: 0, length: qb.length),
                                       offset: 0, shape: QWTensorShape([chunk * geometry.qProjOut], .f16)))
            kBufs.append(QWTensorValue(region: QWMRegion(buffer: kb, base: 0, length: kb.length),
                                       offset: 0, shape: QWTensorShape([chunk * geometry.kvChannels], .f16)))
        }

        func encodeAndWait(_ body: (any MTLComputeCommandEncoder) -> Void) {
            guard let (cb, enc) = hal.begin() else { return XCTFail("cmdbuffer") }
            body(enc)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }

        var filled = 0
        while filled < T {
            encodeAndWait { enc in
                var n = 0
                while filled < T && n < 32 {
                    let s = (filled / chunk) % distinctChunks
                    op.encodeIndexAndCluster(enc, qgate: qBufs[s], kIn: kBufs[s], fullAttnLayer: 0,
                                             pos0: filled, rows: chunk)
                    enc.memoryBarrier(scope: .buffers)
                    filled += chunk
                    n += 1
                }
            }
        }

        func snapshot(rows: Int) -> [[Int32]] {
            let base = op.region.pointer
            let counts = base.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
            let lists = base.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
            return (0..<rows).map { r in
                (0..<Int(counts[r])).map { lists[r * op.listStride + $0] }
            }
        }

        func select(parallel: Bool, pos0: Int, rows: Int, tiled32: Bool) -> [[Int32]] {
            op.useParallelSelect = parallel
            let base = op.region.pointer
            memset(base.advanced(by: op.blockCountScratch.offset), 0, op.maxPrefillChunk * 4)
            encodeAndWait { enc in
                op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: pos0, rows: rows, tiled32: tiled32)
            }
            return snapshot(rows: rows)
        }

        // Fresh index queries for the prefill-chunk rows ending at T.
        encodeAndWait { enc in
            op.encodeIndexAndCluster(enc, qgate: qBufs[1 % distinctChunks], kIn: kBufs[1 % distinctChunks],
                                     fullAttnLayer: 0, pos0: T - chunk, rows: chunk)
        }
        let cases: [(pos0: Int, rows: Int, tiled: Bool)] = [
            (T - chunk, chunk, true),   // NAX tiled prefill
            (T - chunk, 40, false),     // MTP-style multi-row verify
            (T - chunk, 1, false),      // decode
        ]
        for c in cases {
            // Twice in a row: the parallel path must also restore its zero-at-rest scratch.
            let legacy = select(parallel: false, pos0: c.pos0, rows: c.rows, tiled32: c.tiled)
            let fast1 = select(parallel: true, pos0: c.pos0, rows: c.rows, tiled32: c.tiled)
            let fast2 = select(parallel: true, pos0: c.pos0, rows: c.rows, tiled32: c.tiled)
            XCTAssertEqual(fast1, legacy, "parallel selector diverged: rows=\(c.rows) tiled=\(c.tiled)")
            XCTAssertEqual(fast2, legacy, "parallel selector not repeatable: rows=\(c.rows) tiled=\(c.tiled)")
            XCTAssertGreaterThan(legacy.map(\.count).max() ?? 0, 1 + cfg.budgetBlocks)
        }
    }

    func testParityDistinctKeys() throws {
        try runParity(distinctChunks: 12, T: 49_152, budget: 2048)
    }

    /// Periodic keys produce exact score ties straddling the top-k threshold.
    func testParityTiedKeys() throws {
        try runParity(distinctChunks: 4, T: 49_152, budget: 2048)
    }

    func testParitySmallBudget() throws {
        try runParity(distinctChunks: 4, T: 49_152, budget: 512)
    }
}

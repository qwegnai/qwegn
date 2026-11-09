import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

/// Opt-in LSA selector GPU microbenchmark: `QWG_LSA_BENCH=1 swift test --filter QWLSABenchTests`.
/// Prints per-context GPU time and an FNV hash of the selected block lists
/// (used to prove selector optimizations are bit-identical).
final class QWLSABenchTests: XCTestCase {
    func testSelectorScaling() throws {
        guard ProcessInfo.processInfo.environment["QWG_LSA_BENCH"] == "1" else {
            throw XCTSkip("set QWG_LSA_BENCH=1 to run the LSA selector microbenchmark")
        }
        let hal = QWLSATests.hal
        let geometry = QWLSATests.geometry
        let maxT = Int(ProcessInfo.processInfo.environment["QWG_LSA_BENCH_MAXT"] ?? "") ?? 1_048_576
        let rungs = [65_536, 131_072, 262_144, 524_288, 1_048_576].filter { $0 <= maxT }
        let budget = Int(ProcessInfo.processInfo.environment["QWG_LSA_BENCH_BUDGET"] ?? "") ?? 2048
        let cfg = QWLSAConfig(mode: .svd, budget: budget, clusterBlock: 512)
        let inDim = geometry.lsaInputDim
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 101)
        let op = try QWLSAOperator(
            hal: hal, geometry: geometry, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 3, wqIdxF16: basis, wkIdxF16: basis)],
            maxContext: maxT + 1024, maxPrefillChunk: 512)

        let chunk = 512
        var qBufs: [QWTensorValue] = []
        var kBufs: [QWTensorValue] = []
        for s in 0..<8 {
            let q = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: UInt64(0x1000 + s))
            let k = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: UInt64(0x2000 + s))
            let qb = hal.device.makeBuffer(bytes: q, length: q.count * 2, options: .storageModeShared)!
            let kb = hal.device.makeBuffer(bytes: k, length: k.count * 2, options: .storageModeShared)!
            qBufs.append(QWTensorValue(region: QWMRegion(buffer: qb, base: 0, length: qb.length),
                                       offset: 0, shape: QWTensorShape([chunk * geometry.qProjOut], .f16)))
            kBufs.append(QWTensorValue(region: QWMRegion(buffer: kb, base: 0, length: kb.length),
                                       offset: 0, shape: QWTensorShape([chunk * geometry.kvChannels], .f16)))
        }

        var filled = 0
        func fill(to T: Int) {
            while filled < T {
                guard let (cb, enc) = hal.begin() else { return }
                var n = 0
                while filled < T && n < 64 {
                    let s = (filled / chunk) % 8
                    op.encodeIndexAndCluster(enc, qgate: qBufs[s], kIn: kBufs[s], fullAttnLayer: 0,
                                             pos0: filled, rows: chunk)
                    enc.memoryBarrier(scope: .buffers)
                    filled += chunk
                    n += 1
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
            }
        }

        func listHash(rows: Int) -> UInt64 {
            let base = op.region.pointer
            let counts = base.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
            let lists = base.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
            var h: UInt64 = 0xcbf29ce484222325
            for r in 0..<rows {
                let c = Int(counts[r])
                h = (h ^ UInt64(c)) &* 0x100000001b3
                for i in 0..<c { h = (h ^ UInt64(UInt32(bitPattern: lists[r * op.listStride + i]))) &* 0x100000001b3 }
            }
            return h
        }

        func timeGPU(iters: Int, _ body: (any MTLComputeCommandEncoder) -> Void) -> Double {
            guard let (cb, enc) = hal.begin() else { return .nan }
            for _ in 0..<iters {
                body(enc)
                enc.memoryBarrier(scope: .buffers)
            }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(iters)
        }

        let qDec = qBufs[0].slice(byteOffset: 0, count: geometry.qProjOut)
        let kDec = kBufs[0].slice(byteOffset: 0, count: geometry.kvChannels)
        for T in rungs {
            fill(to: T)
            let pos = T - 1
            // Decode: index+select for one row (the per-layer, per-token LSA overhead).
            _ = timeGPU(iters: 3) { enc in
                op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: pos, rows: 1)
            }
            let decMs = timeGPU(iters: 50) { enc in
                op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: pos, rows: 1)
            }
            let decHash = listHash(rows: 1)
            // Prefill: a 512-row chunk with NAX 32-row tiling (16 selections).
            op.encodeIndexAndClusterOnly(hal: hal, qgate: qBufs[1], kIn: kBufs[1], pos0: T - chunk, rows: chunk)
            let preMs = timeGPU(iters: 5) { enc in
                op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: T - chunk, rows: chunk, tiled32: true)
            }
            let preHash = listHash(rows: chunk)
            print(String(format: "LSA-BENCH T=%7d decode_select=%.3f ms prefill_select_512=%.3f ms dec_hash=%016llx pre_hash=%016llx",
                         T, decMs, preMs, decHash, preHash))
            _ = qDec; _ = kDec
        }
    }
}

private extension QWLSAOperator {
    /// Re-project the index queries for `rows` rows (fills `qIdxScratch`) without
    /// disturbing the populated pooled/cluster state beyond `pos0 + rows`.
    func encodeIndexAndClusterOnly(hal: QWHal, qgate: QWTensorValue, kIn: QWTensorValue, pos0: Int, rows: Int) {
        guard let (cb, enc) = hal.begin() else { return }
        encodeIndexAndCluster(enc, qgate: qgate, kIn: kIn, fullAttnLayer: 0, pos0: pos0, rows: rows)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
    }
}

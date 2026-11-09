// QWDxParityTests — bit-exactness door for the device-x (dx) kernel family
// (P7 ablation-race winner: x from DEVICE as half4 + 4×uint4 register
// prefetch; the pinned-FMA fold tree is verbatim, fma operand VALUES are
// Full note: docs/STYLE_NOTES.md#QWDxParityTests.swift.
import XCTest
import Metal
import Foundation
@testable import QWMLP
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

final class QWDxParityTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func pso(_ name: String) throws -> MTLComputePipelineState {
        try XCTUnwrap(try Self.hal.pipeline(name), "kernel \(name) missing from metallib")
    }

    /// row-per-TG q4g GEMV family encode (buffers 0..8 = engine contract).
    private struct Bind { let buf: any MTLBuffer; let base: Int }
    private static func bind(_ v: QWTensorValue) -> Bind {
        Bind(buf: v.region.buffer, base: v.region.base)
    }
    private static func bind(_ b: MTLBuffer) -> Bind { Bind(buf: b, base: 0) }

    private func encode(_ ce: any MTLComputeCommandEncoder, _ p: MTLComputePipelineState,
                        threads: Int, tgs: Int,
                        blocks: Bind, x: Bind, gs: Bind, y32: Bind,
                        n: UInt32, k: UInt32, y16: Bind) {
        ce.setComputePipelineState(p)
        ce.setBuffer(blocks.buf, offset: blocks.base, index: 0)
        ce.setBuffer(x.buf, offset: x.base, index: 1)
        ce.setBuffer(gs.buf, offset: gs.base, index: 2)
        ce.setBuffer(y32.buf, offset: y32.base, index: 3)
        var nn = n, kk = k
        ce.setBytes(&nn, length: 4, index: 4)
        ce.setBytes(&kk, length: 4, index: 5)
        ce.setBuffer(y16.buf, offset: y16.base, index: 8)
        ce.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    private func bytes(_ b: MTLBuffer) -> Data { Data(bytes: b.contents(), count: b.length) }

    private func assertBytesEqual(_ a: Data, _ b: Data, _ msg: String,
                                  file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "\(msg): size mismatch", file: file, line: line)
        if a == b { return }
        var first = min(a.count, b.count)
        for i in 0..<first where a[a.index(a.startIndex, offsetBy: i)] != b[b.index(b.startIndex, offsetBy: i)] {
            first = i; break
        }
        XCTFail("\(msg): first differing byte at \(first)", file: file, line: line)
    }

    // MARK: down-shape GEMV family (n=5120 k=17408 — 544 blocks of 32)

    func testDxAndMtgGEMVBitExactVsShipping() throws {
        let (_, dn) = try QWMLPTests.weights()          // mlp_down blocks
        let n = 5120, k: UInt32 = 17408
        let hIn = QWMLPTests.buf(try QWMLPTests.load("mlp_h_M1.bin"))  // down's x = h (17408 f16)
        let hs = QWMLPTests.zeros(136, .f32)

        let variants: [(String, Int, Int)] = [
            ("qwg_gemv_q4g_v4_h16_dual", 32, 1),           // ship reference — first
            ("qwg_gemv_q4g_dx_h16_dual_t4", 32, 1),        // dx race winner
        ]
        let wv = dn.view
        var ref16 = Data(), ref32 = Data()
        for (v, threads, rows) in variants {
            let p = try pso(v)
            let y16 = QWMLPTests.zeros(n, .f16)
            let y32 = QWMLPTests.zeros(n, .f32)
            let tgs = (n + rows - 1) / rows
            guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            encode(ce, p, threads: threads, tgs: tgs,
                   blocks: Self.bind(wv.blocks), x: Self.bind(hIn), gs: Self.bind(hs),
                   y32: Self.bind(y32), n: UInt32(n), k: k, y16: Self.bind(y16))
            ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if v == variants[0].0 {
                let nz = Data(bytes: y32.contents(), count: n * 4)
                    .withUnsafeBytes { $0.bindMemory(to: Float.self).filter { $0 != 0 }.count }
                XCTAssertGreaterThan(nz, n / 2, "shipping dual wrote ~zeros — fixture wiring broken")
                ref16 = bytes(y16); ref32 = bytes(y32)
            } else {
                assertBytesEqual(bytes(y16), ref16, "\(v): y16 vs shipping dual")
                assertBytesEqual(bytes(y32), ref32, "\(v): y32 vs shipping dual")
            }
        }
    }

    // MARK: gu-shape silu pair family (N=34816 K=5120)

    func testSiluPairDxAndMtg2BitExactVsShipping() throws {
        let (gu, _) = try QWMLPTests.weights()
        let bigN: UInt32 = 2 * 17408, k: UInt32 = 5120
        let x = QWMLPTests.buf(try QWMLPTests.load("mlp_x_M1.bin"))   // ln2 (5120 f16)
        let gs = QWMLPTests.zeros(40, .f32)

        let variants: [(String, Int, Int)] = [
            ("qwg_gemv_q4g_silu_pair_h16", 64, 1),        // ship reference — first
            ("qwg_gemv_q4g_silu_pair_dx_h16", 64, 1),     // dx winner
        ]
        let wv = gu.view
        var ref = Data()
        for (v, threads, pairs) in variants {
            let p = try pso(v)
            let h16 = QWMLPTests.zeros(17408, .f16)
            let dummy = QWMLPTests.zeros(1, .f32)
            let tgs = (17408 + pairs - 1) / pairs
            guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            encode(ce, p, threads: threads, tgs: tgs,
                   blocks: Self.bind(wv.blocks), x: Self.bind(x), gs: Self.bind(gs),
                   y32: Self.bind(dummy), n: bigN, k: k, y16: Self.bind(h16))
            ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if v == variants[0].0 {
                let nz = Data(bytes: h16.contents(), count: 17408 * 2)
                    .withUnsafeBytes { $0.bindMemory(to: UInt16.self).filter { $0 != 0 }.count }
                XCTAssertGreaterThan(nz, 17408 / 2, "shipping silu_pair wrote ~zeros — wiring broken")
                ref = bytes(h16)
            } else {
                assertBytesEqual(bytes(h16), ref, "\(v): h16 vs shipping silu_pair_h16")
            }
        }
    }
}

import Foundation
import Metal
import XCTest
@testable import QWFlashNext
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

/// CPU reference vs GPU Group-Latent MoE parity across decode (T = 1), verify (T = 4), and NAX prefill (T = 64).
final class FNLatentMoETests: XCTestCase {
    private static func fwht32Ortho(_ v: inout [Float]) {
        let invSqrt32 = Float(1.0 / sqrt(32.0))
        var offset = 0
        while offset < v.count {
            var h = 1
            while h < 32 {
                var i = 0
                while i < 32 {
                    for j in i..<(i + h) {
                        let a = v[offset + j]
                        let b = v[offset + j + h]
                        v[offset + j] = a + b
                        v[offset + j + h] = a - b
                    }
                    i += h * 2
                }
                h *= 2
            }
            for k in 0..<32 { v[offset + k] *= invSqrt32 }
            offset += 32
        }
    }

    private static func fwhtCols32(_ mat: [Float], rows: Int, cols: Int) -> [Float] {
        var out = mat
        for r in 0..<rows {
            var rowSlice = Array(out[(r * cols)..<((r + 1) * cols)])
            fwht32Ortho(&rowSlice)
            out.replaceSubrange((r * cols)..<((r + 1) * cols), with: rowSlice)
        }
        return out
    }

    private static func fwhtRows32(_ mat: [Float], rows: Int, cols: Int) -> [Float] {
        var out = mat
        for c in 0..<cols {
            var colVec = (0..<rows).map { out[$0 * cols + c] }
            fwht32Ortho(&colVec)
            for r in 0..<rows { out[r * cols + c] = colVec[r] }
        }
        return out
    }

    private struct PackedMatrix {
        let rotData: Data
        let cpuUnrotated: [Float]
    }

    private static func makeAQMatrix(rows: Int, cols: Int, bits: Int, group: Int, seed: Int,
                                     foldOutputFwht: Bool = false) throws -> PackedMatrix {
        var rotData = Data()
        var deqRot = [Float](repeating: 0, count: rows * cols)
        let qMax = (1 << bits) - 1
        let scale: Float = bits == 8 ? (1.0 / 256.0) : (1.0 / 16.0)
        let bias: Float = bits == 8 ? -0.5 : -0.5
        for r in 0..<rows {
            for g in stride(from: 0, to: cols, by: group) {
                let codes = (0..<group).map { i -> UInt32 in
                    let raw = (r * 13 + (g + i) * 7 + seed * 19) & 0xFFFF
                    return UInt32(raw % (qMax + 1))
                }
                rotData.append(try QWAQBlock.pack(bits: bits, group: group, values: codes, scale: scale, bias: bias))
                for i in 0..<group {
                    deqRot[r * cols + g + i] = Float(codes[i]) * scale + bias
                }
            }
        }
        var cpu = fwhtCols32(deqRot, rows: rows, cols: cols)
        if foldOutputFwht {
            cpu = fwhtRows32(cpu, rows: rows, cols: cols)
        }
        return PackedMatrix(rotData: rotData, cpuUnrotated: cpu)
    }

    private static func makeView(_ arena: QWArena, bytes: Data, rows: Int, cols: Int,
                                 bits: Int, group: Int) throws -> QWWeightView {
        let t = try arena.weightsRegion().allocate(QWTensorShape([bytes.count / 2], .f16))
        arena.copyIn(t, from: bytes)
        return try XCTUnwrap(QWWeightView(blocks: t, n: rows, k: cols, bits: bits, groupSize: group,
                                          isRotated: true, isIncumbentContainer: false))
    }

    func testGroupLatentMoEMatchesCPUAcrossDecodeVerifyAndPrefill() throws {
        let H = 64, NG = 4, L = 64, E = 32, K = 4, I = 64, IS = 64
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dMats = try (0..<NG).map { try Self.makeAQMatrix(rows: L, cols: H, bits: 8, group: 32, seed: 1 + $0, foldOutputFwht: true) }
        let uMats = try (0..<NG).map { try Self.makeAQMatrix(rows: H, cols: L, bits: 8, group: 32, seed: 11 + $0, foldOutputFwht: false) }
        let guMats = try (0..<E).map { try Self.makeAQMatrix(rows: 2 * I, cols: L, bits: 4, group: 64, seed: 21 + $0, foldOutputFwht: false) }
        let dnMats = try (0..<E).map { try Self.makeAQMatrix(rows: L, cols: I, bits: 4, group: 64, seed: 61 + $0, foldOutputFwht: true) }
        let shGate = try Self.makeAQMatrix(rows: IS, cols: H, bits: 8, group: 32, seed: 101)
        let shUp = try Self.makeAQMatrix(rows: IS, cols: H, bits: 8, group: 32, seed: 102)
        let shDn = try Self.makeAQMatrix(rows: H, cols: IS, bits: 8, group: 32, seed: 103)

        let labels = (0..<E).map { Int32($0 % NG) }
        let gateW = (0..<(E * H)).map { Float(($0 * 17) % 19 - 9) * 0.0625 }
        let shGateScalarW = (0..<H).map { Float(($0 * 11) % 13 - 6) * 0.0625 }

        let dData = dMats.reduce(into: Data()) { $0.append($1.rotData) }
        let uData = uMats.reduce(into: Data()) { $0.append($1.rotData) }
        let guData = guMats.reduce(into: Data()) { $0.append($1.rotData) }
        let dnData = dnMats.reduce(into: Data()) { $0.append($1.rotData) }
        var shGUData = Data(shGate.rotData); shGUData.append(shUp.rotData)
        let weightBytes = dData.count + uData.count + guData.count + dnData.count + shGUData.count + shDn.rotData.count + (1 << 20)

        let arena = try QWArena(hal: hal, weights: weightBytes, activations: 16 << 20, kv: 256, gdnStates: 256)
        let wReg = arena.weightsRegion()
        let routerAndSharedBF16 = try wReg.allocate(QWTensorShape([gateW.count + shGateScalarW.count], .f16))
        arena.write(routerAndSharedBF16, from: (gateW + shGateScalarW).map(QWBF16.fromFloatRNE))
        let shScalarF32 = try wReg.allocate(QWTensorShape([H], .f32))
        arena.write(shScalarF32, from: shGateScalarW)
        let labelsT = try wReg.allocate(QWTensorShape([E], .i32))
        arena.write(labelsT, from: labels)

        let gpuMoe = FNGPUMoE(
            gateAndSharedBF16: routerAndSharedBF16,
            gateUp: try Self.makeView(arena, bytes: guData, rows: E * 2 * I, cols: L, bits: 4, group: 64),
            down: try Self.makeView(arena, bytes: dnData, rows: E * L, cols: I, bits: 4, group: 64),
            sharedGateUp: try Self.makeView(arena, bytes: shGUData, rows: 2 * IS, cols: H, bits: 8, group: 32),
            sharedDown: try Self.makeView(arena, bytes: shDn.rotData, rows: H, cols: IS, bits: 8, group: 32),
            sharedGateScalar: shScalarF32,
            latentD: try Self.makeView(arena, bytes: dData, rows: NG * L, cols: H, bits: 8, group: 32),
            latentU: try Self.makeView(arena, bytes: uData, rows: NG * H, cols: L, bits: 8, group: 32),
            latentLabels: labelsT, experts: E, topK: K, intermediate: I, sharedIntermediate: IS,
            numLatentGroups: NG, latentDim: L)

        let cpuSrc = FNArrayExperts(
            experts: E,
            gateUp: guMats.flatMap(\.cpuUnrotated), down: dnMats.flatMap(\.cpuUnrotated),
            latentGroups: NG, latentD: dMats.flatMap(\.cpuUnrotated), latentU: uMats.flatMap(\.cpuUnrotated))
        let cpuMoe = FNSparseMoE(
            hidden: H, experts: E, topK: K, intermediate: I, sharedIntermediate: IS, normTopK: true,
            gate: gateW, source: cpuSrc, sharedGate: shGate.cpuUnrotated, sharedUp: shUp.cpuUnrotated,
            sharedDown: shDn.cpuUnrotated, sharedGateScalar: shGateScalarW,
            latentGroups: NG, latentDim: L, latentLabels: labels.map(Int.init), latentSource: cpuSrc)

        let doors = [
            "qwg_router_gemm", "qwg_moe_router_bf16_f32", "qwg_router_nax",
            "qwg_moe_route", "qwg_moe_route_decode", "qwg_moe_group_pairs_f32",
            "qwg_silu_mul_f32", "qwg_silu_mul_split_f16", "qwg_cast_f32_to_f16",
            "qwg_gemv_aq8_g32_f32_v4", "qwg_gemm_aq8_g32_f32_bm32", "qwg_gemm_aq8_g32_nax_f32",
            "qwg_moe_gateup_aq4_g64_f32_r1", "qwg_moe_down_shared_combine_aq4_g64_f32",
            "qwg_moe_down_shared_combine_aq8_g32_f32", "qwg_moe_nax_gemm_aq4_g64",
            "qwg_moe_nax_split_aq4_g64_f32", "qwg_moe_nax_gemm_aq8_g32", "qwg_moe_down_reduce_combine_f32",
        ]
        let ctx = try FNGPUContext(hal: hal, doors: doors, geometry: .qwen38_27b)

        for rows in [1, 2, 4, 16, 64] {
            let xPlain = (0..<(rows * H)).map { Float(($0 * 7 + rows) % 23 - 11) * 0.1 }
            let xRot = Self.fwhtCols32(xPlain, rows: rows, cols: H)
            let scr = try FNGPUScratch(arena: arena, T: rows, hidden: H, hc: 1, lowrank: 8,
                                       moe: gpuMoe, ple: nil, gdn: nil, attn: nil)
            let act = arena.activationsRegion()
            let mixedT = try act.allocate(QWTensorShape([rows * H], .f32))
            let outT = try act.allocate(QWTensorShape([rows * H], .f32))
            arena.write(mixedT, from: xPlain)
            arena.write(scr.rot32, from: xRot)
            if rows >= 64 {
                arena.write(scr.naxX16, from: xRot.map(Float16.init))
            }
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            var enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .concurrent))
            try FNGPULayerPasses.moe(&enc, ctx: ctx, m: gpuMoe, s: scr, mixed: mixedT, out: outT, rows: rows, cb: cb)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { XCTFail("GPU error at T=\(rows): \(err)") }

            let got = arena.read(outT, as: Float.self, count: rows * H)
            let want = cpuMoe.forward(xPlain, rows: rows)
            let worst = zip(got, want).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            let wantMax = want.reduce(Float(0)) { max($0, abs($1)) }
            let rel = worst / max(1e-6, wantMax)
            XCTAssertLessThan(rel, rows >= 64 ? 0.01 : 0.002,
                              "Mismatch at T=\(rows): maxAbs=\(worst), wantMax=\(wantMax), rel=\(rel)")
        }
    }
}

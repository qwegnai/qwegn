// QWErnie55AttnOptPhase5gTests.swift — Parity & benchmark tests for Phase 5g 8 KiB HCA wo_a per-mode NAX PSOs.
import Foundation
import Metal
import XCTest
@testable import QWAlloc
@testable import QWAttn
@testable import QWCore
@testable import QWHAL
@testable import QWQuant

final class QWErnie55AttnOptPhase5gTests: XCTestCase {
    private static func detRand(_ count: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
        var s = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        return (0..<count).map { _ in
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27
            let r = s &* 0x2545_F491_4F6C_DD1D
            return (Float(r & 0xFFFF) / 65535.0 * 2.0 - 1.0) * scale
        }
    }

    private static func makeF16Buf(_ hal: QWHal, count: Int, seed: UInt64?, scale: Float = 0.25) -> QWTensorValue {
        let bytes = max(256, (count * 2 + 255) & ~255)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        if let s0 = seed {
            let lut = detRand(4096, seed: s0, scale: scale).map { Float16($0) }
            let dst = buf.contents().assumingMemoryBound(to: Float16.self)
            lut.withUnsafeBufferPointer { lBuf in
                guard let lp = lBuf.baseAddress else { return }
                var i = 0
                while i + 4096 <= count {
                    memcpy(dst + i, lp, 4096 * 2)
                    i += 4096
                }
                while i < count { dst[i] = lp[i & 4095]; i += 1 }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes),
                             offset: 0, shape: QWTensorShape([count], .f16))
    }

    private static func maxAbsDiffF16(_ a: QWTensorValue, _ b: QWTensorValue, count: Int) -> Float {
        let pa = a.region.pointer.advanced(by: a.offset).assumingMemoryBound(to: UInt16.self)
        let pb = b.region.pointer.advanced(by: b.offset).assumingMemoryBound(to: UInt16.self)
        var m: Float = 0, i = 0
        while i < count {
            let ua = pa[i], ub = pb[i]
            if ua != ub || (ua & 0x7C00) == 0x7C00 {
                let fa = Float(Float16(bitPattern: ua)), fb = Float(Float16(bitPattern: ub))
                if !(fa.isFinite && fb.isFinite) { return .infinity }
                let d = abs(fa - fb)
                if d > m { m = d }
            }
            i += 1
        }
        return m
    }

    private static func packWeights(hal: QWHal, inc: Int, bits: Int, group: Int,
                                    rows: Int, cols: Int, seed: UInt64) -> QWTensorValue {
        let nGroups = cols / group
        let rowBytes = (inc == 1) ? ((cols / 32) * 20) : (nGroups * (4 + (group * bits) / 8))
        var bytes = Data(count: max(256, (rows * rowBytes + 1) & ~1))
        var rng = seed == 0 ? 0x55AA_55AA_1234_5678 : seed
        func nextByte() -> UInt8 {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return UInt8(truncatingIfNeeded: (rng &* 0x2545_F491_4F6C_DD1D) >> 8)
        }
        bytes.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let gBytes = rowBytes / nGroups
            for r in 0..<rows {
                for g in 0..<nGroups {
                    let gp = ptr.advanced(by: r * rowBytes + g * gBytes)
                    let sc = Float16((bits == 8 && inc == 0 ? 0.002 : 0.03) + Float((r + g) % 5) * 0.0005)
                    let bi = Float16((inc == 1 ? -1031.5 : (bits == 8 ? -127.5 : -7.5)) * Float(sc))
                    UnsafeMutableRawPointer(gp).storeBytes(of: sc.bitPattern, as: UInt16.self)
                    UnsafeMutableRawPointer(gp.advanced(by: 2)).storeBytes(of: bi.bitPattern, as: UInt16.self)
                    for b in 4..<gBytes { gp[b] = nextByte() }
                }
            }
        }
        let buf = hal.device.makeBuffer(length: bytes.count, options: .storageModeShared)!
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress { buf.contents().copyMemory(from: src, byteCount: bytes.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes.count),
                             offset: 0, shape: QWTensorShape([bytes.count / 2], .f16))
    }

    private static func makeGeometry(qHeads: Int = 64, topK: Int = 2048, window: Int = 64, oLoraRank: Int = 64) throws -> QWModelGeometry {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": 256, "intermediate_size": 256,
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
          "num_attention_heads": \(qHeads), "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [128, -2], "default_compress_ratio": 128,
          "q_lora_rank": 128, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 16, "o_lora_rank": \(oLoraRank), "sliding_window": \(window),
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 1000000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": \(qHeads), "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": 128, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 256, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 512,
          "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": \(topK),
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    func testHCAGroupedWoANax8KiBPerModeParityAndBenchmark() throws {
        let hal = try QWHal(geometry: try Self.makeGeometry())
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "8 KiB HCA wo_a NAX requires Metal 4")

        let pMMA = try hal.pipeline("qwg_hca_grouped_wo_a_mma"), pGen = try hal.pipeline("qwg_hca_grouped_wo_a_nax")
        let specs: [(String, String, Int, Int, Int)] = [
            ("hq4", "qwg_hca_grouped_wo_a_nax_hq4", 1, 4, 32),
            ("aq8", "qwg_hca_grouped_wo_a_nax_aq8", 0, 8, 32),
            ("aq4", "qwg_hca_grouped_wo_a_nax_aq4", 0, 4, 64),
        ]

        // 1. T=65 parity across aligned + tail rows and skipFwht={0,1} vs MMA & generic NAX.
        let tPar = 65, oGPar = 8, oRPar = 128, gInPar = 512, outPar = oGPar * oRPar
        let zInP = Self.makeF16Buf(hal, count: tPar * oGPar * gInPar, seed: 0xE01)
        let qGateP = Self.makeF16Buf(hal, count: tPar * outPar, seed: 0xE02)
        let refMma = Self.makeF16Buf(hal, count: tPar * outPar, seed: nil)
        let refGen = Self.makeF16Buf(hal, count: tPar * outPar, seed: nil)
        let candP = Self.makeF16Buf(hal, count: tPar * outPar, seed: nil)
        for (lbl, kName, inc, bits, grp) in specs {
            let pCand = try hal.pipeline(kName)
            let wP = Self.packWeights(hal: hal, inc: inc, bits: bits, group: grp, rows: outPar, cols: gInPar, seed: 0xE10 + UInt64(bits))
            for skip: UInt32 in [0, 0x10000] {
                var prm: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                    UInt32(tPar), UInt32(oGPar), UInt32(oRPar), UInt32(gInPar), UInt32(outPar), 0,
                    UInt32(bits), UInt32(grp), UInt32(inc) | skip
                )
                guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                for (pso, dst, grid) in [
                    (pMMA, refMma, MTLSize(width: oGPar, height: (tPar + 31) / 32, depth: oRPar / 128)),
                    (pGen, refGen, MTLSize(width: oRPar / 64, height: (tPar + 63) / 64, depth: oGPar)),
                    (pCand, candP, MTLSize(width: oRPar / 64, height: (tPar + 63) / 64, depth: oGPar)),
                ] {
                    e.setComputePipelineState(pso)
                    zInP.bind(e, index: 0); wP.bind(e, index: 1); qGateP.bind(e, index: 2); dst.bind(e, index: 3)
                    withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 4) }
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
                hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                XCTAssertLessThan(Self.maxAbsDiffF16(refMma, candP, count: tPar * outPar), 5e-3, "\(lbl) vs mma skip=\(skip)")
                XCTAssertLessThan(Self.maxAbsDiffF16(refGen, candP, count: tPar * outPar), 1e-4, "\(lbl) vs gen skip=\(skip)")
            }
        }

        // 2. T=4087, oGroups=8, oLoraRank=1024, groupIn=512 parity + informational timing vs generic NAX.
        let T = 4087, oGroups = 8, oRank = 1024, groupIn = 512, outCols = oGroups * oRank
        let zIn = Self.makeF16Buf(hal, count: T * oGroups * groupIn, seed: 0xE21)
        let qGate = Self.makeF16Buf(hal, count: T * outCols, seed: 0xE22)
        let outRef = Self.makeF16Buf(hal, count: T * outCols, seed: nil)
        let outMode = Self.makeF16Buf(hal, count: T * outCols, seed: nil)
        let grid = MTLSize(width: oRank / 64, height: (T + 63) / 64, depth: oGroups)
        var modeBest: [String: Double] = [:], genBest: [String: Double] = [:]
        for (lbl, kName, inc, bits, grp) in specs {
            let pMode = try hal.pipeline(kName)
            let wB = Self.packWeights(hal: hal, inc: inc, bits: bits, group: grp, rows: outCols, cols: groupIn, seed: 0xE30 + UInt64(bits))
            var prm: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                UInt32(T), UInt32(oGroups), UInt32(oRank), UInt32(groupIn), UInt32(outCols), 0, UInt32(bits), UInt32(grp), UInt32(inc)
            )
            func run(_ pso: any MTLComputePipelineState, dst: QWTensorValue) throws -> Double {
                guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                e.setComputePipelineState(pso)
                zIn.bind(e, index: 0); wB.bind(e, index: 1); qGate.bind(e, index: 2); dst.bind(e, index: 3)
                withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 4) }
                e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
            }
            _ = try run(pGen, dst: outRef); _ = try run(pMode, dst: outMode)
            XCTAssertLessThan(Self.maxAbsDiffF16(outRef, outMode, count: T * outCols), 1e-4, "T=4087 \(lbl) parity")
            var bM = Double.greatestFiniteMagnitude, bG = Double.greatestFiniteMagnitude
            for iter in 0..<7 {
                let gMs = try run(pGen, dst: outRef), mMs = try run(pMode, dst: outMode)
                if iter >= 2 { bG = min(bG, gMs); bM = min(bM, mMs) }
            }
            modeBest[lbl] = bM; genBest[lbl] = bG
        }
        print(String(format: "[QWErnie55AttnOptPhase5gTests] T=4087 wo_a minGPU: hq4=%.3f ms (gen=%.3f), aq8=%.3f ms (gen=%.3f), aq4=%.3f ms (gen=%.3f)",
                     modeBest["hq4"]!, genBest["hq4"]!, modeBest["aq8"]!, genBest["aq8"]!, modeBest["aq4"]!, genBest["aq4"]!))
    }
}

// QWErnie55MoETests.swift — Golden parity tests for SiTU, dual-split sigmoid routing, and Latent-MoE.
// Verifies SHVQ15_G64 + SHVQ3_G64 + HQ8_G32 latent experts + full-hidden shared experts across decode/verify/NAX.

import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWErnie55MoETests: XCTestCase {
    static let ernieGeometry: QWModelGeometry = {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": 1024,
          "hidden_size": 256,
          "intermediate_size": 256,
          "num_hidden_layers": 2,
          "max_position_embeddings": 1048576,
          "rms_norm_eps": 1e-6,
          "num_attention_heads": 64,
          "num_key_value_heads": 1,
          "head_dim": 512,
          "compress_ratios": [128, -2],
          "default_compress_ratio": 128,
          "q_lora_rank": 256,
          "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448,
          "compress_qk_rope_head_dim": 64,
          "compress_v_head_dim": 512,
          "o_groups": 8,
          "o_lora_rank": 256,
          "sliding_window": 128,
          "compress_rope_theta": 160000.0,
          "hca_rope_type": "yarn",
          "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1
          },
          "rope_theta": 8000000,
          "rotary_percent": 1.0,
          "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa",
          "hybrid_mla_num_attention_heads": 64,
          "hybrid_mla_num_key_value_heads": 64,
          "hybrid_mla_q_lora_rank": 256,
          "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192,
          "hybrid_mla_qk_rope_head_dim": 64,
          "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true,
          "index_n_heads": 64,
          "index_head_dim": 128,
          "index_topk": 2048,
          "use_fast_hadamard": true,
          "use_vha_attention": true,
          "add_full_attention_sink_bias": true,
          "use_attn_merger": true,
          "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true,
          "hc_mult": 4,
          "hc_sinkhorn_iters": 20,
          "hc_eps": 1e-6,
          "hidden_act": "situ",
          "activation_situ_beta": 4.0,
          "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1,
          "n_routed_experts": 32,
          "num_experts_per_tok": 4,
          "moe_intermediate_size": 64,
          "moe_latent_size": 64,
          "latent_moe_use_norm": true,
          "moe_split_feature_routing": true,
          "routed_scaling_factor": 3.0,
          "routed_scaling_factor_learnable": true,
          "n_shared_experts": 1,
          "scoring_func": "sigmoid",
          "topk_method": "noaux_tc",
          "multimax": "lm_head",
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try! QWModelDescriptor.decode(configJSON: cfg))
    }()

    static let ernieHal: QWHal = { try! QWHal(geometry: ernieGeometry) }()

    static let fixtureURL: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out/ernie55_synthetic_goldens.json")

    private func loadGoldens() throws -> [String: Any] {
        let data = try Data(contentsOf: Self.fixtureURL)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.ernieHal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.ernieHal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    private static func fwht32(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
            }
        }
    }

    private static func situRef(gate: Float, up: Float, beta: Float = 4.0, linBeta: Float = 25.0) -> Float {
        let sig = 1.0 / (1.0 + exp(-gate))
        let g = beta * tanh(gate / beta) * sig
        let u = linBeta * tanh(up / linBeta)
        return g * u
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32, scale: Float = 0.0625) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed = seedInit
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return bank
    }

    private func unpackSHVQRows(bits: Int, group: Int, n: Int, k: Int, bank: Data) -> [[Float]] {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        return (0..<n).map { r in
            var row = [Float]()
            row.reserveCapacity(k)
            var tmp = [Float]()
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                bank.subdata(in: off..<(off + gBytes)).withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWSHVQBlock.unpack(group: raw, bits: bits, group: group, into: &tmp)
                }
                row.append(contentsOf: tmp)
            }
            return row
        }
    }

    private func makeAQ8G32Bank(n: Int, k: Int, seedInit: UInt32, scale: Float = 0.002) -> Data {
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        var seed = seedInit
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = scale * (1.0 + 0.1 * Float((r + g) % 4))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-127.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<group {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return bank
    }

    private func unpackAQ8G32Rows(n: Int, k: Int, bank: Data) -> [[Float]] {
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        return (0..<n).map { r in
            var row = [Float]()
            row.reserveCapacity(k)
            var tmp = [Float]()
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                bank.subdata(in: off..<(off + gBytes)).withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWAQBlock.unpack(group: raw, bits: 8, group: group, into: &tmp)
                }
                row.append(contentsOf: tmp)
            }
            return row
        }
    }

    func testSituAndMulF32AndSplitF16MatchPythonGolden() throws {
        let root = try loadGoldens()
        let situ = try XCTUnwrap(root["situ"] as? [String: Any])
        let gateRows = try XCTUnwrap(situ["gate"] as? [[Double]])
        let upRows = try XCTUnwrap(situ["up"] as? [[Double]])
        let wantRows = try XCTUnwrap(situ["out"] as? [[Double]])
        let rows = gateRows.count
        let inter = gateRows[0].count

        var gu32Vals = [Float]()
        var gu16Vals = [Float16]()
        var wantFlat = [Float]()
        for r in 0..<rows {
            let gRow = gateRows[r].map { Float($0) }
            let uRow = upRows[r].map { Float($0) }
            gu32Vals.append(contentsOf: gRow)
            gu32Vals.append(contentsOf: uRow)
            gu16Vals.append(contentsOf: gRow.map { Float16($0) })
            gu16Vals.append(contentsOf: uRow.map { Float16($0) })
            wantFlat.append(contentsOf: wantRows[r].map { Float($0) })
        }

        let arena = try QWArena(hal: Self.ernieHal, weights: 0, activations: 65536, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        let gu32 = try ar.allocate(QWTensorShape([rows * 2 * inter], .f32))
        let act32 = try ar.allocate(QWTensorShape([rows * inter], .f32))
        let gu16 = try ar.allocate(QWTensorShape([rows * 2 * inter], .f16))
        let outHi = try ar.allocate(QWTensorShape([rows * inter], .f16))
        arena.write(gu32, from: gu32Vals)
        arena.write(gu16, from: gu16Vals)

        try step { enc in
            try QWGemm.encodeSiluMulF32(enc, hal: Self.ernieHal, gu: gu32, act: act32, rows: rows, intermediate: inter)
            try QWGemm.encodeSiluMulSplitF16(enc, hal: Self.ernieHal, gu: gu16, outHi: outHi, outLo: outHi,
                                             rows: rows, intermediate: inter)
        }

        var got32 = arena.read(act32, as: Float.self, count: rows * inter)
        Self.fwht32(&got32)
        for i in 0..<wantFlat.count {
            XCTAssertEqual(got32[i], wantFlat[i], accuracy: 2e-4, "SiTU F32 mismatch at \(i)")
        }

        var got16 = arena.read(outHi, as: Float16.self, count: rows * inter).map { Float($0) }
        Self.fwht32(&got16)
        for r in 0..<rows {
            for c in 0..<inter {
                let ref16 = Self.situRef(gate: Float(Float16(Float(gateRows[r][c]))),
                                         up: Float(Float16(Float(upRows[r][c]))))
                XCTAssertEqual(got16[r * inter + c], ref16, accuracy: 8e-2, "SiTU SplitF16 mismatch at (\(r),\(c))")
            }
        }
    }

    func testDualSplitSigmoidRouterAndLearnableScaleMatchPythonGolden() throws {
        let root = try loadGoldens()
        let router = try XCTUnwrap(root["router"] as? [String: Any])
        let xRows = try XCTUnwrap(router["x"] as? [[Double]]).map { $0.map { Float($0) } }
        let w0Rows = try XCTUnwrap(router["w0"] as? [[Double]]).map { $0.map { Float($0) } }
        let w1Rows = try XCTUnwrap(router["w1"] as? [[Double]]).map { $0.map { Float($0) } }
        let biasVals = try XCTUnwrap(router["bias"] as? [Double]).map { Float($0) }
        let scaleVals = try XCTUnwrap(router["scale"] as? [Double]).map { Float($0) }
        let wantWts = try XCTUnwrap(router["weights"] as? [[Double]]).map { $0.map { Float($0) } }
        let wantIds = try XCTUnwrap(router["ids"] as? [[Int]]).map { $0.map { Int32($0) } }

        let M = xRows.count, E = w0Rows.count, H = xRows[0].count, K = wantIds[0].count
        let wFlat = (w0Rows + w1Rows).flatMap { $0 }
        let xFlat = xRows.flatMap { $0 }

        let arena = try QWArena(hal: Self.ernieHal, weights: 65536, activations: 65536, kv: 0, gdnStates: 0)
        let wr = arena.weightsRegion(), ar = arena.activationsRegion()
        let wTV = try wr.allocate(QWTensorShape([2 * E * H], .f32))
        let biasTV = try wr.allocate(QWTensorShape([E], .f32))
        let scaleTV = try wr.allocate(QWTensorShape([E], .f32))
        arena.write(wTV, from: wFlat)
        arena.write(biasTV, from: biasVals)
        arena.write(scaleTV, from: scaleVals)

        let xTV = try ar.allocate(QWTensorShape([M * H], .f32))
        let logitsTV = try ar.allocate(QWTensorShape([M * 2 * E], .f32))
        let idx1 = try ar.allocate(QWTensorShape([K], .i32))
        let wts1 = try ar.allocate(QWTensorShape([K], .f32))
        let idxM = try ar.allocate(QWTensorShape([M * K], .i32))
        let wtsM = try ar.allocate(QWTensorShape([M * K], .f32))
        let idxBud = try ar.allocate(QWTensorShape([M * K], .i32))
        let wtsBud = try ar.allocate(QWTensorShape([M * K], .f32))
        arena.write(xTV, from: xFlat)

        try step { enc in
            try QWGemm.encodeFnRouterGemm(enc, hal: Self.ernieHal, weight: wTV, x: xTV, y: logitsTV,
                                          rows: M, n: 2 * E, k: H)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnMoeRoute(enc, hal: Self.ernieHal, logits: logitsTV, idx: idx1, wts: wts1,
                                        rows: 1, experts: E, topK: K, normTopK: true,
                                        bias: biasTV, splitFeature: true, routedScaleParam: scaleTV)
            try QWGemm.encodeFnMoeRoute(enc, hal: Self.ernieHal, logits: logitsTV, idx: idxM, wts: wtsM,
                                        rows: M, experts: E, topK: K, normTopK: true,
                                        bias: biasTV, splitFeature: true, routedScaleParam: scaleTV)
            try QWGemm.encodeFnMoeRoute(enc, hal: Self.ernieHal, logits: logitsTV, idx: idxBud, wts: wtsBud,
                                        rows: M, experts: E, topK: K, normTopK: true,
                                        bias: biasTV, expertBudget: 6, splitFeature: true,
                                        routedScaleParam: scaleTV)
        }

        let gotIds1 = arena.read(idx1, as: Int32.self, count: K)
        let gotWts1 = arena.read(wts1, as: Float.self, count: K)
        XCTAssertEqual(gotIds1, wantIds[0])
        for k in 0..<K {
            XCTAssertEqual(gotWts1[k], wantWts[0][k], accuracy: 1e-5, "T=1 weight[\(k)]")
        }

        let gotIdsM = arena.read(idxM, as: Int32.self, count: M * K)
        let gotWtsM = arena.read(wtsM, as: Float.self, count: M * K)
        for m in 0..<M {
            for k in 0..<K {
                XCTAssertEqual(gotIdsM[m * K + k], wantIds[m][k], "T=3 id[\(m),\(k)]")
                XCTAssertEqual(gotWtsM[m * K + k], wantWts[m][k], accuracy: 1e-5, "T=3 weight[\(m),\(k)]")
            }
        }

        let gotIdsBud = arena.read(idxBud, as: Int32.self, count: M * K)
        let gotWtsBud = arena.read(wtsBud, as: Float.self, count: M * K)
        XCTAssertEqual(Array(gotIdsBud[0..<K]), wantIds[0], "row 0 never pruned under budget=6")
        let uniqueExperts = Set(gotIdsBud.filter { $0 >= 0 })
        XCTAssertLessThanOrEqual(uniqueExperts.count, 6)
        for m in 0..<M {
            for k in 0..<K where gotIdsBud[m * K + k] < 0 {
                XCTAssertEqual(gotWtsBud[m * K + k], 0.0)
            }
        }
    }

    func testLatentMoEOperatorSHVQ15AndSHVQ3AndSharedExpertsParity() throws {
        let H = 256, L = 64, I = 64, ISh = 64, E = 32, K = 4, maxT = 64
        let fc1Data = makeAQ8G32Bank(n: L, k: H, seedInit: 0x1001)
        let fc2Data = makeAQ8G32Bank(n: H, k: L, seedInit: 0x1002)
        let shGUData = makeAQ8G32Bank(n: 2 * ISh, k: H, seedInit: 0x1003)
        let shDnData = makeAQ8G32Bank(n: H, k: ISh, seedInit: 0x1004)
        let guData = makeSHVQBank(bits: 15, group: 64, n: E * 2 * I, k: L, seedInit: 0x2001)
        let dnData = makeSHVQBank(bits: 3, group: 64, n: E * L, k: I, seedInit: 0x2002)

        let fc1Rows = unpackAQ8G32Rows(n: L, k: H, bank: fc1Data)
        let fc2Rows = unpackAQ8G32Rows(n: H, k: L, bank: fc2Data)
        let shGURows = unpackAQ8G32Rows(n: 2 * ISh, k: H, bank: shGUData)
        let shDnRows = unpackAQ8G32Rows(n: H, k: ISh, bank: shDnData)
        let guRows = unpackSHVQRows(bits: 15, group: 64, n: E * 2 * I, k: L, bank: guData)
        let dnRows = unpackSHVQRows(bits: 3, group: 64, n: E * L, k: I, bank: dnData)

        let routerW: [Float] = (0..<(2 * E * H)).map { i in sin(Float(i + 1) * 0.031) * 0.2 }
        let routerB: [Float] = (0..<E).map { i in cos(Float(i + 3) * 0.17) * 0.05 }
        let routeScale: [Float] = (0..<E).map { i in 2.5 + 0.5 * sin(Float(i + 2) * 0.23) }
        let rawLatentGamma: [Float] = (0..<L).map { i in Float(Float16(1.0 + 0.1 * sin(Float(i) * 0.19))) }
        let shiftedGamma16: [Float16] = rawLatentGamma.map { Float16($0 - 1.0) }

        let weightBytes = fc1Data.count + fc2Data.count + shGUData.count + shDnData.count
            + guData.count + dnData.count + (2 << 20)
        let arena = try QWArena(hal: Self.ernieHal, weights: weightBytes, activations: 16 << 20, kv: 0, gdnStates: 0)
        let wr = arena.weightsRegion(), ar = arena.activationsRegion()

        func loadView(_ d: Data, n: Int, k: Int, bits: Int, group: Int, isSHVQ: Bool) throws -> QWWeightView {
            let b = try wr.allocate(QWTensorShape([(d.count + 1) / 2], .f16))
            arena.copyIn(b, from: d)
            return try XCTUnwrap(QWWeightView(blocks: b, n: n, k: k, bits: bits, groupSize: group,
                                              isRotated: true, isIncumbentContainer: false, isSHVQ: isSHVQ))
        }
        let fc1View = try loadView(fc1Data, n: L, k: H, bits: 8, group: 32, isSHVQ: false)
        let fc2View = try loadView(fc2Data, n: H, k: L, bits: 8, group: 32, isSHVQ: false)
        let shGUView = try loadView(shGUData, n: 2 * ISh, k: H, bits: 8, group: 32, isSHVQ: false)
        let shDnView = try loadView(shDnData, n: H, k: ISh, bits: 8, group: 32, isSHVQ: false)
        let guView = try loadView(guData, n: E * 2 * I, k: L, bits: 15, group: 64, isSHVQ: true)
        let dnView = try loadView(dnData, n: E * L, k: I, bits: 3, group: 64, isSHVQ: true)

        let routerTV = try wr.allocate(QWTensorShape([2 * E * H], .f32))
        let biasTV = try wr.allocate(QWTensorShape([E], .f32))
        let scaleTV = try wr.allocate(QWTensorShape([E], .f32))
        let latGammaTV = try wr.allocate(QWTensorShape([L], .f16))
        arena.write(routerTV, from: routerW)
        arena.write(biasTV, from: routerB)
        arena.write(scaleTV, from: routeScale)
        arena.write(latGammaTV, from: shiftedGamma16)

        let latent = QWMoEOperator.LatentBottleneck(fc1: fc1View, normGamma: latGammaTV, normEps: 1e-6,
                                                    fc2: fc2View, latentDim: L)
        let weights = QWMoEOperator.Weights(
            router: routerTV, routerBias: biasTV, routedScaleParam: scaleTV, splitFeatureRouting: true,
            gateUp: guView, down: dnView, sharedGateUp: shGUView, sharedDown: shDnView, sharedGateScalar: nil,
            latent: latent, hidden: H, experts: E, topK: K, intermediate: I, sharedIntermediate: ISh,
            normTopK: true, reuseExpertsAcrossTokens: true
        )

        let tilePlan = QWGemm.MoETilePlan(experts: E, pairs: maxT * K, gateUpRows: 2 * I, downRows: L)
        let gateTV = try ar.allocate(QWTensorShape([maxT], .f32))
        arena.write(gateTV, from: [Float](repeating: 1.0, count: maxT))
        let scratch = QWMoEOperator.Scratch(
            xRot32: try ar.allocate(QWTensorShape([maxT * H], .f32)),
            xPlain32: try ar.allocate(QWTensorShape([maxT * H], .f32)),
            logits: try ar.allocate(QWTensorShape([maxT * 2 * E], .f32)),
            idx: try ar.allocate(QWTensorShape([maxT * K], .i32)),
            wts: try ar.allocate(QWTensorShape([maxT * K], .f32)),
            gu32: try ar.allocate(QWTensorShape([maxT * K * 2 * I], .f32)),
            act32: try ar.allocate(QWTensorShape([maxT * K * I], .f32)),
            guSh32: try ar.allocate(QWTensorShape([maxT * 2 * ISh], .f32)),
            actSh32: try ar.allocate(QWTensorShape([maxT * ISh], .f32)),
            shared: try ar.allocate(QWTensorShape([maxT * H], .f32)),
            gate: gateTV,
            expOff: try ar.allocate(QWTensorShape([E + 1], .i32)),
            sortedPair: try ar.allocate(QWTensorShape([maxT * K], .i32)),
            sortedTok: try ar.allocate(QWTensorShape([maxT * K], .i32)),
            invPerm: try ar.allocate(QWTensorShape([maxT * K], .i32)),
            tileE: try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32)),
            tileR: try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32)),
            tileCount: try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32)),
            gu16: try ar.allocate(QWTensorShape([maxT * K * 2 * I], .f16)),
            dnHi: try ar.allocate(QWTensorShape([maxT * K * I], .f16)),
            dnAcc: try ar.allocate(QWTensorShape([maxT * K * H], .f32)),
            naxX16: try ar.allocate(QWTensorShape([maxT * (H + ISh)], .f16)),
            naxY16: try ar.allocate(QWTensorShape([maxT * (H + ISh)], .f16)),
            latent: QWMoEOperator.LatentScratch(
                bufA32: try ar.allocate(QWTensorShape([maxT * L], .f32)),
                bufB32: try ar.allocate(QWTensorShape([maxT * L], .f32)),
                bufB16: try ar.allocate(QWTensorShape([maxT * L], .f16))
            ),
            maxTileWidth: H
        )

        let x16In = try ar.allocate(QWTensorShape([maxT * H], .f16))
        let out32 = try ar.allocate(QWTensorShape([maxT * H], .f32))
        let xRotVals: [Float] = (0..<(maxT * H)).map { i in Float(Float16(sin(Float(i + 5) * 0.07) * 0.4)) }
        arena.write(x16In, from: xRotVals.map { Float16($0) })

        func cpuRef(rows T: Int) -> [Float] {
            var out = [Float](repeating: 0, count: T * H)
            for t in 0..<T {
                let xRot = Array(xRotVals[(t * H)..<((t + 1) * H)])
                var xPlain = xRot
                Self.fwht32(&xPlain)

                var scores = [Float](repeating: 0, count: E)
                for e in 0..<E {
                    var d0: Float = 0, d1: Float = 0
                    for h in 0..<H {
                        d0 += routerW[e * H + h] * xPlain[h]
                        d1 += routerW[(E + e) * H + h] * xPlain[h]
                    }
                    scores[e] = (1.0 / (1.0 + exp(-d0))) + (1.0 / (1.0 + exp(-d1)))
                }
                let ranked = (0..<E).sorted { a, b in
                    let sa = scores[a] + routerB[a], sb = scores[b] + routerB[b]
                    return sa != sb ? sa > sb : a < b
                }
                let topIds = Array(ranked[0..<K])
                let sumS = max(topIds.reduce(Float(0)) { $0 + scores[$1] }, 1e-20)
                let topWts = topIds.map { (scores[$0] / sumS) * routeScale[$0] }

                var latIn = (0..<L).map { r in zip(fc1Rows[r], xRot).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
                Self.fwht32(&latIn)

                var latAcc = [Float](repeating: 0, count: L)
                for (rk, eid) in topIds.enumerated() {
                    let wt = topWts[rk]
                    var act = [Float](repeating: 0, count: I)
                    for i in 0..<I {
                        let g = zip(guRows[eid * 2 * I + i], latIn).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                        let u = zip(guRows[eid * 2 * I + I + i], latIn).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                        act[i] = Self.situRef(gate: g, up: u)
                    }
                    Self.fwht32(&act)
                    for l in 0..<L {
                        let d = zip(dnRows[eid * L + l], act).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                        latAcc[l] += wt * d
                    }
                }

                let meanSq = latAcc.reduce(Float(0)) { $0 + $1 * $1 } / Float(L)
                let rstd = 1.0 / (meanSq + 1e-6).squareRoot()
                var latNorm = (0..<L).map { l in latAcc[l] * rstd * (1.0 + Float(shiftedGamma16[l])) }
                Self.fwht32(&latNorm)

                var shAct = [Float](repeating: 0, count: ISh)
                for i in 0..<ISh {
                    let g = zip(shGURows[i], xRot).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    let u = zip(shGURows[ISh + i], xRot).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    shAct[i] = Self.situRef(gate: g, up: u)
                }
                Self.fwht32(&shAct)

                for h in 0..<H {
                    let fc2Val = zip(fc2Rows[h], latNorm).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    let shVal = zip(shDnRows[h], shAct).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    out[t * H + h] = fc2Val + shVal
                }
            }
            return out
        }

        func runMoE(rows T: Int, groupedNax: Bool) throws -> [Float] {
            guard let pair = Self.ernieHal.begin() else { throw XCTSkip("no command buffer") }
            let cb = pair.0
            var enc = pair.1
            try QWMoEOperator.encodeResident(
                &enc, hal: Self.ernieHal, weights: weights, scratch: scratch,
                input: .f16Normed(x16In.slice(byteOffset: 0, count: T * H, dtype: .f16)),
                out32: out32.slice(byteOffset: 0, count: T * H, dtype: .f32),
                rows: T, supportsGroupedNaxPrefill: groupedNax, concurrentBarriers: true,
                rotateEncoder: { innerEnc in try Self.ernieHal.rotateEncoder(cb, &innerEnc) }
            )
            Self.ernieHal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            return arena.read(out32, as: Float.self, count: T * H)
        }

        for T in [1, 3, 64] {
            let ref = cpuRef(rows: T)
            let gotF32 = try runMoE(rows: T, groupedNax: false)
            var diffSq = 0.0, refSq = 0.0
            for i in 0..<ref.count {
                let d = Double(gotF32[i]) - Double(ref[i])
                diffSq += d * d
                refSq += Double(ref[i]) * Double(ref[i])
            }
            let relF32 = (diffSq / max(refSq, 1e-30)).squareRoot()
            XCTAssertLessThan(relF32, 1e-4, "Latent-MoE F32 rel error T=\(T): \(relF32)")

            if T == 64 && Self.ernieHal.caps.isMetal4TensorCapable {
                let gotNax = try runMoE(rows: T, groupedNax: true)
                var naxDiffSq = 0.0
                for i in 0..<ref.count {
                    let d = Double(gotNax[i]) - Double(ref[i])
                    naxDiffSq += d * d
                }
                let relNax = (naxDiffSq / max(refSq, 1e-30)).squareRoot()
                XCTAssertLessThan(relNax, 8e-3, "Latent-MoE Grouped NAX rel error T=64: \(relNax)")
            }
        }
    }
}

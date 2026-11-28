// QWTrunkDecisionEvalTests.swift — Validation & sparse/full parity tests for trunk decision logit extraction.
import CryptoKit
import Foundation
import XCTest
@testable import QWCore
import QWDecision
@testable import QWEngine
import QWHAL
import QWHead
import QWKV

private final class MinimalStubTrunk: QWTrunkExecutor, @unchecked Sendable {
    var hasPrefixCacheTier: Bool = false
    var arenaBytes: Int = 0
    var weightReservedBytes: Int = 0
    var weightUsedBytes: Int = 0
    var activeCapacity: Int = 64
    var steadyStateAllocations: Int = 0
    var dflashDrafter: QWDFlashDrafter? = nil
    var prefixCacheState: QWPrefixCacheState = QWPrefixCacheState(isConfigured: false)
    var lastResult: QWGenerationResult? = nil
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? = nil
    var gpuTurnArbitrator: QWGPUTurnArbitrator? = nil

    func hasCachedPrefix(tokens _: [Int], cacheSalt _: String?) -> Bool { false }
    func buildCapabilities() -> Set<QWKnobCapability> { [] }
    func buildReport() -> QWEngineReport {
        QWEngineReport(
            capacityLog: "", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0,
            steadyAllocations: 0, pipelineSeconds: 0, detailLines: "", weightFormat: "HQ4_G32"
        )
    }
    func runGeneration(
        request _: QWGenerationRequest,
        onToken _: ((Int) -> Bool)?,
        onStopToken _: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
    }
    func configurePrefixCache(_: QWCacheConfig) throws {}
    func buildStandalonePrefixCache(
        tokens _: [Int], outputURL _: URL, pinAsSharedBase _: Bool,
        onPrefillProgress _: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int { 0 }
    func mountStandalonePrefixCache(from _: URL) throws -> (tokens: [Int], byteCount: Int) { ([], 0) }
    func unmountSharedBasePrefixes() {}
    func computeEchoPromptLogprobs(
        promptTokens _: [Int], sampling _: QWSamplerConfig,
        topLogprobsCount _: Int, shouldStop _: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] { [] }
}

final class QWTrunkDecisionEvalTests: XCTestCase {
    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case hq4(shape: [Int], n: Int, k: Int, seed: UInt32, scale: Float, experts: Int? = nil, eHq4: Int? = nil)
    }

    private struct PackConfig {
        let vocab: Int = 128
        let hidden: Int = 128
        let numLayers: Int = 3
        let qHeads: Int = 4
        let kvHeads: Int = 2
        let headDim: Int = 128
        let dRel: Int = 8
        let relExtent: Int = 32
        let slidingWindow: Int = 16
        let localLayerIds: Set<Int> = [0]
        let loopCoreStart: Int = 1
        let loopCoreEnd: Int = 3
        let numPasses: Int = 2
        let gateRank: Int = 16
        let numExperts: Int = 4
        let topK: Int = 2
        let moeInter: Int = 64
    }

    private func requireMetal4Hardware() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Inkling-2 decision eval tests require bare-metal Apple Silicon Metal 4 hardware"
        )
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detVec(_ count: Int, seedInit: UInt64, amp: Float, offset: Float = 0) -> [Float] {
        var rng = seedInit &+ 0x9E37_79B9_7F4A_7C15
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float((rng >> 33) & 0xFFFF) / 32768.0 - 1.0
            out[i] = offset + u * amp
        }
        return out
    }

    private static func bf16Bits(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
        var seed = seedInit
        var out = Data(count: n * groups * 20)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * 20
                    let scVal = Float(Float16(scale * (1.0 + 0.1 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1031.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for i in 0..<16 {
                        seed = seed &* 1_664_525 &+ 1_013_904_223
                        ptr[base + 4 + i] = UInt8((seed >> 24) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func writeQWP1(
        at rootDir: URL,
        provenance: [String: Any],
        planes: [(String, PlaneSpec)]
    ) throws {
        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 60)
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    var le = bf16Bits(val).littleEndian
                    withUnsafeBytes(of: &le) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 4)
                for val in vals {
                    var le = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &le) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .hq4(let shape, let n, let k, let seed, let scale, let experts, let eHq4):
                padTo(16_384)
                let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                var meta: [String: Any] = ["shape": shape, "hq4_off": off]
                if let experts { meta["experts"] = experts }
                if let eHq4 { meta["e_hq4"] = eHq4 }
                entries[name] = meta
            }
        }
        let hdrData = try JSONSerialization.data(
            withJSONObject: ["version": 2, "provenance": provenance, "tensors": entries],
            options: [.sortedKeys]
        )
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (buf: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: buf) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    private func buildSyntheticInkling2Pack(at rootDir: URL, maxPos: Int = 128) throws {
        let c = PackConfig()
        let qDim = c.qHeads * c.headDim
        let kvDim = c.kvHeads * c.headDim
        let rDim = c.qHeads * c.dRel
        let msgLayers = c.loopCoreEnd - c.loopCoreStart
        let cfgDict: [String: Any] = [
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "model_type": "inkling2_loop",
            "eos_token_id": c.vocab - 1,
            "generation_config": ["eos_token_id": c.vocab - 1, "pad_token_id": 0],
            "text_config": [
                "model_type": "inkling2_loop_text",
                "vocab_size": c.vocab,
                "unpadded_vocab_size": c.vocab,
                "hidden_size": c.hidden,
                "num_hidden_layers": c.numLayers,
                "num_attention_heads": c.qHeads,
                "num_key_value_heads": c.kvHeads,
                "head_dim": c.headDim,
                "d_rel": c.dRel,
                "rel_extent": c.relExtent,
                "q_bias": false,
                "o_bias": false,
                "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.25,
                "rms_norm_eps": 1e-6,
                "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.627416997969522,
                "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": Array(c.localLayerIds).sorted(),
                "sconv_kernel_size": 4,
                "final_logit_softcapping": 30.0,
                "swa_head_dim": c.headDim,
                "swa_num_attention_heads": c.qHeads,
                "swa_num_key_value_heads": c.kvHeads,
                "sliding_window_size": c.slidingWindow,
                "n_layers_per_attn_res_block": 1,
                "attn_res_normalize_values": true,
                "n_routed_experts": c.numExperts,
                "num_experts_per_tok": c.topK,
                "n_shared_experts": 0,
                "intermediate_size": c.moeInter,
                "moe_intermediate_size": c.moeInter,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
                "route_scale": 4.0,
                "use_gate_bias": true,
                "max_position_embeddings": maxPos,
                "rope_theta": 1_000_000.0,
                "loop_config": [
                    "core_start": c.loopCoreStart,
                    "core_end": c.loopCoreEnd,
                    "num_passes": c.numPasses,
                    "gate_rank": c.gateRank,
                    "is_varloop": false,
                ],
            ],
        ]
        let cfgJSON = try JSONSerialization.data(withJSONObject: cfgDict, options: [.sortedKeys])
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)
        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": Self.sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [c.vocab, c.hidden], values: Self.detVec(c.vocab * c.hidden, seedInit: 11, amp: 0.25))),
            ("model.embed_norm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(shape: [c.vocab, c.hidden], n: c.vocab, k: c.hidden, seed: 13, scale: 0.02)),
            ("model.norm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: 14, amp: 0.05, offset: 1.0))),
            ("model.final_input.query", .f32(shape: [1, c.hidden], values: Self.detVec(c.hidden, seedInit: 15, amp: 0.1))),
            ("model.message_queries", .f32(shape: [msgLayers, 4, c.hidden], values: Self.detVec(msgLayers * 4 * c.hidden, seedInit: 16, amp: 0.1))),
        ]
        let guPerExpBytes = (c.hidden / 32) * (2 * c.moeInter) * 20
        let downPerExpBytes = (c.moeInter / 32) * c.hidden * 20
        for l in 0..<c.numLayers {
            let lp = "model.layers.\(l)."
            let s = UInt64((l + 1) * 1000)
            let su = UInt32((l + 1) * 1000)
            let isSliding = c.localLayerIds.contains(l)
            if l > 0 {
                let qRows = isSliding ? 1 : 3
                planes.append(("\(lp)attention_input.query", .f32(shape: [qRows, c.hidden], values: Self.detVec(qRows * c.hidden, seedInit: s + 1, amp: 0.1))))
            }
            planes.append(("\(lp)mlp_input.query", .f32(shape: [1, c.hidden], values: Self.detVec(c.hidden, seedInit: s + 2, amp: 0.1))))
            if isSliding {
                planes.append(("\(lp)input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 3, amp: 0.05, offset: 1.0))))
            } else {
                planes.append(("\(lp)q_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 4, amp: 0.05, offset: 1.0))))
                planes.append(("\(lp)k_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 5, amp: 0.05, offset: 1.0))))
                planes.append(("\(lp)v_input_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 6, amp: 0.05, offset: 1.0))))
            }
            planes.append(("\(lp)post_attention_layernorm.weight", .bf16(shape: [c.hidden], values: Self.detVec(c.hidden, seedInit: s + 7, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)attn_sconv.conv.weight", .bf16(shape: [c.hidden, 4], values: Self.detVec(c.hidden * 4, seedInit: s + 8, amp: 0.1))))
            planes.append(("\(lp)mlp_sconv.conv.weight", .bf16(shape: [c.hidden, 4], values: Self.detVec(c.hidden * 4, seedInit: s + 9, amp: 0.1))))
            planes.append(("\(lp)self_attn.q_proj.weight", .hq4(shape: [qDim, c.hidden], n: qDim, k: c.hidden, seed: su + 10, scale: 0.008)))
            planes.append(("\(lp)self_attn.gate_proj.weight", .hq4(shape: [qDim, c.hidden], n: qDim, k: c.hidden, seed: su + 11, scale: 0.008)))
            planes.append(("\(lp)self_attn.r_proj.weight", .hq4(shape: [rDim, c.hidden], n: rDim, k: c.hidden, seed: su + 12, scale: 0.008)))
            planes.append(("\(lp)self_attn.sink_proj.weight", .hq4(shape: [c.qHeads, c.hidden], n: c.qHeads, k: c.hidden, seed: su + 13, scale: 0.008)))
            planes.append(("\(lp)self_attn.k_proj.weight", .hq4(shape: [kvDim, c.hidden], n: kvDim, k: c.hidden, seed: su + 14, scale: 0.008)))
            planes.append(("\(lp)self_attn.v_proj.weight", .hq4(shape: [kvDim, c.hidden], n: kvDim, k: c.hidden, seed: su + 15, scale: 0.008)))
            planes.append(("\(lp)self_attn.o_proj.weight", .hq4(shape: [c.hidden, qDim], n: c.hidden, k: qDim, seed: su + 16, scale: 0.008)))
            planes.append(("\(lp)self_attn.q_scale", .bf16(shape: [qDim], values: Self.detVec(qDim, seedInit: s + 17, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)self_attn.rel_scale", .bf16(shape: [rDim], values: Self.detVec(rDim, seedInit: s + 18, amp: 0.05, offset: 1.0))))
            planes.append(("\(lp)self_attn.rel_proj", .bf16(shape: [c.dRel, c.relExtent], values: Self.detVec(c.dRel * c.relExtent, seedInit: s + 19, amp: 0.05))))
            if isSliding {
                planes.append(("\(lp)self_attn.k_sconv.conv.weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 20, amp: 0.1))))
                planes.append(("\(lp)self_attn.v_sconv.conv.weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 21, amp: 0.1))))
            } else {
                planes.append(("\(lp)self_attn.k_sconv.static_weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 22, amp: 0.1))))
                planes.append(("\(lp)self_attn.v_sconv.static_weight", .bf16(shape: [kvDim, 4], values: Self.detVec(kvDim * 4, seedInit: s + 23, amp: 0.1))))
                planes.append(("\(lp)self_attn.k_sconv.weight_proj.weight", .bf16(shape: [c.kvHeads * 4, c.hidden], values: Self.detVec(c.kvHeads * 4 * c.hidden, seedInit: s + 24, amp: 0.08))))
                planes.append(("\(lp)self_attn.v_sconv.weight_proj.weight", .bf16(shape: [c.kvHeads * 4, c.hidden], values: Self.detVec(c.kvHeads * 4 * c.hidden, seedInit: s + 25, amp: 0.08))))
            }
            if l >= c.loopCoreStart && l < c.loopCoreEnd {
                for branch in ["attn", "k", "v", "mlp"] {
                    let bSeed = s &+ UInt64(branch.utf8.reduce(0, { $0 + Int($1) }) * 17)
                    planes.append(("\(lp)controllers.\(branch).down_weight", .bf16(shape: [c.gateRank, 2 * c.hidden], values: Self.detVec(c.gateRank * 2 * c.hidden, seedInit: bSeed + 1, amp: 0.05))))
                    planes.append(("\(lp)controllers.\(branch).forget_weight", .bf16(shape: [c.hidden, c.gateRank], values: Self.detVec(c.hidden * c.gateRank, seedInit: bSeed + 2, amp: 0.05))))
                    planes.append(("\(lp)controllers.\(branch).gain_weight", .bf16(shape: [c.hidden, c.gateRank], values: Self.detVec(c.hidden * c.gateRank, seedInit: bSeed + 3, amp: 0.05))))
                }
            }
            planes.append(("\(lp)mlp.gate_weight", .bf16(shape: [c.numExperts, c.hidden], values: Self.detVec(c.numExperts * c.hidden, seedInit: s + 30, amp: 0.1))))
            planes.append(("\(lp)mlp.e_score_correction_bias", .f32(shape: [c.numExperts], values: Self.detVec(c.numExperts, seedInit: s + 31, amp: 0.02))))
            planes.append(("\(lp)mlp.experts.gate_up_proj", .hq4(shape: [c.numExperts, 2 * c.moeInter, c.hidden], n: c.numExperts * 2 * c.moeInter, k: c.hidden, seed: su + 32, scale: 0.008, experts: c.numExperts, eHq4: guPerExpBytes)))
            planes.append(("\(lp)mlp.experts.down_proj", .hq4(shape: [c.numExperts, c.hidden, c.moeInter], n: c.numExperts * c.hidden, k: c.moeInter, seed: su + 33, scale: 0.008, experts: c.numExperts, eHq4: downPerExpBytes)))
        }
        try Self.writeQWP1(
            at: rootDir,
            provenance: [
                "model_id": "inkling2_loop",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-decision-eval-test",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": Self.sha256Hex(mfData),
            ],
            planes: planes
        )
    }

    func testTrunkExecutorDefaultThrowsAndSupportExtractsSparseAndFullParity() throws {
        let stub: any QWTrunkExecutor = MinimalStubTrunk()
        XCTAssertThrowsError(
            try stub.evaluateDecisionLogits(tokens: [1, 2], positions: [1], candidateTokenIds: [0], visualSpans: [])
        ) { err in
            guard case QWDecisionError.backendFailed = err else {
                return XCTFail("expected backendFailed, got \(err)")
            }
        }

        let row: [Float] = [0.25, -1.5, 3.0, 0.0, 2.75]
        let candidates = [4, 1, 4, -1, 5]
        row.withUnsafeBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return XCTFail("missing baseAddress") }
            let full = QWTrunkRuntimeSupport.extractDecisionSlotOutput(logitsRow: ptr, vocab: row.count, candidateTokenIds: [])
            let sparse = QWTrunkRuntimeSupport.extractDecisionSlotOutput(logitsRow: ptr, vocab: row.count, candidateTokenIds: candidates)
            XCTAssertEqual(full.fullLogits, row)
            XCTAssertTrue(full.tokenLogits.isEmpty)
            XCTAssertNil(sparse.fullLogits)
            XCTAssertEqual(sparse.tokenLogits, [1: row[1], 4: row[4]])
        }
    }

    func testInkling2DecisionLogitValidationAndSparseFullParity() throws {
        try requireMetal4Hardware()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ink2-decision-eval-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try buildSyntheticInkling2Pack(at: dir, maxPos: 128)

        let runtime = try QWInkling2Runtime(
            packPath: dir.path,
            maxTokens: 64,
            kvMode: .f16,
            rawKVMode: "f16",
            osReserveGiB: 1.0,
            prewarmDoors: [],
            prefixCacheWarmByteLimit: 0,
            maxConcurrentSessions: 1,
            prefillChunkSize: 32
        )

        XCTAssertThrowsError(
            try runtime.evaluateDecisionLogits(tokens: [], positions: [0], candidateTokenIds: [], visualSpans: [])
        ) { err in
            guard case QWDecisionError.invalidQuestion = err else {
                return XCTFail("expected invalidQuestion for empty tokens, got \(err)")
            }
        }
        XCTAssertEqual(
            try runtime.evaluateDecisionLogits(tokens: [1, 2, 3], positions: [], candidateTokenIds: [0, 1], visualSpans: []),
            []
        )
        for badPos in [-1, 3] {
            XCTAssertThrowsError(
                try runtime.evaluateDecisionLogits(tokens: [1, 2, 3], positions: [badPos], candidateTokenIds: [], visualSpans: [])
            ) { err in
                guard case QWDecisionError.invalidQuestion = err else {
                    return XCTFail("expected invalidQuestion for position \(badPos), got \(err)")
                }
            }
        }
        for badTok in [-1, runtime.descriptor.vocab] {
            XCTAssertThrowsError(
                try runtime.evaluateDecisionLogits(tokens: [1, badTok, 2], positions: [0], candidateTokenIds: [], visualSpans: [])
            ) { err in
                XCTAssertEqual(err as? QWEngineDriverError, .badToken(badTok))
            }
        }
        let overflowTokens = Array(repeating: 1, count: runtime.capacity + 1)
        XCTAssertThrowsError(
            try runtime.evaluateDecisionLogits(tokens: overflowTokens, positions: [0], candidateTokenIds: [], visualSpans: [])
        ) { err in
            XCTAssertEqual(err as? QWDecisionError, .contextOverflow(tokens: overflowTokens.count, limit: runtime.capacity))
        }

        let genPrompt = [3, 11, 27, 45, 89, 12, 64, 7]
        let baselineGen = try runtime.runGeneration(
            request: QWGenerationRequest(prompt: genPrompt, maxNew: 4, sampling: .greedy),
            onToken: nil, onStopToken: nil
        )

        let prompt = (0..<40).map { ($0 * 7 + 3) % runtime.descriptor.vocab }
        let positions = [35, 5, 35, 0, 12]
        let fullOutputs = try runtime.evaluateDecisionLogits(
            tokens: prompt, positions: positions, candidateTokenIds: [], visualSpans: []
        )
        let candidateIds = [0, 17, 63, 127, -1, runtime.ink2.unpaddedVocabSize]
        let sparseOutputs = try runtime.evaluateDecisionLogits(
            tokens: prompt, positions: positions, candidateTokenIds: candidateIds, visualSpans: []
        )

        XCTAssertEqual(fullOutputs.count, positions.count)
        XCTAssertEqual(sparseOutputs.count, positions.count)
        XCTAssertEqual(fullOutputs[0], fullOutputs[2])
        XCTAssertEqual(sparseOutputs[0], sparseOutputs[2])
        XCTAssertNotEqual(fullOutputs[0], fullOutputs[1])

        let validCandidates = [0, 17, 63, 127]
        for idx in positions.indices {
            let fullLogits = try XCTUnwrap(fullOutputs[idx].fullLogits)
            XCTAssertEqual(fullLogits.count, runtime.ink2.unpaddedVocabSize)
            XCTAssertTrue(fullLogits.allSatisfy(\.isFinite))
            XCTAssertNil(sparseOutputs[idx].fullLogits)
            XCTAssertEqual(Set(sparseOutputs[idx].tokenLogits.keys), Set(validCandidates))
            for tid in validCandidates {
                XCTAssertEqual(sparseOutputs[idx].tokenLogits[tid], fullLogits[tid])
            }
        }

        let afterGen = try runtime.runGeneration(
            request: QWGenerationRequest(prompt: genPrompt, maxNew: 4, sampling: .greedy),
            onToken: nil, onStopToken: nil
        )
        XCTAssertEqual(afterGen.tokens, baselineGen.tokens)
    }

    func testHyperStreamDecisionEvalValidationAndEmptyCandidateShortCircuit() throws {
        XCTAssertEqual(FNGPUBatchScheduler.maxSlots, 4)
        XCTAssertThrowsError(
            try QWHyperStreamRuntime.prepareDecisionEvaluation(
                tokens: [], positions: [0], candidateTokenIds: [], vocab: 64, activeCapacity: 64
            )
        ) { err in
            guard case QWDecisionError.invalidQuestion = err else {
                return XCTFail("expected invalidQuestion for empty tokens, got \(err)")
            }
        }
        XCTAssertEqual(
            try QWHyperStreamRuntime.prepareDecisionEvaluation(
                tokens: [1, 2, 3], positions: [], candidateTokenIds: [0, 1], vocab: 64, activeCapacity: 64
            ),
            []
        )
        let maxFit = FNGPUDFlash.maxFittingContext(activeCapacity: 16)
        XCTAssertThrowsError(
            try QWHyperStreamRuntime.prepareDecisionEvaluation(
                tokens: Array(repeating: 1, count: maxFit + 1),
                positions: [0], candidateTokenIds: [], vocab: 64, activeCapacity: 16
            )
        ) { err in
            XCTAssertEqual(err as? QWDecisionError, .contextOverflow(tokens: maxFit + 1, limit: maxFit))
        }
        for badTok in [-1, 64] {
            XCTAssertThrowsError(
                try QWHyperStreamRuntime.prepareDecisionEvaluation(
                    tokens: [1, badTok, 2], positions: [0], candidateTokenIds: [], vocab: 64, activeCapacity: 64
                )
            ) { err in
                XCTAssertEqual(err as? QWEngineDriverError, .badToken(badTok))
            }
        }
        for badPos in [-1, 3] {
            XCTAssertThrowsError(
                try QWHyperStreamRuntime.prepareDecisionEvaluation(
                    tokens: [1, 2, 3], positions: [badPos], candidateTokenIds: [], vocab: 64, activeCapacity: 64
                )
            ) { err in
                guard case QWDecisionError.invalidQuestion = err else {
                    return XCTFail("expected invalidQuestion for position \(badPos), got \(err)")
                }
            }
        }
        let shortCircuited = try QWHyperStreamRuntime.prepareDecisionEvaluation(
            tokens: [1, 2, 3], positions: [0, 2], candidateTokenIds: [-1, 64, 128], vocab: 64, activeCapacity: 64
        )
        XCTAssertEqual(shortCircuited, [QWDecisionSlotOutput(tokenLogits: [:]), QWDecisionSlotOutput(tokenLogits: [:])])
        XCTAssertNil(
            try QWHyperStreamRuntime.prepareDecisionEvaluation(
                tokens: [1, 2, 3], positions: [0, 2], candidateTokenIds: [63, 999], vocab: 64, activeCapacity: 64
            )
        )
    }

    func testBatchSchedulerCoalescesMaxBatchSlots4TrunkAndForwardsMountExclusive() throws {
        final class StubHyperTrunkEngine: QWTrunkExecutor, QWInferenceEngine, @unchecked Sendable {
            var hasPrefixCacheTier: Bool = false
            var arenaBytes: Int = 0
            var weightReservedBytes: Int = 0
            var weightUsedBytes: Int = 0
            var activeCapacity: Int = 128
            var maxBatchSlots: Int { FNGPUBatchScheduler.maxSlots }
            var steadyStateAllocations: Int = 0
            var dflashDrafter: QWDFlashDrafter? = nil
            var prefixCacheState: QWPrefixCacheState = QWPrefixCacheState(isConfigured: false)
            var lastResult: QWGenerationResult? = nil
            var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? = nil
            var gpuTurnArbitrator: QWGPUTurnArbitrator? = nil
            var capabilities: Set<QWKnobCapability> { [] }
            var hasPrefixCache: Bool { hasPrefixCacheTier }
            var prefixCache: QWPrefixCacheState { prefixCacheState }
            var report: QWEngineReport {
                QWEngineReport(capacityLog: "", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0)
            }
            var lastTurn: QWGenerationResult? { lastResult }
            var supportsTokenConstraints: Bool { false }

            private let lock = NSLock()
            var soloCalls = 0
            var waveSizes: [Int] = []
            var mountedURLs: [URL] = []
            var unmountCount = 0

            func hasCachedPrefix(tokens _: [Int], cacheSalt _: String?) -> Bool { false }
            func buildCapabilities() -> Set<QWKnobCapability> { [] }
            func buildReport() -> QWEngineReport { report }
            func makeConversation() -> QWConversation? { nil }
            func configurePrefixCache(_: QWCacheConfig) throws {}
            func buildStandalonePrefixCache(
                tokens _: [Int], outputURL _: URL, pinAsSharedBase _: Bool,
                onPrefillProgress _: (@Sendable (Int, Int) -> Void)?
            ) throws -> Int { 0 }
            func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
                lock.withLock { mountedURLs.append(fileURL) }
                return ([1, 2], 128)
            }
            func unmountSharedBasePrefixes() {
                lock.withLock { unmountCount += 1 }
            }
            func computeEchoPromptLogprobs(
                promptTokens _: [Int], sampling _: QWSamplerConfig,
                topLogprobsCount _: Int, shouldStop _: (@Sendable () -> Bool)?
            ) throws -> [QWTokenLogprob] { [] }

            func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
                try runGeneration(request: request, onToken: onToken, onStopToken: nil)
            }

            func runGeneration(
                request: QWGenerationRequest,
                onToken: ((Int) -> Bool)?,
                onStopToken _: ((Int) -> Void)?
            ) throws -> QWGenerationResult {
                lock.withLock { soloCalls += 1 }
                usleep(1000)
                let out = (0..<request.maxNew).map { (request.prompt.first ?? 0) + $0 }
                for tok in out { _ = onToken?(tok) }
                return QWGenerationResult(tokens: out, stepMs: [], stepTokenCounts: [])
            }

            func runScheduledBatch(
                initialJobs: [QWBatchScheduledJob],
                pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
                requeueDelayed _: ([QWBatchScheduledJob]) -> Void
            ) {
                lock.withLock { waveSizes.append(initialJobs.count) }
                var jobs = initialJobs
                if jobs.count < maxBatchSlots {
                    jobs.append(contentsOf: pollPending(maxBatchSlots - jobs.count))
                }
                for job in jobs {
                    let out = (0..<job.request.maxNew).map { (job.request.prompt.first ?? 0) + $0 }
                    job.complete(.success(QWGenerationResult(tokens: out, stepMs: [], stepTokenCounts: [])))
                }
            }
        }

        let trunk = StubHyperTrunkEngine()
        let scheduler = QWBatchScheduler(engine: trunk, trunkExecutor: trunk)
        XCTAssertThrowsError(
            try scheduler.generate(request: QWGenerationRequest(prompt: [1], maxNew: 2, sampling: .greedy, usePrefixCache: true))
        )
        let solo = try scheduler.generate(request: QWGenerationRequest(prompt: [10], maxNew: 3, sampling: .greedy))
        XCTAssertEqual(solo.tokens, [10, 11, 12])
        XCTAssertEqual(trunk.soloCalls, 1)
        XCTAssertTrue(trunk.waveSizes.isEmpty)

        final class BurstBox: @unchecked Sendable {
            private let lock = NSLock()
            private var values: [Int: [Int]] = [:]
            func set(_ idx: Int, _ tokens: [Int]) { lock.withLock { values[idx] = tokens } }
            func get(_ idx: Int) -> [Int]? { lock.withLock { values[idx] } }
        }
        let burstOutputs = BurstBox()
        let group = DispatchGroup()
        QWBatchAdmissionTracker.beginPreparing()
        for idx in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                let req = QWGenerationRequest(prompt: [20 + idx * 10], maxNew: 2, sampling: .greedy)
                if let res = try? scheduler.generate(request: req) {
                    burstOutputs.set(idx, res.tokens)
                }
            }
        }
        usleep(2000)
        QWBatchAdmissionTracker.endPreparing()
        group.wait()

        XCTAssertGreaterThanOrEqual(scheduler.peakCoalescedWaveSize, 2)
        XCTAssertTrue(trunk.waveSizes.contains { $0 >= 2 && $0 <= 4 })
        for idx in 0..<4 {
            XCTAssertEqual(burstOutputs.get(idx), [20 + idx * 10, 21 + idx * 10])
        }
        let fakeURL = URL(fileURLWithPath: "/tmp/hyperstream-mount-test.qwcache")
        let mounted = try scheduler.mountStandalonePrefixCache(from: fakeURL)
        XCTAssertEqual(mounted.tokens, [1, 2])
        XCTAssertEqual(mounted.byteCount, 128)
        scheduler.unmountSharedBasePrefixes()
        XCTAssertEqual(trunk.mountedURLs, [fakeURL])
        XCTAssertEqual(trunk.unmountCount, 1)
    }
}

// QWMTPNextNParityTests.swift — Bit-exact parity for NextN hybrid PLD + commit pair seeding.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWKV
@testable import QWEngine

final class QWMTPNextNParityTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "MTP NextN GPU parity tests require bare-metal Apple Silicon GPU"
        )
    }

    private enum PlaneSpec {
        case f32(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float, isAQ4: Bool = false)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detVec(_ count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float, isAQ4: Bool = false) -> Data {
        let group = 32
        let groups = k / group
        let gBytes = 20
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = Float(Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16((isAQ4 ? -7.5 : -1031.5) * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func buildSyntheticNextNPack(at rootDir: URL, includeMTP: Bool = true) throws {
        let h = 256
        let v = 256
        let inter = 256
        let qHeads = 4
        let kvHeads = 2
        let headDim = 64

        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "mtp_num_hidden_layers": \(includeMTP ? 1 : 0),
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": 2048,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
          "layer_types": ["full_attention"],
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!

        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let qRows = g.qProjOut
        let kvRows = g.kvHeads * g.headDim
        let oCols = g.qHeads * g.headDim

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
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02)),
        ]

        func appendAttnAndMLP(_ blk: String, seedBase: UInt32) {
            planes.append((blk + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase, amp: 0.05, offset: 1.0))))
            planes.append((blk + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase + 1, amp: 0.05, offset: 1.0))))
            planes.append((blk + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: seedBase + 2, scale: 0.008)))
            planes.append((blk + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 3, scale: 0.008)))
            planes.append((blk + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 4, scale: 0.008)))
            planes.append((blk + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: seedBase + 5, scale: 0.008)))
            planes.append((blk + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 6, scale: 0.008)))
            planes.append((blk + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 7, scale: 0.008)))
            planes.append((blk + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: seedBase + 8, scale: 0.008)))
        }

        appendAttnAndMLP("model.layers.0.", seedBase: 100)
        if includeMTP {
            planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 200, scale: 0.006)))
            planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            appendAttnAndMLP("mtp.layers.0.", seedBase: 300)
        }

        try writeQWP(at: rootDir, descriptor: descriptor, configJSON: cfgJSON, planes: planes, modelID: "qwen3_5")
    }

    private static func writeQWP(
        at rootDir: URL,
        descriptor: QWModelDescriptor,
        configJSON: Data,
        planes: [(String, PlaneSpec)],
        modelID: String,
        useHQBlock: Bool = false
    ) throws {
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        try configJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)

        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var file = Data()
        file.append(Data("QWP1".utf8))
        file.append(Data(repeating: 0, count: 8))
        var entries: [String: [String: Any]] = [:]

        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        padTo(256)

        for (name, spec) in planes {
            switch spec {
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 4)
                for val in vals {
                    var le = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &le) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale, let isAQ4):
                padTo(16_384)
                let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale, isAQ4: isAQ4))
                padTo(16_384)
                var meta: [String: Any] = ["shape": [n, k], "hq4_off": off]
                if isAQ4 { meta["hq"] = ["bits": 4, "group": 32] }
                entries[name] = meta
            }
        }

        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": modelID,
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-nextn-parity",
                "sha256_source": "synthetic",
                "quant_recorded": useHQBlock ? "hq4_g32" : "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (raw: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: raw) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    private static func buildSyntheticSenseNovaLoopPack(at rootDir: URL, useHQBlock: Bool = false) throws {
        let h = 256, v = 256, inter = 256
        let qHeads = 16, kvHeads = 4, headDim = useHQBlock ? 128 : 256
        let gdnKHeads = 2, gdnVHeads = 4, gdnKDim = 128, gdnVDim = 128, convK = 4
        let cfgJSON = """
        {
          "architectures": ["SenseNovaLoopForConditionalGeneration"],
          "model_type": "sensenova_loop",
          "loop_settings": { "start": 1, "end": 3, "eval_count": 2, "counts": [2] },
          "mhc": { "start": 1, "end": 3, "streams": 2, "iterations": 20, "fixed_identity": false, "pre_scale": 1.0 },
          "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": \(v), "hidden_size": \(h), "intermediate_size": \(inter),
            "num_hidden_layers": 3, "mtp_num_hidden_layers": 0,
            "num_attention_heads": \(qHeads), "num_key_value_heads": \(kvHeads), "head_dim": \(headDim),
            "attn_output_gate": true, "partial_rotary_factor": 0.25,
            "linear_num_key_heads": \(gdnKHeads), "linear_num_value_heads": \(gdnVHeads),
            "linear_key_head_dim": \(gdnKDim), "linear_value_head_dim": \(gdnVDim), "linear_conv_kernel_dim": \(convK),
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            "max_position_embeddings": 2048, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
            "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2
          }
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let qRows = g.qProjOut, kvRows = g.kvHeads * g.headDim, oCols = g.qHeads * g.headDim
        var planes: [(String, PlaneSpec)] = [
            ("model.language_model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.language_model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02, isAQ4: useHQBlock)),
        ]
        for layer in [0, 1] {
            let b = "model.language_model.layers.\(layer)."
            let s = UInt32(100 + layer * 30)
            planes.append((b + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: s, amp: 0.05, offset: 1.0))))
            planes.append((b + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: s + 1, amp: 0.05, offset: 1.0))))
            planes.append((b + "linear_attn.in_proj_qkv.weight", .hq4(n: g.gdnQKVChannels, k: h, seed: s + 2, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "linear_attn.in_proj_z.weight", .hq4(n: g.gdnZChannels, k: h, seed: s + 3, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "linear_attn.in_proj_a.weight", .hq4(n: g.gdnVHeads, k: h, seed: s + 4, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "linear_attn.in_proj_b.weight", .hq4(n: g.gdnVHeads, k: h, seed: s + 5, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "linear_attn.conv1d.weight", .bf16(shape: [g.gdnQKVChannels, 1, convK],
                                                                 values: detVec(g.gdnQKVChannels * convK, seedInit: s + 6, amp: 0.05))))
            planes.append((b + "linear_attn.norm.weight", .bf16(shape: [g.gdnHeadDim], values: [Float](repeating: 1.0, count: g.gdnHeadDim))))
            planes.append((b + "linear_attn.A_log", .f32(shape: [g.gdnVHeads], values: [Float](repeating: useHQBlock ? 1.4 : -1.0, count: g.gdnVHeads))))
            planes.append((b + "linear_attn.dt_bias", .f32(shape: [g.gdnVHeads], values: [Float](repeating: 0.0, count: g.gdnVHeads))))
            planes.append((b + "linear_attn.out_proj.weight", .hq4(n: h, k: g.gdnZChannels, seed: s + 7, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: s + 8, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: s + 9, scale: 0.008, isAQ4: useHQBlock)))
            planes.append((b + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: s + 10, scale: 0.008, isAQ4: useHQBlock)))
        }
        let b2 = "model.language_model.layers.2."
        planes.append((b2 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 300, amp: 0.05, offset: 1.0))))
        planes.append((b2 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 301, amp: 0.05, offset: 1.0))))
        planes.append((b2 + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: 302, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: 303, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: 304, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: 305, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
        planes.append((b2 + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
        planes.append((b2 + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 306, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 307, scale: 0.008, isAQ4: useHQBlock)))
        planes.append((b2 + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 308, scale: 0.008, isAQ4: useHQBlock)))
        for layer in [1, 2] {
            let b = "model.language_model.layers.\(layer)."
            let s = UInt32(400 + layer * 20)
            for kind in ["mhc_attn", "mhc_ffn"] {
                let kOff = kind == "mhc_attn" ? UInt32(0) : UInt32(5)
                planes.append((b + kind + ".weight", .f32(shape: [8, 2 * h], values: detVec(8 * 2 * h, seedInit: s + kOff, amp: 0.05))))
                planes.append((b + kind + ".bias", .f32(shape: [8], values: detVec(8, seedInit: s + kOff + 1, amp: 0.1))))
            }
        }
        try writeQWP(at: rootDir, descriptor: descriptor, configJSON: cfgJSON, planes: planes, modelID: "sensenova_loop", useHQBlock: useHQBlock)
    }

    func testNextNHybridNgramAndCommitSeedingBitExactParity() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "NextN driver parity requires bare-metal Apple Silicon GPU")
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-nextn-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir)

        let prompt = [1, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40]
        let maxNew = 12

        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let baseSession = try baseDriver.makeSession(promptTokens: prompt.count)
        let baseline = try baseDriver.generate(session: baseSession, prompt: prompt, maxNew: maxNew)
        baseDriver.finishSession(baseSession)
        XCTAssertEqual(baseline.count, maxNew)

        for strategy in ["mtp", "hybrid", "ngram"] {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
                mtpDraftStrategy: strategy, mtpQuadraticSeeding: true, prefillChunkSize: 32)
            let session = try driver.makeSession(promptTokens: prompt.count)
            let tokens = try driver.generateExperimentalGreedy(session: session, prompt: prompt, maxNew: maxNew)
            XCTAssertEqual(session.roundRecord.mtpPhysicalRows, session.position - 1,
                           "MTP physical pair rows must track trunk position - 1 for strategy \(strategy)")
            driver.finishSession(session)
            XCTAssertEqual(tokens, baseline,
                           "NextN strategy \(strategy) must be bit-exact with greedy baseline")
        }
    }

    func testBatchedDecodeParityWithMTPDrafts2AndPostBatchScheduler() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "Batched MTP parity requires bare-metal Apple Silicon GPU")
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-nextn-batch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir)

        // mtpDrafts=2 allocates a 3-row mtpBatchedMLP (< 4), exercising lazy 4-row sharedBatchMLP upgrade.
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 2, mtpAdaptiveDepth: false, prefillChunkSize: 32)
        XCTAssertTrue(driver.canRunBatchedDecode)
        XCTAssertEqual(driver.mtpBatchedMLP?.rows, 3)

        let prompts = [
            [1, 10, 20, 30, 40],
            [2, 15, 25, 35, 45, 55],
            [5, 19, 31, 47, 61],
            [4, 18, 32, 44],
        ]
        let steps = 5
        var serialOutputs: [[Int]] = []
        for p in prompts {
            let res = try driver.generate(request: QWGenerationRequest(prompt: p, maxNew: steps, sampling: .greedy), onToken: nil)
            serialOutputs.append(res.tokens)
        }

        let activeB = min(driver.freeSlots.count, 4)
        for batchSize in 2...activeB {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer { for s in sessions where !s.sessionFinished { driver.finishSession(s) } }
            for b in 0..<batchSize {
                let sess = try driver.makeSession(promptTokens: prompts[b].count)
                sessions.append(sess)
                pending.append(try driver.prefill(sess, tokens: prompts[b]))
            }
            var batched = pending.map { [$0] }
            for _ in 1..<steps {
                let (nextIds, _, scratch) = try driver.forwardBatchStep(sessions, tokens: pending)
                for b in 0..<batchSize {
                    XCTAssertTrue(scratch.logitBuffer(b).allSatisfy(\.isFinite))
                    XCTAssertTrue(sessions[b].batchedDecodeTouched)
                    XCTAssertFalse(driver.canRunSpeculativeRound(sessions[b]))
                    batched[b].append(nextIds[b])
                }
                pending = nextIds
            }
            for b in 0..<batchSize {
                XCTAssertEqual(batched[b], serialOutputs[b], "B=\(batchSize) row \(b) diverged with mtpDrafts=2")
            }
        }

        // Concurrent burst forcing B=3 -> B=1 stepSingleSlot tail (batchedDecodeTouched == true), plus 1-stream speculative.
        let module = DenseModule(engine: driver)
        let engine = try QWEngine(engine: module, descriptor: driver.geometry.descriptor, plan: driver.plan)
        let scheduler = engine.batchScheduler
        let singleSpec = try scheduler.generate(request: QWGenerationRequest(prompt: [1, 10, 20, 30, 40, 10, 20, 30, 40], maxNew: 8, sampling: .greedy))
        let serialLong = try driver.generate(request: QWGenerationRequest(prompt: [1, 10, 20, 30, 40, 10, 20, 30, 40], maxNew: 8, sampling: .greedy), onToken: nil)
        XCTAssertEqual(singleSpec.tokens, serialLong.tokens)
        XCTAssertFalse(singleSpec.stepTokenCounts.isEmpty)

        final class Box: @unchecked Sendable {
            private let lock = NSLock()
            var values: [Int: QWGenerationResult] = [:]
            func set(_ idx: Int, _ val: QWGenerationResult) { lock.withLock { values[idx] = val } }
        }
        let box = Box()
        let group = DispatchGroup()
        let maxNewPerReq = [2, 2, 5]
        QWBatchAdmissionTracker.beginPreparing()
        for idx in 0..<3 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                if let res = try? scheduler.generate(request: QWGenerationRequest(prompt: prompts[idx], maxNew: maxNewPerReq[idx], sampling: .greedy)) {
                    box.set(idx, res)
                }
            }
        }
        usleep(2000)
        QWBatchAdmissionTracker.endPreparing()
        group.wait()
        for idx in 0..<3 {
            XCTAssertEqual(box.values[idx]?.tokens, Array(serialOutputs[idx].prefix(maxNewPerReq[idx])))
        }
    }

    func testWeightFreeNgramParityWithZeroMTPWeights() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "Weight-free ngram parity requires bare-metal Apple Silicon GPU")
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-wf-ngram-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, includeMTP: false)

        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let wfDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 4, mtpBackend: .standard,
            mtpAdaptiveDepth: false, mtpRelaxedMargin: 1.5,
            mtpDraftStrategy: "ngram", prefillChunkSize: 32)
        XCTAssertTrue(wfDriver.isWeightFreeSpeculation)
        XCTAssertNil(wfDriver.mtpCache)
        XCTAssertNil(wfDriver.mtpDeviceEmbedding)
        XCTAssertNil(wfDriver.mtpPromptEmbeddings)

        let prompts: [[Int]] = [
            // Repetitive / periodic prompt triggers context-copy & recycler speculation rounds
            [1, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20],
            // Novel prompt with distinct pairs avoids unigram verify overhead and preserves M=1 parity
            [1, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        ]
        for (idx, prompt) in prompts.enumerated() {
            let maxNew = 14
            let sBase = try baseDriver.makeSession(promptTokens: prompt.count)
            let baseTokens = try baseDriver.generate(session: sBase, prompt: prompt, maxNew: maxNew)
            baseDriver.finishSession(sBase)

            let res = try wfDriver.generate(
                request: QWGenerationRequest(
                    prompt: prompt, maxNew: maxNew, sampling: .greedy,
                    speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 4, mtpBackend: "variational", mtpRelaxedMargin: 1.5, mtpDraftStrategy: "ngram")),
                onToken: nil)
            XCTAssertEqual(res.tokens, baseTokens, "Weight-free .ngram diverged on prompt \(idx)")
            if idx == 0 {
                XCTAssertFalse(res.stepTokenCounts.isEmpty, "Repetitive prompt must exercise speculative verify rounds")
            }
        }
    }

    func testSenseNovaLoopQuantKVVerifyParityAndULP() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "SenseNova-Loop verify parity requires bare-metal Apple Silicon GPU")
        for useHQBlock in [false, true] {
            let rootDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("qw-snloop-verify-ulp-\(useHQBlock)-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: rootDir) }
            try Self.buildSyntheticSenseNovaLoopPack(at: rootDir, useHQBlock: useHQBlock)
            try verifySyntheticSenseNovaLoopPack(at: rootDir, useHQBlock: useHQBlock)
        }
    }

    private func verifySyntheticSenseNovaLoopPack(at rootDir: URL, useHQBlock: Bool) throws {
        for mode in [QWKVMode.f16, QWKVMode.q8, QWKVMode.q4] {
            let baseDriver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: mode, mtpDrafts: 0, prefillChunkSize: 32)
            let wfDriver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: mode, mtpDrafts: 5, mtpBackend: .standard,
                mtpAdaptiveDepth: false, mtpRelaxedMargin: 0,
                mtpDraftStrategy: "ngram", prefillChunkSize: 32)

            let attnDec = try XCTUnwrap(wfDriver.attnDecode)
            do {
                let bS = try baseDriver.makeSession(promptTokens: 64)
                let wS = try wfDriver.makeSession(promptTokens: 64)
                defer { baseDriver.finishSession(bS); wfDriver.finishSession(wS) }
                let g = wfDriver.geometry
                let pre = [1, 2, 3, 4, 5, 6]
                _ = try baseDriver.prefill(bS, tokens: pre)
                _ = try wfDriver.prefill(wS, tokens: pre)
                let kv = try XCTUnwrap(wS.attnKVViews.last)
                let kvB = try XCTUnwrap(bS.attnKVViews.last)
                let part = try XCTUnwrap(wfDriver.mtpSharedAttentionPartials)
                let hal = wfDriver.hal
                let qgBuf = hal.device.makeBuffer(length: 2 * g.qProjOut * 2, options: .storageModeShared)!
                let kBuf = hal.device.makeBuffer(length: 2 * g.kvChannels * 2, options: .storageModeShared)!
                let vBuf = hal.device.makeBuffer(length: 2 * g.kvChannels * 2, options: .storageModeShared)!
                let outCols = g.qHeads * g.headDim
                let oBatBuf = hal.device.makeBuffer(length: 2 * outCols * 2, options: .storageModeShared)!
                let oSerBuf = hal.device.makeBuffer(length: 2 * outCols * 2, options: .storageModeShared)!
                let fill: (MTLBuffer, Int, UInt16) -> Void = { buf, cnt, seed in
                    let p = buf.contents().assumingMemoryBound(to: UInt16.self)
                    for i in 0..<cnt { p[i] = Float16(sin(Float(i + Int(seed)) * 0.17) * 0.4).bitPattern }
                }
                fill(qgBuf, 2 * g.qProjOut, 1); fill(kBuf, 2 * g.kvChannels, 2); fill(vBuf, 2 * g.kvChannels, 3)
                func tv(_ b: MTLBuffer, _ cnt: Int) -> QWTensorValue {
                    QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length), offset: 0, shape: QWTensorShape([cnt], .f16))
                }
                let qv = tv(qgBuf, 2 * g.qProjOut), kvv = tv(kBuf, 2 * g.kvChannels), vvv = tv(vBuf, 2 * g.kvChannels)
                let partB = hal.device.makeBuffer(length: part.length, options: .storageModeShared)!
                let partS = hal.device.makeBuffer(length: part.length, options: .storageModeShared)!
                let regB = QWMRegion(buffer: partB, base: 0, length: part.length)
                let regS = QWMRegion(buffer: partS, base: 0, length: part.length)
                let posTest = 64
                try wfDriver.kvCache.prepareWrites(wS.kv, positions: 6..<posTest + 2)
                try baseDriver.kvCache.prepareWrites(bS.kv, positions: 6..<posTest + 2)
                for pCheck in [6, 64] {
                    let splitsCheck = QWAttnDecode.splits(forKeys: pCheck + 2)
                    if let (cb, enc) = hal.begin() {
                        if pCheck > 6 {
                            let fillBuf = hal.device.makeBuffer(length: (pCheck - 6) * g.kvChannels * 2, options: .storageModeShared)!
                            fill(fillBuf, (pCheck - 6) * g.kvChannels, 9)
                            let fKv = tv(fillBuf, (pCheck - 6) * g.kvChannels)
                            wfDriver.encodePagedKVWrite(enc, kv: kv, k: fKv, v: fKv, startPos: 6, t: pCheck - 6)
                            baseDriver.encodePagedKVWrite(enc, kv: kvB, k: fKv, v: fKv, startPos: 6, t: pCheck - 6)
                            enc.memoryBarrier(scope: .buffers)
                        }
                        wfDriver.encodePagedKVWrite(enc, kv: kv, k: kvv, v: vvv, startPos: pCheck, t: 2)
                        enc.memoryBarrier(scope: .buffers)
                        try attnDec.encodeBatched(enc, qgate: qv, kv: kv, knew: kvv, vnew: vvv, partials: regB,
                                                  out: tv(oBatBuf, 2 * outCols), cBase: pCheck, T: 2,
                                                  splits: splitsCheck, sharedPrefix: true)
                        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    }
                    for r in 0..<2 {
                        if let (cb, enc) = hal.begin() {
                            try attnDec.encode(
                                enc, qgate: qv.slice(byteOffset: r * g.qProjOut * 2, count: g.qProjOut),
                                kv: kvB,
                                knew: kvv.slice(byteOffset: r * g.kvChannels * 2, count: g.kvChannels),
                                vnew: vvv.slice(byteOffset: r * g.kvChannels * 2, count: g.kvChannels),
                                partials: regS,
                                out: tv(oSerBuf, 2 * outCols).slice(byteOffset: r * outCols * 2, count: outCols),
                                C: pCheck + r, splits: splitsCheck)
                            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                        }
                        let pB = oBatBuf.contents().assumingMemoryBound(to: UInt16.self)
                        let pS = oSerBuf.contents().assumingMemoryBound(to: UInt16.self)
                        var mismatch = 0
                        for i in 0..<outCols where pB[r * outCols + i] != pS[r * outCols + i] { mismatch += 1 }
                        XCTAssertEqual(mismatch, 0, "Direct general attention mode=\(mode) C=\(pCheck) row=\(r)")
                    }
                }
            }
            for prefixLen in [6, 7] {
                let prefix = Array(1...prefixLen)
                for activeRows in [2, 3, 4, 5, 6] {
                    let bSess = try baseDriver.makeSession(promptTokens: 64)
                    let wSess = try wfDriver.makeSession(promptTokens: 64)
                    defer { baseDriver.finishSession(bSess); wfDriver.finishSession(wSess) }
                    baseDriver.activeCollectLogprobs = true
                    let b0 = try baseDriver.prefill(bSess, tokens: prefix)
                    let w0 = try wfDriver.prefill(wSess, tokens: prefix)
                    XCTAssertEqual(b0, w0, "mode=\(mode) prefix=\(prefixLen) prefill tok0 mismatch")
                    let stepTokens = [b0] + Array(0..<(activeRows - 1)).map { 60 + $0 }
                    let win = try wfDriver.verifyTokensLocked(wSess, tokens: stepTokens)
                    let bPtr = baseDriver.sc.logits.region.pointer.advanced(by: baseDriver.sc.logits.offset).assumingMemoryBound(to: Float.self)
                    let wPtr = win.logits.region.pointer.advanced(by: win.logits.offset).assumingMemoryBound(to: Float.self)
                    let vocab = baseDriver.geometry.vocab
                    var retained2RecBits: [UInt32]?
                    var retained2RingBits: [UInt16]?
                    for row in 0..<activeRows {
                        _ = try baseDriver.decodeStep(bSess, token: stepTokens[row])
                        if activeRows == 5 && prefixLen == 6 && row == 1 {
                            let bRec = bSess.recSlab.region.pointer.advanced(by: bSess.recSlab.offset)
                                .assumingMemoryBound(to: UInt32.self)
                            retained2RecBits = Array(UnsafeBufferPointer(start: bRec, count: bSess.recSlab.shape.elementCount))
                            let bRing = bSess.ringSlab.region.pointer.advanced(by: bSess.ringSlab.offset)
                                .assumingMemoryBound(to: UInt16.self)
                            retained2RingBits = Array(UnsafeBufferPointer(start: bRing, count: bSess.ringSlab.shape.elementCount))
                        }
                        let rowPtr = wPtr.advanced(by: row * vocab)
                        var maxULP: UInt32 = 0
                        var worstAbs: Float = 0
                        for vIdx in 0..<vocab {
                            let diff = abs(bPtr[vIdx] - rowPtr[vIdx])
                            worstAbs = max(worstAbs, diff)
                            let ub = bPtr[vIdx].bitPattern
                            let uw = rowPtr[vIdx].bitPattern
                            let ulp = ub >= uw ? ub - uw : uw - ub
                            if bPtr[vIdx] != 0 || rowPtr[vIdx] != 0 { maxULP = max(maxULP, ulp) }
                        }
                        XCTAssertLessThanOrEqual(
                            maxULP, 0,
                            "hq=\(useHQBlock) mode=\(mode) cBase=\(prefixLen) M=\(activeRows) row=\(row) maxAbs=\(worstAbs) maxULP=\(maxULP)")
                    }
                    if let expectedRec = retained2RecBits, let expectedRing = retained2RingBits {
                        let chk = try XCTUnwrap(wfDriver.gdnReplayCheckpoint)
                        let stable = try chk.rollbackTapeIsStable(retaining: 2, activeRows: 5)
                        XCTAssertEqual(stable, !useHQBlock, "hq=\(useHQBlock) rollback stability mismatch")
                        let cb = try XCTUnwrap(wfDriver.hal.queue.makeCommandBuffer())
                        try win.encodeRetainedPrefix(
                            cb, hal: wfDriver.hal, checkpoint: chk, decoder: wfDriver.gdnDecode,
                            rec: wSess.recSlab, ring: wSess.ringSlab,
                            trunkHidden: try XCTUnwrap(wSess.trunkHidden), activeRows: 5, retaining: 2)
                        cb.commit()
                        cb.waitUntilCompleted()
                        let wRec = wSess.recSlab.region.pointer.advanced(by: wSess.recSlab.offset)
                            .assumingMemoryBound(to: UInt32.self)
                        let wRing = wSess.ringSlab.region.pointer.advanced(by: wSess.ringSlab.offset)
                            .assumingMemoryBound(to: UInt16.self)
                        if stable {
                            var maxRecAbs: Float = 0
                            for i in 0..<expectedRec.count {
                                maxRecAbs = max(maxRecAbs, abs(Float(bitPattern: wRec[i]) - Float(bitPattern: expectedRec[i])))
                            }
                            XCTAssertLessThan(maxRecAbs, 1e-4, "hq=\(useHQBlock) mode=\(mode) rollback recSlab exceeded 1e-4")
                        } else {
                            var recMismatches = 0
                            for i in 0..<expectedRec.count where wRec[i] != expectedRec[i] { recMismatches += 1 }
                            XCTAssertEqual(recMismatches, 0, "hq=\(useHQBlock) mode=\(mode) partial rejection recSlab diverged")
                        }
                        var ringMismatches = 0
                        for i in 0..<expectedRing.count where wRing[i] != expectedRing[i] { ringMismatches += 1 }
                        XCTAssertEqual(ringMismatches, 0, "hq=\(useHQBlock) mode=\(mode) partial rejection ringSlab diverged")
                    }
                }
            }

            let prompt = [1, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 41, 42, 43]
            let sBase = try baseDriver.makeSession(promptTokens: prompt.count)
            baseDriver.activeCollectLogprobs = false
            let baseTokens = try baseDriver.generate(session: sBase, prompt: prompt, maxNew: 24)
            baseDriver.finishSession(sBase)

            let res = try wfDriver.generate(
                request: QWGenerationRequest(
                    prompt: prompt, maxNew: 24, sampling: .greedy,
                    speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 5, mtpBackend: "standard", mtpRelaxedMargin: 0, mtpDraftStrategy: "ngram")),
                onToken: nil)
            XCTAssertEqual(res.tokens, baseTokens, "SenseNovaLoop weight-free .ngram diverged in mode=\(mode)")
        }
    }
}

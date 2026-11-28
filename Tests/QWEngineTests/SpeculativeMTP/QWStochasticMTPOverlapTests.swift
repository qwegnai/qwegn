// QWStochasticMTPOverlapTests.swift — Async T>0 overlap, rep-penalty fallback, sparse-MoE cap, and prefix-retain tests.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWMTP
@testable import QWEngine

final class QWStochasticMTPOverlapTests: XCTestCase {
    private final class BoundedConstraint: QWTokenConstraint, @unchecked Sendable {
        private static let allowedPerStep: [[Int]] = [
            [10, 11], [20, 21, 22], [30], [40, 41], [50], [60, 61, 62], [70], [80, 81],
        ]
        private(set) var step: Int

        init(step: Int = 0) { self.step = step }
        var allowsSpeculation: Bool { QWConstrainedSpeculationGate.isEnabled }
        var isUnconstrainedNow: Bool { step >= Self.allowedPerStep.count }
        var speculationBoundaryToken: Int? { nil }
        func makeSpeculativeProbe() -> (any QWTokenConstraint)? { BoundedConstraint(step: step) }
        func proposeCanonicalDrafts(maxCount: Int) -> [Int] {
            guard QWConstrainedSpeculationGate.isEnabled, maxCount > 0 else { return [] }
            var out: [Int] = []
            var idx = step
            while out.count < maxCount, idx < Self.allowedPerStep.count {
                out.append(Self.allowedPerStep[idx][0])
                idx += 1
            }
            return out
        }
        func allowedTokens() throws -> [Int]? {
            step < Self.allowedPerStep.count ? Self.allowedPerStep[step] : nil
        }
        func accept(_ token: Int) throws {
            guard step < Self.allowedPerStep.count else { return }
            guard Self.allowedPerStep[step].contains(token) else {
                throw QWTokenConstraintError.rejectedToken(token)
            }
            step += 1
        }
    }

    func testSparseMoENeuralDepthCapMatchesAcrossGreedyAndStochasticModes() {
        for isStochastic in [false, true] {
            XCTAssertEqual(
                QWAdaptiveDraftPolicy.neuralDepthCap(
                    isStochastic: isStochastic, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 288
                ),
                1
            )
            XCTAssertEqual(
                QWMTPAdaptiveDepth.hybridNextNNeuralDepthCap(
                    isStochastic: isStochastic, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 288
                ),
                1
            )
            XCTAssertNil(
                QWMTPAdaptiveDepth.hybridNextNNeuralDepthCap(
                    isStochastic: isStochastic, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 0
                )
            )
            XCTAssertEqual(
                QWMTPAdaptiveDepth.hybridNextNNeuralDepthCap(
                    isStochastic: isStochastic, stageProfiling: false, hasDeviceEmbedding: false, numExperts: 0
                ),
                1
            )
            XCTAssertNil(
                QWMTPAdaptiveDepth.hybridNextNNeuralDepthCap(
                    isStochastic: isStochastic, stageProfiling: true, hasDeviceEmbedding: true, numExperts: 288
                )
            )
        }
    }

    func testStochasticOverlapRepetitionFallbackConstrainedAndGreedyUntouched() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Stochastic MTP GPU overlap tests require bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-stoch-mtp-overlap-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir)

        let prompt = [1, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40]
        let maxNew = 12
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "hybrid", mtpQuadraticSeeding: true, prefillChunkSize: 32
        )
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)

        // 1. T = 0 untouched greedy parity vs serial baseline.
        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32
        )
        let baseSession = try baseDriver.makeSession(promptTokens: prompt.count)
        let greedyBase = try baseDriver.generate(session: baseSession, prompt: prompt, maxNew: maxNew)
        baseDriver.finishSession(baseSession)

        let greedySpecSession = try driver.makeSession(promptTokens: prompt.count)
        let greedySpec = try driver.generateExperimentalGreedy(session: greedySpecSession, prompt: prompt, maxNew: maxNew)
        driver.finishSession(greedySpecSession)
        XCTAssertEqual(greedySpec, greedyBase)

        // 2. T > 0 async GPU LC + drain:false overlap: deterministic replay, early abort, and cached multi-turn.
        let lcConfig = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.8, topP: 0.9, topK: 8, repetitionPenalty: 1.0, seed: 42),
            vocabulary: 256
        )
        let lcS1 = try driver.makeStochasticSession(promptTokens: prompt.count, config: lcConfig)
        let lcOut1 = try driver.generateExperimentalStochastic(session: lcS1, prompt: prompt, maxNew: maxNew, speculative: true)
        driver.finishSession(lcS1)

        let lcS2 = try driver.makeStochasticSession(promptTokens: prompt.count, config: lcConfig)
        let lcOut2 = try driver.generateExperimentalStochastic(session: lcS2, prompt: prompt, maxNew: maxNew, speculative: true)
        driver.finishSession(lcS2)
        XCTAssertEqual(lcOut1.count, maxNew)
        XCTAssertEqual(lcOut1, lcOut2, "async GPU stochastic LC speculative output must be seed-deterministic")

        let abortS = try driver.makeStochasticSession(promptTokens: prompt.count, config: lcConfig)
        var streamed = 0
        let aborted = try driver.generateExperimentalStochastic(
            session: abortS, prompt: prompt, maxNew: maxNew, speculative: true
        ) { _ in
            streamed += 1
            return streamed < 4
        }
        driver.finishSession(abortS)
        XCTAssertLessThanOrEqual(aborted.count, maxNew)
        XCTAssertGreaterThanOrEqual(aborted.count, 4)

        let unseededLC = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.8, topP: 0.9, topK: 8, repetitionPenalty: 1.0, seed: -1),
            vocabulary: 256
        )
        let cachedTurn1 = try driver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: 6, sampling: unseededLC,
                usePrefixCache: true,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 3)
            ),
            onToken: nil
        )
        let cachedTurn2 = try driver.generate(
            request: QWGenerationRequest(
                prompt: prompt + [15, 25], maxNew: 6, sampling: unseededLC,
                usePrefixCache: true,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 3)
            ),
            onToken: nil
        )
        XCTAssertEqual(cachedTurn1.tokens.count, 6)
        XCTAssertEqual(cachedTurn2.tokens.count, 6)

        // 3. repetitionPenalty > 1.0 CPU fallback parity (T = 0 rep=1.25 through stochastic pipeline).
        let repDetConfig = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.0, topP: 1.0, topK: 1, repetitionPenalty: 1.25, seed: 7),
            vocabulary: 256
        )
        let repSerialS = try driver.makeStochasticSession(promptTokens: prompt.count, config: repDetConfig)
        let repSerial = try driver.generateExperimentalStochastic(session: repSerialS, prompt: prompt, maxNew: maxNew, speculative: false)
        driver.finishSession(repSerialS)

        let repSpecS = try driver.makeStochasticSession(promptTokens: prompt.count, config: repDetConfig)
        let repSpec = try driver.generateExperimentalStochastic(session: repSpecS, prompt: prompt, maxNew: maxNew, speculative: true)
        driver.finishSession(repSpecS)
        XCTAssertEqual(repSpec, repSerial, "repetitionPenalty > 1.0 lazy CPU resolver fallback must match serial")

        // 4. Constrained T > 0 (rep == 1.0) zero-escape through lazy CPU resolver guardrail.
        let constrainedRes = try driver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: 8,
                sampling: lcConfig,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 3, mtpBackend: "standard"),
                constraint: BoundedConstraint()
            ),
            onToken: nil
        )
        let verifier = BoundedConstraint()
        for tok in constrainedRes.tokens {
            XCTAssertNoThrow(try verifier.accept(tok))
        }
    }

    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
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

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32, gBytes = 20
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = Float(Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1031.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF); ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF); ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func buildSyntheticNextNPack(at rootDir: URL) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5",
          "vocab_size": \(v), "hidden_size": \(h), "intermediate_size": \(inter),
          "num_hidden_layers": 1, "mtp_num_hidden_layers": 1,
          "num_attention_heads": \(qHeads), "num_key_value_heads": \(kvHeads), "head_dim": \(headDim),
          "max_position_embeddings": 2048, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0, "layer_types": ["full_attention"],
          "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let qRows = g.qProjOut, kvRows = g.kvHeads * g.headDim, oCols = g.qHeads * g.headDim

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
        func appendBlk(_ blk: String, seedBase: UInt32) {
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
        appendBlk("model.layers.0.", seedBase: 100)
        planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 200, scale: 0.006)))
        planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        appendBlk("mtp.layers.0.", seedBase: 300)

        var file = Data("QWP1".utf8)
        file.append(Data(repeating: 0, count: 8))
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        padTo(256)
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(256); let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (rb: UnsafeRawBufferPointer) in raw.append(contentsOf: rb) }
                }
                file.append(raw); padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale):
                padTo(16_384); let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "qwen3_5", "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-stoch-mtp", "sha256_source": "synthetic",
                "quant_recorded": "q4g", "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (rb: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: rb) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }
}

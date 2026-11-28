// QWConstrainedMTPParityTests.swift — Synthetic GPU parity tests for constrained speculative decoding.

import CryptoKit
import Foundation
import XCTest
import QWCore
import QWHAL
import QWHead
@testable import QWEngine

final class QWConstrainedMTPParityTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Constrained MTP parity tests require bare-metal Apple Silicon GPU"
        )
    }

    private final class MockGrammarConstraint: QWTokenConstraint, @unchecked Sendable {
        // Alternates deterministic structural runs [30, 31, 32] with 4-way value choices [40, 41, 42, 43]
        // and canonical proposal [50, 51] where [50, 51, 52] are allowed at step 3.
        private static let schedule: [[Int]] = [
            [30], [31], [32],
            [40, 41, 42, 43],
            [50, 51, 52], [51, 53],
            [30], [31], [32],
            [40, 41, 42, 43],
            [60], [61], [2]
        ]

        private(set) var step: Int
        private(set) var history: [Int]

        init(step: Int = 0, history: [Int] = []) {
            self.step = step
            self.history = history
        }

        var allowsSpeculation: Bool { QWConstrainedSpeculationGate.isEnabled }
        var isUnconstrainedNow: Bool { step >= Self.schedule.count }
        var speculationBoundaryToken: Int? { nil }

        func makeSpeculativeProbe() -> (any QWTokenConstraint)? {
            MockGrammarConstraint(step: step, history: history)
        }

        func proposeCanonicalDrafts(maxCount: Int) -> [Int] {
            guard QWConstrainedSpeculationGate.isEnabled, maxCount > 0 else { return [] }
            let probe = MockGrammarConstraint(step: step, history: history)
            var out: [Int] = []
            while out.count < maxCount, probe.step < Self.schedule.count {
                let allowed = Self.schedule[probe.step]
                // Never propose EOS (token 2) as draft; propose canonical first key on structural runs.
                if allowed == [2] { break }
                if allowed.count == 1 {
                    out.append(allowed[0])
                    probe.step += 1
                } else if probe.step == 4 {
                    out.append(50)
                    probe.step += 1
                } else if probe.step == 5 {
                    out.append(51)
                    probe.step += 1
                } else {
                    break
                }
            }
            return out
        }

        func allowedTokens() throws -> [Int]? {
            guard step < Self.schedule.count else { return nil }
            return Self.schedule[step]
        }

        func accept(_ token: Int) throws {
            if step < Self.schedule.count {
                let allowed = Self.schedule[step]
                guard allowed.contains(token) else {
                    throw QWTokenConstraintError.rejectedToken(token)
                }
                step += 1
            }
            history.append(token)
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

    private static func makeHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
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
                    let bi = Float16(-1031.5 * scVal).bitPattern
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

    private static func buildSyntheticPack(at rootDir: URL) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let cfgString = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "mtp_num_hidden_layers": 1,
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
        """
        let cfgJSON = Data(cfgString.utf8)
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
        func appendBlock(_ blk: String, seedBase: UInt32) {
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
        appendBlock("model.layers.0.", seedBase: 100)
        planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 200, scale: 0.006)))
        planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        appendBlock("mtp.layers.0.", seedBase: 300)

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
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "qwen3_5",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-constrained-mtp",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
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

    func testConstrainedSpeculativeGreedyVariationalStochasticAndBatchParity() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-constrained-mtp-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticPack(at: rootDir)

        let prompt = [1, 10, 20, 30, 40, 50, 30, 31, 32]
        let maxNew = 16
        let stopIDs: Set<Int> = [2]

        let serialDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32
        )
        let serialConstraint = MockGrammarConstraint()
        let serialRes = try serialDriver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: .greedy,
                speculative: QWSpeculativeParameters(enabled: false, mtpDrafts: 0),
                constraint: serialConstraint
            ),
            onToken: nil
        )
        XCTAssertEqual(serialRes.tokens.count, 12, "must emit exact 12 grammar tokens before EOS=2")

        for strategy in ["mtp", "hybrid", "ngram"] {
            let specDriver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: .q4, mtpDrafts: 4, mtpAdaptiveDepth: false,
                mtpDraftStrategy: strategy, prefillChunkSize: 32
            )
            let specConstraint = MockGrammarConstraint()
            let specRes = try specDriver.generate(
                request: QWGenerationRequest(
                    prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                    sampling: .greedy,
                    speculative: QWSpeculativeParameters(
                        enabled: true, mtpDrafts: 4, mtpBackend: "standard",
                        mtpRelaxedMargin: 0, mtpDraftStrategy: strategy
                    ),
                    constraint: specConstraint
                ),
                onToken: nil
            )
            XCTAssertEqual(specRes.tokens, serialRes.tokens, "strategy \(strategy) must match serial greedy bit-exact")
            XCTAssertGreaterThan(specRes.stepTokenCounts.max() ?? 1, 1, "canonical drafts must commit >1 token per round")
            XCTAssertEqual(Array(specConstraint.history.prefix(specRes.tokens.count)), specRes.tokens)
        }

        // Greedy + repetitionPenalty > 1.0 exercises QWStochasticSession constrained path.
        let repSampling = QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1.3)
        let serialRep = try serialDriver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: repSampling,
                speculative: QWSpeculativeParameters(enabled: false, mtpDrafts: 0),
                constraint: MockGrammarConstraint()
            ),
            onToken: nil
        )
        let specDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 4, mtpAdaptiveDepth: false, prefillChunkSize: 32
        )
        let specRep = try specDriver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: repSampling,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 4, mtpBackend: "standard", mtpRelaxedMargin: 0),
                constraint: MockGrammarConstraint()
            ),
            onToken: nil
        )
        XCTAssertEqual(specRep.tokens, serialRep.tokens, "penalized greedy constrained speculation must match serial")

        // Variational & stochastic constrained soundness: every emitted token must pass fresh MockGrammarConstraint.
        let varConstraint = MockGrammarConstraint()
        let varRes = try specDriver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: .greedy,
                speculative: QWSpeculativeParameters(
                    enabled: true, mtpDrafts: 4, mtpBackend: "variational",
                    mtpRelaxedMargin: 2.0, mtpRelaxedMinProb: 0.05
                ),
                constraint: varConstraint
            ),
            onToken: nil
        )
        let verifier1 = MockGrammarConstraint()
        for tok in varRes.tokens { XCTAssertNoThrow(try verifier1.accept(tok)) }

        let stochConstraint = MockGrammarConstraint()
        let stochRes = try specDriver.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: QWSamplerConfig(temperature: 0.8, topK: 4, topP: 0.9, uniformVersion: .upper24, repetitionPenalty: 1.15),
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 4, mtpBackend: "standard"),
                constraint: stochConstraint
            ),
            onToken: nil
        )
        let verifier2 = MockGrammarConstraint()
        for tok in stochRes.tokens { XCTAssertNoThrow(try verifier2.accept(tok)) }

        // Batch scheduler single-slot constrained speculation parity.
        let module = DenseModule(engine: specDriver)
        let engine = try QWEngine(engine: module, descriptor: specDriver.geometry.descriptor, plan: specDriver.plan)
        let batchRes = try engine.batchScheduler.generate(
            request: QWGenerationRequest(
                prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
                sampling: .greedy,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 4, mtpBackend: "standard", mtpRelaxedMargin: 0),
                constraint: MockGrammarConstraint()
            )
        )
        XCTAssertEqual(batchRes.tokens, serialRes.tokens, "DriverBatchDecode constrained single-slot must match serial")
    }
}

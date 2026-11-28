// QWGigaChat4OracleTests.swift — End-to-end 2-layer + NextN MTP PyTorch-to-Metal parity tests for GigaChat4.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWHead
import QWKV
import QWMTP
@testable import QWEngine

final class QWGigaChat4OracleTests: XCTestCase {
    private struct DecodeStepGolden: Decodable {
        let step: Int
        let token_in: Int
        let greedy_token: Int
        let logits: [Float]
    }

    private struct BatchDecodeStepGolden: Decodable {
        let step: Int
        let tokens_in: [Int]
        let greedy_tokens: [Int]
        let logits: [[Float]]
    }

    private struct OracleGoldens: Decodable {
        let pack_dir: String
        let vocab_size: Int
        let hidden_size: Int
        let prompt_tokens: [Int]
        let prefill_greedy_token: Int
        let prefill_last_logits: [Float]
        let decode_steps: [DecodeStepGolden]
        let greedy_sequence_8: [Int]
        let batch_prompts: [[Int]]
        let batch_prefill_tokens: [Int]
        let batch_decode_steps: [BatchDecodeStepGolden]
        let nextn_drafts: [DecodeStepGolden]
    }

    private static let defaultHFSnapshot =
        NSHomeDirectory() + "/.cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B/"
        + "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc"

    private func resolveOraclePython() -> String? {
        if let envPy = ProcessInfo.processInfo.environment["QW_ORACLE_PYTHON"],
           FileManager.default.isExecutableFile(atPath: envPy) {
            return envPy
        }
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let parentCheckout = repoRoot.deletingLastPathComponent().deletingLastPathComponent()
        let candidates = [
            repoRoot.appendingPathComponent(".venv-oracle/bin/python3").path,
            parentCheckout.appendingPathComponent(".venv-oracle/bin/python3").path
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    private func readLogits(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.pointer.advanced(by: tv.offset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var worst: Float = 0
        for i in 0..<a.count {
            worst = max(worst, abs(a[i] - b[i]))
        }
        return worst
    }

    private func argmax(_ v: [Float]) -> Int {
        var bestIdx = 0
        var bestVal = -Float.infinity
        for (i, x) in v.enumerated() where x > bestVal {
            bestVal = x
            bestIdx = i
        }
        return bestIdx
    }

    func testGigaChat4TwoLayerPrefillDecodeBatchedMLAAndNextNMTPParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "GigaChat4 GPU oracle requires bare-metal Apple Silicon Metal GPU"
        )
        guard let py = resolveOraclePython() else {
            throw XCTSkip("Oracle Python (.venv-oracle/bin/python3) not installed on this host")
        }
        let snapshot = ProcessInfo.processInfo.environment["QW_GIGACHAT4_HF_SNAPSHOT"] ?? Self.defaultHFSnapshot
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: snapshot + "/modeling_gigachat4.py"),
            "GigaChat4 HF snapshot modeling_gigachat4.py not present on this host"
        )

        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let scriptURL = repoRoot.appendingPathComponent("Scripts/oracle/gigachat4_2layer_oracle.py")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-gc4-oracle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: workDir) }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: py)
        proc.arguments = [scriptURL.path, "--out-dir", workDir.path, "--hf-snapshot", snapshot]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        let procOut = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let msg = String(data: procOut, encoding: .utf8) ?? "<binary>"
            XCTFail("gigachat4_2layer_oracle.py failed (\(proc.terminationStatus)): \(msg)")
            return
        }

        let goldenURL = workDir.appendingPathComponent("oracle_goldens.json")
        let goldens = try JSONDecoder().decode(OracleGoldens.self, from: Data(contentsOf: goldenURL))
        let vocab = goldens.vocab_size

        // 1. Trunk prefill + 4 serial decode steps + Batched MLA decode (B=2 & B=4) + single-slot reclaim.
        let driver = try QWEngineDriver(
            packPath: goldens.pack_dir,
            maxTokens: 1024,
            ropeRows: 2048,
            kvMode: .f16,
            mtpDrafts: 0,
            prefillChunkSize: 32
        )
        XCTAssertTrue(driver.hasSandwichGatedNorm)
        XCTAssertNotNil(driver.finalGatedNorm)
        XCTAssertTrue(driver.canRunBatchedDecode)

        driver.activeCollectLogprobs = true
        let s0 = try driver.makeSession(promptTokens: goldens.prompt_tokens.count)
        let preTok = try driver.prefill(s0, tokens: goldens.prompt_tokens)
        let preLogits = readLogits(driver.sc.logits, count: vocab)
        let preDiff = maxAbsDiff(preLogits, goldens.prefill_last_logits)
        XCTAssertEqual(preTok, goldens.prefill_greedy_token, "Prefill greedy token mismatch")
        XCTAssertEqual(argmax(preLogits), goldens.prefill_greedy_token)
        XCTAssertLessThan(preDiff, 5e-3, "Prefill max_abs_diff=\(preDiff) >= 5e-3")

        var worstDecodeDiff: Float = 0
        var curTok = preTok
        for step in goldens.decode_steps {
            XCTAssertEqual(curTok, step.token_in)
            let nxt = try driver.decodeStep(s0, token: curTok)
            let stepLogits = readLogits(driver.sc.logits, count: vocab)
            let diff = maxAbsDiff(stepLogits, step.logits)
            worstDecodeDiff = max(worstDecodeDiff, diff)
            XCTAssertEqual(nxt, step.greedy_token, "Serial decode step \(step.step) greedy mismatch")
            XCTAssertEqual(argmax(stepLogits), step.greedy_token)
            XCTAssertLessThan(diff, 5e-3, "Serial decode step \(step.step) max_abs_diff=\(diff) >= 5e-3")
            curTok = nxt
        }
        driver.finishSession(s0)
        driver.activeCollectLogprobs = false

        let serialSeqF16 = try driver.generate(
            request: QWGenerationRequest(prompt: goldens.prompt_tokens, maxNew: 8, sampling: .greedy),
            onToken: nil
        ).tokens
        XCTAssertEqual(serialSeqF16, goldens.greedy_sequence_8)

        let activeB = min(driver.maxBatchSlots, 4)
        XCTAssertGreaterThanOrEqual(activeB, 2)
        for batchSize in [2, activeB] {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer { for s in sessions where !s.sessionFinished { driver.finishSession(s) } }
            for b in 0..<batchSize {
                let bs = try driver.makeSession(promptTokens: goldens.batch_prompts[b].count)
                sessions.append(bs)
                let first = try driver.prefill(bs, tokens: goldens.batch_prompts[b])
                XCTAssertEqual(first, goldens.batch_prefill_tokens[b], "Batch B=\(batchSize) row \(b) prefill token")
                pending.append(first)
            }
            for step in goldens.batch_decode_steps {
                let (nextIds, _, bScratch) = try driver.forwardBatchStep(sessions, tokens: pending)
                for b in 0..<batchSize {
                    let rowLogits = Array(bScratch.logitBuffer(b))
                    let diff = maxAbsDiff(rowLogits, step.logits[b])
                    XCTAssertEqual(nextIds[b], step.greedy_tokens[b], "Batch B=\(batchSize) step \(step.step) row \(b)")
                    XCTAssertLessThan(diff, 5e-3, "Batch B=\(batchSize) step \(step.step) row \(b) diff=\(diff)")
                }
                pending = nextIds
            }
            // Verify lossless reclaim from mlaBatchState back to single-slot serial decode on row 0.
            let tailSingle = try driver.decodeStep(sessions[0], token: pending[0])
            XCTAssertEqual(tailSingle, goldens.greedy_sequence_8[3], "Single-slot reclaim after batched MLA decode diverged")
        }

        // 2. Heterogeneous Layer-2 NextN MTP (dense MLP + denseOnly MLA + 9 GatedNorms + deduplicated embed/head).
        let mtpDriver = try QWEngineDriver(
            packPath: goldens.pack_dir,
            maxTokens: 1024,
            ropeRows: 2048,
            kvMode: .f16,
            mtpDrafts: 3,
            mtpBackend: .standard,
            mtpAdaptiveDepth: false,
            mtpRelaxedMargin: 0,
            mtpDraftStrategy: "mtp",
            prefillChunkSize: 32
        )
        let mtpWeights = try XCTUnwrap(mtpDriver.mtpWeights)
        let mtpDraftEncoder = try XCTUnwrap(mtpDriver.mtpDraftEncoder)
        XCTAssertNil(mtpWeights.moe, "GigaChat4 NextN MTP layer must be dense MLP, not MoE")
        XCTAssertEqual(mtpWeights.mla?.denseOnly, true, "GigaChat4 NextN MTP MLA must be denseOnly")
        XCTAssertEqual(mtpWeights.mla?.oGateInQKV, true, "GigaChat4 NextN MTP MLA must fuse oGateInQKV")
        XCTAssertNotNil(mtpWeights.gatedNorms, "GigaChat4 NextN MTP must carry 9 GatedNorm weights")

        let ms = try mtpDriver.makeSession(promptTokens: goldens.prompt_tokens.count)
        let mFirst = try mtpDriver.prefill(ms, tokens: goldens.prompt_tokens)
        XCTAssertEqual(mFirst, goldens.prefill_greedy_token)
        let mHidden = try XCTUnwrap(ms.trunkHidden)
        let mViews = try XCTUnwrap(ms.mtpKVViews.first)
        var capturedDraftLogits: [[Float]] = []
        let proposed = try QWMTPDraftChain.run(
            hal: mtpDriver.hal,
            draft: mtpDraftEncoder,
            headStorage: mtpDriver.headStorage,
            draftHead: mtpDriver.mtpDraftHead,
            hidden: mHidden,
            kv: mViews,
            pendingToken: mFirst,
            drafts: 3,
            position: ms.position,
            cacheRows: ms.roundRecord.mtpPhysicalRows,
            selection: .cpuReference,
            deviceEmbedding: nil,
            embedding: { try mtpDriver.copyEmbedRow($0, to: $1) },
            sampledToken: { logitsTV in
                let vals = self.readLogits(logitsTV, count: vocab)
                capturedDraftLogits.append(vals)
                return self.argmax(vals)
            },
            gpuFailure: { mtpDriver.commandBufferErrors += 1 }
        )
        mtpDriver.finishSession(ms)

        XCTAssertEqual(capturedDraftLogits.count, 3)
        var worstNextNDiff: Float = 0
        for (idx, goldenDraft) in goldens.nextn_drafts.enumerated() {
            let diff = maxAbsDiff(capturedDraftLogits[idx], goldenDraft.logits)
            worstNextNDiff = max(worstNextNDiff, diff)
            XCTAssertEqual(proposed[idx], goldenDraft.greedy_token, "NextN draft step \(idx) token mismatch")
            XCTAssertLessThan(diff, 5e-3, "NextN draft step \(idx) max_abs_diff=\(diff) >= 5e-3")
        }

        // Full speculative round-trip (verify + partial/all-accept commit seeding) across f16 / q8 / q4 KV modes.
        for mode in [QWKVMode.f16, QWKVMode.q8, QWKVMode.q4] {
            let baseModeDriver = try QWEngineDriver(
                packPath: goldens.pack_dir, maxTokens: 1024, ropeRows: 2048,
                kvMode: mode, mtpDrafts: 0, prefillChunkSize: 32)
            let specModeDriver = try QWEngineDriver(
                packPath: goldens.pack_dir, maxTokens: 1024, ropeRows: 2048,
                kvMode: mode, mtpDrafts: 3, mtpBackend: .standard,
                mtpAdaptiveDepth: false, mtpRelaxedMargin: 0,
                mtpDraftStrategy: "mtp", prefillChunkSize: 32)
            let baseTokens = try baseModeDriver.generate(
                request: QWGenerationRequest(prompt: goldens.prompt_tokens, maxNew: 8, sampling: .greedy),
                onToken: nil
            ).tokens
            let specRes = try specModeDriver.generate(
                request: QWGenerationRequest(
                    prompt: goldens.prompt_tokens, maxNew: 8, sampling: .greedy,
                    speculative: QWSpeculativeParameters(
                        enabled: true, mtpDrafts: 3, mtpBackend: "standard",
                        mtpRelaxedMargin: 0, mtpDraftStrategy: "mtp"
                    )
                ),
                onToken: nil
            )
            XCTAssertEqual(specRes.tokens, baseTokens, "Speculative NextN diverged from base in kvMode=\(mode)")
            XCTAssertFalse(specRes.stepTokenCounts.isEmpty, "Speculative NextN must record stepTokenCounts in kvMode=\(mode)")
        }
        print(String(
            format: "[GigaChat4Oracle] prefill_max_diff=%.6f decode_max_diff=%.6f nextn_max_diff=%.6f",
            preDiff, worstDecodeDiff, worstNextNDiff
        ))
    }

    func testDFlash2DrafterAutoResolutionWithTargetPackPath() throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-gc4-dflash-auto-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        let targetDir = tmpRoot.appendingPathComponent("SyntheticGC4-37B-A4B-hq4g32")
        let drafterRoot = tmpRoot.appendingPathComponent("SyntheticGC4-37B-A4B-DFlash2-shvq4g32")
        let drafterDir = drafterRoot.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: drafterDir, withIntermediateDirectories: true)
        var stubQWP = Data("QWP1".utf8)
        let hdrJSON = Data("{\"provenance\":{},\"tensors\":{}}".utf8)
        var hdrLen = UInt64(hdrJSON.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in stubQWP.append(contentsOf: raw) }
        stubQWP.append(hdrJSON)
        try stubQWP.write(to: drafterRoot.appendingPathComponent("model.qwp"))

        let targetCfgJSON = """
        {
          "architectures": ["GigaChat4ForCausalLM"],
          "model_type": "gigachat4",
          "vocab_size": 128256,
          "hidden_size": 2560,
          "intermediate_size": 8192,
          "moe_intermediate_size": 768,
          "num_hidden_layers": 32,
          "first_k_dense_replace": 1,
          "full_attention_layers": [3, 7, 11, 15, 19, 23, 27, 31],
          "linear_attention_type": "GigaChat4GatedDeltaNet",
          "linear_gating_type": "gated_rmsnorm_sigmoid",
          "linear_sigmoid_gate_scale": 2.0,
          "linear_num_key_heads": 16,
          "linear_num_value_heads": 32,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4,
          "num_attention_heads": 32,
          "num_key_value_heads": 32,
          "q_lora_rank": 1536,
          "kv_lora_rank": 512,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "index_n_heads": 64,
          "index_head_dim": 128,
          "index_topk": 2048,
          "indexer_norm_type": "LlamaRMSNorm",
          "n_routed_experts": 192,
          "n_shared_experts": 1,
          "num_experts_per_tok": 8,
          "routed_scaling_factor": 2.5,
          "norm_type": "GatedNorm",
          "layernorm_type": "pre_post",
          "layernorm_gating_weight": 2.0,
          "gated_attention": true,
          "use_mla_scaling_factor": true,
          "swiglu_limit": 10.0,
          "final_logit_softcapping": 30.0,
          "max_position_embeddings": 1048576,
          "rms_norm_eps": 1e-6,
          "rope_theta": 100000.0,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let targetDesc = try QWModelDescriptor.decode(configJSON: targetCfgJSON)

        let drafterCfg = """
        {
          "architectures": ["GigaChat4DFlashDraftModel"],
          "model_type": "dflash_gigachat4",
          "vocab_size": 128256,
          "hidden_size": 2560,
          "intermediate_size": 6912,
          "num_hidden_layers": 5,
          "num_attention_heads": 32,
          "num_key_value_heads": 32,
          "q_lora_rank": 1536,
          "kv_lora_rank": 512,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "block_size": 16,
          "dflash_config": {
            "target_layer_ids": [1, 8, 15, 22, 29]
          }
        }
        """.data(using: .utf8)!
        try drafterCfg.write(to: drafterDir.appendingPathComponent("config.json"))

        let resolvedSibling = QWDFlashFootprint.resolveDrafterPackPath(
            targetDescriptor: targetDesc,
            explicitPath: nil,
            targetPackPath: targetDir.path,
            environment: [:]
        )
        XCTAssertEqual(
            URL(fileURLWithPath: resolvedSibling).standardizedFileURL.path,
            drafterRoot.standardizedFileURL.path
        )

        let home = NSHomeDirectory()
        let prodDrafter = home + "/.qwg/models/GigaChat4-37B-A4B-DFlash2-shvq4g32"
        if FileManager.default.fileExists(atPath: prodDrafter) {
            let resolvedProd = QWDFlashFootprint.resolveDrafterPackPath(
                targetDescriptor: targetDesc,
                explicitPath: nil,
                targetPackPath: home + "/.qwg/models/GigaChat4-37B-A4B-hq4g32",
                environment: [:]
            )
            XCTAssertEqual(
                URL(fileURLWithPath: resolvedProd).standardizedFileURL.path,
                URL(fileURLWithPath: prodDrafter).standardizedFileURL.path
            )
        }
    }
}

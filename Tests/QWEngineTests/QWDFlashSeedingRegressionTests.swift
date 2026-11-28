// QWDFlashSeedingRegressionTests.swift — Regression tests for DFlash2/DSpine prefill seeding and round guards.
import CryptoKit
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL
import QWHead
import QWMTP
import QWQuant
@testable import QWEngine

final class QWDFlashSeedingRegressionTests: XCTestCase {
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

    private static func writeQWP1(
        at rootDir: URL,
        provenance: [String: Any],
        planes: [(String, PlaneSpec)]
    ) throws {
        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 8)
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
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let hdrData = try JSONSerialization.data(
            withJSONObject: ["provenance": provenance, "tensors": entries],
            options: [.sortedKeys]
        )
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (buf: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: buf) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    private static func buildSyntheticTargetPack(at rootDir: URL, maxPos: Int = 256) throws -> QWModelDescriptor {
        let h = 256, v = 256, inter = 256
        let qHeads = 4, kvHeads = 2, headDim = 64
        let cfgJSON = """
        {
          "architectures": ["Qwen3ForCausalLM"],
          "model_type": "qwen3",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": \(maxPos),
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
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

        let planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02)),
            ("model.layers.0.input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 100, amp: 0.05, offset: 1.0))),
            ("model.layers.0.post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 101, amp: 0.05, offset: 1.0))),
            ("model.layers.0.self_attn.q_proj.weight", .hq4(n: g.qProjOut, k: h, seed: 102, scale: 0.008)),
            ("model.layers.0.self_attn.k_proj.weight", .hq4(n: g.kvHeads * g.headDim, k: h, seed: 103, scale: 0.008)),
            ("model.layers.0.self_attn.v_proj.weight", .hq4(n: g.kvHeads * g.headDim, k: h, seed: 104, scale: 0.008)),
            ("model.layers.0.self_attn.o_proj.weight", .hq4(n: h, k: g.qHeads * g.headDim, seed: 105, scale: 0.008)),
            ("model.layers.0.self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            ("model.layers.0.self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            ("model.layers.0.mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 106, scale: 0.008)),
            ("model.layers.0.mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 107, scale: 0.008)),
            ("model.layers.0.mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 108, scale: 0.008)),
        ]
        try writeQWP1(
            at: rootDir,
            provenance: [
                "model_id": "qwen3",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-dflash-seeding",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            planes: planes
        )
        return descriptor
    }

    private static func buildSyntheticDFlash2Pack(at rootDir: URL, slidingWindow: Int = 64) throws {
        let h = 256, v = 256, inter = 256, r = 256
        let qHeads = 2, kvHeads = 2, headDim = 128
        let convK = 2, convG = 16, convProjDim = 2 * convK * (h / convG)
        let cfgJSON = """
        {
          "model_type": "qwen3",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "sliding_window": \(slidingWindow),
          "dflash_config": {
            "block_size": 8,
            "mask_token_id": 3,
            "target_layer_ids": [0],
            "conv_kernel_size": \(convK),
            "conv_group_size": \(convG),
            "selector_rank": \(r),
            "selector_top_k": 8
          }
        }
        """.data(using: .utf8)!
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        try cfgJSON.write(to: assetsDir.appendingPathComponent("config.json"))
        let drafterDesc = try QWModelDescriptor.decodeDrafter(configJSON: cfgJSON)

        let planes: [(String, PlaneSpec)] = [
            ("fc.weight", .hq4(n: h, k: h, seed: 201, scale: 0.008)),
            ("hidden_norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("candidate_selector.hidden_projection.weight", .bf16(shape: [r, h], values: detVec(r * h, seedInit: 202, amp: 0.05))),
            ("candidate_selector.predecessor_codebook", .bf16(shape: [v, r], values: detVec(v * r, seedInit: 203, amp: 0.05))),
            ("candidate_selector.successor_codebook", .bf16(shape: [v, r], values: detVec(v * r, seedInit: 204, amp: 0.05))),
            ("layers.0.input_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("layers.0.post_attention_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("layers.0.self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            ("layers.0.self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            ("layers.0.attention_conv.base_kernel", .bf16(shape: [2 * convK, h], values: detVec(2 * convK * h, seedInit: 205, amp: 0.05, offset: 0.5))),
            ("layers.0.attention_conv.kernel_projection.weight", .bf16(shape: [convProjDim, h], values: detVec(convProjDim * h, seedInit: 206, amp: 0.02))),
            ("layers.0.mlp_conv.base_kernel", .bf16(shape: [2 * convK, h], values: detVec(2 * convK * h, seedInit: 207, amp: 0.05, offset: 0.5))),
            ("layers.0.mlp_conv.kernel_projection.weight", .bf16(shape: [convProjDim, h], values: detVec(convProjDim * h, seedInit: 208, amp: 0.02))),
            ("layers.0.self_attn.q_proj.weight", .hq4(n: qHeads * headDim, k: h, seed: 209, scale: 0.008)),
            ("layers.0.self_attn.k_proj.weight", .hq4(n: kvHeads * headDim, k: h, seed: 210, scale: 0.008)),
            ("layers.0.self_attn.v_proj.weight", .hq4(n: kvHeads * headDim, k: h, seed: 211, scale: 0.008)),
            ("layers.0.self_attn.o_proj.weight", .hq4(n: h, k: qHeads * headDim, seed: 212, scale: 0.008)),
            ("layers.0.mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 213, scale: 0.008)),
            ("layers.0.mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 214, scale: 0.008)),
            ("layers.0.mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 215, scale: 0.008)),
        ]
        try writeQWP1(
            at: rootDir,
            provenance: ["model_id": "dflash2", "geometry_digest": drafterDesc.geometryDigestV2Hex],
            planes: planes
        )
    }

    func testDFlashConfigValidationAndZeroMTPCacheEstimate() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dflash-cpu-\(UUID().uuidString)")
        let targetDir = baseDir.appendingPathComponent("target")
        let dflashDir = baseDir.appendingPathComponent("drafter-dflash2")
        defer { try? FileManager.default.removeItem(at: baseDir) }

        let targetDesc = try Self.buildSyntheticTargetPack(at: targetDir, maxPos: 256)
        try Self.buildSyntheticDFlash2Pack(at: dflashDir, slidingWindow: 64)
        XCTAssertNoThrow(
            try QWDFlashConfig.validateDrafter(
                targetDescriptor: targetDesc, packPath: dflashDir.path, drafts: 7, selectorTopK: 8)
        )

        let startup = try QWEngineDriver.resolveStartup(
            packPath: targetDir.path, kvMode: .q4, mtpDrafts: 7, mtpBackend: .dflash2,
            prefillScratchEviction: .never, mtpAdaptiveDepth: false,
            mtpAdaptiveEntropyThreshold: nil, dflashDraftPackPath: dflashDir.path, dflashSelectorTopK: 8)
        XCTAssertEqual(startup.mtpBackend, .dflash2)
        XCTAssertEqual(startup.dflashDraftPackPath, dflashDir.path)

        let discovered = QWDFlashConfig.defaultPackPath(for: targetDesc, modelsDirectory: baseDir)
        XCTAssertEqual(
            URL(fileURLWithPath: discovered).resolvingSymlinksInPath().path,
            dflashDir.resolvingSymlinksInPath().path
        )

        let footprint = QWEngineDriver.dflashFootprintBytes(
            draftPackPath: dflashDir.path, descriptor: targetDesc, mtpDrafts: 7,
            isDFlash: true, isDSpineBackend: false, physicalMemoryBytes: 64 << 30)
        let zeroFootprint = QWEngineDriver.dflashFootprintBytes(
            draftPackPath: dflashDir.path, descriptor: targetDesc, mtpDrafts: 0,
            isDFlash: true, isDSpineBackend: false, physicalMemoryBytes: 64 << 30)
        XCTAssertGreaterThan(footprint.weights, 0)
        XCTAssertGreaterThan(footprint.scratch, 0)
        XCTAssertEqual(zeroFootprint.weights, 0)
        XCTAssertEqual(zeroFootprint.scratch, 0)

        let geom = QWModelGeometry(descriptor: targetDesc)
        let dflashKVBytes = QWEngineDriver.kvBudgetBytes(
            maxTokens: 256, mtpDrafts: 0, kvMode: .q4, geometry: geom)
        let nextNKVBytes = QWEngineDriver.kvBudgetBytes(
            maxTokens: 256, mtpDrafts: 7, kvMode: .q4, geometry: geom)
        XCTAssertGreaterThan(nextNKVBytes, dflashKVBytes)
    }

    func testDFlashPrefillSeedingAndHasValidMTPCacheGuardsWithoutPromptEmbeddings() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic DFlash2 driver seeding regression test requires bare-metal Apple Silicon GPU"
        )
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dflash-seeding-\(UUID().uuidString)")
        let targetDir = baseDir.appendingPathComponent("target")
        let dflashDir = baseDir.appendingPathComponent("drafter")
        defer { try? FileManager.default.removeItem(at: baseDir) }

        _ = try Self.buildSyntheticTargetPack(at: targetDir, maxPos: 256)
        try Self.buildSyntheticDFlash2Pack(at: dflashDir, slidingWindow: 64)

        let driver = try QWEngineDriver(
            packPath: targetDir.path, maxTokens: 256, ropeRows: 256, kvMode: .q4,
            mtpDrafts: 4, mtpBackend: .dflash2, mtpAdaptiveDepth: false,
            mtpRelaxedMargin: 0, dflashDraftPackPath: dflashDir.path,
            dflashSelectorTopK: 8, prefillChunkSize: 32)
        XCTAssertNotNil(driver.dflashDrafter)
        XCTAssertNil(driver.mtpCache)
        XCTAssertNil(driver.mtpPromptEmbeddings)
        XCTAssertFalse(driver.isWeightFreeSpeculation)

        // Multi-chunk prefill (32 + 8) with mtpCache == nil && mtpPromptEmbeddings == nil.
        let prompt = (0..<40).map { 10 + ($0 % 200) }
        let session = try driver.makeSession(promptTokens: 64)
        XCTAssertNotNil(session.dflashSlot)
        XCTAssertNil(session.mtpKV)
        XCTAssertEqual(session.dflashCacheOffset, 0)
        _ = try driver.prefill(session, tokens: prompt)
        XCTAssertEqual(session.position, prompt.count)
        XCTAssertEqual(session.dflashCacheOffset, prompt.count)
        XCTAssertTrue(session.trunkHiddenValid)
        XCTAssertTrue(driver.canRunSpeculativeRound(session))

        // Both prepareGreedyRound and commitPreparedVerification accept dflashDrafter != nil && mtpCache == nil.
        let round = try driver.prepareGreedyRound(session, maxNew: 8)
        try driver.commitPreparedVerification(session, round: round, stochastic: false, permit: nil)
        XCTAssertEqual(session.dflashCacheOffset, session.position)

        // Unsynced dflashCacheOffset still fails the boundary guard cleanly.
        session.dflashCacheOffset = 0
        XCTAssertFalse(driver.canRunSpeculativeRound(session))
        XCTAssertThrowsError(try driver.prepareGreedyRound(session, maxNew: 8)) { err in
            XCTAssertEqual(err as? QWEngineDriverError,
                           .state("greedy round requires a seeded boundary and valid output budget"))
        }
        driver.finishSession(session)

        // Full greedy and stochastic multi-chunk speculative generation parity.
        let baseSession = try driver.makeSession(promptTokens: 64)
        let baseline = try driver.generate(session: baseSession, prompt: prompt, maxNew: 10)
        driver.finishSession(baseSession)
        XCTAssertEqual(baseline.count, 10)

        let specSession = try driver.makeSession(promptTokens: 64)
        let specOut = try driver.generateExperimentalGreedy(session: specSession, prompt: prompt, maxNew: 10)
        XCTAssertEqual(specSession.dflashCacheOffset, specSession.position)
        driver.finishSession(specSession)
        XCTAssertEqual(specOut, baseline)

        try driver.configurePrefixCache(byteLimit: 128 << 20, entryLimit: 4)
        XCTAssertNotNil(driver.prefixStore)
        let cachedOut = try driver.generateCachedGreedy(prompt: prompt, maxNew: 10, speculative: true)
        XCTAssertEqual(cachedOut, baseline)

        let sampler = QWSamplerConfig(temperature: 0.7, topK: 8, topP: 0.9, seed: 42, uniformVersion: .upper24)
        let stochSession = try driver.makeStochasticSession(promptTokens: 64, config: sampler)
        let stochOut = try driver.generateExperimentalStochastic(
            session: stochSession, prompt: prompt, maxNew: 10, speculative: true)
        XCTAssertEqual(stochSession.dflashCacheOffset, stochSession.position)
        driver.finishSession(stochSession)
        XCTAssertEqual(stochOut.count, 10)
    }

    func testDFlashLargeChunkPrefillSliceSeedingAt1024Tokens() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic DFlash2 1024-token slice seeding test requires bare-metal Apple Silicon GPU"
        )
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dflash-slice-\(UUID().uuidString)")
        let targetDir = baseDir.appendingPathComponent("target")
        let dflashDir = baseDir.appendingPathComponent("drafter")
        defer { try? FileManager.default.removeItem(at: baseDir) }

        _ = try Self.buildSyntheticTargetPack(at: targetDir, maxPos: 1280)
        try Self.buildSyntheticDFlash2Pack(at: dflashDir, slidingWindow: 64)

        let driver = try QWEngineDriver(
            packPath: targetDir.path, maxTokens: 1280, ropeRows: 1280, kvMode: .q4,
            mtpDrafts: 4, mtpBackend: .dflash2, mtpAdaptiveDepth: false,
            mtpRelaxedMargin: 0, dflashDraftPackPath: dflashDir.path,
            dflashSelectorTopK: 8, prefillChunkSize: 1024)
        XCTAssertNil(driver.mtpPromptEmbeddings)
        XCTAssertNil(driver.mtpCache)

        let prompt = (0..<1024).map { 10 + ($0 % 200) }
        let baseSession = try driver.makeSession(promptTokens: 1040)
        let baseline = try driver.generate(session: baseSession, prompt: prompt, maxNew: 6)
        driver.finishSession(baseSession)

        let specSession = try driver.makeSession(promptTokens: 1040)
        let specOut = try driver.generateExperimentalGreedy(session: specSession, prompt: prompt, maxNew: 6)
        XCTAssertEqual(specSession.position, 1029)
        XCTAssertEqual(specSession.dflashCacheOffset, 1029)
        XCTAssertTrue(specSession.trunkHiddenValid)
        driver.finishSession(specSession)
        XCTAssertEqual(specOut, baseline)
    }

    func testDFlashPrefillCaptureSliceWindowingAndMultiChunkSlotByteParity() throws {
        XCTAssertNil(QWDFlashConfig.prefillCaptureSlice(startPos: 0, rows: 2048, windowEnd: 5000, maxCacheSize: 2047))
        XCTAssertNil(QWDFlashConfig.prefillCaptureSlice(startPos: 0, rows: 2953, windowEnd: 5000, maxCacheSize: 2047))
        XCTAssertEqual(
            QWDFlashConfig.prefillCaptureSlice(startPos: 2048, rows: 2048, windowEnd: 5000, maxCacheSize: 2047),
            QWDFlashConfig.PrefillCaptureSlice(localStart: 905, captureRows: 1143, ctxStartPos: 2953)
        )
        XCTAssertEqual(
            QWDFlashConfig.prefillCaptureSlice(startPos: 4096, rows: 904, windowEnd: 5000, maxCacheSize: 2047),
            QWDFlashConfig.PrefillCaptureSlice(localStart: 0, captureRows: 904, ctxStartPos: 4096)
        )
        XCTAssertEqual(
            QWDFlashConfig.prefillCaptureSlice(startPos: 0, rows: 4096, windowEnd: 4096, maxCacheSize: 2047),
            QWDFlashConfig.PrefillCaptureSlice(localStart: 2049, captureRows: 2047, ctxStartPos: 2049)
        )
        XCTAssertNil(QWDFlashConfig.prefillCaptureSlice(startPos: 0, rows: 0, windowEnd: 512, maxCacheSize: 2047))

        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic DFlash2 windowed capture GPU parity test requires bare-metal Apple Silicon GPU"
        )
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dflash-win-\(UUID().uuidString)")
        let targetDir = baseDir.appendingPathComponent("target")
        let dflashDir = baseDir.appendingPathComponent("drafter")
        defer { try? FileManager.default.removeItem(at: baseDir) }

        _ = try Self.buildSyntheticTargetPack(at: targetDir, maxPos: 256)
        try Self.buildSyntheticDFlash2Pack(at: dflashDir, slidingWindow: 64)

        let driver = try QWEngineDriver(
            packPath: targetDir.path, maxTokens: 256, ropeRows: 256, kvMode: .q4,
            mtpDrafts: 4, mtpBackend: .dflash2, mtpAdaptiveDepth: false,
            mtpRelaxedMargin: 0, dflashDraftPackPath: dflashDir.path,
            dflashSelectorTopK: 8, prefillChunkSize: 64)
        let chunk = driver.prefillChunkSize
        let prompt = (0..<160).map { 10 + ($0 % 200) }

        // Reference: force every chunk to update context (dflashWindowEnd = nil -> pos + T).
        let unwindowed = try driver.makeSession(promptTokens: 256)
        try driver.requestGate.withRequest(flight: driver.flight) { permit in
            for off in stride(from: 0, to: prompt.count, by: chunk) {
                let count = min(chunk, prompt.count - off)
                _ = try driver.prefillChunk(
                    unwindowed, tokens: prompt, start: off, count: count,
                    runsHead: off + count == prompt.count, mode: .chunked, permit: permit,
                    dflashWindowEnd: nil)
            }
        }
        let unwinSlot = try XCTUnwrap(unwindowed.dflashSlot)
        let refK = Data(bytes: unwinSlot.kSlab.region.pointer.advanced(by: unwinSlot.kSlab.offset),
                        count: unwinSlot.kSlab.bytes)
        let refV = Data(bytes: unwinSlot.vSlab.region.pointer.advanced(by: unwinSlot.vSlab.offset),
                        count: unwinSlot.vSlab.bytes)
        let refOffset = unwindowed.dflashCacheOffset
        driver.finishSession(unwindowed)

        // Windowed prefill: chunks before prompt.count - maxCacheSize (192 - 63 = 129) skip capture & ctx update.
        let windowed = try driver.makeSession(promptTokens: 256)
        let winFirst = try driver.prefill(windowed, tokens: prompt)
        let winSlot = try XCTUnwrap(windowed.dflashSlot)
        let gotK = Data(bytes: winSlot.kSlab.region.pointer.advanced(by: winSlot.kSlab.offset),
                        count: winSlot.kSlab.bytes)
        let gotV = Data(bytes: winSlot.vSlab.region.pointer.advanced(by: winSlot.vSlab.offset),
                        count: winSlot.vSlab.bytes)
        XCTAssertEqual(windowed.dflashCacheOffset, refOffset)
        XCTAssertEqual(gotK, refK, "windowed DFlash2 prefill must preserve exact ring K slab bytes")
        XCTAssertEqual(gotV, refV, "windowed DFlash2 prefill must preserve exact ring V slab bytes")
        driver.finishSession(windowed)

        let baseSession = try driver.makeSession(promptTokens: 256)
        let baseline = try driver.generate(session: baseSession, prompt: prompt, maxNew: 8)
        driver.finishSession(baseSession)
        XCTAssertEqual(winFirst, baseline[0])

        let specSession = try driver.makeSession(promptTokens: 256)
        let specOut = try driver.generateExperimentalGreedy(session: specSession, prompt: prompt, maxNew: 8)
        driver.finishSession(specSession)
        XCTAssertEqual(specOut, baseline)
    }
}

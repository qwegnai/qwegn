// QWDFlashMLADrafterTests.swift — Synthetic 2-layer MLA DFlash 2 SafeTensors
// end-to-end execution (propose, enqueueGreedyPropose journal, prefix state) and
// GigaChat4 DFlash 2 pack config/footprint/auto-discovery validation tests.
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWQuant

final class QWDFlashMLADrafterTests: XCTestCase {
    private static func bf16Bits(_ v: Float) -> UInt16 { QWDFlashMLATests.bf16Bits(v) }
    private static func bf16ToF32(_ b: UInt16) -> Float { QWDFlashMLATests.bf16ToF32(b) }

    private func sharedBuffer<T>(_ dev: any MTLDevice, _ values: [T]) throws -> any MTLBuffer {
        let bytes = max(16, values.count * MemoryLayout<T>.stride)
        let buf = try XCTUnwrap(dev.makeBuffer(length: bytes, options: .storageModeShared))
        let dst = buf.contents()
        withExtendedLifetime(buf) {
            values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let src = raw.baseAddress, raw.count > 0 {
                    dst.copyMemory(from: src, byteCount: raw.count)
                }
            }
        }
        return buf
    }

    private func zeroBuffer(_ dev: any MTLDevice, byteCount: Int) throws -> any MTLBuffer {
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(16, byteCount), options: .storageModeShared))
        memset(buf.contents(), 0, buf.length)
        return buf
    }

    private static func writeSafeTensorsFile(
        at url: URL, tensors: [(name: String, shape: [Int], data: [UInt16])]
    ) throws {
        var headerObj: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            let byteLen = t.data.count * 2
            headerObj[t.name] = ["dtype": "BF16", "shape": t.shape, "data_offsets": [offset, offset + byteLen]]
            offset += byteLen
        }
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        var out = Data()
        var lenLE = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &lenLE) { (raw: UnsafeRawBufferPointer) in out.append(raw.bindMemory(to: UInt8.self)) }
        out.append(hdrData)
        for t in tensors {
            t.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in out.append(raw.bindMemory(to: UInt8.self)) }
        }
        try out.write(to: url)
    }

    private func skipUnlessMetal() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "DFlash 2 MLA GPU tests require bare-metal Apple Silicon GPU"
        )
    }

    func testSyntheticEndToEndMLADrafterSafeTensorsProposeJournalAndPrefix() throws {
        try skipUnlessMetal()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_mla_e2e_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let H = 128, I = 256, V = 64, L = 2, qHeads = 2
        let qLora = 64, kvLora = 128, nopeDim = 64, ropeDim = 64, vDim = 64
        let selRank = 32, nRank = 16, convGroups = H / 16, convProjDim = 4 * convGroups
        let cfgJSON = """
        {
          "architectures": ["GigaChat4DFlash2DraftModel"],
          "model_type": "qwen3",
          "hidden_size": \(H),
          "intermediate_size": \(I),
          "num_hidden_layers": \(L),
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": 1,
          "qk_head_dim": \(nopeDim + ropeDim),
          "qk_nope_head_dim": \(nopeDim),
          "qk_rope_head_dim": \(ropeDim),
          "v_head_dim": \(vDim),
          "q_lora_rank": \(qLora),
          "kv_lora_rank": \(kvLora),
          "gated_attention": true,
          "use_mla_scaling_factor": true,
          "norm_type": "GatedNorm",
          "layernorm_gating_weight": 2.0,
          "layernorm_type": "pre_post",
          "swiglu_limit": 10.0,
          "vocab_size": \(V),
          "sliding_window": 32,
          "final_logit_softcapping": 30.0,
          "rope_interleave": true,
          "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 100000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "mscale": 1.0,
            "mscale_all_dim": 1.0
          },
          "dflash_config": {
            "attention_mode": "mla",
            "block_size": 8,
            "mask_token_id": 63,
            "target_layer_ids": [0, 2],
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": \(selRank),
            "selector_top_k": 16,
            "final_logit_softcapping": 30.0,
            "selector_softcap_mode": "before_correction"
          }
        }
        """
        try cfgJSON.write(to: tmpDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        var seed = 1
        func makeBF16(_ shape: [Int], scale: Float = 0.0625) -> [UInt16] {
            let count = shape.reduce(1, *)
            var arr = [UInt16](repeating: 0, count: count)
            for i in 0..<count {
                seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0x7fff_ffff
                arr[i] = Self.bf16Bits(Float((seed % 9) - 4) * scale)
            }
            return arr
        }
        var tensors: [(name: String, shape: [Int], data: [UInt16])] = []
        func addGatedNorm(_ pfx: String, _ dim: Int) {
            tensors.append((pfx + ".weight", [dim], [UInt16](repeating: Self.bf16Bits(1.0), count: dim)))
            tensors.append((pfx + ".gate_up_projection.weight", [nRank, dim], makeBF16([nRank, dim])))
            tensors.append((pfx + ".gate_down_projection.weight", [dim, nRank], makeBF16([dim, nRank])))
        }
        tensors.append(("fc.weight", [H, 2 * H], makeBF16([H, 2 * H])))
        addGatedNorm("hidden_norm", H)
        addGatedNorm("norm", H)
        tensors.append(("candidate_selector.hidden_projection.weight", [selRank, H], makeBF16([selRank, H])))
        tensors.append(("candidate_selector.predecessor_codebook", [V, selRank], makeBF16([V, selRank])))
        tensors.append(("candidate_selector.successor_codebook", [V, selRank], makeBF16([V, selRank])))
        for l in 0..<L {
            let p = "layers.\(l)."
            addGatedNorm(p + "input_layernorm", H)
            addGatedNorm(p + "self_attn.q_a_layernorm", qLora)
            addGatedNorm(p + "self_attn.kv_a_layernorm", kvLora)
            addGatedNorm(p + "post_self_attn_layernorm", H)
            addGatedNorm(p + "post_attention_layernorm", H)
            addGatedNorm(p + "post_feedforward_layernorm", H)
            tensors.append((p + "attention_conv.base_kernel", [2, 2, H], makeBF16([2, 2, H], scale: 0.25)))
            tensors.append((p + "attention_conv.kernel_projection.weight", [convProjDim, H], makeBF16([convProjDim, H])))
            tensors.append((p + "mlp_conv.base_kernel", [2, 2, H], makeBF16([2, 2, H], scale: 0.25)))
            tensors.append((p + "mlp_conv.kernel_projection.weight", [convProjDim, H], makeBF16([convProjDim, H])))
            tensors.append((p + "self_attn.q_a_proj.weight", [qLora, H], makeBF16([qLora, H])))
            tensors.append((p + "self_attn.kv_a_proj_with_mqa.weight", [kvLora + ropeDim, H], makeBF16([kvLora + ropeDim, H])))
            tensors.append((p + "self_attn.gate_proj.weight", [qHeads * vDim, H], makeBF16([qHeads * vDim, H])))
            tensors.append((p + "self_attn.q_b_proj.weight", [qHeads * (nopeDim + ropeDim), qLora],
                            makeBF16([qHeads * (nopeDim + ropeDim), qLora])))
            tensors.append((p + "self_attn.kv_b_proj.weight", [qHeads * (nopeDim + vDim), kvLora],
                            makeBF16([qHeads * (nopeDim + vDim), kvLora])))
            tensors.append((p + "self_attn.o_proj.weight", [H, qHeads * vDim], makeBF16([H, qHeads * vDim])))
            tensors.append((p + "mlp.gate_proj.weight", [I, H], makeBF16([I, H])))
            tensors.append((p + "mlp.up_proj.weight", [I, H], makeBF16([I, H])))
            tensors.append((p + "mlp.down_proj.weight", [H, I], makeBF16([H, I])))
        }
        try Self.writeSafeTensorsFile(at: tmpDir.appendingPathComponent("model.safetensors"), tensors: tensors)

        let targetDesc = QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3", capabilities: [],
            hidden: H, vocab: V, denseIntermediate: I, numLayers: 4,
            layerTypes: Array(repeating: "full_attention", count: 4),
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 2, kvHeads: 2, headDim: 64, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 100_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 1
        )
        let geo = QWModelGeometry(descriptor: targetDesc)
        let hal = try QWHal(geometry: geo, weightBasis: .standard)
        let drafter = try QWDFlashDrafter(geometry: geo, hal: hal, packPath: tmpDir.path, drafts: 7, sessionSlotCount: 2)
        XCTAssertTrue(drafter.config.isMLA)
        XCTAssertEqual(drafter.config.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(drafter.config.swigluLimit, 10.0)

        let dev = hal.device
        let embedBuf = try sharedBuffer(dev, makeBF16([V, H], scale: 0.25))
        let headBuf = try sharedBuffer(dev, makeBF16([V, H], scale: 0.25).map { Float16(Self.bf16ToF32($0)).bitPattern })
        let headStorage = QWHeadStorage.f16(QWTensorValue(
            region: QWMRegion(buffer: headBuf, base: 0, length: headBuf.length),
            offset: 0, shape: QWTensorShape([V * H], .f16)
        ))

        let ctxRows = 4
        let tapBuf = try sharedBuffer(dev, (0..<(ctxRows * H)).map { Float((($0 * 5 + 1) % 7) - 3) * 0.25 })
        let tapTensor = QWTensorValue(
            region: QWMRegion(buffer: tapBuf, base: 0, length: tapBuf.length),
            offset: 0, shape: QWTensorShape([ctxRows * H], .f32)
        )
        let cbCtx = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let encCap = try XCTUnwrap(cbCtx.makeComputeCommandEncoder())
        drafter.encodeCaptureTargetHidden(encCap, source: tapTensor, srcRowOffset: 0, dstRowOffset: 0, rows: ctxRows, slot: 0)
        drafter.encodeCaptureTargetHidden(encCap, source: tapTensor, srcRowOffset: 0, dstRowOffset: 0, rows: ctxRows, slot: 1)
        encCap.endEncoding()
        try drafter.encodeContextUpdate(cbCtx, slot: drafter.sessionSlots[0], startPosition: 0, totalCount: ctxRows)
        cbCtx.commit()
        cbCtx.waitUntilCompleted()

        let headClosure = { (
            enc: any MTLComputeCommandEncoder, xRot16: QWTensorValue, xNorm16: QWTensorValue,
            xRot32: QWTensorValue?, yLogits32: QWTensorValue, rows: Int, vocabRows: Int
        ) throws in
            try QWDFlashDrafter.encodeSlicedHeadLogits(
                enc, hal: hal, headStorage: headStorage, geometry: geo,
                xRot16: xRot16, xNorm16: xNorm16, xRot32: xRot32,
                yLogits32: yLogits32, rows: rows, vocabRows: vocabRows
            )
        }
        let embedClosure = { (
            enc: any MTLComputeCommandEncoder, propX32: QWTensorValue, anchor: Int, mask: Int, rows: Int
        ) throws in
            drafter.encodeStagePropBF16(
                enc, tableBuffer: embedBuf, byteOffset: 0, propX: propX32,
                anchorToken: anchor, maskToken: mask, rows: rows
            )
        }

        let greedy = try drafter.propose(
            hal: hal, slot: drafter.sessionSlots[0], cacheOffset: ctxRows,
            pendingToken: 5, draftCount: 7, headStorage: headStorage,
            encodeHeadLogits: headClosure, encodeAnchorMaskEmbed: embedClosure
        )
        XCTAssertEqual(greedy.drafts.count, 7)
        XCTAssertTrue(greedy.drafts.allSatisfy { (0..<V).contains($0) })

        let journalBuf = try zeroBuffer(dev, byteCount: 64)
        let cbJ = try drafter.enqueueGreedyPropose(
            hal: hal, slot: drafter.sessionSlots[0], cacheOffset: ctxRows,
            pendingToken: 5, draftCount: 7, headStorage: headStorage,
            encodeHeadLogits: headClosure, encodeAnchorMaskEmbed: embedClosure,
            journalBuffer: journalBuf, journalByteOffset: 8
        )
        cbJ.waitUntilCompleted()
        let jPtr = journalBuf.contents().advanced(by: 8).assumingMemoryBound(to: Int32.self)
        XCTAssertEqual((0..<7).map { Int(jPtr[$0]) }, greedy.drafts)

        let stream = QWMTPRandomStream(seed: 42)
        try stream.beginGumbelRound(drafts: 7, vocabulary: V)
        let stoch = try drafter.propose(
            hal: hal, slot: drafter.sessionSlots[0], cacheOffset: ctxRows,
            pendingToken: 5, draftCount: 7, headStorage: headStorage,
            encodeHeadLogits: headClosure, encodeAnchorMaskEmbed: embedClosure,
            stochastic: (temperature: 0.7, stream: stream)
        )
        XCTAssertEqual(stoch.drafts.count, 7)
        XCTAssertEqual(stoch.proposals.count, 7)

        let blob = drafter.capturePrefixBlob(slot: drafter.sessionSlots[0], tokenCount: ctxRows)
        XCTAssertTrue(QWDFlashDrafter.validatePrefixBlob(
            data: blob, expectedMarker: drafter.prefixGeometryMarker(), expectedTokens: ctxRows, config: drafter.config
        ))
        XCTAssertEqual(drafter.restorePrefixBlob(blob, into: drafter.sessionSlots[1], expectedTokens: ctxRows), ctxRows)
        let restored = try drafter.propose(
            hal: hal, slot: drafter.sessionSlots[1], cacheOffset: ctxRows,
            pendingToken: 5, draftCount: 7, headStorage: headStorage,
            encodeHeadLogits: headClosure, encodeAnchorMaskEmbed: embedClosure
        )
        XCTAssertEqual(restored.drafts, greedy.drafts)
    }

    func testRealGigaChat4DFlash2PackConfigValidationFootprintAndAutoDiscovery() throws {
        let home = NSHomeDirectory()
        let realPack = home + "/.qwg/models/GigaChat4-37B-A4B-DFlash2-shvq4g32"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: realPack))

        let url = URL(fileURLWithPath: realPack)
        let cfg = QWDFlashConfig.load(from: url)
        XCTAssertEqual(cfg.attentionMode, .mla)
        XCTAssertTrue(cfg.isMLA)
        XCTAssertEqual(cfg.hiddenSize, 2560)
        XCTAssertEqual(cfg.numLayers, 5)
        XCTAssertEqual(cfg.numAttentionHeads, 32)
        XCTAssertEqual(cfg.headDim, 192)
        XCTAssertEqual(cfg.qLoraRank, 1536)
        XCTAssertEqual(cfg.kvLoraRank, 512)
        XCTAssertEqual(cfg.qkNopeHeadDim, 128)
        XCTAssertEqual(cfg.qkRopeHeadDim, 64)
        XCTAssertEqual(cfg.vHeadDim, 128)
        XCTAssertTrue(cfg.gatedAttention)
        XCTAssertTrue(cfg.useMLAScalingFactor)
        XCTAssertEqual(cfg.alphaQ, Float((2560.0 / 1536.0).squareRoot()), accuracy: 1e-6)
        XCTAssertEqual(cfg.alphaKV, Float((2560.0 / 512.0).squareRoot()), accuracy: 1e-6)
        XCTAssertEqual(cfg.gatedNormRank, 16)
        XCTAssertEqual(cfg.layernormGatingWeight, 2.0)
        XCTAssertTrue(cfg.useSandwichNorm)
        XCTAssertEqual(cfg.swigluLimit, 10.0)
        XCTAssertTrue(cfg.ropeInterleave)
        XCTAssertEqual(cfg.ropeYarn?.factor, 32.0)
        XCTAssertEqual(cfg.ropeYarn?.originalMaxPositionEmbeddings, 32768)
        XCTAssertEqual(cfg.softmaxScale, 0.1308608, accuracy: 1e-5)
        XCTAssertEqual(cfg.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(cfg.selectorSoftcapMode, .beforeCorrection)
        XCTAssertTrue(cfg.shiftLabel)
        XCTAssertEqual(cfg.targetLayerIds, [1, 8, 15, 22, 29])
        XCTAssertEqual(cfg.maskTokenId, 128016)
        XCTAssertEqual(QWDFlashFootprint.kvBytesPerTokenPerLayer(config: cfg), 1152)
        XCTAssertEqual(QWDFlashFootprint.kvBytesPerToken(config: cfg), 5760)

        let gc4Target = QWModelDescriptor(
            family: .qwen3MoE, modelType: "qwen3_moe", capabilities: [.moe],
            hidden: 2560, vocab: 128256, denseIntermediate: 8192, numLayers: 30,
            layerTypes: Array(repeating: "full_attention", count: 30),
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 131072,
            qHeads: 32, kvHeads: 1, headDim: 192, attnOutputGate: true,
            partialRotaryFactor: 1.0, ropeTheta: 100_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 128, expertsPerTok: 8, moeIntermediate: 1024,
            sharedExpertIntermediate: 2048, expertGateType: "sigmoid", hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2
        )
        XCTAssertNoThrow(try QWDFlashConfig.validateDrafter(
            targetDescriptor: gc4Target, packPath: realPack, drafts: 7, selectorTopK: 16
        ))
        let wBytes = QWDFlashFootprint.drafterWeightBytes(packPath: realPack, isDSpine: false)
        XCTAssertGreaterThan(wBytes, 450 << 20)
        XCTAssertLessThan(wBytes, 600 << 20)

        let autoSibling = QWDFlashConfig.resolveAutoDrafterPath(
            for: gc4Target,
            targetPackPath: home + "/.qwg/models/GigaChat4-37B-A4B-shvq4g32"
        )
        XCTAssertEqual(URL(fileURLWithPath: autoSibling).standardizedFileURL.path, url.standardizedFileURL.path)

        let hfSnap = home + "/.cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B/snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc"
        if FileManager.default.fileExists(atPath: hfSnap + "/dflash_draft/config.json") {
            let autoNested = QWDFlashConfig.resolveAutoDrafterPath(for: gc4Target, targetPackPath: hfSnap)
            XCTAssertEqual(
                URL(fileURLWithPath: autoNested).standardizedFileURL.path,
                URL(fileURLWithPath: hfSnap + "/dflash_draft").standardizedFileURL.path
            )
            XCTAssertNoThrow(try QWDFlashConfig.validateDrafter(
                targetDescriptor: gc4Target, packPath: autoNested, drafts: 7
            ))
        }
    }

    func testPyTorchOracleMLADrafterParity() throws {
        try skipUnlessMetal()
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let parentCheckout = repoRoot.deletingLastPathComponent().deletingLastPathComponent()
        let pyCandidates = [
            ProcessInfo.processInfo.environment["QW_ORACLE_PYTHON"],
            repoRoot.appendingPathComponent(".venv-oracle/bin/python3").path,
            parentCheckout.appendingPathComponent(".venv-oracle/bin/python3").path,
        ].compactMap { $0 }
        guard let py = pyCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw XCTSkip("Oracle Python (.venv-oracle/bin/python3) not installed on this host")
        }
        let scriptURL = repoRoot.appendingPathComponent("Scripts/oracle/gigachat4_dflash2_oracle.py")
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_gc4_dflash2_oracle_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: py)
        proc.arguments = [scriptURL.path, "--out-dir", dir.path]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        let procOut = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let msg = String(data: procOut, encoding: .utf8) ?? "<binary>"
            XCTFail("gigachat4_dflash2_oracle.py failed (\(proc.terminationStatus)): \(msg)")
            return
        }

        let H = 128, I = 256, V = 64, ctxRows = 4
        let targetDesc = QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3", capabilities: [],
            hidden: H, vocab: V, denseIntermediate: I, numLayers: 4,
            layerTypes: Array(repeating: "full_attention", count: 4),
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 2, kvHeads: 2, headDim: 64, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 100_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 1
        )
        let geo = QWModelGeometry(descriptor: targetDesc)
        let hal = try QWHal(geometry: geo, weightBasis: .standard)
        let drafter = try QWDFlashDrafter(geometry: geo, hal: hal, packPath: dir.path, drafts: 7, sessionSlotCount: 1)
        XCTAssertTrue(drafter.config.shiftLabel)

        let dev = hal.device
        func loadBin(_ name: String) throws -> any MTLBuffer {
            let data = try Data(contentsOf: dir.appendingPathComponent(name))
            let buf = try XCTUnwrap(dev.makeBuffer(length: max(16, data.count), options: .storageModeShared))
            let dst = buf.contents()
            withExtendedLifetime(buf) {
                data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    if let src = raw.baseAddress, raw.count > 0 { dst.copyMemory(from: src, byteCount: raw.count) }
                }
            }
            return buf
        }
        let embedBuf = try loadBin("embed_bf16.bin")
        let headBuf = try loadBin("head_f16.bin")
        let tap0Buf = try loadBin("tap0_f16.bin")
        let tap1Buf = try loadBin("tap1_f16.bin")
        let expTokensData = try Data(contentsOf: dir.appendingPathComponent("draft_tokens.bin"))
        let expTokens = expTokensData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self)).map(Int.init)
        }
        let expHiddenData = try Data(contentsOf: dir.appendingPathComponent("pred_hidden.bin"))
        let expHidden = expHiddenData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Float.self))
        }

        let tap0 = QWTensorValue(region: QWMRegion(buffer: tap0Buf, base: 0, length: tap0Buf.length),
                                 offset: 0, shape: QWTensorShape([ctxRows * H], .f16))
        let tap1 = QWTensorValue(region: QWMRegion(buffer: tap1Buf, base: 0, length: tap1Buf.length),
                                 offset: 0, shape: QWTensorShape([ctxRows * H], .f16))
        let headStorage = QWHeadStorage.f16(QWTensorValue(
            region: QWMRegion(buffer: headBuf, base: 0, length: headBuf.length),
            offset: 0, shape: QWTensorShape([V * H], .f16)
        ))

        let cbCtx = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let encCap = try XCTUnwrap(cbCtx.makeComputeCommandEncoder())
        drafter.encodeCaptureTargetHidden(encCap, source: tap0, srcRowOffset: 0, dstRowOffset: 0, rows: ctxRows, slot: 0)
        drafter.encodeCaptureTargetHidden(encCap, source: tap1, srcRowOffset: 0, dstRowOffset: 0, rows: ctxRows, slot: 1)
        encCap.endEncoding()
        try drafter.encodeContextUpdate(cbCtx, slot: drafter.sessionSlots[0], startPosition: 0, totalCount: ctxRows)
        cbCtx.commit()
        cbCtx.waitUntilCompleted()

        let greedy = try drafter.propose(
            hal: hal, slot: drafter.sessionSlots[0], cacheOffset: ctxRows,
            pendingToken: 5, draftCount: 7, headStorage: headStorage,
            encodeHeadLogits: { enc, xRot16, xNorm16, xRot32, yLogits32, rows, vocabRows in
                try QWDFlashDrafter.encodeSlicedHeadLogits(
                    enc, hal: hal, headStorage: headStorage, geometry: geo,
                    xRot16: xRot16, xNorm16: xNorm16, xRot32: xRot32,
                    yLogits32: yLogits32, rows: rows, vocabRows: vocabRows
                )
            },
            encodeAnchorMaskEmbed: { enc, propX32, anchor, mask, rows in
                drafter.encodeStagePropBF16(
                    enc, tableBuffer: embedBuf, byteOffset: 0, propX: propX32,
                    anchorToken: anchor, maskToken: mask, rows: rows
                )
            }
        )
        let normPtr = drafter.propNorm.region.buffer.contents()
            .advanced(by: drafter.propNorm.region.base + drafter.propNorm.offset)
            .assumingMemoryBound(to: UInt16.self)
        var maxDiff: Float = 0
        for i in 0..<(7 * H) {
            maxDiff = max(maxDiff, abs(Float(Float16(bitPattern: normPtr[i])) - expHidden[i]))
        }
        XCTAssertLessThan(maxDiff, 5.0e-2, "MLA DFlash2 pred_hidden diverged from PyTorch SGLang reference: \(maxDiff)")
        XCTAssertEqual(greedy.drafts, expTokens)
    }

    func testValidateDrafterRejectsMLASRAMCeilingOverflow() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_mla_sram_guard_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let badCfg = """
        {
          "architectures": ["GigaChat4DFlash2DraftModel"],
          "model_type": "qwen3",
          "hidden_size": 2560,
          "intermediate_size": 8192,
          "num_hidden_layers": 2,
          "num_attention_heads": 32,
          "num_key_value_heads": 1,
          "qk_head_dim": 192,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "q_lora_rank": 1536,
          "kv_lora_rank": 1024,
          "norm_type": "GatedNorm",
          "vocab_size": 64,
          "dflash_config": {"block_size": 8, "mask_token_id": 1, "target_layer_ids": [1, 2]}
        }
        """
        try Data(badCfg.utf8).write(to: tmpDir.appendingPathComponent("config.json"))
        let target = QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3", capabilities: [],
            hidden: 2560, vocab: 64, denseIntermediate: 8192, numLayers: 4,
            layerTypes: Array(repeating: "full_attention", count: 4),
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 32, kvHeads: 8, headDim: 128, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 100_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 1
        )
        XCTAssertThrowsError(try QWDFlashConfig.validateDrafter(
            targetDescriptor: target, packPath: tmpDir.path, drafts: 7
        )) { err in
            XCTAssertTrue("\(err)".contains("SRAM ceilings"), "\(err)")
        }
    }
}

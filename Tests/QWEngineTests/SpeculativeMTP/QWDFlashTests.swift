import CryptoKit
import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWMTP
import QWQuant
@testable import QWEngine

final class QWDFlashTests: XCTestCase {
    func testDFlashConfigConstantsAndPipelines() throws {
        let cfg = QWDFlashConfig()
        XCTAssertEqual(cfg.numLayers, 5)
        XCTAssertEqual(cfg.blockSize, 8)
        XCTAssertEqual(cfg.hiddenSize, 5120)
        XCTAssertEqual(cfg.intermediateSize, 17408)
        XCTAssertEqual(cfg.targetLayerIds, [5, 19, 33, 47, 61])
        XCTAssertEqual(cfg.slidingWindow, 2048)
        XCTAssertEqual(cfg.maxCacheSize, 2047)
        XCTAssertEqual(cfg.maskTokenId, 248070)
        XCTAssertEqual(cfg.selectorRank, 256)
        XCTAssertEqual(cfg.selectorTopK, 16)

        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "DFlash pipeline check requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        for name in [
            "qwg_dflash_capture_target_hidden",
            "qwg_dflash_rmsnorm",
            "qwg_dflash_f16_gemv_batched",
            "qwg_dflash_conv_apply",
            "qwg_dflash_rope_kv_ctx",
            "qwg_dflash_rope_qk_prop",
            "qwg_dflash_attn",
            "qwg_dflash_top16_stage1",
            "qwg_dflash_top16_stage2",
            "qwg_dflash_selector_greedy"
        ] {
            XCTAssertNoThrow(try hal.pipeline(name), "Missing DFlash pipeline \(name)")
        }
    }

    func testDFlashDrafterLoadsPackedCheckpoint() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Local ~/.qwg/models pack check requires bare-metal Apple Silicon GPU"
        )
        let packPath = QWDFlashConfig.defaultPackPath(for: .qwen38_27b)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: packPath),
                          "DFlash 2 pack not installed at \(packPath)")
        let hal = try QWHal(geometry: .qwen38_27b)
        let drafter = try QWDFlashDrafter(geometry: QWTestGeometry.g27, hal: hal, packPath: packPath, selectorTopK: 16)
        let expectedLayerIds = drafter.config.isDSpine ? [1, 16, 31, 46, 61] : [5, 19, 33, 47, 61]
        XCTAssertEqual(drafter.config.numLayers, 5)
        XCTAssertEqual(drafter.config.targetLayerIds, expectedLayerIds)
        XCTAssertEqual(drafter.sessionSlots.count, 2)
        for lid in expectedLayerIds {
            XCTAssertGreaterThanOrEqual(drafter.targetLayerSlotMap[lid], 0)
        }
    }

    func testValidateDrafterFailsFastOnIncompatibleTargetModelOrMissingPack() throws {
        let flashNextDesc = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.moe],
            hidden: 2560, vocab: 248320,
            denseIntermediate: nil, numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 262144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 512, expertsPerTok: 10, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248044)
        let dense27BDesc = QWModelDescriptor.qwen38_27b

        // 1. Incompatible primary model (48-layer Flash-Next vs 5120/64-layer or mismatched pack) must fail with QWDFlashError.state
        let dense27BDrafterPath = QWDFlashConfig.defaultPackPath(for: dense27BDesc)
        let mismatchPackPath: String
        var cleanupMismatchDir: URL?
        if FileManager.default.fileExists(atPath: dense27BDrafterPath) {
            mismatchPackPath = dense27BDrafterPath
        } else {
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("qwdflash_mismatch_\(UUID().uuidString)")
            let assets = tmp.appendingPathComponent("assets")
            try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
            let cfg: [String: Any] = [
                "hidden_size": 1024, "num_hidden_layers": 5, "vocab_size": 248320,
                "dflash_config": ["block_size": 8, "target_layer_ids": [1, 5, 10, 15, 20]]
            ]
            try JSONSerialization.data(withJSONObject: cfg).write(to: assets.appendingPathComponent("config.json"))
            FileManager.default.createFile(atPath: tmp.appendingPathComponent("model.qwp").path, contents: Data())
            mismatchPackPath = tmp.path
            cleanupMismatchDir = tmp
        }
        defer { if let dir = cleanupMismatchDir { try? FileManager.default.removeItem(at: dir) } }
        XCTAssertThrowsError(
            try QWDFlashConfig.validateDrafter(
                targetDescriptor: flashNextDesc,
                packPath: mismatchPackPath,
                drafts: 4,
                selectorTopK: 16
            )
        ) { error in
            guard case QWDFlashError.state(let msg) = error else {
                XCTFail("Expected QWDFlashError.state, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("does not match target") || msg.contains("out of range"), msg)
        }

        // 2. Non-existent drafter pack path must fail fast with .packMissing
        XCTAssertThrowsError(
            try QWDFlashConfig.validateDrafter(
                targetDescriptor: dense27BDesc,
                packPath: "/nonexistent/dflash-pack",
                drafts: 4,
                selectorTopK: 16
            )
        ) { error in
            guard case QWDFlashError.packMissing = error else {
                XCTFail("Expected QWDFlashError.packMissing, got \(error)")
                return
            }
        }

        // 3. Invalid draft count (> maxDrafts) must fail fast with .state
        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == nil,
           FileManager.default.fileExists(atPath: dense27BDrafterPath) {
            let cfg = QWDFlashConfig.load(from: URL(fileURLWithPath: dense27BDrafterPath))
            let expectedMax = max(7, cfg.blockSize - 1)
            XCTAssertThrowsError(
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: dense27BDesc,
                    packPath: dense27BDrafterPath,
                    drafts: expectedMax + 1,
                    selectorTopK: 16
                )
            ) { error in
                guard case QWDFlashError.state(let msg) = error else {
                    XCTFail("Expected QWDFlashError.state, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("1...\(expectedMax)"))
            }

            // 4. Valid 27B Dense + installed DFlash2/DSpine pack succeeds on CPU in <1ms
            XCTAssertNoThrow(
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: dense27BDesc,
                    packPath: dense27BDrafterPath,
                    drafts: expectedMax,
                    selectorTopK: 16
                )
            )
        }
    }

    func testDSpineConfigParsingAndPipelines() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_dspine_cfg_\(UUID().uuidString)")
        let assetsDir = tmpDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let dspineConfigJSON = """
        {
          "num_hidden_layers": 5,
          "block_size": 16,
          "hidden_size": 5120,
          "intermediate_size": 17408,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000000.0,
          "partial_rotary_factor": 0.25,
          "sliding_window": 8192,
          "dflash_config": {
            "target_layer_ids": [1, 16, 31, 46, 61],
            "mask_token_id": 248070,
            "draft_block_causal": true,
            "selected_prefix_code_bridge": {
              "code_size": 1024,
              "eps": 1e-6,
              "kappa": 10.0,
              "rank": 512
            },
            "selected_prefix_last_write_refine": true,
            "code_bridge_second_pass": false
          }
        }
        """
        try dspineConfigJSON.write(to: assetsDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let cfg = QWDFlashConfig.load(from: tmpDir)
        XCTAssertTrue(cfg.isDSpine)
        XCTAssertTrue(cfg.draftBlockCausal)
        XCTAssertEqual(cfg.numLayers, 5)
        XCTAssertEqual(cfg.blockSize, 16)
        XCTAssertEqual(cfg.targetLayerIds, [1, 16, 31, 46, 61])
        XCTAssertEqual(cfg.slidingWindow, 8192)
        XCTAssertEqual(cfg.maxCacheSize, 8191)
        XCTAssertEqual(cfg.bridgeCodeSize, 1024)
        XCTAssertEqual(cfg.bridgeRank, 512)
        XCTAssertEqual(cfg.bridgeEps, 1e-6, accuracy: 1e-12)

        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "DSpine pipeline check requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        for name in [
            "qwg_dspine_bridge_state",
            "qwg_dspine_bridge_write_add",
            "qwg_dspine_refine_code_to_msg",
            "qwg_dspine_refine_write",
            "qwg_dspine_refine_scores",
            "qwg_dspine_refine_chain"
        ] {
            XCTAssertNoThrow(try hal.pipeline(name), "Missing DSpine pipeline \(name)")
        }
    }

    func testDSpineFusedBridgeAndRefinementKernelsParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "DSpine kernel parity test requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let dev = hal.device
        let H5120 = 5120
        let H = 64
        let C = 32
        let R = 32
        let M = 3
        let D = 2
        let topK = 16
        let eps: Float = 1e-6

        let pRms = try hal.pipeline("qwg_dflash_rmsnorm")
        let pState = try hal.pipeline("qwg_dspine_bridge_state")
        let pWrite = try hal.pipeline("qwg_dspine_bridge_write_add")
        let pRefMsg = try hal.pipeline("qwg_dspine_refine_code_to_msg")
        let pRefWrite = try hal.pipeline("qwg_dspine_refine_write")
        let pRefScores = try hal.pipeline("qwg_dspine_refine_scores")
        let pRefChain = try hal.pipeline("qwg_dspine_refine_chain")

        // 1. Test qwg_dflash_rmsnorm mode 4 (x += dOut, save rms(f32), y = x / (rms + eps))
        let x5120Buf = dev.makeBuffer(length: M * H5120 * 4, options: .storageModeShared)!
        let dOut5120Buf = dev.makeBuffer(length: M * H5120 * 2, options: .storageModeShared)!
        let y5120Buf = dev.makeBuffer(length: M * H5120 * 2, options: .storageModeShared)!
        let rmsBuf = dev.makeBuffer(length: M * 4, options: .storageModeShared)!

        let x5120Ptr = x5120Buf.contents().bindMemory(to: Float.self, capacity: M * H5120)
        let d5120Ptr = dOut5120Buf.contents().bindMemory(to: Float16.self, capacity: M * H5120)
        for i in 0..<(M * H5120) {
            x5120Ptr[i] = Float((i % 13) - 6) * 0.1
            d5120Ptr[i] = Float16(Float((i % 7) - 3) * 0.05)
        }

        let cb1 = hal.queue.makeCommandBuffer()!
        let enc1 = cb1.makeComputeCommandEncoder()!
        var vEps = eps, vMode4: UInt32 = 4, vHidden = UInt32(H5120)
        enc1.setComputePipelineState(pRms)
        enc1.setBuffer(x5120Buf, offset: 0, index: 0)
        enc1.setBuffer(dOut5120Buf, offset: 0, index: 1)
        enc1.setBuffer(y5120Buf, offset: 0, index: 2)
        enc1.setBuffer(rmsBuf, offset: 0, index: 3)
        enc1.setBytes(&vEps, length: 4, index: 4)
        enc1.setBytes(&vMode4, length: 4, index: 5)
        enc1.setBytes(&vHidden, length: 4, index: 6)
        enc1.setBuffer(rmsBuf, offset: 0, index: 7)
        enc1.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc1.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        let y5120Ptr = y5120Buf.contents().bindMemory(to: Float16.self, capacity: M * H5120)
        let rmsPtr = rmsBuf.contents().bindMemory(to: Float.self, capacity: M)
        for m in 0..<M {
            var ss: Float = 0
            for i in 0..<H5120 {
                let raw = Float(((m * H5120 + i) % 13) - 6) * 0.1 + Float(d5120Ptr[m * H5120 + i])
                ss += raw * raw
            }
            let rms = sqrt(ss / Float(H5120))
            XCTAssertEqual(rmsPtr[m], rms, accuracy: 1e-4)
            for i in [0, 17, 511, 5119] {
                let expectedY = x5120Ptr[m * H5120 + i] / (rms + eps)
                XCTAssertEqual(Float(y5120Ptr[m * H5120 + i]), expectedY, accuracy: 1e-2)
            }
        }

        // 2. Test qwg_dspine_bridge_state (mode 0 & mode 1) + qwg_dspine_bridge_write_add
        let codebookBuf = dev.makeBuffer(length: 64 * C * 2, options: .storageModeShared)!
        let cbPtr = codebookBuf.contents().bindMemory(to: UInt16.self, capacity: 64 * C)
        for i in 0..<(64 * C) {
            let f = Float((i % 11) - 5) * 0.2
            cbPtr[i] = UInt16(f.bitPattern >> 16)
        }
        let projBuf = dev.makeBuffer(length: D * C * 4, options: .storageModeShared)!
        let prPtr = projBuf.contents().bindMemory(to: Float.self, capacity: D * C)
        for i in 0..<(D * C) { prPtr[i] = Float((i % 9) - 4) * 0.25 }
        let codesBuf = dev.makeBuffer(length: M * C * 2, options: .storageModeShared)!
        let packedBuf = dev.makeBuffer(length: M * 3 * R * 4, options: .storageModeShared)!
        let pkPtr = packedBuf.contents().bindMemory(to: Float.self, capacity: M * 3 * R)
        for i in 0..<(M * 3 * R) { pkPtr[i] = Float((i % 5) - 2) * 0.3 }
        let gateBiasBuf = dev.makeBuffer(length: R * 2, options: .storageModeShared)!
        let gbPtr = gateBiasBuf.contents().bindMemory(to: Float16.self, capacity: R)
        for r in 0..<R { gbPtr[r] = Float16(0.1) }
        let msgBuf = dev.makeBuffer(length: D * R * 2, options: .storageModeShared)!
        let wWriteBuf = dev.makeBuffer(length: H * R * 2, options: .storageModeShared)!
        let wwPtr = wWriteBuf.contents().bindMemory(to: Float16.self, capacity: H * R)
        for i in 0..<(H * R) { wwPtr[i] = Float16(Float((i % 7) - 3) * 0.05) }
        let xDraftBuf = dev.makeBuffer(length: D * H * 4, options: .storageModeShared)!
        let xDraftPtr = xDraftBuf.contents().bindMemory(to: Float.self, capacity: D * H)
        for i in 0..<(D * H) { xDraftPtr[i] = Float((i % 11) - 5) * 0.1 }
        let xOutBuf = dev.makeBuffer(length: D * H * 4, options: .storageModeShared)!

        var anchorTok: Int32 = 3
        var vH = UInt32(H), vC = UInt32(C), vR = UInt32(R), vD = UInt32(D), vTopK = UInt32(topK)
        var vMode0: UInt32 = 0, vMode1: UInt32 = 1
        let cb2 = hal.queue.makeCommandBuffer()!
        let enc2 = cb2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(pState)
        enc2.setBuffer(projBuf, offset: 0, index: 0)
        enc2.setBuffer(codebookBuf, offset: 0, index: 1)
        enc2.setBuffer(codesBuf, offset: 0, index: 2)
        enc2.setBytes(&anchorTok, length: 4, index: 3)
        enc2.setBytes(&vC, length: 4, index: 4)
        enc2.setBytes(&vEps, length: 4, index: 5)
        enc2.setBytes(&vMode0, length: 4, index: 6)
        enc2.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        enc2.setBuffer(packedBuf, offset: 0, index: 0)
        enc2.setBuffer(gateBiasBuf, offset: 0, index: 1)
        enc2.setBuffer(msgBuf, offset: 0, index: 2)
        enc2.setBytes(&anchorTok, length: 4, index: 3)
        enc2.setBytes(&vR, length: 4, index: 4)
        enc2.setBytes(&vEps, length: 4, index: 5)
        enc2.setBytes(&vMode1, length: 4, index: 6)
        enc2.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        enc2.setComputePipelineState(pWrite)
        enc2.setBuffer(wWriteBuf, offset: 0, index: 0)
        enc2.setBuffer(msgBuf, offset: 0, index: 1)
        enc2.setBuffer(xDraftBuf, offset: 0, index: 2)
        enc2.setBuffer(rmsBuf, offset: 4, index: 3)
        enc2.setBuffer(xOutBuf, offset: 0, index: 4)
        enc2.setBytes(&vH, length: 4, index: 5)
        enc2.setBytes(&vR, length: 4, index: 6)
        enc2.setBytes(&vD, length: 4, index: 7)
        enc2.dispatchThreadgroups(MTLSize(width: H / 4, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        let codesPtr = codesBuf.contents().bindMemory(to: Float16.self, capacity: M * C)
        let msgPtr = msgBuf.contents().bindMemory(to: Float16.self, capacity: D * R)
        let sqrtC = sqrt(Float(C))
        for c in 0..<C {
            let expectedAnchor = sqrtC * Float(bitPattern: UInt32(cbPtr[Int(anchorTok) * C + c]) << 16)
            XCTAssertEqual(Float(codesPtr[c]), expectedAnchor, accuracy: 1e-2)
        }
        for d in 0..<D {
            for r in 0..<R {
                let q = pkPtr[(d + 1) * 3 * R + r]
                let s = pkPtr[d * 3 * R + R + r]
                let v = pkPtr[d * 3 * R + 2 * R + r]
                let b = Float(gbPtr[r])
                let g = 1.0 / (1.0 + exp(-(q + s + b)))
                XCTAssertEqual(Float(msgPtr[d * R + r]), g * v, accuracy: 1e-2)
            }
        }

        // 3. Test fused refinement tail (code_to_msg + refine_write + refine_scores + refine_chain)
        let candBuf = dev.makeBuffer(length: D * 16 * 4, options: .storageModeShared)!
        let candPtr = candBuf.contents().bindMemory(to: Int32.self, capacity: D * 16)
        for d in 0..<D {
            for c in 0..<16 { candPtr[d * 16 + c] = Int32((d * 16 + c) % 64) }
        }
        let refMsgBuf = dev.makeBuffer(length: D * 16 * R * 2, options: .storageModeShared)!
        let refStatesBuf = dev.makeBuffer(length: D * 16 * H * 4, options: .storageModeShared)!
        let refActBuf = dev.makeBuffer(length: D * 16 * H * 2, options: .storageModeShared)!
        let wPackedBuf = dev.makeBuffer(length: 3 * R * C * 2, options: .storageModeShared)!
        let wpPtr = wPackedBuf.contents().bindMemory(to: Float16.self, capacity: 3 * R * C)
        for i in 0..<(3 * R * C) { wpPtr[i] = Float16(0.0) }
        let lmHeadBuf = dev.makeBuffer(length: 64 * H * 2, options: .storageModeShared)!
        let lmPtr = lmHeadBuf.contents().bindMemory(to: Float16.self, capacity: 64 * H)
        for i in 0..<(64 * H) { lmPtr[i] = Float16(0.0) }
        for i in 0..<H {
            lmPtr[5 * H + i] = Float16(xDraftPtr[0 * H + i] >= 0 ? 1.0 : -1.0)
            lmPtr[27 * H + i] = Float16(xDraftPtr[1 * H + i] >= 0 ? 1.0 : -1.0)
        }
        let tableBuf = dev.makeBuffer(length: D * 256 * 4, options: .storageModeShared)!
        let outTokensBuf = dev.makeBuffer(length: D * 4, options: .storageModeShared)!

        let cb3 = hal.queue.makeCommandBuffer()!
        let enc3 = cb3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(pRefMsg)
        enc3.setBuffer(wPackedBuf, offset: 0, index: 0)
        enc3.setBuffer(codebookBuf, offset: 0, index: 1)
        enc3.setBuffer(candBuf, offset: 0, index: 2)
        enc3.setBuffer(packedBuf, offset: 0, index: 3)
        enc3.setBuffer(gateBiasBuf, offset: 0, index: 4)
        enc3.setBuffer(refMsgBuf, offset: 0, index: 5)
        enc3.setBytes(&vR, length: 4, index: 6)
        enc3.setBytes(&vC, length: 4, index: 7)
        enc3.setBytes(&vTopK, length: 4, index: 8)
        enc3.dispatchThreadgroups(MTLSize(width: R / 4, height: D, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        enc3.setComputePipelineState(pRefWrite)
        enc3.setBuffer(wWriteBuf, offset: 0, index: 0)
        enc3.setBuffer(refMsgBuf, offset: 0, index: 1)
        enc3.setBuffer(xDraftBuf, offset: 0, index: 2)
        enc3.setBuffer(rmsBuf, offset: 4, index: 3)
        enc3.setBuffer(refStatesBuf, offset: 0, index: 4)
        enc3.setBytes(&vH, length: 4, index: 5)
        enc3.setBytes(&vR, length: 4, index: 6)
        enc3.setBytes(&vTopK, length: 4, index: 7)
        enc3.dispatchThreadgroups(MTLSize(width: H / 4, height: D, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc3.endEncoding()
        cb3.commit()
        cb3.waitUntilCompleted()

        let rsPtr = refStatesBuf.contents().bindMemory(to: Float.self, capacity: D * 16 * H)
        let raPtr = refActBuf.contents().bindMemory(to: Float16.self, capacity: D * 16 * H)
        for i in 0..<(D * 16 * H) { raPtr[i] = Float16(rsPtr[i]) }

        var vHeadMode: UInt32 = 0
        let cb4 = hal.queue.makeCommandBuffer()!
        let enc4 = cb4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(pRefScores)
        enc4.setBuffer(lmHeadBuf, offset: 0, index: 0)
        enc4.setBuffer(refActBuf, offset: 0, index: 1)
        enc4.setBuffer(candBuf, offset: 0, index: 2)
        enc4.setBuffer(tableBuf, offset: 0, index: 3)
        enc4.setBytes(&vH, length: 4, index: 4)
        enc4.setBytes(&vTopK, length: 4, index: 5)
        enc4.setBytes(&vHeadMode, length: 4, index: 6)
        enc4.dispatchThreadgroups(MTLSize(width: 16, height: D, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        enc4.setComputePipelineState(pRefChain)
        enc4.setBuffer(tableBuf, offset: 0, index: 0)
        enc4.setBuffer(candBuf, offset: 0, index: 1)
        enc4.setBuffer(outTokensBuf, offset: 0, index: 2)
        enc4.setBytes(&vD, length: 4, index: 3)
        enc4.setBytes(&vTopK, length: 4, index: 4)
        enc4.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc4.endEncoding()
        cb4.commit()
        cb4.waitUntilCompleted()

        let outTokPtr = outTokensBuf.contents().bindMemory(to: Int32.self, capacity: D)
        XCTAssertEqual(outTokPtr[0], 5)
        XCTAssertEqual(outTokPtr[1], 27)
    }

    func testComputeWeightCapacityPreservesRaw2DAndSub2BitPlanes() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_dflash_cap_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let hdrJSON = """
        {
          "provenance": {"source": "unit-test"},
          "tensors": {
            "selected_prefix_bridge.codebook": {"shape": [4096, 1024], "raw_off": 64},
            "fc.weight": {
              "shape": [128, 256],
              "hq4_off": 8388672,
              "hq": {"bits": 15, "group": 64, "vq": "shvq1"}
            }
          }
        }
        """
        let hdrBytes = Data(hdrJSON.utf8)
        var data = Data("QWP1".utf8)
        let len = UInt64(hdrBytes.count)
        for shift in stride(from: 0, to: 64, by: 8) {
            data.append(UInt8(truncatingIfNeeded: len >> shift))
        }
        data.append(Data(count: 64))
        data.append(hdrBytes)
        let qwpURL = tmpDir.appendingPathComponent("model.qwp")
        try data.write(to: qwpURL)

        let pack = try QWQuant.Pack(url: tmpDir, descriptor: .qwen38_27b, skipIdentityCheck: true)
        let cap = QWDFlashDrafter.computeWeightCapacity(pack: pack, isDSpine: true)
        let rawCodebookBytes = 4096 * 1024 * 2
        let shvq15Bytes = 128 * (256 / 64) * 14
        XCTAssertGreaterThanOrEqual(cap, rawCodebookBytes + shvq15Bytes + 64 * 1024 * 1024)
        XCTAssertLessThan(cap, 128 * 1024 * 1024)
    }

    func testDFlashTop16Stage1And2SanitizesNonFiniteLogits() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "DFlash top16 NaN sanitization kernel check requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let pStage1 = try hal.pipeline("qwg_dflash_top16_stage1")
        let pStage2 = try hal.pipeline("qwg_dflash_top16_stage2")
        let D = 2, V = 1024, topK = 16
        let logitsBuf = hal.device.makeBuffer(length: D * V * 4, options: .storageModeShared)!
        let partValBuf = hal.device.makeBuffer(length: D * 256 * 16 * 4, options: .storageModeShared)!
        let partIdBuf = hal.device.makeBuffer(length: D * 256 * 16 * 4, options: .storageModeShared)!
        let unaryBuf = hal.device.makeBuffer(length: D * 16 * 4, options: .storageModeShared)!
        let candBuf = hal.device.makeBuffer(length: D * 16 * 4, options: .storageModeShared)!

        let lPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: D * V)
        for i in 0..<V { lPtr[i] = .nan }
        for i in V..<(2 * V) { lPtr[i] = -.infinity }

        var vocabU = UInt32(V), topKU = UInt32(topK)
        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        enc.setComputePipelineState(pStage1)
        enc.setBuffer(logitsBuf, offset: 0, index: 0)
        enc.setBuffer(partValBuf, offset: 0, index: 1)
        enc.setBuffer(partIdBuf, offset: 0, index: 2)
        enc.setBytes(&vocabU, length: 4, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pStage2)
        enc.setBuffer(partValBuf, offset: 0, index: 0)
        enc.setBuffer(partIdBuf, offset: 0, index: 1)
        enc.setBuffer(unaryBuf, offset: 0, index: 2)
        enc.setBuffer(candBuf, offset: 0, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let cPtr = candBuf.contents().bindMemory(to: Int32.self, capacity: D * 16)
        let uPtr = unaryBuf.contents().bindMemory(to: Float.self, capacity: D * 16)
        for i in 0..<(D * topK) {
            XCTAssertNotEqual(cPtr[i], 0x7fff_ffff)
            XCTAssertTrue((0..<Int32(V)).contains(cPtr[i]), "Out-of-range candidate \(cPtr[i])")
            XCTAssertTrue(uPtr[i].isFinite, "Non-finite unary score at \(i)")
        }
    }

    func testAdmitSessionPrefersDFlashSlotsAcrossSequentialAndConcurrentSessions() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Driver DFlash slot admission test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg-dflash-slots-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildTinyDensePack(at: rootDir)

        let driver = try QWEngineDriver(packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048, kvMode: .q4, mtpDrafts: 0)
        let dev = driver.hal.device
        let dummySlabBuf = dev.makeBuffer(length: 4096, options: .storageModeShared)!
        let dummyTensor = QWTensorValue(region: QWMRegion(buffer: dummySlabBuf, base: 0, length: 4096),
                                        offset: 0, shape: QWTensorShape([128], .f16))
        let dummySlot = QWDFlashSessionSlot(kSlab: dummyTensor, vSlab: dummyTensor, kLayers: [dummyTensor], vLayers: [dummyTensor])
        while driver.freeSlots.count < 4 {
            let tmpl = driver.freeSlots[0]
            driver.freeSlots.append(QWEngineDriver.GdnSlot(recSlab: tmpl.recSlab, ringSlab: tmpl.ringSlab,
                                                           rec: tmpl.rec, ring: tmpl.ring,
                                                           trunkHidden: tmpl.trunkHidden, dflashSlot: nil))
        }
        for i in driver.freeSlots.indices {
            driver.freeSlots[i].dflashSlot = (i < 2) ? dummySlot : nil
        }

        // More than 4 sequential admissions must always claim a dflashSlot != nil slot.
        for seqIdx in 0..<8 {
            let s = try driver.admitSession(promptTokens: 4)
            XCTAssertNotNil(s.dflashSlot, "Sequential session \(seqIdx) missed dflashSlot")
            driver.finishSession(s)
        }

        // 4 concurrent sessions on 2 DFlash slots: first 2 hold dflashSlot, slots 2 & 3 stay valid with nil.
        var active: [QWEngineSession] = []
        defer { for s in active where !s.sessionFinished { driver.finishSession(s) } }
        for _ in 0..<4 {
            active.append(try driver.admitSession(promptTokens: 4))
        }
        XCTAssertNotNil(active[0].dflashSlot)
        XCTAssertNotNil(active[1].dflashSlot)
        XCTAssertNil(active[2].dflashSlot)
        XCTAssertNil(active[3].dflashSlot)
        XCTAssertFalse(driver.canRunSpeculativeRound(active[2]))
    }

    func testDFlashDrafterAllocatesDisjointSessionSlotsForRequestedCount() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Local ~/.qwg/models pack check requires bare-metal Apple Silicon GPU"
        )
        let packPath = QWDFlashConfig.defaultPackPath(for: .qwen38_27b)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: packPath),
                          "DFlash 2 pack not installed at \(packPath)")
        let hal = try QWHal(geometry: .qwen38_27b)
        let drafter2 = try QWDFlashDrafter(
            geometry: QWTestGeometry.g27, hal: hal, packPath: packPath, selectorTopK: 16, sessionSlotCount: 2)
        XCTAssertEqual(drafter2.sessionSlots.count, 2)

        let drafter4 = try QWDFlashDrafter(
            geometry: QWTestGeometry.g27, hal: hal, packPath: packPath, selectorTopK: 16, sessionSlotCount: 4)
        XCTAssertEqual(drafter4.sessionSlots.count, 4)

        var ranges: [(start: Int, end: Int)] = []
        for slot in drafter4.sessionSlots {
            XCTAssertEqual(slot.kLayers.count, drafter4.config.numLayers)
            XCTAssertEqual(slot.vLayers.count, drafter4.config.numLayers)
            for slab in [slot.kSlab, slot.vSlab] {
                let start = slab.region.base + slab.offset
                ranges.append((start, start + slab.bytes))
            }
        }
        let sorted = ranges.sorted { $0.start < $1.start }
        for idx in 1..<sorted.count {
            XCTAssertGreaterThanOrEqual(sorted[idx].start, sorted[idx - 1].end, "DFlash session slabs overlap")
        }
    }

    private static func buildTinyDensePack(at rootDir: URL) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let cfgJSON = """
        {"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","vocab_size":\(v),"hidden_size":\(h),
         "intermediate_size":\(inter),"num_hidden_layers":1,"num_attention_heads":\(qHeads),
         "num_key_value_heads":\(kvHeads),"head_dim":\(headDim),"max_position_embeddings":2048,
         "rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false,"bos_token_id":1,"eos_token_id":2}
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: desc)
        let assets = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        try cfgJSON.write(to: assets.appendingPathComponent("config.json"))
        try Data("{\"version\":\"1.0\"}".utf8).write(to: assets.appendingPathComponent("tokenizer.json"))
        try Data("bos".utf8).write(to: assets.appendingPathComponent("chat_template.jinja"))
        var mf: [String: [String: Any]] = [:]
        for n in ["config.json", "tokenizer.json", "chat_template.jinja"] {
            let d = try Data(contentsOf: assets.appendingPathComponent(n))
            let hex = SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
            mf[n] = ["sha256": hex, "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: mf, options: [.sortedKeys])
        try mfData.write(to: assets.appendingPathComponent("manifest.json"))
        let planes: [(String, Bool, Int, Int)] = [
            ("model.embed_tokens.weight", false, v, h), ("model.norm.weight", false, 1, h),
            ("lm_head.weight", true, v, h),
            ("model.layers.0.input_layernorm.weight", false, 1, h),
            ("model.layers.0.post_attention_layernorm.weight", false, 1, h),
            ("model.layers.0.self_attn.q_proj.weight", true, g.qProjOut, h),
            ("model.layers.0.self_attn.k_proj.weight", true, kvHeads * headDim, h),
            ("model.layers.0.self_attn.v_proj.weight", true, kvHeads * headDim, h),
            ("model.layers.0.self_attn.o_proj.weight", true, h, qHeads * headDim),
            ("model.layers.0.self_attn.q_norm.weight", false, 1, headDim),
            ("model.layers.0.self_attn.k_norm.weight", false, 1, headDim),
            ("model.layers.0.mlp.gate_proj.weight", true, inter, h),
            ("model.layers.0.mlp.up_proj.weight", true, inter, h),
            ("model.layers.0.mlp.down_proj.weight", true, h, inter),
        ]
        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 8)
        var entries: [String: [String: Any]] = [:]
        func pad(_ align: Int) { let r = file.count % align; if r != 0 { file.append(Data(repeating: 0, count: align - r)) } }
        for (name, isQ4, n, k) in planes {
            if isQ4 {
                pad(16_384); let off = file.count; file.append(Data(count: n * (k / 32) * 20)); pad(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            } else {
                pad(256); let off = file.count; let count = name == "model.embed_tokens.weight" ? v * h : n * k
                file.append(Data(count: count * 2)); pad(256)
                entries[name] = ["shape": name.contains("norm") ? [k] : [v, h], "kind": "bf16", "raw_off": off]
            }
        }
        let mfHex = SHA256.hash(data: mfData).map { String(format: "%02x", $0) }.joined()
        let hdr = try JSONSerialization.data(withJSONObject: [
            "provenance": ["model_id": "qwen3", "geometry_digest": desc.geometryDigestV2Hex,
                           "source": "synthetic", "sha256_source": "synthetic", "quant_recorded": "q4g",
                           "assets_dir": "assets", "assets_manifest_sha256": mfHex],
            "tensors": entries
        ], options: [.sortedKeys])
        var hdrLen = UInt64(hdr.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: raw) }
        file.append(hdr)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    func testVirtualLayerCaptureValidationAndSlotMappingForLoopedTarget() throws {
        let lt = (0..<32).map { ($0 + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"" }.joined(separator: ",")
        let loopCfg = """
        {"architectures":["SenseNovaLoopForConditionalGeneration"],"model_type":"sensenova_loop",
         "loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},
         "mhc":{"start":4,"end":28,"streams":2,"iterations":20,"fixed_identity":false,"pre_scale":1.0},
         "text_config":{"model_type":"qwen3_5_text","hidden_size":256,"intermediate_size":256,
          "num_hidden_layers":32,"mtp_num_hidden_layers":0,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,
          "attn_output_gate":true,"vocab_size":256,"rms_norm_eps":1e-6,"rope_theta":1000000,"partial_rotary_factor":0.25,
          "max_position_embeddings":32768,"full_attention_interval":4,"layer_types":[\(lt)],"tie_word_embeddings":false,
          "bos_token_id":1,"eos_token_id":2,"linear_conv_kernel_dim":4,"linear_key_head_dim":32,
          "linear_num_key_heads":2,"linear_num_value_heads":4,"linear_value_head_dim":32}}
        """.data(using: .utf8)!
        let targetDesc = try QWModelDescriptor.decode(configJSON: loopCfg)
        XCTAssertEqual(targetDesc.numLayers, 32)
        XCTAssertEqual(targetDesc.executionPlan.count, 56)

        let modelsDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg-dflash-loop-\(UUID().uuidString)")
        let packDir = modelsDir.appendingPathComponent("SenseNova-Loop-Test-DFlash2-hq4g32")
        let assets = packDir.appendingPathComponent("assets")
        defer { try? FileManager.default.removeItem(at: modelsDir) }
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)

        func makeDraftCfg(_ tapIds: [Int]) -> Data {
            """
            {"model_type":"qwen3","vocab_size":256,"hidden_size":256,"intermediate_size":256,
             "num_hidden_layers":\(tapIds.count),"num_attention_heads":2,"num_key_value_heads":2,"head_dim":128,
             "rms_norm_eps":1e-6,"rope_theta":1000000,"sliding_window":64,
             "dflash_config":{"block_size":8,"target_layer_ids":\(tapIds),"conv_kernel_size":0,"selector_rank":64,"selector_top_k":8}}
            """.data(using: .utf8)!
        }
        let cfg41 = makeDraftCfg([41])
        try cfg41.write(to: assets.appendingPathComponent("config.json"))
        XCTAssertTrue(QWDFlashConfig.load(from: packDir).usesVirtualCapture(for: targetDesc))
        let drafterDesc = try QWModelDescriptor.decodeDrafter(configJSON: cfg41)

        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 8)
        var entries: [String: [String: Any]] = [:]
        func pad(_ a: Int) { let r = file.count % a; if r != 0 { file.append(Data(repeating: 0, count: a - r)) } }
        for n in ["fc.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                  "layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
                  "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight"] {
            pad(16_384); let off = file.count; file.append(Data(count: 256 * 8 * 20)); pad(16_384)
            entries[n] = ["shape": [256, 256], "hq4_off": off]
        }
        let bf16Specs: [(String, [Int])] = [
            ("hidden_norm.weight", [256]), ("norm.weight", [256]),
            ("candidate_selector.hidden_projection.weight", [64, 256]),
            ("candidate_selector.predecessor_codebook", [256, 64]), ("candidate_selector.successor_codebook", [256, 64]),
            ("layers.0.input_layernorm.weight", [256]), ("layers.0.post_attention_layernorm.weight", [256]),
            ("layers.0.self_attn.q_norm.weight", [128]), ("layers.0.self_attn.k_norm.weight", [128]),
        ]
        for (n, shape) in bf16Specs {
            pad(256); let off = file.count; file.append(Data(count: shape.reduce(1, *) * 2)); pad(256)
            entries[n] = ["shape": shape, "kind": "bf16", "raw_off": off]
        }
        let hdr = try JSONSerialization.data(withJSONObject: [
            "provenance": ["model_id": "dflash2", "geometry_digest": drafterDesc.geometryDigestV2Hex],
            "tensors": entries
        ])
        var hdrLen = UInt64(hdr.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: raw) }
        file.append(hdr)
        try file.write(to: packDir.appendingPathComponent("model.qwp"))
        XCTAssertEqual(URL(fileURLWithPath: QWDFlashConfig.defaultPackPath(for: targetDesc, modelsDirectory: modelsDir))
            .resolvingSymlinksInPath().path, packDir.resolvingSymlinksInPath().path)
        XCTAssertNoThrow(try QWDFlashConfig.validateDrafter(targetDescriptor: targetDesc, packPath: packDir.path))
        if QWDeviceCaps.current?.supportsSimdgroupMMA == true {
            let hal = try QWHal(geometry: QWModelGeometry(descriptor: targetDesc))
            let drafter = try QWDFlashDrafter(geometry: QWModelGeometry(descriptor: targetDesc), hal: hal, packPath: packDir.path)
            XCTAssertTrue(drafter.usesVirtualLayerCapture)
            XCTAssertEqual(drafter.targetLayerSlotMap.count, 56)
            XCTAssertEqual(drafter.targetLayerSlotMap[41], 0)
            XCTAssertEqual(drafter.targetLayerSlotMap[9], -1)
        }
        try makeDraftCfg([57]).write(to: assets.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try QWDFlashConfig.validateDrafter(targetDescriptor: targetDesc, packPath: packDir.path))
    }
}

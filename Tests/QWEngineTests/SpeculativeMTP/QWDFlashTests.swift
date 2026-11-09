import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
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
        let packPath = QWDFlashDrafter.defaultPackPath
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

        // 1. Incompatible primary model (Flash-Next) must fail fast with .state
        XCTAssertThrowsError(
            try QWDFlashConfig.validateDrafter(
                targetDescriptor: flashNextDesc,
                packPath: QWDFlashDrafter.defaultPackPath,
                drafts: 4,
                selectorTopK: 16
            )
        ) { error in
            guard case QWEngineDriverError.state(let msg) = error else {
                XCTFail("Expected QWEngineDriverError.state, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("only compatible with dense"))
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
            guard case QWEngineDriverError.packMissing = error else {
                XCTFail("Expected packMissing, got \(error)")
                return
            }
        }

        // 3. Invalid draft count (> maxDrafts) must fail fast with .state
        if FileManager.default.fileExists(atPath: QWDFlashDrafter.defaultPackPath) {
            let cfg = QWDFlashConfig.load(from: URL(fileURLWithPath: QWDFlashDrafter.defaultPackPath))
            let expectedMax = max(7, cfg.blockSize - 1)
            XCTAssertThrowsError(
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: dense27BDesc,
                    packPath: QWDFlashDrafter.defaultPackPath,
                    drafts: expectedMax + 1,
                    selectorTopK: 16
                )
            ) { error in
                guard case QWEngineDriverError.state(let msg) = error else {
                    XCTFail("Expected QWEngineDriverError.state, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("1...\(expectedMax)"))
            }

            // 4. Valid 27B Dense + installed DFlash2/DSpine pack succeeds on CPU in <1ms
            XCTAssertNoThrow(
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: dense27BDesc,
                    packPath: QWDFlashDrafter.defaultPackPath,
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
        var vEps = eps, vMode4: UInt32 = 4
        enc1.setComputePipelineState(pRms)
        enc1.setBuffer(x5120Buf, offset: 0, index: 0)
        enc1.setBuffer(dOut5120Buf, offset: 0, index: 1)
        enc1.setBuffer(y5120Buf, offset: 0, index: 2)
        enc1.setBuffer(rmsBuf, offset: 0, index: 3)
        enc1.setBytes(&vEps, length: 4, index: 4)
        enc1.setBytes(&vMode4, length: 4, index: 5)
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
}

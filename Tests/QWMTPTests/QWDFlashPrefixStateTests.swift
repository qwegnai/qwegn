// QWDFlashPrefixStateTests.swift — Tests for unified DFlash2 prefix-state
// serialization, geometry-hash validation, tokenCount == 0, and resolveDraftVocab.
import Darwin
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWAlloc
import QWCore
import QWHAL

final class QWDFlashPrefixStateTests: XCTestCase {
    private static func bf16Bits(_ value: Float) -> UInt16 {
        UInt16((value.bitPattern &+ 0x8000) >> 16)
    }

    private static func writeSafeTensorsFile(
        to url: URL,
        tensors: [(name: String, shape: [Int], data: [UInt16])]
    ) throws {
        var header: [String: Any] = [:]
        var body = Data()
        for t in tensors {
            let start = body.count
            let byteCount = t.data.count * 2
            t.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                body.append(contentsOf: raw)
            }
            header[t.name] = [
                "dtype": "BF16",
                "shape": t.shape,
                "data_offsets": [start, start + byteCount],
            ]
        }
        var headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + headerBytes.count) % 8 != 0 {
            headerBytes.append(0x20)
        }
        var out = Data()
        var hdrLen = UInt64(headerBytes.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { out.append(contentsOf: $0) }
        out.append(headerBytes)
        out.append(body)
        try out.write(to: url)
    }

    private static func makeTargetDescriptor(vocab: Int = 300, hidden: Int = 128, numLayers: Int = 4) throws -> QWModelDescriptor {
        let cfg: [String: Any] = [
            "model_type": "inkling2_loop",
            "eos_token_id": 2,
            "text_config": [
                "model_type": "inkling2_loop_text",
                "hidden_size": hidden,
                "vocab_size": vocab,
                "intermediate_size": hidden * 2,
                "num_hidden_layers": numLayers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "model_max_length": 1024,
                "rms_norm_eps": 1e-6,
                "local_layer_ids": [0, 1],
                "sliding_window_size": 256,
                "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 2,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "loop_config": [
                    "core_start": 1,
                    "core_end": 3,
                    "num_passes": 2,
                    "gate_rank": 16,
                ],
            ],
        ]
        return try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg))
    }

    private static func makeSyntheticDrafter(
        at dir: URL,
        target: QWModelDescriptor,
        hal: QWHal,
        slidingWindow: Int = 17,
        targetLayerIds: [Int] = [0, 2]
    ) throws -> QWDFlashDrafter {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let H = target.hidden, I = 256, V = target.vocab, R = 64, convProj = 2 * 2 * (H / 16)
        let cfgJSON: [String: Any] = [
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "hidden_size": H,
            "intermediate_size": I,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": V,
            "sliding_window": slidingWindow,
            "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10000],
            "dflash_config": [
                "block_size": 8,
                "mask_token_id": 270,
                "target_layer_ids": targetLayerIds,
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": R,
                "selector_top_k": 8,
            ],
        ]
        try JSONSerialization.data(withJSONObject: cfgJSON).write(to: dir.appendingPathComponent("config.json"))
        let ones = [UInt16](repeating: bf16Bits(1.0), count: H)
        let fill = { (n: Int) in [UInt16](repeating: bf16Bits(0.02), count: n) }
        try writeSafeTensorsFile(
            to: dir.appendingPathComponent("model.safetensors"),
            tensors: [
                ("fc.weight", [H, targetLayerIds.count * H], fill(H * targetLayerIds.count * H)),
                ("hidden_norm.weight", [H], ones),
                ("norm.weight", [H], ones),
                ("candidate_selector.hidden_projection.weight", [R, H], fill(R * H)),
                ("candidate_selector.predecessor_codebook", [V, R], fill(V * R)),
                ("candidate_selector.successor_codebook", [V, R], fill(V * R)),
                ("layers.0.input_layernorm.weight", [H], ones),
                ("layers.0.post_attention_layernorm.weight", [H], ones),
                ("layers.0.self_attn.q_norm.weight", [128], ones),
                ("layers.0.self_attn.k_norm.weight", [128], ones),
                ("layers.0.attention_conv.base_kernel", [2, 2, H], fill(2 * 2 * H)),
                ("layers.0.attention_conv.kernel_projection.weight", [convProj, H], fill(convProj * H)),
                ("layers.0.mlp_conv.base_kernel", [2, 2, H], fill(2 * 2 * H)),
                ("layers.0.mlp_conv.kernel_projection.weight", [convProj, H], fill(convProj * H)),
                ("layers.0.self_attn.q_proj.weight", [128, H], fill(128 * H)),
                ("layers.0.self_attn.k_proj.weight", [128, H], fill(128 * H)),
                ("layers.0.self_attn.v_proj.weight", [128, H], fill(128 * H)),
                ("layers.0.self_attn.o_proj.weight", [H, 128], fill(H * 128)),
                ("layers.0.mlp.gate_proj.weight", [I, H], fill(I * H)),
                ("layers.0.mlp.up_proj.weight", [I, H], fill(I * H)),
                ("layers.0.mlp.down_proj.weight", [H, I], fill(H * I)),
            ]
        )
        return try QWDFlashDrafter(
            geometry: QWModelGeometry(descriptor: target),
            hal: hal,
            packPath: dir.path,
            drafts: 7,
            sessionSlotCount: 1,
            headRotated: false
        )
    }

    private func makeSlot(device: any MTLDevice, config: QWDFlashConfig) throws -> QWDFlashSessionSlot {
        let kvDim = config.numKeyValueHeads * config.headDim
        let layerElems = config.maxCacheSize * kvDim
        let bytes = config.numLayers * layerElems * 2
        guard let kBuf = device.makeBuffer(length: bytes, options: .storageModeShared),
              let vBuf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw XCTSkip("failed to allocate synthetic KV buffers")
        }
        let kReg = QWMRegion(buffer: kBuf, base: 0, length: bytes)
        let vReg = QWMRegion(buffer: vBuf, base: 0, length: bytes)
        let kSlab = QWTensorValue(region: kReg, offset: 0, shape: QWTensorShape([config.numLayers * layerElems], .f16))
        let vSlab = QWTensorValue(region: vReg, offset: 0, shape: QWTensorShape([config.numLayers * layerElems], .f16))
        let kLayers = (0..<config.numLayers).map {
            QWTensorValue(region: kReg, offset: $0 * layerElems * 2, shape: QWTensorShape([layerElems], .f16))
        }
        let vLayers = (0..<config.numLayers).map {
            QWTensorValue(region: vReg, offset: $0 * layerElems * 2, shape: QWTensorShape([layerElems], .f16))
        }
        return QWDFlashSessionSlot(kSlab: kSlab, vSlab: vSlab, kLayers: kLayers, vLayers: vLayers)
    }

    private func populateSlot(_ slot: QWDFlashSessionSlot, config: QWDFlashConfig, seed: Int) {
        let rowBytes = config.numKeyValueHeads * config.headDim * 2
        for (l, (kT, vT)) in zip(slot.kLayers, slot.vLayers).enumerated() {
            let kPtr = kT.region.buffer.contents().advanced(by: kT.region.base + kT.offset).assumingMemoryBound(to: UInt8.self)
            let vPtr = vT.region.buffer.contents().advanced(by: vT.region.base + vT.offset).assumingMemoryBound(to: UInt8.self)
            withExtendedLifetime(kT.region.buffer) {
                withExtendedLifetime(vT.region.buffer) {
                    for b in 0..<(config.maxCacheSize * rowBytes) {
                        kPtr[b] = UInt8((seed + l * 29 + b) & 0xFF)
                        vPtr[b] = UInt8((seed + l * 13 + b * 5) & 0xFF)
                    }
                }
            }
        }
    }

    func testResolveDraftVocabWithDefaultsAndEnvOverride() {
        let prev = getenv("QWG_DRAFT_VOCAB").map { String(cString: $0) }
        defer {
            if let prev { setenv("QWG_DRAFT_VOCAB", prev, 1) } else { unsetenv("QWG_DRAFT_VOCAB") }
        }
        unsetenv("QWG_DRAFT_VOCAB")
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320))
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320, defaultCap: nil))
        XCTAssertEqual(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320, defaultCap: 98_304), 98_304)
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 98_304, defaultCap: 98_304))
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 32_768, defaultCap: 98_304))

        setenv("QWG_DRAFT_VOCAB", "65536", 1)
        XCTAssertEqual(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320, defaultCap: nil), 65_536)
        XCTAssertEqual(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320, defaultCap: 98_304), 65_536)
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 32_768, defaultCap: 98_304))

        setenv("QWG_DRAFT_VOCAB", "0", 1)
        XCTAssertNil(QWDFlashDrafter.resolveDraftVocab(fullVocab: 248_320, defaultCap: 98_304))
    }

    func testPrefixPayloadBytesAndGeometryHash() {
        let cfgA = QWDFlashConfig(
            hiddenSize: 256, numLayers: 3, numAttentionHeads: 4, numKeyValueHeads: 2,
            headDim: 64, intermediateSize: 512, vocabSize: 256,
            slidingWindow: 17, blockSize: 8, maskTokenId: 250, targetLayerIds: [0, 1, 2]
        )
        let cfgB = QWDFlashConfig(
            hiddenSize: 256, numLayers: 3, numAttentionHeads: 4, numKeyValueHeads: 2,
            headDim: 64, intermediateSize: 512, vocabSize: 256,
            slidingWindow: 17, blockSize: 8, maskTokenId: 250, targetLayerIds: [0, 1, 3]
        )
        let hdrBytes = MemoryLayout<QWDFlashPrefixState>.size
        let rowBytes = cfgA.numKeyValueHeads * cfgA.headDim * 2
        XCTAssertEqual(QWDFlashDrafter.prefixPayloadBytes(config: cfgA, tokenCount: 0), hdrBytes)
        XCTAssertEqual(
            QWDFlashDrafter.prefixPayloadBytes(config: cfgA, tokenCount: 5),
            hdrBytes + 2 * 3 * 5 * rowBytes
        )
        XCTAssertEqual(
            QWDFlashDrafter.prefixPayloadBytes(config: cfgA, tokenCount: 50),
            hdrBytes + 2 * 3 * cfgA.maxCacheSize * rowBytes
        )
        let markerA = QWDFlashDrafter.prefixGeometryMarker(config: cfgA)
        let markerB = QWDFlashDrafter.prefixGeometryMarker(config: cfgB)
        XCTAssertNotEqual(QWDFlashDrafter.prefixGeometryHash(markerA), QWDFlashDrafter.prefixGeometryHash(markerB))
        XCTAssertEqual(QWDFlashPrefixState.magic, 0x3156_4b46_4457_5151)
    }

    func testStaticAppendRestoreAndValidationEdgeCases() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let cfg = QWDFlashConfig(
            hiddenSize: 256, numLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2,
            headDim: 64, intermediateSize: 512, vocabSize: 256,
            slidingWindow: 17, blockSize: 8, maskTokenId: 250, targetLayerIds: [0, 1]
        )
        let marker = QWDFlashDrafter.prefixGeometryMarker(config: cfg)
        let srcSlot = try makeSlot(device: dev, config: cfg)
        let dstSlot = try makeSlot(device: dev, config: cfg)
        populateSlot(srcSlot, config: cfg, seed: 11)

        let rowBytes = cfg.numKeyValueHeads * cfg.headDim * 2
        for tokenCount in [0, 6, 40] {
            let expectedBytes = QWDFlashDrafter.prefixPayloadBytes(config: cfg, tokenCount: tokenCount)
            var blob = Data(count: expectedBytes)
            var wOff = 0
            blob.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress else { return }
                QWDFlashDrafter.appendPrefixKV(
                    to: ptr, at: &wOff, slot: srcSlot, config: cfg, marker: marker, tokenCount: tokenCount
                )
            }
            XCTAssertEqual(wOff, expectedBytes)
            XCTAssertTrue(QWDFlashDrafter.validatePrefixBlob(
                data: blob, expectedMarker: marker, expectedTokens: tokenCount, config: cfg
            ))
            XCTAssertFalse(QWDFlashDrafter.validatePrefixBlob(
                data: blob, expectedMarker: marker + ":bad", expectedTokens: tokenCount, config: cfg
            ))
            XCTAssertFalse(QWDFlashDrafter.validatePrefixBlob(
                data: blob, expectedMarker: marker, expectedTokens: tokenCount + 1, config: cfg
            ))

            var rOff = 0
            let restored = blob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
                guard let ptr = raw.baseAddress else { return nil }
                return QWDFlashDrafter.restorePrefixKV(
                    from: ptr, dataCount: raw.count, offset: &rOff, slot: dstSlot,
                    config: cfg, expectedMarker: marker, expectedTokens: tokenCount
                )
            }
            XCTAssertEqual(restored, tokenCount)
            XCTAssertEqual(rOff, expectedBytes)

            let activeBytes = min(tokenCount, cfg.maxCacheSize) * rowBytes
            if activeBytes > 0 {
                for l in 0..<cfg.numLayers {
                    let kS = srcSlot.kLayers[l], kD = dstSlot.kLayers[l]
                    let vS = srcSlot.vLayers[l], vD = dstSlot.vLayers[l]
                    let kSrc = Data(bytes: kS.region.buffer.contents().advanced(by: kS.region.base + kS.offset), count: activeBytes)
                    let kDst = Data(bytes: kD.region.buffer.contents().advanced(by: kD.region.base + kD.offset), count: activeBytes)
                    let vSrc = Data(bytes: vS.region.buffer.contents().advanced(by: vS.region.base + vS.offset), count: activeBytes)
                    let vDst = Data(bytes: vD.region.buffer.contents().advanced(by: vD.region.base + vD.offset), count: activeBytes)
                    XCTAssertEqual(kDst, kSrc)
                    XCTAssertEqual(vDst, vSrc)
                }
            }
        }
    }

    func testDrafterCaptureAndRestorePrefixBlobRoundTripAndGeometryMismatch() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_df_prefix_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let target = try Self.makeTargetDescriptor()
        let hal = try QWHal(geometry: QWModelGeometry(descriptor: target), weightBasis: .rotated)
        let drafterA = try Self.makeSyntheticDrafter(
            at: tmp.appendingPathComponent("a"), target: target, hal: hal, slidingWindow: 17, targetLayerIds: [0, 2]
        )
        let drafterB = try Self.makeSyntheticDrafter(
            at: tmp.appendingPathComponent("b"), target: target, hal: hal, slidingWindow: 17, targetLayerIds: [1, 3]
        )
        let srcSlot = drafterA.sessionSlots[0]
        let dstSlotA = try makeSlot(device: hal.device, config: drafterA.config)
        let dstSlotB = drafterB.sessionSlots[0]
        populateSlot(srcSlot, config: drafterA.config, seed: 37)

        let rowBytes = drafterA.config.numKeyValueHeads * drafterA.config.headDim * 2
        for tokenCount in [0, 7, 40] {
            let blob = drafterA.capturePrefixBlob(slot: srcSlot, tokenCount: tokenCount)
            XCTAssertEqual(blob.count, QWDFlashDrafter.prefixPayloadBytes(config: drafterA.config, tokenCount: tokenCount))
            XCTAssertEqual(drafterA.restorePrefixBlob(blob, into: dstSlotA, expectedTokens: tokenCount), tokenCount)
            XCTAssertNil(drafterA.restorePrefixBlob(blob, into: dstSlotA, expectedTokens: tokenCount + 1))
            XCTAssertNil(drafterB.restorePrefixBlob(blob, into: dstSlotB, expectedTokens: tokenCount))

            let activeBytes = min(tokenCount, drafterA.config.maxCacheSize) * rowBytes
            if activeBytes > 0 {
                for l in 0..<drafterA.config.numLayers {
                    let kS = srcSlot.kLayers[l], kD = dstSlotA.kLayers[l]
                    let vS = srcSlot.vLayers[l], vD = dstSlotA.vLayers[l]
                    let kSrc = Data(bytes: kS.region.buffer.contents().advanced(by: kS.region.base + kS.offset), count: activeBytes)
                    let kDst = Data(bytes: kD.region.buffer.contents().advanced(by: kD.region.base + kD.offset), count: activeBytes)
                    let vSrc = Data(bytes: vS.region.buffer.contents().advanced(by: vS.region.base + vS.offset), count: activeBytes)
                    let vDst = Data(bytes: vD.region.buffer.contents().advanced(by: vD.region.base + vD.offset), count: activeBytes)
                    XCTAssertEqual(kDst, kSrc)
                    XCTAssertEqual(vDst, vSrc)
                }
            }
        }
    }
}

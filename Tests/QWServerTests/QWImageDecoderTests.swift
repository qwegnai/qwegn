import XCTest
@testable import QWServer
import QWCore
import QWMM
import QWVision
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

private final class MockOnDemandVisionBackend: QWVisionBackend, @unchecked Sendable {
    let outDim: Int
    var isLoaded: Bool = false
    var loadTransitions: Int = 0
    var unloadTransitions: Int = 0

    init(outDim: Int = 16) {
        self.outDim = outDim
    }

    func prepareForPrompt(hasImages: Bool) throws {
        if hasImages {
            if !isLoaded {
                isLoaded = true
                loadTransitions += 1
            }
        } else if isLoaded {
            isLoaded = false
            unloadTransitions += 1
        }
    }

    func encode(patches: [Float], grid: QWVisionGridPlan,
                bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float] {
        try prepareForPrompt(hasImages: true)
        return [Float](repeating: 0.25, count: grid.nToken * outDim)
    }
}

final class QWImageDecoderTests: XCTestCase {
    private func makeRedPngDataUri(width: Int = 4, height: Int = 4) -> String {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[i * 4 + 0] = 255
            rgba[i * 4 + 3] = 255
        }
        let ctx = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let cgImage = ctx.makeImage()!
        let pngData = NSMutableData()
        let dest = CGImageDestinationCreateWithData(
            pngData as CFMutableData,
            UTType.png.identifier as CFString,
            1,
            nil
        )!
        CGImageDestinationAddImage(dest, cgImage, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        let base64 = (pngData as Data).base64EncodedString()
        return "data:image/png;base64,\(base64)"
    }

    func testDecodeSyntheticImageFromBase64DataUrl() throws {
        let dataUri = makeRedPngDataUri(width: 4, height: 4)
        let decoded = try QWImageDecoder.decode(urlOrDataUri: dataUri)
        XCTAssertEqual(decoded.width, 4)
        XCTAssertEqual(decoded.height, 4)
        XCTAssertEqual(decoded.rgb.count, 4 * 4 * 3)
        XCTAssertEqual(decoded.rgb[0], 255)
        XCTAssertEqual(decoded.rgb[1], 0)
        XCTAssertEqual(decoded.rgb[2], 0)
    }

    func testDecodeSmartResizedAndPrepareMultimodalTurnLoadsAndUnloadsVision() async throws {
        let dataUri = makeRedPngDataUri(width: 40, height: 40)
        let resized = try QWImageDecoder.decodeSmartResized(urlOrDataUri: dataUri)
        XCTAssertEqual(resized.image.width % 32, 0)
        XCTAssertEqual(resized.image.height % 32, 0)
        XCTAssertEqual(resized.image.rgb.count, resized.image.width * resized.image.height * 3)

        let backend = MockOnDemandVisionBackend()
        XCTAssertFalse(backend.isLoaded)

        // Prompt with an image -> loads vision on demand and expands <|image_pad|> (248056)
        let promptWithPad = [10, 248056, 20]
        let turn1 = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: promptWithPad,
            imageUrls: [dataUri],
            hiddenDim: 16,
            visionBackend: backend
        )
        XCTAssertTrue(backend.isLoaded)
        XCTAssertTrue(turn1.loadedOnDemand)
        XCTAssertFalse(turn1.unloadedOnText)
        XCTAssertEqual(turn1.visualSpans.count, 1)
        XCTAssertEqual(turn1.visualSpans[0].tokenOffset, 1)
        XCTAssertEqual(turn1.visualSpans[0].embeddings.count, turn1.visualSpans[0].tokenCount * 16)

        // Subsequent text-only prompt -> unloads vision to free RAM
        let turn2 = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 20, 30],
            imageUrls: [],
            hiddenDim: 16,
            visionBackend: backend
        )
        XCTAssertFalse(backend.isLoaded)
        XCTAssertFalse(turn2.loadedOnDemand)
        XCTAssertTrue(turn2.unloadedOnText)
        XCTAssertTrue(turn2.visualSpans.isEmpty)
        XCTAssertEqual(backend.loadTransitions, 1)
        XCTAssertEqual(backend.unloadTransitions, 1)
    }

    func testChatMessageContentInsertsVisionPadForImageParts() throws {
        let json = """
        {
          "model": "qwen3.8-flash-next",
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Describe this image."}
              ]
            }
          ]
        }
        """.data(using: .utf8)!
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.extractImageUrls(), ["data:image/png;base64,AAAA"])
        let qwMessages = req.toQWJsonMessages()
        if case .string(let renderedContent) = qwMessages[0].get("content") {
            XCTAssertTrue(renderedContent.contains("<|vision_start|><|image_pad|><|vision_end|>"))
            XCTAssertTrue(renderedContent.contains("Describe this image."))
        } else {
            XCTFail("Expected string content in rendered message")
        }
    }

    func testInvalidDataUrlSchemeThrows() {
        XCTAssertThrowsError(try QWImageDecoder.decode(urlOrDataUri: "data:missingcomma")) { error in
            XCTAssertTrue(error is QWImageDecoderError)
        }
    }

    func testNonExistentFileThrows() {
        XCTAssertThrowsError(try QWImageDecoder.decode(urlOrDataUri: "/non/existent/image.jpg")) { error in
            XCTAssertEqual(error as? QWImageDecoderError, .fileNotFound("/non/existent/image.jpg"))
        }
    }

    func testCanonicalModelNamePreservesVersionDotsAndStripsKnownExtensions() {
        XCTAssertEqual(
            QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: "/tmp/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit")),
            "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
        )
        XCTAssertEqual(
            QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: "/tmp/models/Qwen3.8-Flash-Next-hq8g32-hq4g64/model.qwp")),
            "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        )
        XCTAssertEqual(
            QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: "/tmp/models/Qwen3.8-27B-DFlash2-hq4g32.qwp")),
            "Qwen3.8-27B-DFlash2-hq4g32"
        )
    }

    func testResolveModelReferencePrefersActiveModelCaseInsensitively() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("qwg-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let primaryDir = tmp.appendingPathComponent("Qwen3.8-Flash-Next-hq8g32-hq4g64")
        try FileManager.default.createDirectory(at: primaryDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: primaryDir.appendingPathComponent("model.qwp").path, contents: Data())

        var settings = QWServerSettings()
        settings.modelName = "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        settings.modelPackPath = primaryDir.path
        settings.modelAlias = "Qwen3.8-Flash-Next"
        settings.modelAliases = [
            "Qwen3.8-Flash-Next-hq8g32-hq4g64": "Qwen3.8-Flash-Next",
            "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit": "qwen3.8-flash-next"
        ]

        let resolved = settings.resolveModelReference("qwen3.8-flash-next")
        XCTAssertEqual(resolved.modelName, "Qwen3.8-Flash-Next-hq8g32-hq4g64")
        XCTAssertEqual(resolved.packPath, primaryDir.path)
    }

    func testQwen36_35BA3BVisionConfigAdvertisesVisionCapability() throws {
        let packPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".qwg/models/Qwen3.6-35B-A3B-hq4g32").path
        let cfgPath = packPath + "/assets/config.json"
        guard FileManager.default.fileExists(atPath: cfgPath) else {
            throw XCTSkip("requires a local Qwen3.6-35B-A3B model pack; skipped")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: cfgPath))
        let desc = try QWModelDescriptor.decode(configJSON: data)
        XCTAssertEqual(desc.family, .qwen38Dense)
        XCTAssertTrue(desc.capabilities.contains(.vision))
        XCTAssertEqual(desc.visDepth, 27)
        XCTAssertEqual(desc.visHidden, 1152)
        XCTAssertEqual(desc.visOut, 2048)
        XCTAssertEqual(desc.hidden, 2048)
        XCTAssertTrue(QWVisionOnDemandBackend.hasVisionWeights(modelPath: packPath))
    }

    func testSetVisionLoadedTogglesResidencyInPlaceWithoutReloadingOnTextTurn() async throws {
        let toggleable = try QWVisionOnDemandBackend(
            source: .custom {
                try Self.makeTinyVisionWeights()
            },
            keepResident: false,
            allowOnDemand: false
        )
        XCTAssertFalse(toggleable.isLoaded)
        XCTAssertFalse(toggleable.keepResident)

        // When loadVision is false and allowOnDemand is false, image turns are refused until vision is enabled
        XCTAssertThrowsError(try toggleable.prepareForPrompt(hasImages: true))

        // Toggling loadVision ON loads weights in place and keeps them resident across text turns
        try toggleable.setVisionLoaded(true)
        XCTAssertTrue(toggleable.isLoaded)
        XCTAssertTrue(toggleable.keepResident)
        XCTAssertEqual(toggleable.loadCount, 1)

        try toggleable.prepareForPrompt(hasImages: false)
        XCTAssertTrue(toggleable.isLoaded, "resident vision tower must stay loaded across text turns")
        XCTAssertEqual(toggleable.unloadCount, 0)

        // Toggling loadVision OFF unloads vision weights in place
        try toggleable.setVisionLoaded(false)
        XCTAssertFalse(toggleable.isLoaded)
        XCTAssertFalse(toggleable.keepResident)
        XCTAssertEqual(toggleable.unloadCount, 1)
    }

    func testEncodedImageLRUCacheSkipsRepeatEncodeAndVisionReload() async throws {
        QWImageDecoder.clearEncodedImageCache()
        let dataUri = makeRedPngDataUri(width: 32, height: 32)
        let backend = MockOnDemandVisionBackend()
        let promptWithPad = [10, 248056, 20]

        var firstEncodeMs: Double = -1
        let turn1 = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: promptWithPad,
            imageUrls: [dataUri],
            hiddenDim: 16,
            visionBackend: backend,
            onImageEncoded: { _, _, _, _, ms in firstEncodeMs = ms }
        )
        XCTAssertEqual(backend.loadTransitions, 1)
        XCTAssertGreaterThanOrEqual(firstEncodeMs, 0)

        // Unload vision on a text turn, then re-send the same image in conversation history:
        _ = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 20],
            imageUrls: [],
            hiddenDim: 16,
            visionBackend: backend
        )
        XCTAssertFalse(backend.isLoaded)

        var cachedEncodeMs: Double = -1
        let turn3 = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: promptWithPad + [30, 40],
            imageUrls: [dataUri],
            hiddenDim: 16,
            visionBackend: backend,
            onImageEncoded: { _, _, _, _, ms in cachedEncodeMs = ms }
        )
        XCTAssertEqual(cachedEncodeMs, 0.0, "Repeat turn with identical image must hit encoded LRU cache in 0 ms")
        XCTAssertEqual(backend.loadTransitions, 1, "Cached image turn must not reload vision tower")
        XCTAssertEqual(turn3.visualSpans, turn1.visualSpans)
    }

    func testVisualSpanSaltedCacheIdentityDistinguishesDifferentImages() {
        let history: [Int64] = [100, 248056, 248056, 200]
        let prompt: [Int] = [100, 248056, 248056, 200]
        let spanA = QWVisualSpan(tokenOffset: 1, tokenCount: 2, embeddings: [Float](repeating: 0.25, count: 32))
        let spanB = QWVisualSpan(tokenOffset: 1, tokenCount: 2, embeddings: [Float](repeating: 0.75, count: 32))

        let saltedHistA1 = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [spanA])
        let saltedHistA2 = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [spanA])
        let saltedHistB = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [spanB])
        XCTAssertEqual(saltedHistA1, saltedHistA2)
        XCTAssertNotEqual(saltedHistA1, saltedHistB)
        XCTAssertEqual(saltedHistA1[0], 100)
        XCTAssertEqual(saltedHistA1[3], 200)

        let saltedTokA = QWVisualSpan.saltedCacheTokens(prompt: prompt, visualSpans: [spanA], vocab: 248320)
        let saltedTokB = QWVisualSpan.saltedCacheTokens(prompt: prompt, visualSpans: [spanB], vocab: 248320)
        XCTAssertNotEqual(saltedTokA, saltedTokB)
        XCTAssertTrue(saltedTokA.allSatisfy { (0..<248320).contains($0) })
    }

    func testCanonicalModelFamilyStemMatchesSiblingQuantizations() {
        let stemHQ = QWEngineLifecycleActor.canonicalModelFamilyStem("Qwen3.8-Flash-Next-hq8g32-hq4g64")
        let stemMLX = QWEngineLifecycleActor.canonicalModelFamilyStem("ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit")
        let stemAlias = QWEngineLifecycleActor.canonicalModelFamilyStem("Qwen3.8-Flash-Next")
        let stemOther = QWEngineLifecycleActor.canonicalModelFamilyStem("Qwen3.6-35B-A3B-hq4g32")
        XCTAssertEqual(stemHQ, "qwen3.8-flash-next")
        XCTAssertEqual(stemMLX, "qwen3.8-flash-next")
        XCTAssertEqual(stemAlias, "qwen3.8-flash-next")
        XCTAssertEqual(stemOther, "qwen3.6-35b-a3b")
    }

    func testVisionEncoderNonZeroDeterminism() throws {
        let weights = try Self.makeTinyVisionWeights(fillBF16: 0x3C00)
        let ref = QWVisionRef(weights: weights)
        let grid = try QWVisionGrid(t: 1, h: 4, w: 4)
        let patches = (0..<(grid.patches * 1536)).map { i in
            Float((i % 17) - 8) * 0.05
        }
        let out1 = ref.encode(patches: patches, plan: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
        let out2 = ref.encode(patches: patches, plan: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
        XCTAssertEqual(out1.count, grid.tokens * 16)
        XCTAssertEqual(out1, out2)
        XCTAssertTrue(out1.allSatisfy { $0.isFinite })
        XCTAssertTrue(out1.contains { abs($0) > 1e-5 })

        if ProcessInfo.processInfo.environment["QWG_BENCH_VISION_44X80"] == "1" {
            let fullW = try Self.makeTinyVisionWeights(fillBF16: 0x3C00, numBlocks: 27, outHidden: 2048)
            let fullRef = QWVisionRef(weights: fullW)
            let g44x80 = try QWVisionGrid(t: 1, h: 44, w: 80)
            let p44x80 = [Float](repeating: 0.1, count: g44x80.patches * 1536)
            let bins = g44x80.interpBins, pos = g44x80.ropePositions
            var samplesMs: [Double] = []
            for r in 1...3 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let feat = fullRef.encode(patches: p44x80, plan: g44x80.plan, bins: bins, posIds: pos)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
                XCTAssertEqual(feat.count, g44x80.tokens * 2048)
                samplesMs.append(ms)
                print(String(format: "[VisionBench] round=%d patches=44x80 (3520) blocks=27 encode_ms=%.1f", r, ms))
            }
            if let minMs = samplesMs.min(), let maxMs = samplesMs.max() {
                print(String(format: "[VisionBench] min_ms=%.1f max_ms=%.1f speedup_vs_79883ms=%.1fx",
                             minMs, maxMs, 79883.9 / max(minMs, 0.1)))
            }
        }
    }

    private static func makeTinyVisionWeights(fillBF16: UInt16 = 0, numBlocks: Int = 1, outHidden: Int = 16) throws -> QWVisionWeights {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("qwg-vis-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let d = QWVisionShape.dim
        let mlp = QWVisionShape.mlpInner
        var tensors: [(String, [Int])] = [
            ("model.visual.pos_embed.weight", [2304, d]),
            ("model.visual.patch_embed.proj.weight", [d, 3, 2, 16, 16]),
            ("model.visual.patch_embed.proj.bias", [d]),
            ("model.visual.merger.norm.weight", [d]),
            ("model.visual.merger.norm.bias", [d]),
            ("model.visual.merger.linear_fc1.weight", [4608, 4608]),
            ("model.visual.merger.linear_fc1.bias", [4608]),
            ("model.visual.merger.linear_fc2.weight", [outHidden, 4608]),
            ("model.visual.merger.linear_fc2.bias", [outHidden])
        ]
        for b in 0..<numBlocks {
            tensors.append(contentsOf: [
                ("model.visual.blocks.\(b).norm1.weight", [d]),
                ("model.visual.blocks.\(b).norm1.bias", [d]),
                ("model.visual.blocks.\(b).norm2.weight", [d]),
                ("model.visual.blocks.\(b).norm2.bias", [d]),
                ("model.visual.blocks.\(b).attn.qkv.weight", [3 * d, d]),
                ("model.visual.blocks.\(b).attn.qkv.bias", [3 * d]),
                ("model.visual.blocks.\(b).attn.proj.weight", [d, d]),
                ("model.visual.blocks.\(b).attn.proj.bias", [d]),
                ("model.visual.blocks.\(b).mlp.linear_fc1.weight", [mlp, d]),
                ("model.visual.blocks.\(b).mlp.linear_fc1.bias", [mlp]),
                ("model.visual.blocks.\(b).mlp.linear_fc2.weight", [d, mlp]),
                ("model.visual.blocks.\(b).mlp.linear_fc2.bias", [d])
            ])
        }
        tensors.sort { $0.0 < $1.0 }
        var headerObj: [String: Any] = [:]
        var offset = 0
        for (name, shape) in tensors {
            let bytes = shape.reduce(1, *) * 2
            headerObj[name] = ["dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj)
        var fileData = Data(count: 8 + hdrData.count + offset)
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { fileData.replaceSubrange(0..<8, with: $0) }
        fileData.replaceSubrange(8..<(8 + hdrData.count), with: hdrData)
        if fillBF16 != 0 {
            let base = 8 + hdrData.count
            fileData.withUnsafeMutableBytes { raw in
                guard let ptr = raw.baseAddress?.advanced(by: base).assumingMemoryBound(to: UInt16.self) else { return }
                let count = offset / 2
                for i in 0..<count {
                    ptr[i] = ((i % 7) == 0) ? (fillBF16 | 0x8000) : fillBF16
                }
            }
        }
        try fileData.write(to: tmp.appendingPathComponent("model.safetensors"))
        return try QWVisionWeights(dir: tmp.path)
    }
}

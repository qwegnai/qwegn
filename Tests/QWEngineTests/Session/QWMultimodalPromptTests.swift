import XCTest
import Foundation
@testable import QWEngine
import QWCore
import QWFlashNext
import QWMM
import QWVision

final class QWMultimodalPromptTests: XCTestCase {

    func testMultimodalPromptBuildingAndInterleaving() throws {
        // Construct a prompt with text tokens and an image placeholder
        // image placeholder token is QWMMConfig.imageToken (248056)
        let textBefore = [101, 102]
        let imageToken = QWMMConfig.imageToken
        let textAfter = [201, 202, 203]

        let promptTokens = textBefore + [imageToken] + textAfter

        // Grid 1 x 2 x 2 -> 4 patches -> with merge=2, exactly 1 token!
        let grid = try QWVisionGrid(t: 1, h: 2, w: 2)
        XCTAssertEqual(grid.tokens, 1)

        // 1 visual token with hiddenDim = 5120 floats
        let hiddenDim = QWModelGeometry.qwen38_27b.hidden // 5120
        var visualFeatures = [Float](repeating: 0, count: grid.tokens * hiddenDim)
        for i in 0..<hiddenDim {
            visualFeatures[i] = Float(i + 1) * 0.01
        }

        // Build multimodal prompt
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens,
            images: [(grid: grid, features: visualFeatures)]
        )

        // The expanded prompt tokens should have 2 (before) + 1 (image) + 3 (after) = 6 tokens
        XCTAssertEqual(mmPrompt.expandedTokens.count, 6)
        XCTAssertEqual(mmPrompt.expandedTokens, [101, 102, imageToken, 201, 202, 203])
        XCTAssertEqual(mmPrompt.plan.spans.count, 3)

        // Test embedding resolution
        // Dummy text embedder closure that returns token * 1.0 for each hidden dimension
        func dummyTextEmbedder(_ token: Int) -> [Float] {
            [Float](repeating: Float(token), count: hiddenDim)
        }

        let x0 = try mmPrompt.resolveX0(textEmbedder: dummyTextEmbedder)
        XCTAssertEqual(x0.count, 6 * hiddenDim)

        // Row 0: token 101
        XCTAssertEqual(x0[0], 101.0)
        // Row 1: token 102
        XCTAssertEqual(x0[hiddenDim], 102.0)
        // Row 2: image token -> must match visualFeatures
        XCTAssertEqual(x0[2 * hiddenDim], 0.01)
        XCTAssertEqual(x0[2 * hiddenDim + 5119], Float(5120) * 0.01)
        // Row 3: token 201
        XCTAssertEqual(x0[3 * hiddenDim], 201.0)
    }

    func testMultimodalPromptRejectsMismatchedFeatures() throws {
        let grid = try QWVisionGrid(t: 1, h: 2, w: 2)
        // Feature count is short (e.g. 100 instead of 5120)
        let shortFeatures = [Float](repeating: 1.0, count: 100)

        XCTAssertThrowsError(
            try QWMultimodalPrompt(
                promptTokens: [QWMMConfig.imageToken],
                images: [(grid: grid, features: shortFeatures)]
            )
        )
    }

    func testFlashNextVisualSpansAndSubchunkExpansion() throws {
        let hidden = 2560
        let hc = 4
        let grid = try QWVisionGrid(t: 1, h: 4, w: 4) // 4 visual tokens
        var features = [Float](repeating: 0, count: grid.tokens * hidden)
        for tok in 0..<grid.tokens {
            for d in 0..<hidden {
                features[tok * hidden + d] = Float((tok + 1) * 1000 + d)
            }
        }
        let promptTokens = [10, 11, QWMMConfig.imageToken, 20, 21, 22]
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens,
            images: [(grid: grid, features: features)],
            hiddenDim: hidden
        )
        XCTAssertEqual(mmPrompt.expandedTokens.count, 2 + 4 + 3)
        let spans = mmPrompt.visualSpans
        XCTAssertEqual(spans.count, 1)
        XCTAssertEqual(spans[0].tokenOffset, 2)
        XCTAssertEqual(spans[0].tokenCount, 4)
        XCTAssertEqual(spans[0].embeddings.count, 4 * hidden)

        // Subchunk 0: leading text [0..<2]
        let sub0 = FNGPUPrefillStack.resolveSubchunk(
            pos: 0, limitPos: 9, chunkSize: 64, hidden: hidden, hc: hc, visualSpans: spans
        )
        XCTAssertEqual(sub0.rows, 2)
        XCTAssertNil(sub0.visualStreamIn)

        // Subchunk 1: visual span [2..<6] expanded across hc=4 streams
        let sub1 = FNGPUPrefillStack.resolveSubchunk(
            pos: 2, limitPos: 9, chunkSize: 64, hidden: hidden, hc: hc, visualSpans: spans
        )
        XCTAssertEqual(sub1.rows, 4)
        let streamIn = try XCTUnwrap(sub1.visualStreamIn)
        XCTAssertEqual(streamIn.count, 4 * hc * hidden)
        for r in 0..<4 {
            for c in 0..<hc {
                XCTAssertEqual(streamIn[r * (hc * hidden) + c * hidden + 0], Float((r + 1) * 1000))
                XCTAssertEqual(streamIn[r * (hc * hidden) + c * hidden + (hidden - 1)], Float((r + 1) * 1000 + (hidden - 1)))
            }
        }

        // Subchunk 2: trailing text [6..<9]
        let sub2 = FNGPUPrefillStack.resolveSubchunk(
            pos: 6, limitPos: 9, chunkSize: 64, hidden: hidden, hc: hc, visualSpans: spans
        )
        XCTAssertEqual(sub2.rows, 3)
        XCTAssertNil(sub2.visualStreamIn)
    }

    func testVisionOnDemandBackendLoadAndUnloadLifecycle() throws {
        // Create a minimal synthetic safetensors checkpoint directory in a temporary folder
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_vis_ondemand_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Self.writeSyntheticVisionSafetensors(to: tmpDir, outHidden: 2560)

        let backend = try QWVisionOnDemandBackend(modelPath: tmpDir.path)
        // By default vision head is NOT loaded
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)
        XCTAssertEqual(backend.loadCount, 0)
        XCTAssertEqual(backend.unloadCount, 0)

        // Prompt 1 has an image -> loads vision head once
        try backend.prepareForPrompt(hasImages: true)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentBytes, 0)
        XCTAssertEqual(backend.loadCount, 1)
        XCTAssertEqual(backend.unloadCount, 0)

        // Second image in the same prompt reuses loaded weights without reloading
        try backend.prepareForPrompt(hasImages: true)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertEqual(backend.loadCount, 1)

        // Prompt 2 does NOT have an image -> unloads vision head to free RAM
        try backend.prepareForPrompt(hasImages: false)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)
        XCTAssertEqual(backend.loadCount, 1)
        XCTAssertEqual(backend.unloadCount, 1)

        // Prompt 3 has an image -> reloads on demand
        try backend.prepareForPrompt(hasImages: true)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentBytes, 0)
        XCTAssertEqual(backend.loadCount, 2)
        XCTAssertEqual(backend.unloadCount, 1)
    }

    private static func writeSyntheticVisionSafetensors(to dir: URL, outHidden: Int) throws {
        let d = QWVisionShape.dim // 1152
        let mlp = QWVisionShape.mlpInner // 4304
        let specs: [(String, [Int])] = [
            ("model.visual.pos_embed.weight", [2304, d]),
            ("model.visual.patch_embed.proj.weight", [d, 3, 2, 16, 16]),
            ("model.visual.patch_embed.proj.bias", [d]),
            ("model.visual.blocks.0.norm1.weight", [d]),
            ("model.visual.blocks.0.norm1.bias", [d]),
            ("model.visual.blocks.0.norm2.weight", [d]),
            ("model.visual.blocks.0.norm2.bias", [d]),
            ("model.visual.blocks.0.attn.qkv.weight", [3 * d, d]),
            ("model.visual.blocks.0.attn.qkv.bias", [3 * d]),
            ("model.visual.blocks.0.attn.proj.weight", [d, d]),
            ("model.visual.blocks.0.attn.proj.bias", [d]),
            ("model.visual.blocks.0.mlp.linear_fc1.weight", [mlp, d]),
            ("model.visual.blocks.0.mlp.linear_fc1.bias", [mlp]),
            ("model.visual.blocks.0.mlp.linear_fc2.weight", [d, mlp]),
            ("model.visual.blocks.0.mlp.linear_fc2.bias", [d]),
            ("model.visual.merger.norm.weight", [d]),
            ("model.visual.merger.norm.bias", [d]),
            ("model.visual.merger.linear_fc1.weight", [4608, 4608]),
            ("model.visual.merger.linear_fc1.bias", [4608]),
            ("model.visual.merger.linear_fc2.weight", [outHidden, 4608]),
            ("model.visual.merger.linear_fc2.bias", [outHidden]),
        ]
        var header: [String: Any] = [:]
        var offset = 0
        for (name, shape) in specs {
            let bytes = shape.reduce(1, *) * 2
            header[name] = [
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            ]
            offset += bytes
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var lenLE = UInt64(headerJSON.count).littleEndian
        var fileData = Data(bytes: &lenLE, count: 8)
        fileData.append(headerJSON)
        fileData.append(Data(count: offset))
        try fileData.write(to: dir.appendingPathComponent("model.safetensors"))
    }
}

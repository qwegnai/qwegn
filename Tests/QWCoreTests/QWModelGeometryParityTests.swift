import XCTest
@testable import QWCore

/// Parity oracle for the P0 migration: the config-derived geometry must equal
/// the compiled 27B constants field by field, and the per-layer kinds must
/// match the config's layer_types. The config is a stub, not an installed pack.
final class QWModelGeometryParityTests: XCTestCase {
    private func load27B() throws -> QWModelGeometry {
        try QWStubModelConfig.dense27BGeometry()
    }

    func testTwentySevenBConfigGeometryEqualsCompiledConstants() throws {
        let g = try load27B()
        XCTAssertEqual(g.hidden, QWModelGeometry.qwen38_27b.hidden)
        XCTAssertEqual(g.vocab, QWModelGeometry.qwen38_27b.vocab)
        XCTAssertEqual(g.intermediate, QWModelGeometry.qwen38_27b.intermediate)
        XCTAssertEqual(g.numLayers, QWModelGeometry.qwen38_27b.numLayers)
        XCTAssertEqual(g.fullAttentionInterval, QWModelGeometry.qwen38_27b.fullAttentionInterval)
        XCTAssertEqual(g.qHeads, QWModelGeometry.qwen38_27b.qHeads)
        XCTAssertEqual(g.kvHeads, QWModelGeometry.qwen38_27b.kvHeads)
        XCTAssertEqual(g.headDim, QWModelGeometry.qwen38_27b.headDim)
        XCTAssertEqual(g.attnOutputGate, QWModelGeometry.qwen38_27b.attnOutputGate)
        XCTAssertEqual(g.qProjOut, QWModelGeometry.qwen38_27b.qProjOut)
        XCTAssertEqual(g.rotaryDims, QWModelGeometry.qwen38_27b.rotaryDims)
        XCTAssertEqual(g.partialRotaryFactor, QWModelGeometry.qwen38_27b.partialRotaryFactor, accuracy: 1e-12)
        XCTAssertEqual(g.ropeTheta, QWModelGeometry.qwen38_27b.ropeTheta, accuracy: 1)
        XCTAssertEqual(g.rmsEps, QWModelGeometry.qwen38_27b.rmsEps, accuracy: 1e-9)
        XCTAssertEqual(g.maxPositionEmbeddings, QWModelGeometry.qwen38_27b.maxPositionEmbeddings)
        XCTAssertEqual(g.gdnQKHeads, QWModelGeometry.qwen38_27b.gdnQKHeads)
        XCTAssertEqual(g.gdnVHeads, QWModelGeometry.qwen38_27b.gdnVHeads)
        XCTAssertEqual(g.gdnHeadDim, QWModelGeometry.qwen38_27b.gdnHeadDim)
        XCTAssertEqual(g.gdnQKVChannels, QWModelGeometry.qwen38_27b.gdnQKVChannels)
        XCTAssertEqual(g.gdnZChannels, QWModelGeometry.qwen38_27b.gdnZChannels)
        XCTAssertEqual(g.gdnConvKernel, QWModelGeometry.qwen38_27b.gdnConvKernel)
        XCTAssertEqual(g.mtpLayers, QWModelGeometry.qwen38_27b.mtpLayers)
        XCTAssertEqual(g.endOfTextTokenId, QWModelGeometry.qwen38_27b.endOfTextTokenId)
        XCTAssertEqual(g.imStartTokenId, QWModelGeometry.qwen38_27b.imStartTokenId)
        XCTAssertEqual(g.imEndTokenId, QWModelGeometry.qwen38_27b.imEndTokenId)
        XCTAssertEqual(g.thinkTokenId, QWModelGeometry.qwen38_27b.thinkTokenId)
        XCTAssertEqual(g.endThinkTokenId, QWModelGeometry.qwen38_27b.endThinkTokenId)
    }

    func testPerLayerKindsMatchCompiledCadence() throws {
        let g = try load27B()
        for l in 0..<g.numLayers {
            XCTAssertEqual(g.isFullAttention(l), QWModelGeometry.qwen38_27b.isFullAttention(l), "layer \(l)")
        }
    }
}

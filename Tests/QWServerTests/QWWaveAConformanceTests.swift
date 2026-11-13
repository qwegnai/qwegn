// QWWaveAConformanceTests.swift — Tests for explicit-seed gating, BPE prefill healing,
// balanced multi-tool-call parsing, and pre-flight vision HTTP 400 errors.

import Foundation
import XCTest
import QWCore
import QWEngine
import QWHead
@testable import QWServer

final class QWWaveAConformanceTests: XCTestCase {
    func testHasExplicitSeedPropagation() {
        let defaultSampling = QWSamplingParameters()
        XCTAssertFalse(defaultSampling.hasExplicitSeed)
        XCTAssertFalse(QWSamplerConfig(defaultSampling, vocabulary: 16).hasExplicitSeed)

        let zeroSeedSampling = QWSamplingParameters(temperature: 0.7, seed: 0)
        XCTAssertTrue(zeroSeedSampling.hasExplicitSeed)
        XCTAssertTrue(QWSamplerConfig(zeroSeedSampling, vocabulary: 16).hasExplicitSeed)

        let seeded = QWSamplingParameters(temperature: 0.0, seed: 42)
        XCTAssertTrue(seeded.hasExplicitSeed)
        XCTAssertTrue(QWSamplerConfig(seeded, vocabulary: 16).hasExplicitSeed)
    }

    func testPrefixHealingConstraintAvoidsDeadEndsAndReleasesPrefix() throws {
        // 0:"P", 1:"Par", 2:"Paris", 3:"Part", 4:"is", 5:"!"
        let piecesNoAr = ["P", "Par", "Paris", "Part", "is", "!"]
        let tableNoAr = QWJSONTokenTable(
            tokenBytes: piecesNoAr.map { Array($0.utf8) },
            vocabulary: piecesNoAr.count
        )
        let c1 = QWPrefixHealingConstraint(prefixBytes: Array("Par".utf8), table: tableNoAr)
        XCTAssertTrue(c1.hasSupport)
        XCTAssertTrue(c1.allowsSpeculation)
        XCTAssertFalse(c1.isUnconstrainedNow)
        // "P" (0) must be excluded because "ar" has no completion in piecesNoAr.
        XCTAssertEqual(try c1.allowedTokens(), [1, 2, 3])
        XCTAssertThrowsError(try c1.accept(0))
        try c1.accept(2) // "Paris" completes "Par"
        XCTAssertTrue(c1.isUnconstrainedNow)
        XCTAssertNil(try c1.allowedTokens())

        // 0:"P", 1:"ar", 2:"Par", 3:"Paris", 4:"is"
        let piecesWithAr = ["P", "ar", "Par", "Paris", "is"]
        let tableWithAr = QWJSONTokenTable(
            tokenBytes: piecesWithAr.map { Array($0.utf8) },
            vocabulary: piecesWithAr.count
        )
        let c2 = QWPrefixHealingConstraint(prefixBytes: Array("Par".utf8), table: tableWithAr)
        XCTAssertEqual(try c2.allowedTokens(), [0, 2, 3])
        try c2.accept(0) // "P"
        XCTAssertFalse(c2.isUnconstrainedNow)
        XCTAssertEqual(try c2.allowedTokens(), [1]) // only "ar" remains
        try c2.accept(1)
        XCTAssertTrue(c2.isUnconstrainedNow)
        XCTAssertNil(try c2.allowedTokens())
    }

    func testToolCallParserSplitsProseSeparatedAndLiteralCloseCalls() {
        let proseSplit = """
        <tool_call>
        {"name": "read_file", "arguments": {"path": "a.swift"}}
        </tool_call>
        Let me inspect the second file next:
        <tool_call>
        {"name": "read_file", "arguments": {"path": "b.swift"}}
        </tool_call>
        """
        let r1 = QWToolCallParser.parse(text: proseSplit, tools: nil)
        XCTAssertEqual(r1.toolCalls?.count, 2)
        XCTAssertEqual(r1.toolCalls?[0].function.name, "read_file")
        XCTAssertEqual(r1.toolCalls?[1].function.name, "read_file")
        XCTAssertTrue(r1.toolCalls?[0].function.arguments.contains("a.swift") == true)
        XCTAssertTrue(r1.toolCalls?[1].function.arguments.contains("b.swift") == true)

        let literalClose = """
        <tool_call>
        {"name": "search", "arguments": {"pattern": "</tool_call>"}}
        </tool_call>
        Trailing prose
        <tool_call>
        {"name": "done", "arguments": {}}
        </tool_call>
        """
        let r2 = QWToolCallParser.parse(text: literalClose, tools: nil)
        XCTAssertEqual(r2.toolCalls?.count, 2)
        XCTAssertEqual(r2.toolCalls?[0].function.name, "search")
        XCTAssertTrue(r2.toolCalls?[0].function.arguments.contains("</tool_call>") == true)
        XCTAssertEqual(r2.toolCalls?[1].function.name, "done")
        let endIdx = QWToolCallParser.firstCompleteToolCallEnd(in: literalClose)
        XCTAssertNotNil(endIdx)
        XCTAssertTrue(String(literalClose[..<endIdx!]).contains("\"search\""))
        XCTAssertFalse(String(literalClose[..<endIdx!]).contains("\"done\""))

        let xmlProse = """
        <tool_call>
        <function=alpha>
        <parameter=x>1</parameter>
        </function>
        </tool_call>
        Now second:
        <tool_call>
        <function=beta>
        <parameter=y>2</parameter>
        </function>
        </tool_call>
        """
        let r3 = QWToolCallParser.parse(text: xmlProse, tools: nil)
        XCTAssertEqual(r3.toolCalls?.count, 2)
        XCTAssertEqual(r3.toolCalls?[0].function.name, "alpha")
        XCTAssertEqual(r3.toolCalls?[1].function.name, "beta")
    }

    func testValidateRequestCapabilitiesRejectsImagesWhenVisionDisabled() throws {
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "dummy",
            host: "127.0.0.1",
            port: 0,
            loadVision: false
        )
        let actor = QWEngineLifecycleActor(config: cfg, logger: QWServerLogger(level: .none))
        let imgJson = """
        {
            "model": "dummy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
                    ]
                }
            ]
        }
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(imgJson.utf8))
        XCTAssertThrowsError(try actor.validateRequestCapabilities(req)) { error in
            XCTAssertTrue(error is QWMultimodalError)
        }
    }
}

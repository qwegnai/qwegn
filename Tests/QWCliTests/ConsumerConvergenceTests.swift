// ConsumerConvergenceTests.swift — The consumers converge on one
// implementation each, and the CLI names no engine type.
// The rules: one router, one tok/s formula, an m1 producer, a bounded
// record, and a CLI that drives the neutral engine seam.

import Foundation
import XCTest
@testable import qwg
import QWEngine

final class ConsumerConvergenceTests: XCTestCase {
    /// The repository root, found from this file rather than the working
    /// directory: a test must not depend on where it was launched.
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// Every Swift file the CLI owns, recursively: a shared helper in a
    /// subdirectory is part of the consumer surface.
    private func cliSourceFiles() throws -> [String] {
        let dir = repoRoot.appendingPathComponent("Sources/Apps/qwg")
        guard let walker = FileManager.default.enumerator(atPath: dir.path) else { return [] }
        var files: [String] = []
        for case let name as String in walker where name.hasSuffix(".swift") {
            files.append(dir.appendingPathComponent(name).path)
        }
        return files.sorted()
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    /// The boundary between thinking and answer is found once. Two
    /// routers are two turns that disagree about where thinking stopped.
    func testOnlyOneTokenRouterRemains() throws {
        let router = repoRoot.appendingPathComponent("Sources/Apps/qwg/Shared/QWTokenRouter.swift").path
        var hits: [String] = []
        for file in try cliSourceFiles() where file != router {
            if try source(file).contains("firstIndex(of:") { hits.append(file) }
        }
        XCTAssertEqual(hits.count, 0, "the thinking/answer boundary is found in \(hits)")
        // And the one router is the shared one.
        XCTAssertTrue(try source(router).contains("static func split"))
    }

    /// The tokens-per-second formula is computed once. Two copies are two
    /// numbers that can disagree.
    func testOnlyOneTokPerSecondFormulaRemains() throws {
        let formatter = repoRoot.appendingPathComponent("Sources/Apps/qwg/Shared/QWStatFormat.swift").path
        let pattern = try NSRegularExpression(pattern: "Double\\([^)]*(oken|ount)[^)]*\\)\\s*/")
        var hits: [String] = []
        for file in try cliSourceFiles() where file != formatter {
            let text = try source(file)
            // The regular expression works in UTF-16 offsets: the range
            // must be measured the same way the matcher counts.
            let range = NSRange(location: 0, length: (text as NSString).length)
            if pattern.firstMatch(in: text, range: range) != nil { hits.append(file) }
        }
        XCTAssertEqual(hits.count, 0, "the tok/s formula is duplicated in \(hits)")
        XCTAssertTrue(try source(formatter).contains("static func tokensPerSecond"))
    }

    /// The `m1` gate has a producer: `make m1` and `make bench-m1` can
    /// run to completion.
    func testMakeM1HasAProducer() throws {
        let pkg = try source(repoRoot.appendingPathComponent("Package.swift").path)
        XCTAssertTrue(pkg.contains("executableTarget(name: \"m1\""),
                      "the m1 target is missing — make m1 and make bench-m1 cannot run")
        let makefile = try source(repoRoot.appendingPathComponent("Makefile").path)
        XCTAssertTrue(makefile.contains("m1_soak_") || makefile.contains("--export"),
                      "make m1 does not produce the soak artifact bench-m1 verifies")
    }

    /// The record ring is bounded: a server that remembered every
    /// request would be a transcript, not a measurement. That the record
    /// carries no prompt content is proved in the engine's suite.
    func testTheRequestRecordRingIsBounded() throws {
        let store = QWRequestRecordStore()
        for index in 0..<200 {
            store.append(QWRequestRecord(
                id: "req-\(index)", time: Date(), modelId: "m",
                promptTokens: index, generatedTokens: 1))
        }
        XCTAssertLessThanOrEqual(store.recent(1_000).count, 64)
    }

    /// The CLI drives the neutral seam: it names no engine type, and it
    /// does not switch on the family.
    func testTheCLINamesNoEngineType() throws {
        for file in try cliSourceFiles() {
            let text = try source(file)
            XCTAssertFalse(text.contains("FNGPUEngine"), "\(file) names the Flash-Next engine")
            XCTAssertFalse(text.contains("QWEngineDriver"), "\(file) names the dense engine")
            XCTAssertFalse(text.contains(".qwen38Dense"), "\(file) switches on the family")
            XCTAssertFalse(text.contains(".qwen38FlashNext"), "\(file) switches on the family")
        }
    }

    /// Both consumers call the shared helpers: the CLI and the REPL must
    /// not each find the boundary their own way.
    func testBothConsumersCallTheSharedHelpers() throws {
        let cli = try source(repoRoot.appendingPathComponent("Sources/Apps/qwg/qwg.swift").path)
        let repl = try source(repoRoot.appendingPathComponent("Sources/Apps/qwg/QwgReplAndServe.swift").path)
        for text in [cli, repl] {
            XCTAssertTrue(text.contains("QWTokenRouter.split"))
            XCTAssertTrue(text.contains("QWStatFormat.tokensPerSecond"))
            XCTAssertTrue(text.contains("QWStatFormat.rate"))
        }
        XCTAssertTrue(cli.contains("QWTokenRouter.thinkingBudgetNote"))
        XCTAssertTrue(repl.contains("QWTokenRouter.thinkingBudgetNote"))
    }
}

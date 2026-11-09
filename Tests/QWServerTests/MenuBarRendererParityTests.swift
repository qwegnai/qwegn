// MenuBarRendererParityTests.swift — The MenuBar draws the published
// model, and only that. A renderer that reads a raw telemetry field, or
// a second endpoint, is a renderer that can disagree with the
// dashboard.

import XCTest
@testable import QWServer
import Foundation

final class MenuBarRendererParityTests: XCTestCase {
    private let menuBarDirectory = "Sources/Apps/QWMenuBar"
    private let serverDirectory = "Sources/Apps/QWServer"
    /// The files a renderer draws its numbers from.
    private let renderers = ["TelemetryCards.swift", "TelemetryObservable.swift",
                             "CupertinoTelemetryView.swift"]
    /// The literals the old renderer asserted. Each was a fact nobody
    /// measured.
    private let invented = [
        "Metal 4 (Tensor)",
        "Qwen3.8-27B",
        "HQ4_G32_M5",
        "585e9",
        "dramBwGBs: 585"
    ]

    func testNoMenuBarSourceReadsRawTelemetryFields() throws {
        for file in ["TelemetryCards.swift", "TelemetryObservable.swift"] {
            let source = try Self.read("\(menuBarDirectory)/\(file)")
            XCTAssertFalse(source.contains("metrics.cache.hitsWarmTotal"),
                           "\(file) must render the view model, not raw fields")
            XCTAssertFalse(source.contains("hw.gpuUsagePercent"),
                           "\(file) must render the view model, not raw fields")
            XCTAssertFalse(source.contains("metrics.throughput.display"),
                           "\(file) must render the published cards, not raw fields")
        }
    }

    /// The MenuBar draws the view model and nothing else. A second
    /// endpoint is a second answer to the same question, and the two
    /// answers can differ.
    func testTheMenuBarDrawsNothingFromTheEngineInfoEndpoint() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        XCTAssertFalse(source.contains("engineInfo"),
                       "the card draws the published model, not a second endpoint")
    }

    func testMenuBarAndDashboardAgreeOnCardIds() throws {
        let model = QWDashboardViewModel.build(
            lifecycle: .init(phase: .loaded, since: nil, model: "m", detail: nil),
            modelId: "m", capabilities: [.prefixCache])
        let menuBarIds = Set(MenuBarCardCatalogue.ids)
        let dashboardIds = Set(model.cards.map(\.id))
        XCTAssertTrue(menuBarIds.isSubset(of: dashboardIds),
                      "the MenuBar renders cards the dashboard does not publish")
    }

    func testEveryCardTheMenuBarDrawsIsInTheCatalogue() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        let drawn = Self.cardIds(in: source)
        XCTAssertFalse(drawn.isEmpty, "the MenuBar draws no published card at all")
        for id in drawn {
            XCTAssertTrue(MenuBarCardCatalogue.ids.contains(id),
                          "the MenuBar draws '\(id)', which the catalogue does not name")
        }
    }

    /// A renderer that invents a value shows the user a fact nobody
    /// measured. The literals the old renderer asserted are gone — from
    /// the client *and* from the server that fed it.
    func testNobodyAssertsAnInventedValue() throws {
        for file in renderers {
            let source = try Self.read("\(menuBarDirectory)/\(file)")
            for literal in invented {
                XCTAssertFalse(source.contains(literal),
                               "\(file) must not assert the literal '\(literal)'")
            }
        }
        // The server fed the renderer its worst lie: a quantization for
        // a model that was not resident.
        for file in ["Core/QWHttpServer.swift", "Schemas/QWServerSchema.swift"] {
            let source = try Self.read("\(serverDirectory)/\(file)")
            for literal in invented {
                XCTAssertFalse(source.contains(literal),
                               "\(file) must not publish the literal '\(literal)'")
            }
        }
    }

    /// The canvas is the glass *behind* the content, and it must stay
    /// out of the glass container: the container composites its glass
    /// as one layer, drawn above the plain content it holds.
    func testTheGlassCanvasIsNotInsideTheGlassContainer() throws {
        let source = try Self.read("\(menuBarDirectory)/CupertinoTelemetryView.swift")
        // With the canvas inside, it covered the whole panel and every
        // card under it was never seen: the panel rendered empty, with
        // only the glass buttons showing.
        let container = try XCTUnwrap(
            Self.block(of: "GlassEffectContainer", in: source),
            "the popover no longer has a glass container")
        XCTAssertFalse(container.contains("SystemVibrancyView"),
                       "the glass canvas must be drawn behind the glass container, "
                       + "or the container composites it above the content and the "
                       + "panel renders empty")
        XCTAssertTrue(source.contains("SystemVibrancyView"),
                      "the popover still needs its canvas")
    }

    /// The model on the card is the one the server says is engaged.
    func testWeightsCardNamesTheServerModelNotTheConfiguredOne() throws {        let source = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        XCTAssertFalse(source.contains("settings.modelName"),
                       "the card names the model the server loaded, not the configured name")
        XCTAssertTrue(source.contains("lifecycle.model"),
                      "the card takes the model name from the published lifecycle")
    }

    /// Reachability is one fact. Two are a disagreement waiting to
    /// happen: a green dot over a red menu bar icon.
    func testConnectionIsTheOnlySourceOfReachability() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryObservable.swift")
        XCTAssertFalse(source.contains("@Published public var isConnected"),
                       "the connection is one fact, published once")
        XCTAssertFalse(source.contains("@Published public var serverStatus"),
                       "the status colour is derived from the connection and the phase")
    }

    /// An unobserved phase is unknown, not idle. "Idle" asserts a phase
    /// nobody reported.
    func testUnobservedPhaseIsNotAsserted() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryObservable.swift")
        XCTAssertFalse(source.contains("return \"Idle\""),
                       "a phase nobody reported is unknown")
        XCTAssertFalse(source.contains("return \"Stopped\""),
                       "an unreachable server is not a stopped engine")
    }

    /// A rejected load is a rejected load. Discarding the status turns
    /// a server 500 into a green "Active". The check has to be in the
    /// load and the unload themselves, and it has to *show* the failure.
    func testLoadAndUnloadCheckTheStatus() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryObservable.swift")
        for function in ["unloadModel", "reloadModel"] {
            let body = try XCTUnwrap(Self.body(of: function, in: source),
                                      "\(function) is not in the source")
            XCTAssertTrue(body.contains("http.statusCode == 200"),
                          "\(function) must check what the server answered")
            XCTAssertFalse(body.contains("_ = try await URLSession.shared.data(for: req)"),
                           "\(function) discards the status, and a discarded status is a discarded failure")
            // The rejected branch must record the failure: a check
            // that notices the 500 and says nothing is a failure
            // swallowed. The branch is the guard's `else` block.
            let branch = try XCTUnwrap(Self.elseBranch(of: body, after: "http.statusCode == 200"),
                                       "\(function) has no branch for a rejected request")
            XCTAssertTrue(branch.contains("self.fail("),
                          "\(function) must show the failure in the branch that found it")
        }
    }

    /// Each endpoint keeps its own generation token. One shared counter
    /// looked right and was wrong: the poll fires both fetches at once,
    /// so a shared counter dropped every response but the last.
    func testEachEndpointKeepsItsOwnGenerationToken() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryObservable.swift")
        XCTAssertFalse(source.contains("private var requestGeneration"),
                       "a shared counter drops every fetch but the last")
        for counter in ["dashboardGeneration", "engineInfoGeneration"] {
            XCTAssertTrue(source.contains("private var \(counter)"),
                          "\(counter) must have its own counter")
            XCTAssertTrue(source.contains("guard generation == \(counter) else { return }"),
                          "a response must be checked against \(counter)")
        }
    }

    /// The card ids a source asks for, by the only way it asks for one.
    static func cardIds(in source: String) -> Set<String> {
        let marker = "card(\""
        var ids: Set<String> = []
        var rest = source[...]
        while let start = rest.range(of: marker) {
            rest = rest[start.upperBound...]
            guard let end = rest.firstIndex(of: "\"") else { break }
            ids.insert(String(rest[rest.startIndex..<end]))
            rest = rest[end...]
        }
        return ids
    }

    /// The body of a function in a source file: from its signature to
    /// the brace that closes it.
    static func body(of function: String, in source: String) -> String? {
        guard let signature = source.range(of: "func \(function)(") else { return nil }
        guard let open = source[signature.upperBound...].firstIndex(of: "{") else { return nil }
        var depth = 0
        var index = open
        while index < source.endIndex {
            if source[index] == "{" { depth += 1 }
            if source[index] == "}" {
                depth -= 1
                if depth == 0 { return String(source[signature.lowerBound..<index]) }
            }
            index = source.index(after: index)
        }
        return nil
    }

    /// The block a call opens: from the call to the brace that closes it.
    static func block(of call: String, in source: String) -> String? {
        guard let signature = source.range(of: call) else { return nil }
        guard let open = source[signature.upperBound...].firstIndex(of: "{") else { return nil }
        var depth = 0
        var index = open
        while index < source.endIndex {
            if source[index] == "{" { depth += 1 }
            if source[index] == "}" {
                depth -= 1
                if depth == 0 { return String(source[signature.lowerBound..<index]) }
            }
            index = source.index(after: index)
        }
        return nil
    }

    /// The `else` block of the guard that follows a marker: the branch a
    /// rejected request takes.
    static func elseBranch(of body: String, after marker: String) -> String? {
        guard let markerRange = body.range(of: marker) else { return nil }
        let rest = body[markerRange.upperBound...]
        guard let elseRange = rest.range(of: "else {") else { return nil }
        // The block's own brace is the last character of "else {".
        guard let open = rest.index(elseRange.upperBound, offsetBy: -1, limitedBy: rest.startIndex),
              rest[open] == "{" else { return nil }
        var depth = 0
        var index = open
        while index < rest.endIndex {
            if rest[index] == "{" { depth += 1 }
            if rest[index] == "}" {
                depth -= 1
                if depth == 0 { return String(rest[open..<index]) }
            }
            index = rest.index(after: index)
        }
        return nil
    }

    /// Reads a source file relative to the package root, which is two
    /// directories above this file, so the test does not depend on the
    /// working directory the runner happens to use.
    static func read(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // QWServerTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
        let url = root.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}

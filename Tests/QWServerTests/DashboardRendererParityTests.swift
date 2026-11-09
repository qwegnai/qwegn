// DashboardRendererParityTests.swift — The web dashboard draws the
// published view model. A renderer that computes its own rates can
// disagree with the engine about what happened, and a renderer that
// invents a value claims something the server never said.
//
// These are source-text assertions: they read the page the browser
// will run. Each one was proved against the broken code it forbids.

import XCTest
@testable import QWServer
import Foundation

final class DashboardRendererParityTests: XCTestCase {
    private func html() throws -> String {
        try Self.read("Sources/Apps/QWServer/Resources/controlbar.html")
    }

    private func script() throws -> String {
        let html = try html()
        return try XCTUnwrap(Self.script(of: html), "the dashboard has no script")
    }

    func testDashboardFetchesTheViewModel() throws {
        let html = try html()
        XCTAssertTrue(html.contains("/v1/dashboard"),
                      "the dashboard must consume the published view model")
    }

    func testDashboardDoesNotComputeItsOwnRates() throws {
        let html = try html()
        for forbidden in ["tokensOut / decodeSec", "emittedTokens /", "1000 / p50"] {
            XCTAssertFalse(html.contains(forbidden),
                           "the dashboard must not recompute \(forbidden)")
        }
    }

    func testDashboardRendersUnavailableAsEmDash() throws {
        let script = try script()
        // The dash comes from the renderer. The markup's dashes are
        // only what the page shows before the first answer arrives, so
        // an assertion about the markup proves nothing.
        let text = try XCTUnwrap(Self.function("cardText", in: script), "there is no card text")
        XCTAssertTrue(text.contains("if (!card) return \"—\""),
                      "a missing card must render as an em dash")
        XCTAssertNotNil(text.range(
            of: #"card\.value === null \|\| card\.value === undefined\) return "—""#,
            options: .regularExpression),
            "a card without a value must render as an em dash")
        XCTAssertFalse(text.contains("|| 0"),
                       "a missing value is not a zero")
        let cell = try XCTUnwrap(Self.function("metricCellNode", in: script), "there is no metric cell")
        XCTAssertTrue(cell.contains("\"—\""),
                      "an unavailable reading draws a dash in the cell too")
    }

    /// The renderer is driven by the card list: it reads the cards, and
    /// it does not know what a card is called.
    func testTheRendererIsDrivenByTheCards() throws {
        let html = try html()
        XCTAssertTrue(html.contains("model.cards"),
                      "the renderer walks the published cards")
        XCTAssertTrue(html.contains("data-card"),
                      "the rows declare which card they show")
    }

    /// A metric the dashboard renders must come from a card the server
    /// published. The rows may name a card; the script may not name a
    /// metric of its own.
    func testTheScriptNamesNoMetricOfItsOwn() throws {
        let script = try script()
        // The script must not read a metric by name: it asks the model.
        for forbidden in ["throughput.display_prefill_tokens_per_sec",
                          "throughput.display_decode_tokens_per_sec",
                          "hardware.cpu_percent",
                          "gpu.busy_ratio",
                          "cache.warm_bytes",
                          "context.active_tokens"] {
            XCTAssertFalse(script.contains(forbidden),
                           "the script must not name the metric '\(forbidden)'")
        }
    }

    /// The actions the model cannot perform are not drawn.
    func testTheActionsComeFromTheModel() throws {
        let script = try script()
        XCTAssertTrue(script.contains("model.actions"),
                      "the available actions come from the published model")
        // The plan keeps the action endpoints; what the model decides
        // is which of them are offered and whether they are enabled.
        XCTAssertTrue(script.contains("el.disabled = !action.isEnabled"),
                      "the model decides whether an action is enabled")
        XCTAssertTrue(script.contains("el.hidden = !action"),
                      "an action the model does not offer is not drawn")
    }

    /// The view-model stream replaced the metrics stream.
    func testTheDashboardStreamsTheViewModel() throws {
        let html = try html()
        XCTAssertTrue(html.contains("/v1/dashboard/stream"),
                      "the dashboard subscribes to the published model")
    }

    /// The page claims nothing the server did not publish. Every value
    /// below was a fabrication: a bandwidth nobody measured, a legend
    /// the server never published, a miss rate nobody measures.
    func testThePageInventsNoValue() throws {
        let html = try html()
        for forbidden in ["585 GB/s",
                          "Q4g • Metal 4 (Tensor)",
                          "Sampling: 500ms",
                          "Prefill Burst",
                          "Generation Stream",
                          "Misses",
                          "CACHE MISS",
                          "v0.4.0-phase4",
                          "Running • localhost:8080",
                          "req_019a84",
                          "Qwen3-Q4g",
                          "Apple Silicon",
                          "Inference Activity (60s Trace)",
                          "last 60 seconds",
                          "-60 sec",
                          "Real-Time HTTP / IPC"] {
            XCTAssertFalse(html.contains(forbidden),
                           "the page must not claim '\(forbidden)'")
        }
    }

    /// The trace legend names the series the server published. A
    /// legend that names a series the model did not publish is a
    /// caption, not a reading.
    func testTheTraceLegendComesFromTheModel() throws {
        let script = try script()
        XCTAssertTrue(script.contains("trace-legend"),
                      "the legend is drawn from the published cards")
        let body = try XCTUnwrap(Self.function("renderTraceLegend", in: script),
                                 "the legend has no renderer")
        XCTAssertTrue(body.contains("card.title"),
                      "the legend takes the title the server published")
        XCTAssertTrue(body.contains("hasSample"),
                      "the legend names only the series that have samples")
    }

    /// A bar is drawn for a percentage, and only for a percentage. An
    /// absent reading has no bar at all: a zero-width bar would look
    /// like a measured zero.
    func testABarIsDrawnOnlyForAPercentage() throws {
        let script = try script()
        XCTAssertTrue(script.contains("card.unit === \"%\"") || script.contains("unit === \"%\""),
                      "a bar is drawn for a percentage reading")
        XCTAssertFalse(script.contains("number === null ? \"0%\""),
                       "an absent reading must not draw a zero-width bar")
    }

    /// A failed fetch clears the values. A stale number is not a
    /// current one, and the page must not keep the last answer on
    /// screen as though it were still true.
    func testAFailedFetchClearsTheValues() throws {
        let script = try script()
        let body = try XCTUnwrap(Self.function("fetchDashboard", in: script),
                                 "the dashboard has no fetch")
        XCTAssertTrue(body.contains("renderDisconnected()"),
                      "a failed fetch clears the dashboard")
        let clear = try XCTUnwrap(Self.function("renderDisconnected", in: script),
                                  "there is no way to clear the dashboard")
        XCTAssertTrue(clear.contains("textContent = \"—\""),
                      "the cleared values are dashes")
        // Every published surface is cleared, not just the cards: the
        // hero, the fact chips, the context meter and the trace.
        for surface in ["hero-model-title", "chip-cores", "ctx-tokens-label",
                        "sparkline-series-", "opacity-0", "vel-status-text"] {
            XCTAssertTrue(clear.contains(surface),
                          "the disconnect path must clear '\(surface)'")
        }
        // The trace is emptied, not left at the shape it had.
        XCTAssertTrue(clear.contains("setAttribute(\"d\", \"\")"),
                      "the trace's paths are emptied, not left as they were")
    }

    /// The published phase is the phase. A failed load is a failure,
    /// not an unload, and the server's reason is shown.
    func testThePublishedPhaseIsNotCollapsed() throws {
        let script = try script()
        let body = try XCTUnwrap(Self.function("updateStatusIndicators", in: script),
                                 "there is no status renderer")
        for phase in ["failed", "loading", "unloading", "unloaded"] {
            XCTAssertTrue(body.contains("\"\(phase)\""),
                          "the published phase '\(phase)' must have its own state")
        }
        XCTAssertTrue(body.contains("detail"),
                      "the server's reason for the phase is shown")
        XCTAssertFalse(body.contains("Stopped"),
                       "a server that stopped answering is not stopped")
    }

    /// A context over its budget is over its budget. The number is
    /// true; only the shape that draws it is clipped.
    func testAnOverCapacityContextIsNotClamped() throws {
        let script = try script()
        let body = try XCTUnwrap(Self.function("renderContext", in: script),
                                 "there is no context renderer")
        XCTAssertFalse(body.contains("Math.min(100, (active / capacity)"),
                       "the published percentage must not be clamped")
        XCTAssertFalse(body.contains("Math.min(100, Math.round"),
                       "the percentage on screen must not be clamped")
        XCTAssertFalse(body.contains("Math.max(0, Math.round"),
                       "the percentage on screen must not be clamped")
    }

    /// The playground measures the browser's own request. It is not the
    /// engine's time to first token and not the engine's rate, and it
    /// says so.
    func testThePlaygroundLabelsItsOwnMeasurements() throws {
        let script = try script()
        XCTAssertTrue(script.contains("browser round trip"),
                      "the playground's own measurements are labelled as such")
        // The engine's rate and its latency carry the engine's names:
        // they are the cards, and only the cards.
        XCTAssertFalse(script.contains("tok/s"),
                       "the dashboard must not claim the engine's rate")
        XCTAssertFalse(script.contains("TTFT:"),
                       "the dashboard must not claim the engine's latency")
    }

    /// A rejected action is a rejected action. The status is read
    /// before anything is claimed about the engine.
    func testTheActionsReadTheStatus() throws {
        let script = try script()
        for action in ["unloadModel", "reloadOrLoadModel", "restartEngine"] {
            let body = try XCTUnwrap(Self.function(action, in: script),
                                     "\(action) has no body")
            XCTAssertTrue(body.contains("res.ok"),
                          "\(action) must read the status before it claims success")
        }
    }

    /// The server exposes no cache purge, so the dashboard offers no
    /// cache purge. A button that claims an action the server cannot
    /// perform is a lie with a click.
    func testNoActionIsOfferedThatTheServerCannotPerform() throws {
        let html = try html()
        XCTAssertFalse(html.contains("purgeCache"),
                       "the server has no cache purge: the page must not offer one")
    }

    /// A model with no prefix cache has no cache cards, and no cache
    /// section either.
    func testAModelWithoutAPrefixCacheHasNoCacheSection() throws {
        let script = try script()
        let body = try XCTUnwrap(Self.function("renderCacheSections", in: script),
                                 "the cache sections have no renderer")
        XCTAssertTrue(body.contains("section.hidden = cards.length === 0"),
                      "a model with no prefix cache has no cache section")
    }

    /// The script body of the document: the last script block, which
    /// is the dashboard's. Taking the first would make every assertion
    /// below vacuous the day another script is added above it.
    static func script(of html: String) -> String? {
        var blocks: [String] = []
        var rest = html[html.startIndex...]
        while let open = rest.range(of: "<script>") {
            let after = rest[open.upperBound...]
            guard let close = after.range(of: "</script>") else { break }
            blocks.append(String(after[..<close.lowerBound]))
            rest = after[close.upperBound...]
        }
        return blocks.last { $0.contains("API_BASE") } ?? blocks.last
    }

    /// The body of one function in the script: from its name to the
    /// next function. An async function ends at the next async one
    /// too, or a guard would be satisfied by its neighbour's code.
    static func function(_ name: String, in script: String) -> String? {
        guard let start = script.range(of: "function \(name)") else { return nil }
        let rest = script[start.upperBound...]
        guard let end = rest.range(of: #"\n    (async )?function "#,
                                   options: .regularExpression) else {
            return String(rest)
        }
        return String(rest[..<end.lowerBound])
    }

    /// Reads a file relative to the package root, which is two
    /// directories above this file.
    static func read(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // QWServerTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
        let url = root.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}

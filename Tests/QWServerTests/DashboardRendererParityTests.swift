// DashboardRendererParityTests.swift — The web dashboard draws the
// published view model. A renderer that computes its own rates can
// disagree with the engine about what happened, and a renderer that
// invents a value claims something the server never said.
//
// These are source-text assertions: they read the page the browser
// will run. Each one was proved against the broken code it forbids.

import Darwin
import Foundation
import JavaScriptCore
import XCTest
@testable import QWServer

private extension QWHttpServer {
    var boundPort: UInt16? {
        listener?.port?.rawValue
    }
}

final class DashboardRendererParityTests: XCTestCase {
    private static let stylesheetPaths = [
        "Sources/Apps/QWServer/Resources/css/base.css",
        "Sources/Apps/QWServer/Resources/css/panels.css",
        "Sources/Apps/QWServer/Resources/css/chat.css",
        "Sources/Apps/QWServer/Resources/css/decisions.css",
    ]
    private static let scriptPaths = [
        "Sources/Apps/QWServer/Resources/js/state.js",
        "Sources/Apps/QWServer/Resources/js/dashboard.js",
        "Sources/Apps/QWServer/Resources/js/markdown.js",
        "Sources/Apps/QWServer/Resources/js/chat-presets.js", "Sources/Apps/QWServer/Resources/js/chat-media.js",
        "Sources/Apps/QWServer/Resources/js/chat.js",
        "Sources/Apps/QWServer/Resources/js/decisions-views.js",
        "Sources/Apps/QWServer/Resources/js/decisions.js",
        "Sources/Apps/QWServer/Resources/js/actions.js",
        "Sources/Apps/QWServer/Resources/js/settings.js",
        "Sources/Apps/QWServer/Resources/js/updates.js",
    ]

    private func html() throws -> String {
        try Self.read("Sources/Apps/QWServer/Resources/controlbar.html")
    }

    private func styles() throws -> String {
        try Self.stylesheetPaths.map { try Self.read($0) }.joined(separator: "\n")
    }

    private func script() throws -> String {
        try Self.scriptPaths.map { try Self.read($0) }.joined(separator: "\n")
    }

    private func allWebUISource() throws -> String {
        try [html(), styles(), script()].joined(separator: "\n")
    }

    func testDashboardFetchesTheViewModel() throws {
        let ui = try allWebUISource()
        XCTAssertTrue(ui.contains("/v1/dashboard"),
                      "the dashboard must consume the published view model")
    }

    func testDashboardDoesNotComputeItsOwnRates() throws {
        let ui = try allWebUISource()
        for forbidden in ["tokensOut / decodeSec", "emittedTokens /", "1000 / p50"] {
            XCTAssertFalse(ui.contains(forbidden),
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
        let ui = try allWebUISource()
        XCTAssertTrue(ui.contains("model.cards"),
                      "the renderer walks the published cards")
        XCTAssertTrue(ui.contains("data-card"),
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
        let ui = try allWebUISource()
        XCTAssertTrue(ui.contains("/v1/dashboard/stream"),
                      "the dashboard subscribes to the published model")
    }

    /// The page claims nothing the server did not publish. Every value
    /// below was a fabrication: a bandwidth nobody measured, a legend
    /// the server never published, a miss rate nobody measures.
    func testThePageInventsNoValue() throws {
        let ui = try allWebUISource()
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
            XCTAssertFalse(ui.contains(forbidden),
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
        let ui = try allWebUISource()
        XCTAssertFalse(ui.contains("purgeCache"),
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

    /// The workbench UI must load zero external CDN scripts, stylesheets, or URLs.
    func testDashboardHasNoExternalCDNDependencies() throws {
        let paths = ["Sources/Apps/QWServer/Resources/controlbar.html"]
            + Self.stylesheetPaths + Self.scriptPaths
        for path in paths {
            let source = try Self.read(path)
            XCTAssertFalse(source.contains("https://"),
                           "\(path) must remain strictly offline with zero https URLs")
            XCTAssertFalse(source.contains("cdn.tailwindcss.com"),
                           "\(path) must not reference Tailwind CDN")
            XCTAssertFalse(source.contains("fonts.googleapis.com"),
                           "\(path) must not reference Google Fonts CDN")
        }
    }

    /// All four QWG processor vector marks must stay free of the left-pin 340 bug.
    func testProcessorVectorMarksHaveNoLeftPin340Bug() throws {
        for path in [
            "Sources/Apps/QWServer/Resources/logo.svg",
            "Sources/Apps/QWServer/Resources/controlbar.html",
            "Sources/Apps/QWMenuBar/QGProcessorLogoView.swift",
            "Scripts/gen_app_icon.swift",
        ] {
            let source = try Self.read(path)
            XCTAssertFalse(source.contains("x2=\"340\""),
                           "\(path) must not contain x2=\"340\" left-pin regression")
            XCTAssertFalse(source.contains("x: 340"),
                           "\(path) must not contain x: 340 left-pin regression")
        }
    }

    /// Prefill banner tracks live percentage, holds 100% completion briefly, and uses compact M/comma token formatting.
    func testPrefillBannerTracksProgressHoldsCompletionAndResetsOnDisconnect() throws {
        let html = try html()
        let script = try script()
        XCTAssertTrue(html.contains("<div id=\"prefill-banner\" class=\"prefill-banner\">"))
        XCTAssertTrue(html.contains("data-card=\"prefill.progress_tokens\" data-card-unit=\"separate\""))
        XCTAssertTrue(html.contains("id=\"prefill-pct\""))

        let formatter = try XCTUnwrap(Self.function("formatTokenCount", in: script))
        XCTAssertFalse(formatter.contains("|| 0"), "formatTokenCount must never coerce unavailable counts to zero")
        XCTAssertTrue(formatter.contains("1000000"), "formatTokenCount must switch to M indicator at 1,000,000")

        let contextFn = try XCTUnwrap(Self.function("renderContext", in: script))
        XCTAssertTrue(contextFn.contains("formatTokenCount("))
        XCTAssertFalse(contextFn.contains("toLocaleString()"))

        let bannerFn = try XCTUnwrap(Self.function("renderPrefillBanner", in: script))
        XCTAssertTrue(bannerFn.contains("Prefill Complete"))
        XCTAssertTrue(bannerFn.contains("prefillHoldTimer"))
        XCTAssertTrue(bannerFn.contains("100%"))
        XCTAssertFalse(bannerFn.contains("banner.hidden = true"))

        let disconnectFn = try XCTUnwrap(Self.function("renderDisconnected", in: script))
        XCTAssertTrue(disconnectFn.contains("prefillHoldTimer"))
        XCTAssertTrue(disconnectFn.contains("prefill-progress-fill"))
        XCTAssertTrue(disconnectFn.contains("prefill-pct"))
        XCTAssertFalse(disconnectFn.contains("prefillBanner.hidden = true"))
    }

    /// Smoke probe output parses Markdown safely into styled elements while neutralizing raw HTML and unsafe URLs.
    func testSmokeProbeRendersMarkdownAndEscapesUntrustedHtml() throws {
        let html = try html()
        let script = try script()
        XCTAssertTrue(html.contains("id=\"probe-output\""))
        XCTAssertTrue(html.contains("id=\"probe-mode-btn\""))
        XCTAssertTrue(html.contains("toggleSmokeProbeMode()"))
        XCTAssertTrue(script.contains("renderSmokeProbeMarkdown("))

        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in
            jsError = exc?.toString() ?? "unknown JS error"
        }
        for name in [
            "escapeHtml",
            "sanitizeMarkdownUrl",
            "splitMarkdownTableCells",
            "isMarkdownTableDelimiter",
            "applyInlineMarkdownEmphasis",
            "renderInlineMarkdown",
            "renderSmokeProbeMarkdown",
        ] {
            let fnBody = try XCTUnwrap(Self.function(name, in: script), "missing \(name)")
            ctx.evaluateScript("function \(name)\(fnBody)")
            XCTAssertNil(jsError, "JS error evaluating \(name): \(jsError ?? "")")
        }

        let fixture = """
        # LSA Kernel Implementation in Qwegn

        ---

        ## Core LSA Attention Kernels

        | File | Role |
        |------|------|
        | **`QWLSAOperator.swift`** | Uses `ShvqTgLut<1|2|3>` |

        ### 1. Coarse Selection
        - **32 KiB-TG** via `qwg_shvq_read_elem_t<bool SUB2>`
          - Nested item for vision_encode_ms

        ```
        QWEngine
        └─ QWLSAOperator.select()
        ```

        <img src=x onerror=alert(1)> <script>evil()</script> [bad](javascript:alert(1)) [top](/ui#top)
        """
        let fn = try XCTUnwrap(ctx.objectForKeyedSubscript("renderSmokeProbeMarkdown"))
        let result = try XCTUnwrap(fn.call(withArguments: [fixture]))
        XCTAssertNil(jsError, "JS execution failed: \(jsError ?? "")")
        let rendered = try XCTUnwrap(result.toString())

        XCTAssertTrue(rendered.contains("<h1 class=\"md-h1\">LSA Kernel Implementation in Qwegn</h1>"))
        XCTAssertTrue(rendered.contains("<hr class=\"md-hr\" />"))
        XCTAssertTrue(rendered.contains("<h2 class=\"md-h2\">Core LSA Attention Kernels</h2>"))
        XCTAssertTrue(rendered.contains("<table class=\"ledger md-table\">"))
        XCTAssertTrue(rendered.contains("<strong><code class=\"md-inline-code\">QWLSAOperator.swift</code></strong>"))
        XCTAssertTrue(rendered.contains("<code class=\"md-inline-code\">ShvqTgLut&lt;1|2|3&gt;</code>"))
        XCTAssertTrue(rendered.contains("<h3 class=\"md-h3\">1. Coarse Selection</h3>"))
        XCTAssertTrue(rendered.contains("<ul><li><strong>32 KiB-TG</strong>"))
        XCTAssertTrue(rendered.contains("&lt;bool SUB2&gt;"))
        XCTAssertTrue(rendered.contains("vision_encode_ms"))
        XCTAssertFalse(rendered.contains("<em>encode</em>"))
        XCTAssertTrue(rendered.contains("<pre class=\"mono\"><code>QWEngine\n└─ QWLSAOperator.select()</code></pre>"))
        XCTAssertTrue(rendered.contains("href=\"/ui#top\""))

        XCTAssertTrue(rendered.contains("&lt;img src=x onerror=alert(1)&gt;"))
        XCTAssertTrue(rendered.contains("&lt;script&gt;evil()&lt;/script&gt;"))
        XCTAssertFalse(rendered.contains("<img"))
        XCTAssertFalse(rendered.contains("<script"))
        XCTAssertNil(rendered.range(of: #"<[^>]*\bonerror\s*="#, options: [.regularExpression, .caseInsensitive]))
        XCTAssertFalse(rendered.lowercased().contains("javascript:"))

        let extractFnBody = try XCTUnwrap(Self.function("extractSmokeProbeParts", in: script))
        ctx.evaluateScript("function extractSmokeProbeParts\(extractFnBody)")
        let extractFn = try XCTUnwrap(ctx.objectForKeyedSubscript("extractSmokeProbeParts"))
        let extracted = try XCTUnwrap(extractFn.call(withArguments: [
            ["content": "<think>Step 1</think>Answer body"],
            ["choices": []],
        ])?.toDictionary())
        XCTAssertEqual(extracted["reasoning"] as? String, "Step 1")
        XCTAssertEqual(extracted["content"] as? String, "Answer body")
    }

    func testWebUIAssetsAreModularAndServedWithProperMimeTypes() async throws {
        let markup = try html()
        XCTAssertFalse(markup.lowercased().contains("<style"))
        XCTAssertNil(markup.range(of: "<script>", options: .caseInsensitive))
        let assetSpecs: [(route: String, mime: String, sourcePath: String)] = [
            ("/ui", "text/html", "Sources/Apps/QWServer/Resources/controlbar.html"),
            ("/ui/logo.svg", "image/svg+xml", "Sources/Apps/QWServer/Resources/logo.svg"),
            ("/ui/css/base.css", "text/css", "Sources/Apps/QWServer/Resources/css/base.css"),
            ("/ui/css/panels.css", "text/css", "Sources/Apps/QWServer/Resources/css/panels.css"),
            ("/ui/css/chat.css", "text/css", "Sources/Apps/QWServer/Resources/css/chat.css"),
            ("/ui/js/state.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/state.js"),
            ("/ui/js/dashboard.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/dashboard.js"),
            ("/ui/js/markdown.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/markdown.js"),
            ("/ui/js/chat-presets.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/chat-presets.js"), ("/ui/js/chat-media.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/chat-media.js"),
            ("/ui/js/chat.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/chat.js"),
            ("/ui/js/actions.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/actions.js"),
            ("/ui/js/settings.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/settings.js"),
            ("/ui/js/updates.js", "application/javascript", "Sources/Apps/QWServer/Resources/js/updates.js"),
        ]
        for spec in assetSpecs {
            let lineCount = try Self.read(spec.sourcePath).components(separatedBy: "\n").count
            XCTAssertLessThanOrEqual(lineCount, 800, "\(spec.sourcePath) must stay <= 800 lines")
            if spec.route != "/ui" && spec.route != "/ui/logo.svg" {
                XCTAssertTrue(markup.contains(spec.route), "controlbar.html must reference \(spec.route)")
            }
        }

        let cfg = QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test-model")
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: logger)
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.boundPort)
        let session = URLSession.shared
        for spec in assetSpecs {
            let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)\(spec.route)"))
            let (data, resp) = try await session.data(from: url)
            let http = try XCTUnwrap(resp as? HTTPURLResponse)
            XCTAssertEqual(http.statusCode, 200, "Expected 200 for \(spec.route)")
            let ct = try XCTUnwrap(http.value(forHTTPHeaderField: "Content-Type"))
            XCTAssertTrue(ct.hasPrefix(spec.mime), "Unexpected Content-Type '\(ct)' for \(spec.route)")
            XCTAssertEqual(http.value(forHTTPHeaderField: "Cache-Control"), "no-cache")
            XCTAssertEqual(String(data: data, encoding: .utf8), try Self.read(spec.sourcePath))
        }

        let dirURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/ui/css"))
        let (_, dirResp) = try await session.data(from: dirURL)
        XCTAssertEqual((dirResp as? HTTPURLResponse)?.statusCode, 503)

        let traversalResp = Self.rawLoopbackHTTP(port: port, requestLine: "GET /ui/../Package.swift HTTP/1.1")
        XCTAssertTrue(traversalResp.hasPrefix("HTTP/1.1 403 "), "Expected 403 traversal guard, got: \(traversalResp)")
    }

    func testSparklineConnectsPrefillAcrossActiveDecodeAndSplitsTrueGaps() throws {
        let script = try script()
        let updateFnBody = try XCTUnwrap(Self.function("updateSparkline", in: script), "missing updateSparkline")

        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in
            jsError = exc?.toString() ?? "unknown JS error"
        }
        ctx.evaluateScript("""
        var _attrs = {};
        var _classes = {};
        var document = {
          getElementById: function(id) {
            if (!_attrs[id]) _attrs[id] = {};
            if (!_classes[id]) _classes[id] = {};
            return {
              setAttribute: function(k, v) { _attrs[id][k] = String(v); },
              classList: {
                add: function(c) { _classes[id][c] = true; },
                remove: function(c) { delete _classes[id][c]; }
              }
            };
          }
        };
        function updateSparkline\(updateFnBody)
        """)
        XCTAssertNil(jsError, "JS error setting up updateSparkline: \(jsError ?? "")")

        ctx.evaluateScript("updateSparkline([1420, null, null, 1580, null, 1510], [80, 81, 82, null, 83, 81]);")
        XCTAssertNil(jsError, "JS error invoking updateSparkline: \(jsError ?? "")")
        let attrs = try XCTUnwrap(ctx.objectForKeyedSubscript("_attrs")?.toDictionary())
        let aAttrs = try XCTUnwrap(attrs["sparkline-series-a-path"] as? [String: Any])
        let aArea = try XCTUnwrap(attrs["sparkline-series-a-area"] as? [String: Any])
        let aGuide = try XCTUnwrap(attrs["sparkline-series-a-guide"] as? [String: Any])
        let aMarkers = try XCTUnwrap(attrs["sparkline-series-a-markers"] as? [String: Any])
        let bAttrs = try XCTUnwrap(attrs["sparkline-series-b-path"] as? [String: Any])

        let pathA = try XCTUnwrap(aAttrs["d"] as? String)
        XCTAssertTrue(pathA.hasPrefix("M "), "Expected continuous prefill line path, got: \(pathA)")
        XCTAssertTrue(pathA.contains(" C "), "Expected cubic Bezier segments in prefill line: \(pathA)")
        XCTAssertFalse((aArea["d"] as? String ?? "").isEmpty)
        XCTAssertEqual(aGuide["d"] as? String, "")
        let markersA = try XCTUnwrap(aMarkers["d"] as? String)
        XCTAssertEqual(markersA.components(separatedBy: "M ").count - 1, 1,
                       "Expected only endpoint marker, not isolated per-turn dots: \(markersA)")
        XCTAssertTrue((bAttrs["d"] as? String ?? "").hasPrefix("M "))

        ctx.evaluateScript("updateSparkline([1420, null, 1580], [80, null, 82]);")
        XCTAssertNil(jsError, "JS error testing dual gap: \(jsError ?? "")")
        let gapAttrs = try XCTUnwrap(ctx.objectForKeyedSubscript("_attrs")?.toDictionary())
        XCTAssertEqual((gapAttrs["sparkline-series-a-path"] as? [String: Any])?["d"] as? String, "")
        XCTAssertEqual((gapAttrs["sparkline-series-b-path"] as? [String: Any])?["d"] as? String, "")
    }

    func testHeroModelTitlePreservesMarqueeAcrossTelemetryTicksAndContainsOverflow() throws {
        let styles = try styles()
        let script = try script()

        XCTAssertTrue(styles.contains(".hero-model-marquee"))
        XCTAssertTrue(styles.contains("@keyframes qwg-hero-marquee"))
        XCTAssertTrue(styles.contains("@media (prefers-reduced-motion: reduce)"))
        XCTAssertTrue(styles.contains(".hero-top > div:first-child"))
        XCTAssertTrue(styles.contains("min-width: 0"))

        let factsFn = try XCTUnwrap(Self.function("renderFacts", in: script), "missing renderFacts")
        XCTAssertTrue(factsFn.contains("dataset.sig"),
                      "hero title must keep its signature so 1Hz ticks do not restart running text")
        XCTAssertTrue(factsFn.contains("hero-model-marquee"))
        XCTAssertTrue(factsFn.contains("syncHeroTitleMarquee"))

        let disconnectFn = try XCTUnwrap(Self.function("renderDisconnected", in: script))
        XCTAssertTrue(disconnectFn.contains("dataset.sig"),
                      "disconnect must clear hero signature so reconnect re-renders cleanly")
    }

    func testChatTabAndWorkbenchControlsExistInMarkup() throws {
        let doc = try html()
        for required in [
            "id=\"nav-tab-chat\"",
            "id=\"view-chat\"",
            "id=\"probe-open-chat-btn\"",
            "/ui/css/chat.css",
            "/ui/js/chat-presets.js", "/ui/js/chat-media.js",
            "/ui/js/chat.js",
            "id=\"chat-mempack-badge\"",
            "id=\"chat-workspace-path\"",
            "id=\"chat-mempack-mount-btn\"",
            "id=\"chat-harness-bar\"",
            "id=\"chat-system-drawer\"",
            "id=\"chat-params-drawer\"",
            "id=\"chat-param-cache-prompt\"",
            "id=\"chat-param-cache-key\"",
            "id=\"chat-param-store\"",
            "id=\"chat-param-exact-spec\"",
            "id=\"chat-param-metadata\"",
            "id=\"chat-tools-json\"",
            "id=\"chat-attach-btn\"",
            "id=\"chat-stop-btn\"",
        ] {
            XCTAssertTrue(doc.contains(required), "missing chat workbench markup: \(required)")
        }

        let stateScript = try Self.read("Sources/Apps/QWServer/Resources/js/state.js")
        let switchFn = try XCTUnwrap(Self.function("switchTab", in: stateScript), "missing switchTab")
        XCTAssertTrue(switchFn.contains("\"chat\""), "switchTab must accept chat tab")
    }

    func testChatWorkbenchPayloadSSEThinkingAndToolHelpersInJSContext() throws {
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in
            jsError = exc?.toString() ?? "unknown JS error"
        }

        let allScript = try script()
        for name in ["escapeHtml", "formatTokenCount", "sanitizeMarkdownUrl", "splitMarkdownTableCells",
                     "isMarkdownTableDelimiter", "applyInlineMarkdownEmphasis", "renderInlineMarkdown",
                     "renderSmokeProbeMarkdown"] {
            let fnBody = try XCTUnwrap(Self.function(name, in: allScript), "missing \(name)")
            ctx.evaluateScript("function \(name)\(fnBody)")
            XCTAssertNil(jsError, "JS error evaluating \(name): \(jsError ?? "")")
        }

        let presetsScript = try Self.read("Sources/Apps/QWServer/Resources/js/chat-presets.js")
        ctx.evaluateScript(presetsScript)
        XCTAssertNil(jsError, "JS error evaluating chat-presets.js: \(jsError ?? "")")

        let chatScript = try Self.read("Sources/Apps/QWServer/Resources/js/chat.js")
        ctx.evaluateScript(chatScript)
        XCTAssertNil(jsError, "JS error evaluating chat.js: \(jsError ?? "")")

        let presetCount = ctx.evaluateScript("getAllSystemPresets().length")?.toInt32() ?? 0
        XCTAssertGreaterThanOrEqual(presetCount, 5)

        let harnessCheck = ctx.evaluateScript("""
        (() => {
          const s = createBlankChatSession("H", "server-default", "");
          applyPresetHarnessToSession(s, "coding-agent");
          const coding = Object.assign({}, s.params);
          applyPresetHarnessToSession(s, "deep-reasoning");
          const deep = Object.assign({}, s.params);
          applyPresetHarnessToSession(s, "concise-technical");
          const concise = Object.assign({}, s.params);
          applyPresetHarnessToSession(s, "json-extractor");
          const jsonEx = Object.assign({}, s.params);
          const detected = detectWorkspacePathInSystemPrompt("Working in /Users/dev/Developer/ai/qwegine.");
          s.presetId = "coding-agent";
          s.systemPrompt = getPresetById("coding-agent").prompt;
          s.workspacePath = detected;
          s.messages.push({ id: "u1", role: "user", content: "Explain QWMemoryPack" });
          globalThis.publishedModel = { facts: { mountedMemoryPack: "qwegine (Q4)", mountedMemoryPackRoot: detected, mountedMemoryPackKV: "q4" } };
          const builtPrompt = buildHarnessedSystemPrompt(s);
          const builtPayload = buildChatCompletionPayload(s, {});
          delete globalThis.publishedModel;
          return {
            coding, deep, concise, jsonEx, detected, builtPrompt,
            sysRole: builtPayload.messages[0].role,
            sysContent: builtPayload.messages[0].content,
            validOk: validateJsonTurnContent('{"ok":true}').valid,
            fencedOk: validateJsonTurnContent('```json\\n{}\\n```').valid
          };
        })()
        """)?.toDictionary()
        XCTAssertNil(jsError, "harness check threw: \(jsError ?? "")")
        let codingP = try XCTUnwrap(harnessCheck?["coding"] as? [String: Any])
        XCTAssertEqual(codingP["enableThinking"] as? String, "true")
        XCTAssertEqual(codingP["reasoningEffort"] as? String, "high")
        XCTAssertEqual(codingP["temperature"] as? Double ?? -1, 0.2, accuracy: 0.001)
        let deepP = try XCTUnwrap(harnessCheck?["deep"] as? [String: Any])
        XCTAssertEqual(deepP["temperature"] as? Double ?? -1, 0.6, accuracy: 0.001)
        XCTAssertEqual(deepP["enableThinking"] as? String, "true")
        let conciseP = try XCTUnwrap(harnessCheck?["concise"] as? [String: Any])
        XCTAssertEqual(conciseP["enableThinking"] as? String, "false")
        XCTAssertEqual(conciseP["reasoningEffort"] as? String, "low")
        let jsonP = try XCTUnwrap(harnessCheck?["jsonEx"] as? [String: Any])
        XCTAssertEqual(jsonP["responseFormat"] as? String, "json_object")
        XCTAssertEqual(jsonP["temperature"] as? Double ?? -1, 0.0, accuracy: 0.001)
        XCTAssertEqual(harnessCheck?["detected"] as? String, "/Users/dev/Developer/ai/qwegine")
        let builtSys = try XCTUnwrap(harnessCheck?["sysContent"] as? String)
        XCTAssertTrue(builtSys.contains("<workspace_context>"))
        XCTAssertTrue(builtSys.contains("Workspace directory: /Users/dev/Developer/ai/qwegine"))
        XCTAssertTrue(builtSys.contains("<memory_pack_harness>"))
        XCTAssertEqual(harnessCheck?["validOk"] as? Bool, true)
        XCTAssertEqual(harnessCheck?["fencedOk"] as? Bool, false)

        let payloadCheck = ctx.evaluateScript("""
        (() => {
          const baseSession = createBlankChatSession("Test", "server-default", "");
          baseSession.messages.push({
            id: "u1", role: "user", content: "Inspect this",
            images: [{ dataUrl: "data:image/png;base64,AAAA" }]
          });
          const p1 = buildChatCompletionPayload(baseSession, {});
          baseSession.systemPrompt = "Be strict";
          baseSession.params.enableThinking = "true";
          baseSession.params.reasoningEffort = "xhigh";
          baseSession.params.responseFormat = "json_object";
          baseSession.params.toolsJson = '[{"type":"function","function":{"name":"ping"}}]';
          baseSession.messages.push({ id: "a1", role: "assistant", content: "Partial" });
          const p2 = buildChatCompletionPayload(baseSession, { continueLastAssistant: true });
          return { p1, p2 };
        })()
        """)?.toDictionary()
        XCTAssertNil(jsError, "payload builder threw: \(jsError ?? "")")
        let p1 = try XCTUnwrap(payloadCheck?["p1"] as? [String: Any])
        let p1Msgs = try XCTUnwrap(p1["messages"] as? [[String: Any]])
        XCTAssertEqual(p1Msgs.count, 1, "empty systemPrompt should omit leading system turn")
        let firstParts = p1Msgs[0]["content"] as? [[String: Any]]
        XCTAssertEqual(firstParts?.count, 2, "multimodal turn should preserve text and image_url")

        let p2 = try XCTUnwrap(payloadCheck?["p2"] as? [String: Any])
        let p2Msgs = try XCTUnwrap(p2["messages"] as? [[String: Any]])
        XCTAssertEqual(p2Msgs.first?["role"] as? String, "system")
        XCTAssertEqual(p2["enable_thinking"] as? Bool, true)
        XCTAssertEqual(p2["reasoning_effort"] as? String, "xhigh")
        XCTAssertEqual((p2["response_format"] as? [String: Any])?["type"] as? String, "json_object")
        XCTAssertEqual((p2["tools"] as? [[String: Any]])?.count, 1)
        XCTAssertEqual(p2["continue_final_message"] as? Bool, true)
        XCTAssertNil(p2["add_generation_prompt"])

        let sseDone = ctx.evaluateScript("parseChatSSELine('data: [DONE]')")?.toDictionary()
        XCTAssertEqual(sseDone?["done"] as? Bool, true)

        let sseErr = ctx.evaluateScript(#"parseChatSSELine('data: {"type":"server_error","message":"boom"}')"#)?.toDictionary()
        XCTAssertEqual(sseErr?["error"] as? String, "boom")

        let mergedToolName = ctx.evaluateScript("""
        (() => {
          let calls = mergeToolCallChunks([], [{ index: 0, id: "call_1", type: "function", function: { name: "read_", arguments: '{"path":' } }]);
          calls = mergeToolCallChunks(calls, [{ index: 0, function: { name: "file", arguments: '"a.swift"}' } }]);
          return calls[0].function.name + "|" + calls[0].function.arguments;
        })()
        """)?.toString()
        XCTAssertEqual(mergedToolName, #"read_file|{"path":"a.swift"}"#)

        let thinkSplit = ctx.evaluateScript(#"splitInlineThinkTags("<think>step 1</think>\nFinal answer", "", false)"#)?.toDictionary()
        XCTAssertEqual(thinkSplit?["reasoning"] as? String, "step 1")
        XCTAssertEqual(thinkSplit?["answer"] as? String, "Final answer")

        let mdFn = try XCTUnwrap(ctx.objectForKeyedSubscript("renderSmokeProbeMarkdown"))
        let mdRendered = try XCTUnwrap(mdFn.call(withArguments: ["```swift\nlet x = 1\n```"])?.toString())
        XCTAssertTrue(mdRendered.contains("md-code-copy-btn"))
        XCTAssertTrue(mdRendered.contains("<pre class=\"mono\"><code>let x = 1</code></pre>"))
    }

    private static func rawLoopbackHTTP(port: UInt16, requestLine: String) -> String {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return "" }
        defer { Darwin.close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let connectOK = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
        guard connectOK else { return "" }
        let req = "\(requestLine)\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        _ = req.withCString { Darwin.send(fd, $0, strlen($0), 0) }
        var buf = [UInt8](repeating: 0, count: 512)
        let n = Darwin.recv(fd, &buf, buf.count, 0)
        return n > 0 ? String(decoding: buf.prefix(n), as: UTF8.self) : ""
    }

    /// The script body of the document: the last inline script block,
    /// falling back to the modular external JS files when split.
    static func script(of html: String) -> String? {
        var blocks: [String] = []
        var rest = html[html.startIndex...]
        while let open = rest.range(of: "<script>") {
            let after = rest[open.upperBound...]
            guard let close = after.range(of: "</script>") else { break }
            blocks.append(String(after[..<close.lowerBound]))
            rest = after[close.upperBound...]
        }
        if let inline = blocks.last(where: { $0.contains("API_BASE") }) ?? blocks.last {
            return inline
        }
        return try? scriptPaths.map { try read($0) }.joined(separator: "\n")
    }

    /// The body of one function in the script: from its name to the
    /// next top-level function definition.
    static func function(_ name: String, in script: String) -> String? {
        guard let start = script.range(of: "function \(name)") else { return nil }
        let rest = script[start.upperBound...]
        guard let end = rest.range(of: #"\n(?:    )?(?:async )?function "#,
                                   options: .regularExpression) else {
            return String(rest)
        }
        return String(rest[..<end.lowerBound])
    }

    static func read(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = root.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}

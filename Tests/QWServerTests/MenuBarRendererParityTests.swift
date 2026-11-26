// MenuBarRendererParityTests.swift — The MenuBar draws the published
// model, and only that. A renderer that reads a raw telemetry field, or
// a second endpoint, is a renderer that can disagree with the
// dashboard.

import XCTest
@testable import QWServer
import QWCore
import Foundation

final class MenuBarRendererParityTests: XCTestCase {
    private let menuBarDirectory = "Sources/Apps/QWMenuBar"
    private let serverDirectory = "Sources/Apps/QWServer"
    /// The files a renderer draws its numbers from.
    private let renderers = ["TelemetryCards.swift", "TelemetryObservable.swift",
                             "CupertinoTelemetryView.swift", "DualSparklineView.swift",
                             "MarqueeText.swift"]
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
        let menuBarIds = Set(Self.menuBarCardCatalogueIds)
        let dashboardIds = Set(model.cards.map(\.id))
        XCTAssertTrue(menuBarIds.isSubset(of: dashboardIds),
                      "the MenuBar renders cards the dashboard does not publish")
    }

    func testEveryCardTheMenuBarDrawsIsInTheCatalogue() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        let drawn = Self.cardIds(in: source)
        XCTAssertFalse(drawn.isEmpty, "the MenuBar draws no published card at all")
        for id in drawn {
            XCTAssertTrue(Self.menuBarCardCatalogueIds.contains(id),
                          "the MenuBar draws '\(id)', which the catalogue does not name")
        }
    }

    private static let menuBarCardCatalogueIds: [String] = [
        "gpu.busy_ratio",
        "gpu.device",
        "hardware.cpu_percent",
        "hardware.memory_percent",
        "hardware.memory_pressure_level",
        "hardware.gpu_temp_celsius",
        "hardware.fan_rpm",
        "hardware.system_power_watts",
        "hardware.thermal_state",
        "throughput.display_prefill_tokens_per_sec",
        "throughput.display_decode_tokens_per_sec",
        "prefill.progress_tokens",
        "prefill.total_tokens",
        "latency.ttft_ms",
        "context.active_tokens",
        "cache.warm_bytes",
        "cache.ssd_bytes",
        "cache.tokens_reused_total",
        "memory.resident_bytes",
        "memory.arena_reserved_bytes",
    ]

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

    /// The app bundle must ship signed `qwg` and `qwg-smc-helper` sibling executables plus LaunchDaemon plist.
    func testAppBundleShipsAndSignsQWGSiblingExecutable() throws {
        let script = try Self.read("Scripts/build_app.sh")
        XCTAssertTrue(script.contains("$APP/Contents/MacOS/qwg\""),
                      "build_app.sh must copy qwg into Contents/MacOS/qwg")
        XCTAssertTrue(script.contains("codesign") && script.contains("Contents/MacOS/qwg"),
                      "build_app.sh must sign Contents/MacOS/qwg")
        XCTAssertTrue(script.contains("$APP/Contents/MacOS/qwg-smc-helper\""),
                      "build_app.sh must copy qwg-smc-helper into Contents/MacOS/qwg-smc-helper")
        XCTAssertTrue(script.contains("Contents/Library/LaunchDaemons/ai.qwegn.smc-helper.plist"),
                      "build_app.sh must bundle LaunchDaemons/ai.qwegn.smc-helper.plist")
        XCTAssertTrue(script.contains("codesign") && script.contains("Contents/MacOS/qwg-smc-helper"),
                      "build_app.sh must sign Contents/MacOS/qwg-smc-helper")
        let uninstall = try Self.read("Scripts/uninstall.sh")
        XCTAssertTrue(uninstall.contains("ai.qwegn.smc-helper"),
                      "uninstall.sh must unload and remove ai.qwegn.smc-helper")
    }

    /// QWMenuBarApp must use pure AppKit startup without a SwiftUI Settings scene so ⌘, never opens a blank second window.
    func testMenuBarAppDoesNotDeclareEmptySwiftUISettingsScene() throws {
        let appSource = try Self.read("Sources/Apps/QWMenuBar/QWMenuBarApp.swift")
        XCTAssertFalse(appSource.contains("Settings {"),
                       "QWMenuBarApp.swift must not declare a SwiftUI Settings scene")
        XCTAssertFalse(appSource.contains("EmptyView()"),
                       "QWMenuBarApp.swift must not host an EmptyView() scene")
        XCTAssertTrue(appSource.contains("QWMenuBarShortcutRouter"),
                      "QWMenuBarApp.swift must route hotkeys through QWMenuBarShortcutRouter")
        XCTAssertTrue(appSource.contains("buildMainMenu()"),
                      "QWMenuBarApp.swift must install an AppKit mainMenu for standard shortcuts")
    }

    func testWeightsCardUsesRunningTextMarqueeNotMiddleTruncation() throws {
        let cardsSource = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        let cardBody = try XCTUnwrap(cardsSource.range(of: "var loadedWeightsCard: some View {")
            .map { String(cardsSource[$0.lowerBound...]) })
        let sectionEnd = cardBody.range(of: "private struct TraitPill")?.lowerBound ?? cardBody.endIndex
        let loadedCard = String(cardBody[..<sectionEnd])

        XCTAssertTrue(loadedCard.contains("MarqueeText("),
                      "resident model name must use running text MarqueeText")
        XCTAssertFalse(loadedCard.contains(".truncationMode(.middle)"),
                       "long resident model name must scroll cleanly instead of middle elision")
        XCTAssertTrue(loadedCard.contains("marqueeFactPill("),
                      "wide quantization pill must shrink and scroll inside the 440pt card")

        let popoverSource = try Self.read("\(menuBarDirectory)/CupertinoTelemetryView.swift")
        let innerFrame = try XCTUnwrap(
            popoverSource.range(of: ".frame(width: 420)"),
            "popover content stack must set inner 420pt width before outer padding"
        )
        let outerPadding = try XCTUnwrap(
            popoverSource.range(of: ".padding(10)"),
            "popover content stack must apply 10pt outer padding around 420pt tiles"
        )
        XCTAssertLessThan(
            innerFrame.lowerBound,
            outerPadding.lowerBound,
            ".frame(width: 420) must precede .padding(10) so outer inset stays 10pt instead of doubling"
        )
        XCTAssertTrue(popoverSource.contains(".clipped()"),
                      "root popover container must clip overflowing child frames cleanly")
    }

    /// Resident Model card must stay model-centric: no alias clutter, constant host chip, or kernel mode.
    func testResidentModelCardOmitsStaticHostHardwareAndAlias() throws {
        let source = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        for forbidden in ["modelAlias", "gpuName", "metalFamily", "hostLogicalCores", "gdnPrefillMode"] {
            XCTAssertFalse(source.contains(forbidden),
                           "TelemetryCards.swift must not render '\(forbidden)' on the Resident Model card")
        }
    }

    /// Popover header status must not duplicate prefill/decode rates already shown on hero tiles.
    func testPopoverHeaderOmitsDuplicateThroughputRates() throws {
        let cardsSource = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        let observableSource = try Self.read("\(menuBarDirectory)/TelemetryObservable+MenuBar.swift")
        XCTAssertFalse(cardsSource.contains("liveMenuBarRateText"),
                       "TelemetryCards.swift header must not render duplicate liveMenuBarRateText next to status")
        XCTAssertFalse(observableSource.contains("liveMenuBarRateText"),
                       "Dead liveMenuBarRateText helper must not remain in TelemetryObservable+MenuBar.swift")
        if let headerRange = cardsSource.range(of: "var headerSection: some View"),
           let residentRange = cardsSource.range(of: "// MARK: - Resident Model Card"),
           headerRange.lowerBound < residentRange.lowerBound {
            let headerSlice = String(cardsSource[headerRange.lowerBound..<residentRange.lowerBound])
            XCTAssertFalse(headerSlice.contains("tok/s"),
                           "Popover headerSection must not render inline tok/s speeds")
        } else {
            XCTFail("Expected headerSection and Resident Model Card markers in TelemetryCards.swift")
        }
    }

    /// Selectable menu bar telemetry must honor visibility, generation gating, and unavailable gaps.
    func testMenuBarStatusFormatterHonorsSelectedMetricsAndHonesty() {
        func makeCard(_ id: String, value: String?, unit: String?, number: Double? = nil) -> QWDashboardCard {
            QWDashboardCard(
                id: id,
                title: id,
                value: value,
                unit: unit,
                number: number,
                provenance: value == nil ? .unavailable : .measured,
                sampledAt: value == nil ? nil : Date(),
                age: value == nil ? nil : 0.1,
                isStale: false,
                reason: value == nil ? "not measured" : nil,
                series: []
            )
        }

        let loadedDashboard = QWDashboardViewModel(
            sampledAt: Date(),
            lifecycle: .init(phase: .loaded, since: Date(), model: "m", detail: nil),
            modelId: "m",
            capabilities: [QWKnobCapability.prefixCache.rawValue],
            generation: .idle,
            facts: nil,
            cards: [
                makeCard("throughput.display_decode_tokens_per_sec", value: "91.9", unit: "tok/s", number: 91.9),
                makeCard("throughput.display_prefill_tokens_per_sec", value: "93.3", unit: "tok/s", number: 93.3),
                makeCard("memory.arena_reserved_bytes", value: "25.01", unit: "GiB", number: 25.01),
                makeCard("memory.resident_bytes", value: "24.20", unit: "GiB", number: 24.20),
                makeCard("context.active_tokens", value: "38", unit: "tok", number: 38),
                makeCard("hardware.gpu_temp_celsius", value: "54.2", unit: "°C", number: 54.2),
            ],
            actions: [],
            knobs: ["contextTokens": .int(262_144)],
            knobSources: [:]
        )

        // Disconnected or unloaded or empty selection yields nil (icon only).
        XCTAssertNil(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: false, enabledMetrics: [.decodeSpeed], hideSpeedsWhenIdle: false
        ))
        let unloaded = QWDashboardViewModel.build(
            lifecycle: .init(phase: .unloaded, since: nil, model: nil, detail: nil), modelId: nil
        )
        XCTAssertNil(QWMenuBarStatusFormatter.statusTitle(
            dashboard: unloaded, isConnected: true, enabledMetrics: [.decodeSpeed], hideSpeedsWhenIdle: false
        ))
        XCTAssertNil(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [], hideSpeedsWhenIdle: false
        ))

        // Individual metrics and combined fixed-width formatting.
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [.decodeSpeed], hideSpeedsWhenIdle: false
        ), " 91.9 tok/s")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [.prefillSpeed], hideSpeedsWhenIdle: false
        ), "↑  93.3 tok/s")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [.engineMemory], hideSpeedsWhenIdle: false
        ), " 25.01 GiB")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [.contextSize], hideSpeedsWhenIdle: false
        ), "   38/256k")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: [.gpuTemperature], hideSpeedsWhenIdle: false
        ), " 54°C")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: Set(QWMenuBarMetric.allCases),
            hideSpeedsWhenIdle: false
        ), "↓ 91.9 t/s · ↑  93.3 t/s ·  25.01 GiB ·    38/256k ·  54°C")

        // Idle speed hiding keeps non-speed metrics and resumes speeds on decode/prefill.
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: loadedDashboard, isConnected: true, enabledMetrics: Set(QWMenuBarMetric.allCases),
            hideSpeedsWhenIdle: true
        ), " 25.01 GiB ·    38/256k ·  54°C")

        let activeDecode = QWDashboardViewModel(
            sampledAt: Date(), lifecycle: loadedDashboard.lifecycle, modelId: "m",
            capabilities: [], generation: .decode, facts: nil, cards: loadedDashboard.cards,
            actions: [], knobs: loadedDashboard.knobs, knobSources: [:]
        )
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: activeDecode, isConnected: true, enabledMetrics: Set(QWMenuBarMetric.allCases),
            hideSpeedsWhenIdle: true
        ), "↓ 91.9 t/s · ↑  93.3 t/s ·  25.01 GiB ·    38/256k ·  54°C")

        // Cached prefill + missing active_tokens + absent memory preserve slot widths & honesty.
        let unmeasured = QWDashboardViewModel(
            sampledAt: Date(), lifecycle: loadedDashboard.lifecycle, modelId: "m",
            capabilities: [], generation: .idle, facts: nil,
            cards: [
                makeCard("throughput.display_prefill_tokens_per_sec", value: "from cache", unit: nil),
                makeCard("memory.arena_reserved_bytes", value: nil, unit: "GiB"),
                makeCard("memory.resident_bytes", value: nil, unit: "GiB"),
                makeCard("context.active_tokens", value: nil, unit: "tok"),
            ],
            actions: [], knobs: ["contextTokens": .int(262_144)], knobSources: [:]
        )
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: unmeasured, isConnected: true, enabledMetrics: [.prefillSpeed], hideSpeedsWhenIdle: false
        ), "↑  from cache")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: unmeasured, isConnected: true, enabledMetrics: [.engineMemory], hideSpeedsWhenIdle: false
        ), "     — GiB")
        XCTAssertEqual(QWMenuBarStatusFormatter.statusTitle(
            dashboard: unmeasured, isConnected: true, enabledMetrics: [.contextSize], hideSpeedsWhenIdle: false
        ), "    —/256k")
        XCTAssertEqual(QWMenuBarStatusFormatter.samplePreviewTitle(
            enabledMetrics: [.decodeSpeed], hideSpeedsWhenIdle: true
        ), " 91.9 tok/s (active)")
        XCTAssertEqual(QWMenuBarStatusFormatter.samplePreviewTitle(
            enabledMetrics: [.prefillSpeed], hideSpeedsWhenIdle: true
        ), "↑  93.3 tok/s (active)")
        XCTAssertEqual(QWMenuBarStatusFormatter.samplePreviewTitle(
            enabledMetrics: [.decodeSpeed, .prefillSpeed], hideSpeedsWhenIdle: true
        ), "↓ 91.9 t/s · ↑  93.3 t/s (active)")
    }

    func testMenuBarStatusFormatterKeepsConstantWidthAcrossDigitRegimes() throws {
        func makeCard(_ id: String, value: String?, unit: String?, number: Double? = nil) -> QWDashboardCard {
            QWDashboardCard(
                id: id, title: id, value: value, unit: unit, number: number,
                provenance: value == nil ? .unavailable : .measured,
                sampledAt: value == nil ? nil : Date(),
                age: value == nil ? nil : 0.1,
                isStale: false, reason: value == nil ? "not measured" : nil, series: []
            )
        }
        func makeDashboard(
            decode: Double?,
            prefill: Double?,
            prefillText: String? = nil,
            arena: Double?,
            ctx: Double?,
            temp: Double?
        ) -> QWDashboardViewModel {
            let prefillVal = prefillText ?? prefill.map { String(format: "%.1f", $0) }
            let prefillUnit: String? = prefillText == nil ? "tok/s" : nil
            return QWDashboardViewModel(
                sampledAt: Date(),
                lifecycle: .init(phase: .loaded, since: Date(), model: "m", detail: nil),
                modelId: "m",
                capabilities: [],
                generation: .decode,
                facts: nil,
                cards: [
                    makeCard("throughput.display_decode_tokens_per_sec", value: decode.map { String(format: "%.1f", $0) }, unit: "tok/s", number: decode),
                    makeCard("throughput.display_prefill_tokens_per_sec", value: prefillVal, unit: prefillUnit, number: prefill),
                    makeCard("memory.arena_reserved_bytes", value: arena.map { String(format: "%.2f", $0) }, unit: "GiB", number: arena),
                    makeCard("context.active_tokens", value: ctx.map { String(Int($0)) }, unit: "tok", number: ctx),
                    makeCard("hardware.gpu_temp_celsius", value: temp.map { String(format: "%.1f", $0) }, unit: "°C", number: temp),
                ],
                actions: [],
                knobs: ["contextTokens": .int(262_144)],
                knobSources: [:]
            )
        }
        func occurrences(of needle: String, in haystack: String) -> [Int] {
            var indices: [Int] = []
            var start = haystack.startIndex
            while let range = haystack.range(of: needle, range: start..<haystack.endIndex) {
                indices.append(haystack.distance(from: haystack.startIndex, to: range.lowerBound))
                start = range.upperBound
            }
            return indices
        }

        let regimes: [QWDashboardViewModel] = [
            makeDashboard(decode: 9.5, prefill: 9.5, arena: 8.40, ctx: 38, temp: 48.2),
            makeDashboard(decode: 91.9, prefill: 93.3, arena: 25.01, ctx: 512, temp: 54.2),
            makeDashboard(decode: 248.3, prefill: 2450.0, arena: 108.42, ctx: 4096, temp: 102.0),
            makeDashboard(decode: 1240.0, prefill: 12500.0, arena: 1024.0, ctx: 128_000, temp: 115.0),
            makeDashboard(decode: 84.0, prefill: nil, prefillText: "from cache", arena: 25.01, ctx: 10_240, temp: 64.0),
            makeDashboard(decode: nil, prefill: nil, arena: nil, ctx: nil, temp: nil),
        ]

        let metricSets: [Set<QWMenuBarMetric>] = [
            [.decodeSpeed],
            [.prefillSpeed],
            [.decodeSpeed, .prefillSpeed],
            [.engineMemory],
            [.contextSize],
            [.gpuTemperature],
            Set(QWMenuBarMetric.allCases),
        ]
        for metrics in metricSets {
            let titles = try regimes.map {
                try XCTUnwrap(QWMenuBarStatusFormatter.statusTitle(
                    dashboard: $0, isConnected: true, enabledMetrics: metrics, hideSpeedsWhenIdle: false
                ))
            }
            let expectedLength = titles[0].count
            for title in titles {
                XCTAssertEqual(title.count, expectedLength, "Width drift in '\(title)' for \(metrics)")
            }
        }

        let allMetrics = Set(QWMenuBarMetric.allCases)
        let fullTitles = try regimes.map {
            try XCTUnwrap(QWMenuBarStatusFormatter.statusTitle(
                dashboard: $0, isConnected: true, enabledMetrics: allMetrics, hideSpeedsWhenIdle: false
            ))
        }
        for landmark in ["↓", " t/s · ↑", " · ", "GiB", "/256k", "°C"] {
            let expectedOffsets = occurrences(of: landmark, in: fullTitles[0])
            XCTAssertFalse(expectedOffsets.isEmpty, "Missing landmark '\(landmark)' in '\(fullTitles[0])'")
            for title in fullTitles.dropFirst() {
                XCTAssertEqual(
                    occurrences(of: landmark, in: title),
                    expectedOffsets,
                    "Landmark '\(landmark)' shifted in '\(title)' vs '\(fullTitles[0])'"
                )
            }
        }
    }

    func testMenuBarRendersPrefillProgressCards() throws {
        XCTAssertTrue(Self.menuBarCardCatalogueIds.contains("prefill.progress_tokens"))
        XCTAssertTrue(Self.menuBarCardCatalogueIds.contains("prefill.total_tokens"))
        let cardsSource = try Self.read("\(menuBarDirectory)/TelemetryCards.swift")
        XCTAssertTrue(cardsSource.contains("card(\"prefill.progress_tokens\")"))
        XCTAssertTrue(cardsSource.contains("card(\"prefill.total_tokens\")"))
        XCTAssertTrue(cardsSource.contains("PREFILL PROGRESS"))
        XCTAssertTrue(cardsSource.contains("prefillProgressBarRow("))
        XCTAssertFalse(cardsSource.contains("if showPrefillProgress"))
        let primitivesSource = try Self.read("\(menuBarDirectory)/TelemetryCardPrimitives.swift")
        XCTAssertTrue(primitivesSource.contains("QWTokenCountFormatter.format("))
    }

    func testDualSparklineGeometryHandlesSingleTurnGapsAndWideDynamicRange() throws {
        // Single completed turn: isolated points and guides for both prefill and decode,
        // normalized independently so decode is not crushed near the bottom axis.
        let single = QWSparklineGeometry.layout(
            prefill: [1756.2], decode: [54.0], width: 388, height: 50
        )
        XCTAssertEqual(single.totalSlots, 1)
        XCTAssertEqual(single.prefill.isolatedPoints.count, 1)
        XCTAssertEqual(single.decode.isolatedPoints.count, 1)
        let prefillGuide = try XCTUnwrap(single.prefill.singleSlotGuideY)
        let decodeGuide = try XCTUnwrap(single.decode.singleSlotGuideY)
        XCTAssertLessThan(prefillGuide, decodeGuide - 6.0)
        XCTAssertLessThan(decodeGuide, 30.0)

        // In-flight Turn 2: slot 1 nil prefill remains isolated without bridging,
        // while decode forms a 2-point rising run.
        let inflight = QWSparklineGeometry.layout(
            prefill: [1756.2, nil], decode: [54.0, 71.7], width: 388, height: 50
        )
        XCTAssertTrue(inflight.prefill.runs.isEmpty)
        XCTAssertEqual(inflight.prefill.isolatedPoints.map(\.index), [0])
        XCTAssertEqual(inflight.decode.runs.count, 1)
        XCTAssertEqual(inflight.decode.runs[0].count, 2)
        XCTAssertLessThan(inflight.decode.runs[0][1].y, inflight.decode.runs[0][0].y)

        // Gapped alternating turns keep isolated points unbridged and omit single-slot guides.
        let gapped = QWSparklineGeometry.layout(
            prefill: [1756.2, nil, 1810.0], decode: [], width: 388, height: 50
        )
        XCTAssertTrue(gapped.prefill.runs.isEmpty)
        XCTAssertEqual(gapped.prefill.isolatedPoints.count, 2)
        XCTAssertNil(gapped.prefill.singleSlotGuideY)

        // All-nil series remain empty rather than inventing zeros.
        let empty = QWSparklineGeometry.layout(
            prefill: [nil, nil], decode: [nil, nil], width: 388, height: 50
        )
        XCTAssertTrue(empty.prefill.runs.isEmpty)
        XCTAssertTrue(empty.prefill.isolatedPoints.isEmpty)
        XCTAssertNil(empty.prefill.latestPoint)
        XCTAssertNil(empty.prefill.singleSlotGuideY)
        XCTAssertTrue(empty.decode.runs.isEmpty)
        XCTAssertTrue(empty.decode.isolatedPoints.isEmpty)
        XCTAssertNil(empty.decode.latestPoint)
        XCTAssertNil(empty.decode.singleSlotGuideY)
    }

    func testDualSparklineBridgesActiveCompanionPhaseSlotsWhileSplittingTrueGaps() throws {
        // Multi-turn 1Hz trace: sub-second prefills appear as single non-nil slots
        // separated by active 1Hz decode samples. Neither phase fragments into dots.
        let multiTurn = QWSparklineGeometry.layout(
            prefill: [1420.0, nil, nil, nil, 1580.0, nil, nil, 1510.0, nil],
            decode: [80.0, 81.2, 82.0, 80.5, nil, 83.1, 81.8, 80.9, 82.4],
            width: 388,
            height: 50
        )
        XCTAssertEqual(multiTurn.prefill.runs.count, 1)
        XCTAssertEqual(multiTurn.prefill.runs[0].map(\.index), [0, 4, 7])
        XCTAssertTrue(multiTurn.prefill.isolatedPoints.isEmpty)
        XCTAssertNil(multiTurn.prefill.singleSlotGuideY)
        XCTAssertEqual(multiTurn.decode.runs.count, 1)
        XCTAssertEqual(multiTurn.decode.runs[0].map(\.index), [0, 1, 2, 3, 5, 6, 7, 8])
        XCTAssertTrue(multiTurn.decode.isolatedPoints.isEmpty)

        // True unmeasured gap across both series still splits into isolated points.
        let dualGap = QWSparklineGeometry.layout(
            prefill: [1420.0, nil, 1580.0],
            decode: [80.0, nil, 82.0],
            width: 388,
            height: 50
        )
        XCTAssertTrue(dualGap.prefill.runs.isEmpty)
        XCTAssertEqual(dualGap.prefill.isolatedPoints.map(\.index), [0, 2])
        XCTAssertTrue(dualGap.decode.runs.isEmpty)
        XCTAssertEqual(dualGap.decode.isolatedPoints.map(\.index), [0, 2])
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

    func testSMCAbiLayoutAndCoolingPolicyCoherence() throws {
        XCTAssertEqual(MemoryLayout<QWSMCClient.SMCKeyData>.stride, 80, "AppleSMC ABI struct must remain 80 bytes")
        let fltBytes: QWSMCClient.SMCBytes32 = (
            0x00, 0x00, 0x58, 0x42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        let decodedTemp = QWSMCClient.decode(bytes: fltBytes, size: 4, type: QWSMCClient.fourCC("flt "))
        XCTAssertEqual(decodedTemp ?? 0, 54.0, accuracy: 0.001)

        XCTAssertEqual(QWSettingsCoherence.scope(of: "coolingPolicy"), .live)
        XCTAssertFalse(QWSettingsCoherence.needsServerRestart("coolingPolicy"))

        let settings = QWServerSettings(coolingPolicy: "proactive")
        let encoded = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(QWServerSettings.self, from: encoded)
        XCTAssertEqual(decoded.coolingPolicy, "proactive")

        let legacyJSON = Data("{}".utf8)
        let legacyDecoded = try JSONDecoder().decode(QWServerSettings.self, from: legacyJSON)
        XCTAssertEqual(legacyDecoded.coolingPolicy, "auto")

        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true" {
            let smc = QWFanController.shared.sampleAndReconcile()
            XCTAssertNotNil(smc.gpuTemperatureCelsius, "Expected live AppleSMC GPU temperature on bare-metal Apple Silicon")
            XCTAssertGreaterThan(smc.gpuTemperatureCelsius ?? 0, 15.0)
            XCTAssertNotNil(smc.fanSpeedRPM, "Expected live AppleSMC fan telemetry on bare-metal Apple Silicon")
            XCTAssertNotNil(smc.systemPowerWatts, "Expected live AppleSMC PSTR power telemetry on bare-metal Apple Silicon")
        }
    }
}

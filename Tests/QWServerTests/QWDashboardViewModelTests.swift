// QWDashboardViewModelTests.swift — The card list, computed from what
// was measured. No test here reaches for a real model: the cards are
// built from the telemetry registries and the knob registry alone.

import XCTest
@testable import QWServer
import QWCore
import QWEngine
import Foundation

final class QWDashboardViewModelTests: XCTestCase {
    override func tearDown() {
        QWTelemetryRegistry.shared.reset()
        // The server's registry is a singleton: a rate measured by one
        // test must not be the reading another one sees.
        QWEngineTelemetryRegistry.shared.resetToIdle()
        super.tearDown()
    }

    private func lifecycle(phase: QWEngineLifecyclePhase = .loaded) -> QWEngineLifecycleInfo {
        QWEngineLifecycleInfo(phase: phase, since: Date(), model: "Qwen3.8-27B-hq4g32", detail: nil)
    }

    func testUnavailableMetricBecomesUnavailableCardNotZero() throws {
        QWTelemetryRegistry.shared.recordUnavailable("gpu.device",
                                                     reason: "IOAccelerator unavailable on this host")
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "gpu.device" })
        XCTAssertNil(card.value, "an unavailable metric has no value, not a zero")
        XCTAssertEqual(card.provenance, .unavailable)
        XCTAssertNotNil(card.reason)
    }

    func testAMetricNobodyMeasuredIsUnavailableWithAReason() throws {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "latency.ttft_ms" })
        XCTAssertNil(card.value)
        XCTAssertEqual(card.provenance, .unavailable)
        XCTAssertNotNil(card.reason)
    }

    func testModelWithoutPrefixCacheHasNoCacheCard() {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "fn", capabilities: [])
        XCTAssertFalse(model.cards.contains { $0.id.hasPrefix("cache.") })
        XCTAssertFalse(model.actions.contains { $0.id == "enablePrefixCache" })
    }

    func testModelWithPrefixCacheHasTheCacheCardAndAction() {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "dense",
                                               capabilities: [.prefixCache])
        XCTAssertTrue(model.cards.contains { $0.id.hasPrefix("cache.") })
        XCTAssertTrue(model.actions.contains { $0.id == "enablePrefixCache" })
    }

    func testTheCardsAreTheSameForEveryRenderer() {
        let first = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "dense",
                                               capabilities: [.prefixCache])
        let second = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "dense",
                                                capabilities: [.prefixCache])
        XCTAssertEqual(first.cards.map(\.id), second.cards.map(\.id))
        XCTAssertEqual(first.schemaVersion, QWDashboardViewModel.currentSchemaVersion)
    }

    /// The model is either loaded or it is not. Publishing both a load
    /// and an unload puts a button on the screen that cannot be
    /// pressed, and leaves the reader to work out which one is real.
    func testTheModelActionIsNeverBothALoadAndAnUnload() {
        for phase in [QWEngineLifecyclePhase.unloaded, .loading, .loaded, .failed] {
            let model = QWDashboardViewModel.build(lifecycle: lifecycle(phase: phase), modelId: "m")
            XCTAssertFalse(model.actions.contains { $0.id == "load" }
                && model.actions.contains { $0.id == "unload" },
                "the phase \(phase) published a load and an unload")
        }
    }

    func testTheActionsFollowThePhase() {        let loaded = QWDashboardViewModel.build(lifecycle: lifecycle(phase: .loaded), modelId: "m")
        XCTAssertTrue(loaded.actions.first { $0.id == "unload" }?.isEnabled ?? false)
        // A loaded model offers Unload, not a greyed-out Load: one
        // button that can be pressed, not two where one cannot.
        XCTAssertFalse(loaded.actions.contains { $0.id == "load" })

        let unloaded = QWDashboardViewModel.build(lifecycle: lifecycle(phase: .unloaded), modelId: "m")
        XCTAssertTrue(unloaded.actions.first { $0.id == "load" }?.isEnabled ?? false)
        XCTAssertFalse(unloaded.actions.contains { $0.id == "unload" })

        let failed = QWDashboardViewModel.build(lifecycle: lifecycle(phase: .failed), modelId: "m")
        XCTAssertTrue(failed.actions.first { $0.id == "restart" }?.isEnabled ?? false)
    }

    func testTheKnobSourcesAreAClosedSet() {
        let sources = QWKnobSource.sources(from: [
            "temperature": "cli (--temperature 0.7)",
            "maxContext": "config-file (~/.qwg/settings.json)",
            "kvMode": "env (QWG_KV_MODE)",
            "modelName": "none (using defaults)",
            "pack": "discovered (~/.qwg/models/Qwen3.8-27B-hq4g32)"
        ], for: ["temperature", "contextTokens", "kvMode", "seed"])
        XCTAssertEqual(sources["temperature"], QWKnobSource.explicit.rawValue)
        XCTAssertEqual(sources["contextTokens"], QWKnobSource.profile.rawValue)
        XCTAssertEqual(sources["kvMode"], QWKnobSource.deprecatedEnv.rawValue)
        XCTAssertEqual(sources["seed"], QWKnobSource.default.rawValue)
        for value in sources.values {
            XCTAssertTrue(QWKnobSource(rawValue: value) != nil, "\(value) is not a known source")
        }
    }

    func testACardWithNoValueAlwaysHasAReason() {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        for card in model.cards where card.value == nil {
            XCTAssertNotNil(card.reason, "\(card.id) has no value and no reason")
        }
    }

    func testTheRateCardsSayWhichRateTheyShow() throws {
        // The server measured a rate: the card must say which kind it is.
        QWEngineTelemetryRegistry.shared.recordGeneration(
            tokensIn: 100, tokensOut: 50, prefillSec: 0.5, decodeSec: 1.0, ttftSec: nil)
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_decode_tokens_per_sec" })
        // The label says which rate this is: live, average or last. A
        // number without that label is a number nobody can place.
        XCTAssertNotNil(card.value)
        XCTAssertTrue(card.title.contains("("), "the rate card must name its rate")
        XCTAssertEqual(card.unit, "tok/s")
    }

    func testAnUnmeasuredRateCardClaimsNoRate() throws {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_decode_tokens_per_sec" })
        // There is no last rate: the card names no rate it does not have.
        XCTAssertNil(card.value)
        XCTAssertFalse(card.title.contains("("))
    }

    /// The panel must not show a dash for a turn that happened: a
    /// prompt the cache served entirely had no prefill phase, so the
    /// card says where the prompt came from — with no unit.
    func testThePrefillCardSaysWhenThePromptCameFromTheCache() throws {
        let registry = QWEngineTelemetryRegistry.shared
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_prefill_tokens_per_sec" })
        XCTAssertEqual(card.value, "from cache")
        XCTAssertNil(card.unit, "a word is not a rate: it carries no unit")
        XCTAssertNil(card.number)
        XCTAssertEqual(card.provenance, .derived)
        XCTAssertNotNil(card.reason)
    }

    /// A turn that computed a prefill still shows the rate: the state
    /// stands in for the rate only when there is no rate to show.
    func testThePrefillCardShowsTheRateWhenOneWasMeasured() throws {
        let registry = QWEngineTelemetryRegistry.shared
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_prefill_tokens_per_sec" })
        XCTAssertNotNil(card.value)
        XCTAssertEqual(card.unit, "tok/s")
    }

    /// The tile describes the most recent turn: a turn the cache served
    /// entirely shows where the prompt came from, not the speed an older
    /// turn reached.
    func testThePrefillCardDescribesTheLastTurnNotAnOlderOne() throws {
        let registry = QWEngineTelemetryRegistry.shared
        // A cold turn: the engine measured a prefill rate.
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        // The next turn's prompt came from the cache: no prefill ran.
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_prefill_tokens_per_sec" })
        XCTAssertEqual(card.value, "from cache")
        XCTAssertNil(card.number, "the older turn's rate is not the last turn's rate")
    }

    func testACacheTheEngineCannotMeasureIsUnavailableNotZero() throws {
        let registry = QWEngineTelemetryRegistry.shared
        // A backend that cannot report its cache says nothing about it.
        registry.updateCacheUsage(warmBytes: nil, warmLimit: 1 << 30,
                                  ssdBytes: nil, ssdLimit: 1 << 30)
        defer {
            // Restore it as unmeasured: a measured zero would tell the
            // next test that the cache was measured.
            registry.updateCacheUsage(warmBytes: nil, warmLimit: 1 << 30,
                                      ssdBytes: nil, ssdLimit: 1 << 30)
        }
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "fn",
                                               capabilities: [.prefixCache])
        let card = try XCTUnwrap(model.cards.first { $0.id == "cache.warm_bytes" })
        XCTAssertNil(card.value, "a cache nobody measured is not an empty cache")
        XCTAssertEqual(card.provenance, .unavailable)
        XCTAssertNotNil(card.reason)
    }

    /// The trace behind a rate card is the server's, and it survives the
    /// merge with the engine's own reading of the same number.
    func testTheRateCardCarriesThePublishedTrace() throws {
    let registry = QWEngineTelemetryRegistry.shared
    registry.resetToIdle()
    registry.recordGeneration(tokensIn: 100, tokensOut: 50,
                              prefillSec: 0.5, decodeSec: 1.0, ttftSec: nil)
    // The engine's own registry reads that rate too. Its number wins;
    // the trace behind it stays.
    QWTelemetryRegistry.shared.record("throughput.display_decode_tokens_per_sec", 49.0)
    defer {
        registry.resetToIdle()
        QWTelemetryRegistry.shared.reset()
    }
    let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
    let card = try XCTUnwrap(model.cards.first { $0.id == "throughput.display_decode_tokens_per_sec" })
    XCTAssertFalse(card.series.isEmpty, "the card carries the trace the sparkline draws")
}

    /// The GPU card shows a percentage. The engine measures a ratio, so
    /// the card must scale it — a ratio shown as a percentage must not be
    /// a hundredth of one.
    func testTheGPUCardShowsTheMeasuredRatioAsAPercentage() throws {
    QWTelemetryRegistry.shared.record("gpu.busy_ratio", 0.42, provenance: .measured)
    defer { QWTelemetryRegistry.shared.reset() }
    let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
    let card = try XCTUnwrap(model.cards.first { $0.id == "gpu.busy_ratio" })
    XCTAssertEqual(card.unit, "%")
    XCTAssertEqual(card.value, "42.0")
    XCTAssertEqual(card.number ?? 0, 42.0, accuracy: 0.001)
}

    /// A payload from before the facts and the generation phase existed
    /// still decodes: the new fields are additive, and an older server's
    /// dashboard is still read.
    func testAPayloadWithoutTheFactsStillDecodes() throws {
    let json = """
    {"schemaVersion":1,"sampledAt":"2026-10-25T00:00:00Z",
     "lifecycle":{"phase":"loaded","since":"2026-10-25T00:00:00Z","model":"m"},
     "modelId":"m","capabilities":[],"cards":[],"actions":[],
     "knobs":{},"knobSources":{}}
    """
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let model = try decoder.decode(QWDashboardViewModel.self, from: Data(json.utf8))
    XCTAssertNil(model.facts)
    XCTAssertNil(model.generation)
}

    func testTheViewModelSurvivesTheWire() throws {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "dense",
                                               capabilities: [.prefixCache],
                                               knobs: ["temperature": .double(0.7)],
                                               knobSources: ["temperature": "explicit"])
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let data = try encoder.encode(model)
        let decoded = try decoder.decode(QWDashboardViewModel.self, from: data)
        XCTAssertEqual(decoded.cards.count, model.cards.count)
        XCTAssertEqual(decoded.modelId, "dense")
        XCTAssertEqual(decoded.knobs["temperature"], .double(0.7))
        XCTAssertEqual(decoded.knobSources["temperature"], "explicit")
        XCTAssertEqual(decoded.lifecycle.phase, .loaded)
    }

    func testTheContextIsAbsentUntilATurnHasRun() throws {
        // The loader asserts an empty context; it does not measure one.
        // A card that says "0 tok" would be a zero for something nobody
        // looked at.
        let registry = QWEngineTelemetryRegistry.shared
        registry.updateContext(active: nil, capacity: 4096)
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "m")
        let card = try XCTUnwrap(model.cards.first { $0.id == "context.active_tokens" })
        XCTAssertNil(card.value)
        XCTAssertEqual(card.provenance, .unavailable)
        XCTAssertNotNil(card.reason)
    }

    func testTheModelIdIsNilWhenNoModelIsResident() {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(phase: .unloaded), modelId: nil)
        XCTAssertNil(model.modelId)
    }

    func testTheCapabilitiesArePublished() {
        let model = QWDashboardViewModel.build(lifecycle: lifecycle(), modelId: "dense",
                                               capabilities: [.prefixCache, .speculative])
        XCTAssertEqual(model.capabilities, ["prefixCache", "speculative"])
    }
}

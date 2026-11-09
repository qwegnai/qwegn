// QWTelemetryHonestyTests.swift — The telemetry must be what it says it is.
//
// A reading is measured, or it is unavailable. It is never a number the
// engine made up, and never the previous turn's number wearing today's
// label.

import XCTest
@testable import QWServer
import QWCore
import QWEngine

final class QWTelemetryHonestyTests: XCTestCase {
    /// The repository root, from this file's path.
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // QWTelemetryHonestyTests.swift
            .deletingLastPathComponent()   // QWServerTests
            .deletingLastPathComponent()   // Tests
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOf: repoRoot.appendingPathComponent(path), encoding: .utf8)
    }

    func testNoSyntheticGPUPercentageIsEverProduced() throws {
        let sampler = try source("Sources/Apps/QWServer/Telemetry/QWSystemMetricsSampler.swift")
        XCTAssertFalse(sampler.contains("40.0 + (intensity"),
                       "the fabricated GPU percentage must be deleted, not relabelled")
        XCTAssertFalse(sampler.contains("lastGpuPercent"),
                       "the carried-forward GPU value must be deleted")
        XCTAssertFalse(sampler.contains("tokensPerSecond"),
                       "the engine's speed is not a hardware reading")
        XCTAssertFalse(sampler.contains("max(65.0, usagePct)"),
                       "the fabricated memory-pressure percentage must be deleted")
        XCTAssertFalse(sampler.contains("min(usagePct * 0.65, 45.0)"),
                       "the fabricated memory-pressure percentage must be deleted")
        XCTAssertFalse(sampler.contains("return lastCpuPercent"),
                       "the CPU load must not be carried forward as if it were current")
    }

    func testUnavailableHostMetricIsNotZero() {
        let value = QWTelemetryValue.unavailable("IOAccelerator unavailable on this host")
        XCTAssertNil(value.value)
        XCTAssertEqual(value.provenance, .unavailable)
    }

    func testEndRequestCarriesCachedTokens() throws {
        let actor = try source("Sources/Apps/QWServer/Core/QWEngineLifecycleActor.swift")
            + "\n" + (try source("Sources/Apps/QWServer/Core/QWEngineLifecycleActor+Chat.swift"))
        XCTAssertTrue(actor.contains("cachedTokens: metrics.cacheHitTokens"),
                      "endRequest must receive the cache-served token count")
    }

    func testPrefillDurationIsNotFallingBackToTTFT() throws {
        let actor = try source("Sources/Apps/QWServer/Core/QWEngineLifecycleActor.swift")
        XCTAssertFalse(actor.contains("metrics.prefillSec > 0 ? metrics.prefillSec : (ttftSec ?? 0.0)"),
                       "a fully-cached turn has no prefill phase; TTFT must not stand in for it")
        XCTAssertFalse(actor.contains("totalDuration - (ttftSec ?? 0.0)"),
                       "the server must not re-derive the engine's decode duration")
    }

    func testAFullyCachedTurnHasNoPrefillRate() {
        // Every token came from the prefix cache: there was no prefill
        // phase, so the rate is undefined — not the previous turn's rate.
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        let snapshot = registry.snapshot()
        let metric = QWTelemetryMetric.lastPrefillTokensPerSec.rawValue
        XCTAssertEqual(snapshot.provenance[metric], .unavailable)
        XCTAssertNil(snapshot.throughput.lastPrefillTokensPerSec)
        XCTAssertNil(snapshot.throughput.displayPrefillTokensPerSec,
                     "a rate that was never measured has no number to show")
        let display = QWTelemetryMetric.displayPrefillTokensPerSec.rawValue
        XCTAssertEqual(snapshot.provenance[display], .unavailable)
        XCTAssertFalse(snapshot.unavailableReasons[display]?.isEmpty ?? true,
                       "an unavailable metric must say why")
    }

    /// A turn the cache served entirely had no prefill phase: the
    /// server says where the prompt came from, so the card is never
    /// blank for a turn that really happened.
    func testAFullyCachedTurnPublishesWhereThePromptCameFrom() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        let snapshot = registry.snapshot()
        XCTAssertTrue(snapshot.throughput.prefillFromCache)
        let metric = QWTelemetryMetric.prefillFromCache.rawValue
        XCTAssertEqual(snapshot.provenance[metric], .derived)
        XCTAssertNotNil(snapshot.sampledAt[metric])
        XCTAssertFalse(snapshot.unavailableReasons[metric]?.isEmpty ?? true,
                       "a state that stands in for a number must say what it means")
    }

    /// A turn that computed a prefill has a rate to show: the card
    /// must not claim the prompt came from the cache.
    func testATurnThatComputedAPrefillIsNotACacheTurn() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        XCTAssertFalse(registry.snapshot().throughput.prefillFromCache)
    }

    /// A prompt the cache partly served still had a prefill phase: the
    /// rate is the computed tokens over the engine's own duration.
    func testAPartlyCachedTurnStillHasAPrefillRate() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 60,
                            prefillSec: 0.2, decodeSec: 1.0, ttftSec: 0.3)
        let snapshot = registry.snapshot()
        XCTAssertFalse(snapshot.throughput.prefillFromCache)
        XCTAssertEqual(snapshot.throughput.lastPrefillTokensPerSec ?? 0, 200, accuracy: 0.001)
    }

    /// A turn with no prompt at all computed nothing, but the cache did
    /// not serve it either: there was no prompt to come from the cache.
    func testATurnWithNoPromptIsNotACacheTurn() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 0)
        registry.beginDecode(promptTokens: 0)
        registry.endRequest(tokensIn: 0, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        XCTAssertFalse(registry.snapshot().throughput.prefillFromCache,
                       "an empty prompt was not served from the cache")
    }

    /// The state belongs to the model that ran the turn: a reset clears
    /// it, because the next model starts with its own.
    func testAResetClearsTheCacheTurnState() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        XCTAssertTrue(registry.snapshot().throughput.prefillFromCache)
        registry.resetToIdle()
        XCTAssertFalse(registry.snapshot().throughput.prefillFromCache)
    }

    /// A turn that measured no rate is a gap in the trace. Holding the
    /// last known value would draw a flat line where nothing happened.
    func testATurnWithNoRateIsAGapInTheTrace() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        // The next turn was served entirely by the cache: no prefill ran.
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 100,
                            prefillSec: 0, decodeSec: 1.0, ttftSec: 0.2)
        let snapshot = registry.snapshot()
        XCTAssertEqual(snapshot.throughput.prefillRateSeries.count, 2)
        XCTAssertNotNil(snapshot.throughput.prefillRateSeries[0])
        XCTAssertNil(snapshot.throughput.prefillRateSeries[1],
                     "a turn with no measured rate is a gap, never the last value")
    }

    func testThePrefillRateIsMeasuredFromTheEnginesOwnDuration() {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        let snapshot = registry.snapshot()
        let metric = QWTelemetryMetric.lastPrefillTokensPerSec.rawValue
        XCTAssertEqual(snapshot.provenance[metric], .derived)
        XCTAssertEqual(snapshot.throughput.lastPrefillTokensPerSec ?? 0, 200, accuracy: 0.001)
        XCTAssertNotNil(snapshot.sampledAt[metric])
        XCTAssertEqual(snapshot.throughput.displayPrefillTokensPerSec ?? 0, 200, accuracy: 0.001)
        XCTAssertEqual(snapshot.throughput.prefillDisplayMode, .average)
    }

    /// The live rate is the median inter-token latency: 1000 / p50(ms).
    /// The intervals are deliberately uneven, so a running average cannot be
    /// mistaken for the median.
    func testTheDecodeRateComesFromTheInterTokenLatencies() throws {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 8)
        registry.beginDecode(promptTokens: 8)
        var intervals: [Double] = []
        var last = Date()
        for wait in [0.01, 0.01, 0.01, 0.15, 0.15] {
            Thread.sleep(forTimeInterval: wait)
            let now = Date()
            intervals.append(now.timeIntervalSince(last) * 1000.0)
            last = now
            registry.recordDecodeToken()
        }
        let sorted = intervals.sorted()
        let expected = 1000.0 / sorted[sorted.count / 2]
        let runningAverage = 1000.0 * Double(intervals.count) / intervals.reduce(0, +)

        let snapshot = registry.snapshot()
        let metric = QWTelemetryMetric.liveDecodeTokensPerSec.rawValue
        XCTAssertEqual(snapshot.provenance[metric], .derived)
        let rate = try XCTUnwrap(snapshot.throughput.liveDecodeTokensPerSec)
        XCTAssertEqual(rate, expected, accuracy: expected * 0.2,
                       "the rate must be 1000 / p50 of the measured inter-token latencies")
        XCTAssertGreaterThan(rate - runningAverage, runningAverage * 0.2,
                             "the intervals are uneven: a running average must not pass for the median")
    }

    /// A rate measured long ago is stale: it is not the engine's speed now.
    /// The turn-average never goes stale, so the test looks at a turn in
    /// flight, where the display falls back to the last measurement.
    func testAnOldRateIsMarkedStale() {
        let registry = QWEngineTelemetryRegistry(rateStaleAfter: 0.0)
        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        XCTAssertEqual(registry.snapshot().throughput.prefillDisplayMode, .average)

        registry.beginPrefill(promptTokens: 100)
        let snapshot = registry.snapshot()
        XCTAssertEqual(snapshot.throughput.prefillDisplayMode, .stale)
        let display = QWTelemetryMetric.displayPrefillTokensPerSec.rawValue
        XCTAssertEqual(snapshot.provenance[display], .stale)
        XCTAssertNotNil(snapshot.unavailableReasons[display])
    }

    /// A series the server did not measure is absent from the exposition.
    /// A zero would say "idle", which is a claim the server cannot make.
    func testPrometheusOmitsWhatWasNotMeasured() {
        let registry = QWEngineTelemetryRegistry()
        let empty = registry.formatPrometheus()
        XCTAssertFalse(empty.contains("qwegine_speed_tokens_per_second"),
                       "an unmeasured rate must have no series")
        XCTAssertFalse(empty.contains("qwegine_latency_ttft_ms"),
                       "an unmeasured latency must have no series")
        XCTAssertTrue(empty.contains("qwegine_tokens_total"),
                      "a counter that was measured keeps its series")

        registry.beginPrefill(promptTokens: 100)
        registry.beginDecode(promptTokens: 100)
        registry.endRequest(tokensIn: 100, tokensOut: 8, cachedTokens: 0,
                            prefillSec: 0.5, decodeSec: 1.0, ttftSec: 0.6)
        let measured = registry.formatPrometheus()
        XCTAssertTrue(measured.contains("qwegine_speed_tokens_per_second{phase=\"prefill\",window=\"last\"}"))
        XCTAssertTrue(measured.contains("qwegine_latency_ttft_ms{type=\"last\"}"))
    }

    /// The host's readings are optional, and an absent one says why.
    func testTheHostReadingsAreOptionalWithAReason() {
        let reading = QWSystemMetricsSampler.shared.current()
        if reading.cpuUsagePercent == nil {
            XCTAssertNotNil(reading.cpuUsageUnavailableReason)
        } else {
            XCTAssertGreaterThanOrEqual(reading.cpuUsagePercent ?? -1, 0)
            XCTAssertLessThanOrEqual(reading.cpuUsagePercent ?? 101, 100)
        }
        if reading.memoryPressureLevel == nil {
            XCTAssertNotNil(reading.memoryPressureUnavailableReason,
                            "a pressure level that was not read must say so, not say 'Normal'")
        }
    }

    func testTheHostReadingSaysWhenItWasTaken() {
        let reading = QWSystemMetricsSampler.shared.current()
        XCTAssertGreaterThan(reading.sampledAt.timeIntervalSince1970, 0,
                             "a reading without a timestamp cannot be judged for staleness")
        // The cached reading is served again, with the same timestamp: a
        // consumer can tell it is the reading it already had.
        let again = QWSystemMetricsSampler.shared.current()
        XCTAssertEqual(again.sampledAt, reading.sampledAt)
    }

    func testTheGPUReadingIsAbsentWhenTheHostDoesNotPublishIt() {
        // The host may not publish the GPU's utilization. That is a gap, and
        // the gap must not look like an idle GPU.
        let reading = QWSystemMetricsSampler.shared.current()
        if reading.gpuUsagePercent == nil {
            XCTAssertNotNil(reading.gpuUsageUnavailableReason)
        } else {
            XCTAssertGreaterThanOrEqual(reading.gpuUsagePercent ?? -1, 0)
            XCTAssertLessThanOrEqual(reading.gpuUsagePercent ?? 101, 100)
        }
    }

    /// The dashboard and the menu bar must not turn a missing reading into a
    /// zero: the law says a gap is a gap, in every surface.
    func testTheSurfacesDoNotCoerceMissingReadingsToZero() throws {
        let dashboard = try source("Sources/Apps/QWServer/Resources/controlbar.html")
        XCTAssertFalse(dashboard.contains("util.gpuUsagePercent || 0"),
                       "the dashboard must not turn a missing GPU reading into a zero")
        XCTAssertFalse(dashboard.contains("util.memoryUsedGB || 0"),
                       "the dashboard must not turn a missing memory reading into a zero")
        XCTAssertFalse(dashboard.contains("util.memoryPressurePercent || 0"),
                       "the dashboard must not invent a pressure index")
        XCTAssertFalse(dashboard.contains("util.cpuUsagePercent || 0"),
                       "the dashboard must not turn a missing CPU reading into a zero")
        XCTAssertFalse(dashboard.contains("tp.livePrefillTokensPerSec || 0.0"),
                       "the dashboard must not turn a missing rate into a zero")
        // The client keeps no history of its own: the server publishes
        // the trace, gaps and all. A client-side history would be a
        // second, disagreeing idea of what happened.
        XCTAssertFalse(dashboard.contains("new Array(60)"),
                       "the client must not keep a sparkline history of its own")
        XCTAssertTrue(dashboard.contains("val === null || val === undefined"),
                      "a sample that was not measured is a gap, never a zero")

        let menuBar = try source("Sources/Apps/QWMenuBar/TelemetryCards.swift")
        XCTAssertFalse(menuBar.contains("prefillRate > 0 ? String(format"),
                       "the menu bar must not turn a missing rate into a zero")
        XCTAssertFalse(menuBar.contains("decodeRate > 0 ? String(format"),
                       "the menu bar must not turn a missing rate into a zero")
    }

    /// The median of the inter-token latencies, without a copy of the
    /// window: the decode loop allocates nothing.
    func testTheMedianNeedsNoCopy() {
        var samples: [Double] = []
        samples.reserveCapacity(4)
        QWEngineTelemetryRegistry.insertSorted(3, into: &samples, capacity: 4)
        QWEngineTelemetryRegistry.insertSorted(1, into: &samples, capacity: 4)
        QWEngineTelemetryRegistry.insertSorted(2, into: &samples, capacity: 4)
        XCTAssertEqual(samples, [1, 2, 3])
        // The window is full: the next sample pushes the oldest out, and the
        // buffer never grows past its reserved capacity.
        QWEngineTelemetryRegistry.insertSorted(0, into: &samples, capacity: 4)
        XCTAssertEqual(samples, [0, 1, 2, 3])
        QWEngineTelemetryRegistry.insertSorted(9, into: &samples, capacity: 4)
        XCTAssertEqual(samples, [1, 2, 3, 9])
        XCTAssertEqual(samples.count, 4)
        XCTAssertLessThanOrEqual(samples.capacity, 4, "the buffer must not grow past its reserved capacity")
    }

    /// A menu bar may talk to a server that is a version behind. The
    /// snapshot it gets must still decode: a field that is not there was
    /// never measured, not zero.
    func testAnOlderServerPayloadStillDecodes() throws {
        let older = """
        {"cache":{"warmBytes":1,"warmLimitBytes":2,"ssdBytes":3,"ssdLimitBytes":4,"lookupsTotal":5,\
        "hitsWarmTotal":6,"hitsSSDTotal":7,"missesTotal":8,"tokensReusedTotal":9,\
        "evictionsWarmTotal":10,"evictionsSSDTotal":11},
        "throughput":{"phase":"decode","tokensInTotal":12,"tokensOutTotal":13},
        "latency":{},"context":{"activeTokens":14,"maxTokens":15},
        "memory":{"residentBytes":16,"arenaReservedBytes":17}}
        """
        let snapshot = try JSONDecoder().decode(
            QWEngineTelemetryRegistry.MetricsSnapshot.self, from: Data(older.utf8))
        XCTAssertEqual(snapshot.throughput.phase, "decode")
        XCTAssertNil(snapshot.throughput.liveDecodeTokensPerSec)
        XCTAssertNil(snapshot.throughput.displayDecodeTokensPerSec)
        XCTAssertEqual(snapshot.throughput.decodeDisplayMode, .last)
        XCTAssertTrue(snapshot.provenance.isEmpty)
        // A counter the old server does not have is absent, not zero.
        XCTAssertNil(snapshot.cache.hitsUnclassifiedTotal)
    }

    func testSpeculativeBurstAndChronologicalEvictionInLiveDecodeRate() throws {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 16)
        registry.beginDecode(promptTokens: 16)

        // 20 slow steps (25ms ITL) then 15 bursts (30ms round, 3 tokens 0.1ms
        // apart -> ~10ms/tok, 100 tok/s), evicting the window's old entries.
        // Injected timestamps, not sleeps: arithmetic, not scheduler luck.
        var t = Date()
        for _ in 0..<20 {
            t = t.addingTimeInterval(0.025)
            registry.recordDecodeToken(at: t)
        }
        for _ in 0..<15 {
            t = t.addingTimeInterval(0.030)
            registry.recordDecodeToken(at: t)
            registry.recordDecodeToken(at: t.addingTimeInterval(0.0001))
            registry.recordDecodeToken(at: t.addingTimeInterval(0.0002))
        }

        let rate = try XCTUnwrap(registry.snapshot().throughput.liveDecodeTokensPerSec)
        XCTAssertGreaterThan(rate, 70.0,
                             "speculative 3-token bursts of ~30ms must report ~90-100 tok/s, not round rate (~30 tok/s)")
    }
}

import XCTest
@testable import QWServer
@testable import QWEngine
import Foundation

final class QWServerMetricsTests: XCTestCase {
    /// A tier this build does not name is still a hit. Counting it as
    /// a miss would assert that the cache did not serve a prompt it
    /// served, and would drop the tokens it reused.
    func testAnUnnamedTierIsAHitAndNeverAMiss() {
        let registry = QWEngineTelemetryRegistry()
        registry.recordCacheLookup(hitTier: "cold_ssd", matchedTokens: 64)
        registry.recordCacheLookup(hitTier: "a_tier_this_build_does_not_name", matchedTokens: 32)
        registry.recordCacheLookup(hitTier: nil, matchedTokens: 0)
        let snap = registry.snapshot()
        XCTAssertEqual(snap.cache.lookupsTotal, 3)
        XCTAssertEqual(snap.cache.hitsSSDTotal, 1)
        XCTAssertEqual(snap.cache.hitsUnclassifiedTotal, 1)
        XCTAssertEqual(snap.cache.missesTotal, 1)
        // the reused tokens of both hits are counted
        XCTAssertEqual(snap.cache.tokensReusedTotal, 96)
        // the server knows its own counter, so it reports the tier
        XCTAssertTrue(registry.formatPrometheus()
            .contains("qwegine_cache_hits_total{tier=\"unclassified\"} 1"))
    }

    func testTelemetryRegistryAndPrometheusOutput() {
        let registry = QWEngineTelemetryRegistry()

        // 1. Initial State
        let snap0 = registry.snapshot()
        XCTAssertEqual(snap0.cache.lookupsTotal, 0)
        XCTAssertEqual(snap0.cache.hitsWarmTotal, 0)
        XCTAssertEqual(snap0.cache.hitsSSDTotal, 0)
        XCTAssertEqual(snap0.cache.missesTotal, 0)

        // 2. Record lookups and evictions
        registry.recordCacheLookup(hitTier: "warm_ram", matchedTokens: 32)
        registry.recordCacheLookup(hitTier: "cold_ssd", matchedTokens: 64)
        registry.recordCacheLookup(hitTier: nil, matchedTokens: 0)
        registry.recordWarmEviction()
        registry.recordSSDEviction()

        registry.updateCacheUsage(warmBytes: 1024 * 1024 * 50, warmLimit: 1024 * 1024 * 100,
                                  ssdBytes: 1024 * 1024 * 200, ssdLimit: 1024 * 1024 * 500)

        // 3. Record generation
        registry.recordGeneration(tokensIn: 100, tokensOut: 50, prefillSec: 0.1, decodeSec: 0.5, ttftSec: 0.08)
        registry.updateContext(active: 150, capacity: 4096)
        registry.updateArena(reserved: 1024 * 1024 * 1024)

        let snap1 = registry.snapshot()
        XCTAssertEqual(snap1.cache.lookupsTotal, 3)
        XCTAssertEqual(snap1.cache.hitsWarmTotal, 1)
        XCTAssertEqual(snap1.cache.hitsSSDTotal, 1)
        XCTAssertEqual(snap1.cache.missesTotal, 1)
        XCTAssertEqual(snap1.cache.tokensReusedTotal, 96)
        XCTAssertEqual(snap1.cache.evictionsWarmTotal, 1)
        XCTAssertEqual(snap1.cache.evictionsSSDTotal, 1)
        XCTAssertEqual(snap1.throughput.tokensInTotal, 100)
        XCTAssertEqual(snap1.throughput.tokensOutTotal, 50)
        XCTAssertEqual(snap1.context.activeTokens, 150)
        XCTAssertEqual(snap1.context.maxTokens, 4096)

        // 4. Test Prometheus text rendering
        let promText = registry.formatPrometheus()
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"warm_ram\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"cold_ssd\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"miss\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_tokens_reused_total 96"))
        XCTAssertTrue(promText.contains("qwegine_tokens_total{direction=\"in\"} 100"))
        XCTAssertTrue(promText.contains("qwegine_tokens_total{direction=\"out\"} 50"))
        XCTAssertTrue(promText.contains("qwegine_context_tokens{state=\"active\"} 150"))
        XCTAssertTrue(promText.contains("qwegine_context_tokens{state=\"capacity\"} 4096"))
    }

    func testHttpServerMetricsEndpoints() async throws {
        let port = 19097
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "test-metrics-model",
            host: "127.0.0.1",
            port: port
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: config, logger: logger)
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: port, logger: logger)

        try server.start()
        defer { server.stop() }

        // Allow listener to bind
        try await Task.sleep(nanoseconds: 100_000_000)

        let session = URLSession.shared
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!

        // 1. Test GET /metrics (Prometheus format)
        let promURL = baseURL.appendingPathComponent("metrics")
        let (promData, promResp) = try await session.data(from: promURL)
        let httpProm = promResp as! HTTPURLResponse
        XCTAssertEqual(httpProm.statusCode, 200)
        XCTAssertTrue((httpProm.value(forHTTPHeaderField: "Content-Type") ?? "").contains("text/plain"))
        let promString = String(data: promData, encoding: .utf8) ?? ""
        XCTAssertTrue(promString.contains("qwegine_cache_hits_total"))
        XCTAssertTrue(promString.contains("qwegine_tokens_total"))

        // 2. Test GET /v1/metrics (JSON format)
        let jsonURL = baseURL.appendingPathComponent("v1/metrics")
        let (jsonData, jsonResp) = try await session.data(from: jsonURL)
        let httpJson = jsonResp as! HTTPURLResponse
        XCTAssertEqual(httpJson.statusCode, 200)
        XCTAssertTrue((httpJson.value(forHTTPHeaderField: "Content-Type") ?? "").contains("application/json"))
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        XCTAssertNotNil(parsed?["cache"])
        XCTAssertNotNil(parsed?["throughput"])
        XCTAssertNotNil(parsed?["latency"])
        XCTAssertNotNil(parsed?["context"])
        XCTAssertNotNil(parsed?["memory"])
        XCTAssertNotNil(parsed?["hardware"])
    }

    func testEngineInferencePhaseAndLiveRates() {
        let registry = QWEngineTelemetryRegistry()

        // 1. Initial state must be idle with no live rates: a rate that was
        // not measured is absent, not zero.
        var snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)

        // 2. Prefill phase
        registry.beginPrefill(promptTokens: 100)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "prefill")

        // 3. Decode phase transition. The prefill's own duration is not
        // known here: the time to the first token is not the prefill's
        // duration, and a fully cached prompt has no prefill phase at all.
        registry.beginDecode(promptTokens: 100)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "decode")
        XCTAssertNil(snap.throughput.lastPrefillTokensPerSec)
        XCTAssertEqual(snap.provenance[QWTelemetryMetric.lastPrefillTokensPerSec.rawValue], .unavailable)

        // 4. Decode token generation
        registry.recordDecodeToken()
        registry.recordDecodeToken()

        // 5. End request: the engine's prefill duration is the only one that
        // may stand for the prefill phase (100 tok / 0.2 s = 500 tok/s).
        registry.endRequest(tokensIn: 100, tokensOut: 10, prefillSec: 0.2, decodeSec: 0.3, ttftSec: 0.2)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)
        XCTAssertEqual(snap.throughput.lastPrefillTokensPerSec ?? 0, 500.0)
        XCTAssertGreaterThan(snap.throughput.lastDecodeTokensPerSec ?? 0, 0.0)

        // 6. Reset to idle preserves the last recorded rates while the live
        // rates stay absent.
        registry.resetToIdle()
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)
        XCTAssertEqual(snap.throughput.lastPrefillTokensPerSec ?? 0, 500.0)
        XCTAssertNotNil(snap.hardware)
    }
}


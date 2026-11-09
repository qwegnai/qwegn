import XCTest
@testable import QWEngine

/// The engine-owned telemetry registry: provenance, age, and honest gaps.
final class QWTelemetryRegistryTests: XCTestCase {
    func testUnavailableValueCarriesReasonAndNoNumber() {
        let registry = QWTelemetryRegistry()
        registry.recordUnavailable("gpu.device", reason: "IOAccelerator unavailable on this host")
        let value = registry.value("gpu.device")
        XCTAssertNil(value?.value)
        XCTAssertEqual(value?.provenance, .unavailable)
        XCTAssertNotNil(value?.reason)
    }

    func testRecordedValueCarriesProvenance() {
        let registry = QWTelemetryRegistry()
        registry.record("decode.tps", 42.0)
        XCTAssertEqual(registry.value("decode.tps")?.provenance, .measured)
        XCTAssertEqual(registry.value("decode.tps")?.value, 42.0)
        registry.record("prefill.tps", 100.0, provenance: .derived)
        XCTAssertEqual(registry.value("prefill.tps")?.provenance, .derived)
    }

    func testSeriesRecordsExplicitGapsNotZeros() {
        var series = QWTelemetrySeries(capacity: 8)
        let t0 = Date()
        series.append(10.0, at: t0)
        series.append(nil, at: t0.addingTimeInterval(1))       // a gap
        series.append(12.0, at: t0.addingTimeInterval(2))
        let bucket = series.bucket(1)
        XCTAssertEqual(bucket.count, 3)
        XCTAssertEqual(bucket[0], 10.0)
        XCTAssertNil(bucket[1], "a gap must stay nil")
        XCTAssertEqual(bucket[2], 12.0)
    }

    func testBucketWithAnyGapIsNil() {
        var series = QWTelemetrySeries(capacity: 8)
        let t0 = Date()
        series.append(1.0, at: t0)
        series.append(nil, at: t0.addingTimeInterval(0.25))
        series.append(3.0, at: t0.addingTimeInterval(0.5))
        let bucket = series.bucket(1)
        XCTAssertEqual(bucket.count, 1)
        XCTAssertNil(bucket[0], "a bucket containing a gap must not be filled with the mean of the rest")
    }

    func testEmptyWindowIsNilNotZero() {
        var series = QWTelemetrySeries(capacity: 8)
        let t0 = Date()
        series.append(5.0, at: t0)
        series.append(7.0, at: t0.addingTimeInterval(120))
        let bucket = series.bucket(1)
        XCTAssertEqual(bucket.count, 121)
        XCTAssertEqual(bucket[0], 5.0)
        XCTAssertNil(bucket[60], "an unsampled window is a gap, not a zero")
        XCTAssertEqual(bucket[120], 7.0)
    }

    func testSeriesIsBoundedAndKeepsTheNewestSamples() {
        var series = QWTelemetrySeries(capacity: 4)
        for i in 0..<10 { series.append(Double(i)) }
        XCTAssertEqual(series.samples.count, 4)
        XCTAssertEqual(series.samples.last?.value, 9)
    }

    func testValueCarriesAgeAndStaleness() {
        let registry = QWTelemetryRegistry()
        let past = Date().addingTimeInterval(-120)
        registry.record("decode.tps", 42.0, provenance: .measured)
        let value = registry.value("decode.tps")
        XCTAssertNotNil(value?.age)
        XCTAssertFalse(value!.isStale(now: Date(), bound: 30))
        XCTAssertTrue(QWTelemetryValue(value: 1.0, provenance: .measured, sampledAt: past, reason: nil)
                        .isStale(now: Date(), bound: 30))
    }

    func testUnavailableValueIsAlwaysStale() {
        let value = QWTelemetryValue.unavailable("no device")
        XCTAssertTrue(value.isStale(now: Date(), bound: 3600))
        XCTAssertNil(value.age)
    }

    func testRegistryKeepsSeriesAlongsideValues() {
        let registry = QWTelemetryRegistry()
        registry.appendSeries("decode.tps", 1.0)
        registry.appendSeries("decode.tps", nil)
        let series = registry.series("decode.tps")
        XCTAssertEqual(series?.samples.count, 2)
        XCTAssertNil(series?.samples.last?.value)
        registry.reset()
        XCTAssertNil(registry.series("decode.tps"))
        XCTAssertNil(registry.value("decode.tps"))
    }

    func testSnapshotReturnsEveryRecordedValue() {
        let registry = QWTelemetryRegistry()
        registry.record("a", 1.0)
        registry.recordUnavailable("b", reason: "nope")
        let snapshot = registry.snapshot()
        XCTAssertEqual(snapshot.count, 2)
        XCTAssertEqual(snapshot["b"]?.provenance, .unavailable)
    }
}

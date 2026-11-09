// QWTelemetryRegistry.swift — The engine's own record of what it measured.
// Pure: the registry stores what the engine reported, with its provenance and
// age. It never invents a reading and never turns a gap into a zero.

import Foundation

/// The engine-side registry of telemetry readings and series.
public final class QWTelemetryRegistry: @unchecked Sendable {
    /// The process-wide registry the engine writes to.
    public static let shared = QWTelemetryRegistry()

    private let lock = NSLock()
    private var values: [String: QWTelemetryValue] = [:]
    private var series: [String: QWTelemetrySeries] = [:]

    public init() {}

    /// Records a reading with its provenance.
    public func record(_ name: String, _ value: Double, provenance: QWTelemetryProvenance = .measured) {
        lock.lock(); defer { lock.unlock() }
        values[name] = .init(value: value, provenance: provenance, sampledAt: Date(), reason: nil)
    }

    /// Records that a reading is not available, and why.
    public func recordUnavailable(_ name: String, reason: String) {
        lock.lock(); defer { lock.unlock() }
        values[name] = .unavailable(reason)
    }

    /// The last reading for a name, with its provenance and age.
    public func value(_ name: String) -> QWTelemetryValue? {
        lock.lock(); defer { lock.unlock() }
        return values[name]
    }

    /// The recorded series for a name.
    public func series(_ name: String) -> QWTelemetrySeries? {
        lock.lock(); defer { lock.unlock() }
        return series[name]
    }

    /// Appends to the series for a name; a nil value records an explicit gap.
    public func appendSeries(_ name: String, _ value: Double?, at: Date = Date()) {
        lock.lock(); defer { lock.unlock() }
        var s = series[name] ?? QWTelemetrySeries()
        s.append(value, at: at)
        series[name] = s
    }

    /// Every recorded reading, by name.
    public func snapshot() -> [String: QWTelemetryValue] {
        lock.lock(); defer { lock.unlock() }
        return values
    }

    /// Forgets every reading and series.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        values.removeAll()
        series.removeAll()
    }
}

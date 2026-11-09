// QWStatFormat.swift — One way to format a measurement.
//
// A rate is computed once. Two copies of the formula are two numbers
// that can disagree, and a statistic nobody measured is absent rather
// than a zero.

import Foundation

/// The shared formulas behind the CLI's statistics.
enum QWStatFormat {
    /// Tokens per second. Nil when the time was not measured.
    static func tokensPerSecond(tokens: Int, seconds: Double?) -> Double? {
        guard let seconds, seconds > 0 else { return nil }
        return Double(tokens) / seconds
    }

    /// The percentile of a sorted sample. Nil when the sample is empty.
    static func percentile(_ sorted: [Double], _ q: Double) -> Double? {
        guard !sorted.isEmpty else { return nil }
        return sorted[min(sorted.count - 1, Int(q * Double(sorted.count)))]
    }

    /// The mean of a sample. Nil when the sample is empty.
    static func mean(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    /// A number, formatted. "unavailable" when there was none: the
    /// statistics never print a zero nobody measured.
    static func number(_ value: Double?, _ format: String) -> String {
        guard let value else { return "unavailable" }
        return String(format: format, value)
    }

    /// A measured duration. "unavailable" when nothing measured it.
    static func seconds(_ value: Double?, _ format: String = "%.2f") -> String {
        guard let value else { return "unavailable" }
        return String(format: format, value) + " s"
    }

    /// A measured rate. "unavailable" when nothing measured it.
    static func rate(_ value: Double?, _ format: String = "%.1f") -> String {
        guard let value else { return "unavailable" }
        return String(format: format, value) + " tok/s"
    }

    /// A byte count in MiB. "unavailable" when there was none.
    static func mebibytes(_ bytes: Int?) -> String {
        guard let bytes else { return "unavailable" }
        return String(format: "%.1f", Double(bytes) / (1024 * 1024)) + " MB"
    }

    /// The latency distribution of a sorted sample of step times.
    static func stepLatency(_ sorted: [Double]) -> String {
        guard !sorted.isEmpty else { return "unavailable" }
        return "mean \(number(mean(sorted), "%.0f")) ms"
            + " · p50 \(number(percentile(sorted, 0.50), "%.0f")) ms"
            + " · p95 \(number(percentile(sorted, 0.95), "%.0f")) ms"
    }

    /// The percentiles of a sorted sample of round times.
    static func percentiles(_ sorted: [Double]) -> String {
        guard !sorted.isEmpty else { return "unavailable" }
        return "p50 \(number(percentile(sorted, 0.50), "%.0f")) ms"
            + " · p95 \(number(percentile(sorted, 0.95), "%.0f")) ms"
    }
}

// QWDashboardBuilder.swift — The card list, computed from what was measured.
// A card exists because the engine declared the metric, not because a
// renderer asked for one. A metric that was never measured still has a
// card: it says it is unavailable, and why.

import Foundation
import QWCore
import QWEngine

/// One declared card.
private struct DashboardCardSpec {
    /// The metric's name. It is the card's id.
    let id: String
    /// What the value is called.
    let title: String
    /// The unit the value is in, when it has one.
    let unit: String?
    /// The value is divided by this before it is shown.
    let scale: Double
    /// How many decimals the shown value carries.
    let digits: Int
    /// Past this age the reading no longer describes the present.
    let staleAfter: TimeInterval
    /// The capability the model needs for the card to exist at all.
    let capability: QWKnobCapability?
}

/// A reading a card can show: the number, where it came from, and when.
struct CardReading {
    let number: Double?
    /// A reading that is a word rather than a number — the host's memory
    /// pressure level, for instance. A card with one carries no number.
    let text: String?
    let provenance: QWTelemetryProvenance
    let sampledAt: Date?
    let reason: String?
    /// What the number is, when the title alone does not say it.
    let note: String?
    /// The published trace. A nil is a gap, never a zero.
    var series: [Double?] = []
}

extension QWDashboardViewModel {
    /// Builds the dashboard from the telemetry the engine measured and the
    /// knobs the registry declares.
    public static func build(lifecycle: QWEngineLifecycleInfo,
                              modelId: String?,
                              capabilities: [QWKnobCapability] = [],
                              knobs: [String: QWKnobValue] = [:],
                              knobSources: [String: String] = [:],
                              facts: QWDashboardFacts? = nil,
                              requests: [QWRequestRecord]? = nil,
                              now: Date = Date()) -> QWDashboardViewModel {
        let allReadings = readings()
        let cards = cardSpecs.compactMap { spec -> QWDashboardCard? in
            // A model without the capability has no such card at all: the
            // renderer must not draw a dash for a feature that does not
            // exist.
            if let capability = spec.capability, !capabilities.contains(capability) { return nil }
            return card(for: spec, reading: allReadings[spec.id], now: now)
        }
        // The engine's own phase: a renderer that shows the engine at
        // work reads it here, not from a counter it counts itself.
        let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
        let generation = QWEngineTelemetryRegistry.EngineInferencePhase(
            rawValue: snapshot.throughput.phase) ?? .idle
        return QWDashboardViewModel(
            sampledAt: now,
            lifecycle: lifecycle,
            modelId: modelId,
            capabilities: capabilities.map(\.rawValue).sorted(),
            generation: generation,
            facts: facts,
            cards: cards,
            actions: dashboardActions(lifecycle: lifecycle, capabilities: capabilities, knobs: knobs),
            knobs: knobs,
            knobSources: knobSources,
            requests: requests)
    }

    /// The cards the dashboard knows about. Declared once, here.
    private static let cardSpecs: [DashboardCardSpec] = [
        .init(id: "gpu.busy_ratio", title: "Engine GPU", unit: "%",
              // The engine measures a ratio; the card shows a
              // percentage, so the value is divided by a hundredth —
              // not by a hundred, which would show a hundredth of it.
              scale: 0.01, digits: 1, staleAfter: QWTelemetryFreshness.busyRatioStaleAfter,
              capability: nil),
        .init(id: "gpu.device", title: "Device GPU", unit: "%",
              scale: 1, digits: 1, staleAfter: QWTelemetryFreshness.hardwareStaleAfter,
              capability: nil),
        .init(id: "hardware.cpu_percent", title: "CPU", unit: "%",
              scale: 1, digits: 1, staleAfter: QWTelemetryFreshness.hardwareStaleAfter,
              capability: nil),
        .init(id: "hardware.memory_percent", title: "Memory", unit: "%",
              scale: 1, digits: 1, staleAfter: QWTelemetryFreshness.hardwareStaleAfter,
              capability: nil),
        .init(id: "throughput.display_prefill_tokens_per_sec", title: "Prefill", unit: "tok/s",
              scale: 1, digits: 1, staleAfter: QWTelemetryFreshness.rateStaleAfter,
              capability: nil),
        .init(id: "throughput.display_decode_tokens_per_sec", title: "Decode", unit: "tok/s",
              scale: 1, digits: 1, staleAfter: QWTelemetryFreshness.rateStaleAfter,
              capability: nil),
        .init(id: "latency.ttft_ms", title: "Time to first token", unit: "ms",
              scale: 1, digits: 0, staleAfter: QWTelemetryFreshness.latencyStaleAfter,
              capability: nil),
        .init(id: "cache.warm_bytes", title: "Warm cache", unit: "GiB",
              scale: 1_073_741_824, digits: 2, staleAfter: QWTelemetryFreshness.cacheStaleAfter,
              capability: .prefixCache),
        .init(id: "cache.ssd_bytes", title: "SSD cache", unit: "GiB",
              scale: 1_073_741_824, digits: 2, staleAfter: QWTelemetryFreshness.cacheStaleAfter,
              capability: .prefixCache),
        .init(id: "cache.tokens_reused_total", title: "Tokens reused", unit: "tok",
              scale: 1, digits: 0, staleAfter: QWTelemetryFreshness.cacheStaleAfter,
              capability: .prefixCache),
        .init(id: "context.active_tokens", title: "Last turn tokens", unit: "tok",
              scale: 1, digits: 0, staleAfter: QWTelemetryFreshness.rateStaleAfter,
              capability: nil),
        .init(id: "hardware.memory_pressure_level", title: "Memory pressure", unit: nil,
              scale: 1, digits: 0, staleAfter: QWTelemetryFreshness.hardwareStaleAfter,
              capability: nil)
    ]

    /// The card for one declared metric.
    private static func card(for spec: DashboardCardSpec,
                              reading: CardReading?,
                              now: Date) -> QWDashboardCard {
        let provenance = reading?.provenance ?? .unavailable
        let sampledAt = reading?.sampledAt
        let age = sampledAt.map { now.timeIntervalSince($0) }
        // A metric that was never measured is unavailable, not stale. A
        // measurement that is simply old is stale, and says how old.
        let isStale: Bool = (reading?.number == nil && reading?.text == nil)
            ? false
            : (age.map { $0 > spec.staleAfter } ?? true)
        let scaled = reading?.number.map { $0 / spec.scale }
        // A reading that is a word is shown as it is; a reading that is
        // a number is formatted.
        let value = reading?.text ?? scaled.map { format($0, digits: spec.digits) }
        // A word carries no unit: "from cache" is not a number of
        // tokens per second, and "normal" is not a pressure.
        let unit = reading?.text == nil ? spec.unit : nil
        // The trace is scaled like the value it belongs to, so it reads
        // in the same unit as the number above it.
        let series = (reading?.series ?? []).suffix(60).map { $0.map { $0 / spec.scale } }
        let title = reading?.note.map { "\(spec.title) (\($0))" } ?? spec.title
        return QWDashboardCard(
            id: spec.id,
            title: title,
            value: value,
            unit: unit,
            number: scaled,
            provenance: provenance,
            sampledAt: sampledAt,
            age: age,
            isStale: isStale,
            reason: reason(for: reading, isStale: isStale),
            series: series)
    }

    /// Why a card has no value. There is always one: a dash without a
    /// reason is a shrug.
    private static func reason(for reading: CardReading?, isStale: Bool) -> String? {
        guard let reading else { return "the engine has not measured this yet" }
        if let reason = reading.reason { return reason }
        if reading.number == nil && reading.text == nil { return "not measured" }
        return isStale ? "the reading is older than the freshness bound" : nil
    }

    /// The number as it is shown.
    private static func format(_ value: Double, digits: Int) -> String {
        String(format: "%.\(digits)f", value)
    }
}

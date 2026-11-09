// QWDashboardReadings.swift — Where the cards' numbers come from.
// The engine's own registry first — what the engine measured about
// itself — and the server's measurements after it. A reading that was
// never taken is unavailable, with the reason it is not there.

import Foundation
import QWCore
import QWEngine

extension QWDashboardViewModel {
    /// Every reading the cards can show, by metric name.
    static func readings() -> [String: CardReading] {
        var out: [String: CardReading] = [:]
        // The server's own measurements first, the engine's after: what
        // the engine measured about itself is the more specific reading.
        for (name, reading) in serverReadings(QWEngineTelemetryRegistry.shared.snapshot()) {
            out[name] = reading
        }
        for (name, value) in QWTelemetryRegistry.shared.snapshot() {
            // The engine's reading wins on the number; the note — which
            // rate this is — stays, because the engine does not know it.
            // The trace stays too: the engine's reading replaces the
            // number, not the history behind it.
            out[name] = CardReading(number: value.value, text: out[name]?.text,
                                     provenance: value.provenance,
                                     sampledAt: value.sampledAt, reason: value.reason,
                                     note: out[name]?.note, series: out[name]?.series ?? [])
        }
        return out
    }

    /// The server's own measurements, in the same vocabulary.
    private static func serverReadings(
        _ snap: QWEngineTelemetryRegistry.MetricsSnapshot
    ) -> [String: CardReading] {
        var out: [String: CardReading] = [:]
        let hardware = snap.hardware
        let hardwareAt = hardware?.sampledAt

        func put(_ name: String, _ number: Double?, _ at: Date?, _ reason: String?,
                 _ note: String? = nil, registry: String? = nil, series: [Double?] = []) {
            // The registry's own name for this metric, when the card's
            // name differs: the provenance and the reason live under it.
            let metric = registry ?? name
            guard let number else {
                // A reading with no number always carries the reason it
                // has no number: a dash without a reason is a shrug.
                // It carries no note either: an unmeasured rate is not
                // "the last rate" — there is no last rate.
                out[name] = CardReading(number: nil, text: nil, provenance: .unavailable,
                                         sampledAt: nil,
                                         reason: reason ?? snap.unavailableReasons[metric]
                                             ?? "the engine has not measured this yet",
                                         note: nil)
                return
            }
            guard let at else {
                // A number with no moment is not a measurement: the engine
                // never reported this, so the reading is absent.
                out[name] = CardReading(number: nil, text: nil, provenance: .unavailable,
                                         sampledAt: nil,
                                         reason: snap.unavailableReasons[metric]
                                             ?? "the engine has not measured this yet",
                                         note: note)
                return
            }
            // The registry's own reason, when it has one: a reading that
            // has grown old says so, in its own words.
            out[name] = CardReading(number: number, text: nil,
                                     provenance: snap.provenance[metric] ?? .measured,
                                     sampledAt: at, reason: snap.unavailableReasons[metric],
                                     note: note, series: series)
        }

        /// A reading that is a word, not a number: the host's memory
        /// pressure level. macOS reports a level, never a percentage.
        func putText(_ name: String, _ text: String?, _ at: Date?, _ reason: String?,
                     registry: String? = nil, series: [Double?] = []) {
            let metric = registry ?? name
            guard let text, let at else {
                out[name] = CardReading(number: nil, text: nil, provenance: .unavailable,
                                         sampledAt: nil,
                                         reason: reason ?? snap.unavailableReasons[metric]
                                             ?? "the engine has not measured this yet",
                                         note: nil)
                return
            }
            out[name] = CardReading(number: nil, text: text,
                                     provenance: snap.provenance[metric] ?? .measured,
                                     sampledAt: at,
                                     reason: snap.unavailableReasons[metric] ?? reason,
                                     note: nil, series: series)
        }

        put("gpu.busy_ratio", snap.gpuBusyRatio,
            snap.sampledAt[QWTelemetryMetric.gpuBusyRatio.rawValue], nil)
        put("gpu.device", hardware?.gpuUsagePercent, hardwareAt,
            hardware?.gpuUsageUnavailableReason,
            registry: QWTelemetryMetric.gpuPercent.rawValue)
        put("hardware.cpu_percent", hardware?.cpuUsagePercent, hardwareAt,
            hardware?.cpuUsageUnavailableReason,
            registry: QWTelemetryMetric.cpuPercent.rawValue)
        put("hardware.memory_percent", hardware?.memoryUsagePercent, hardwareAt,
            hardware?.memoryUnavailableReason,
            registry: QWTelemetryMetric.memoryPercent.rawValue)
        // The prefill card shows a rate when the last turn had one. A
        // turn whose prompt came entirely from the cache ran no prefill
        // phase: the card says where the prompt came from, not a rate.
        let prefillMetric = QWTelemetryMetric.prefillFromCache.rawValue
        if snap.throughput.prefillFromCache, let at = snap.sampledAt[prefillMetric] {
            putText("throughput.display_prefill_tokens_per_sec", "from cache", at,
                    snap.unavailableReasons[prefillMetric], registry: prefillMetric,
                    series: snap.throughput.prefillRateSeries)
        } else {
            put("throughput.display_prefill_tokens_per_sec",
                snap.throughput.displayPrefillTokensPerSec,
                snap.sampledAt[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue],
                snap.unavailableReasons[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue],
                snap.throughput.prefillDisplayMode.rawValue,
                series: snap.throughput.prefillRateSeries)
        }
        put("throughput.display_decode_tokens_per_sec",
            snap.throughput.displayDecodeTokensPerSec,
            snap.sampledAt[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue],
            snap.unavailableReasons[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue],
            snap.throughput.decodeDisplayMode.rawValue,
            series: snap.throughput.decodeRateSeries)
        put("latency.ttft_ms", snap.latency.lastTtftMs,
            snap.sampledAt[QWTelemetryMetric.ttftMs.rawValue],
            snap.unavailableReasons[QWTelemetryMetric.ttftMs.rawValue])
        put("cache.warm_bytes", snap.cache.warmBytes.map(Double.init), snap.cache.sampledAt, nil)
        put("cache.ssd_bytes", snap.cache.ssdBytes.map(Double.init), snap.cache.sampledAt, nil)
        put("cache.tokens_reused_total", Double(snap.cache.tokensReusedTotal),
            snap.cache.sampledAt, nil)
        put("context.active_tokens", snap.context.activeTokens.map(Double.init),
            snap.context.sampledAt, nil)
        putText("hardware.memory_pressure_level", hardware?.memoryPressureLevel, hardwareAt,
                hardware?.memoryPressureUnavailableReason,
                registry: QWTelemetryMetric.memoryPressure.rawValue)
        return out
    }

    /// The actions this model can perform. An action the model cannot
    /// perform is not published at all: a renderer must not offer it.
    static func dashboardActions(lifecycle: QWEngineLifecycleInfo,
                                  capabilities: [QWKnobCapability],
                                  knobs: [String: QWKnobValue]) -> [QWDashboardAction] {
        var actions: [QWDashboardAction] = []
        let phase = lifecycle.phase

        // The model is either loaded or it is not, so the panel offers
        // the one action that applies. Publishing both — one of them
        // greyed out — puts a button on the screen that cannot be
        // pressed and leaves the reader to work out which one is real.
        switch phase {
        case .loaded:
            actions.append(QWDashboardAction(id: "unload", title: "Unload model", knobId: nil,
                                             isEnabled: true, reason: nil))
        case .loading:
            actions.append(QWDashboardAction(id: "load", title: "Load model", knobId: nil,
                                             isEnabled: false, reason: "a load is under way"))
        default:
            actions.append(QWDashboardAction(id: "load", title: "Load model", knobId: nil,
                                             isEnabled: true, reason: nil))
        }

        // A restart is only offered where it does something: with a
        // model up, or with a failed one to try again.
        if phase == .loaded || phase == .failed {
            actions.append(QWDashboardAction(id: "restart", title: "Restart model", knobId: nil,
                                             isEnabled: true, reason: nil))
        }

        // A model without a prefix cache has no such action to offer,
        // and neither has one whose cache is already on.
        if capabilities.contains(.prefixCache) {
            let isOn = knobs["prefixCacheEnabled"]?.boolValue ?? false
            if !isOn {
                actions.append(QWDashboardAction(
                    id: "enablePrefixCache", title: "Enable prefix cache",
                    knobId: "prefixCacheEnabled",
                    isEnabled: true, reason: nil))
            }
        }
        return actions
    }
}

// QWTelemetrySnapshotMaps.swift — The snapshot's own truth about its metrics.
// One place decides how each metric was obtained, when it was taken, and
// why it is missing. The maps cannot drift from the values they describe.

import Foundation
import QWEngine

extension QWEngineTelemetryRegistry {
    /// The rate a UI should show, and what it is. A measurement that is too
    /// old to describe the present is stale, not the engine's current speed.
    static func displayRate(live: Double?, average: Double?, last: Double?,
                             sampledAt: Date?, now: Date,
                             staleAfter: TimeInterval) -> (Double?, QWRateDisplayMode) {
        if let live { return (live, .live) }
        if let average { return (average, .average) }
        if let last {
            let age = sampledAt.map { now.timeIntervalSince($0) } ?? .infinity
            return (last, age > staleAfter ? .stale : .last)
        }
        return (nil, .last)
    }

    /// How each metric in the snapshot was obtained.
    static func provenanceMap(livePrefill: Double?, liveDecode: Double?,
                                      lastPrefill: Double?, lastDecode: Double?,
                                      emaPrefill: Double?, emaDecode: Double?,
                                      avgPrefill: Double?, avgDecode: Double?,
                                      displayPrefill: Double?, displayDecode: Double?,
                                      prefillMode: QWRateDisplayMode, decodeMode: QWRateDisplayMode,
                                      ttft: Bool, hw: QWSystemHardwareUtilization, hwStale: Bool,
                                      busyRatio: Double?, busyStale: Bool,
                                      prefillFromCache: Bool) -> [String: QWTelemetryProvenance] {
        var map: [String: QWTelemetryProvenance] = [:]
        map[QWTelemetryMetric.livePrefillTokensPerSec.rawValue] = livePrefill != nil ? .derived : .unavailable
        map[QWTelemetryMetric.liveDecodeTokensPerSec.rawValue] = liveDecode != nil ? .derived : .unavailable
        map[QWTelemetryMetric.lastPrefillTokensPerSec.rawValue] = lastPrefill != nil ? .derived : .unavailable
        map[QWTelemetryMetric.lastDecodeTokensPerSec.rawValue] = lastDecode != nil ? .derived : .unavailable
        map[QWTelemetryMetric.ema1mPrefillTokensPerSec.rawValue] = emaPrefill != nil ? .derived : .unavailable
        map[QWTelemetryMetric.ema1mDecodeTokensPerSec.rawValue] = emaDecode != nil ? .derived : .unavailable
        map[QWTelemetryMetric.avgPrefillTokensPerSec.rawValue] = avgPrefill != nil ? .derived : .unavailable
        map[QWTelemetryMetric.avgDecodeTokensPerSec.rawValue] = avgDecode != nil ? .derived : .unavailable
        map[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue] =
            displayPrefill == nil ? .unavailable : (prefillMode == .stale ? .stale : .derived)
        map[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue] =
            displayDecode == nil ? .unavailable : (decodeMode == .stale ? .stale : .derived)
        map[QWTelemetryMetric.ttftMs.rawValue] = ttft ? .measured : .unavailable
        map[QWTelemetryMetric.cpuPercent.rawValue] = hw.cpuUsagePercent == nil ? .unavailable : (hwStale ? .stale : .measured)
        map[QWTelemetryMetric.gpuPercent.rawValue] = hw.gpuUsagePercent == nil ? .unavailable : (hwStale ? .stale : .measured)
        map[QWTelemetryMetric.memoryPercent.rawValue] = hw.memoryUsagePercent == nil ? .unavailable : (hwStale ? .stale : .measured)
        map[QWTelemetryMetric.memoryPressure.rawValue] = hw.memoryPressureLevel == nil ? .unavailable : (hwStale ? .stale : .measured)
        map[QWTelemetryMetric.gpuBusyRatio.rawValue] = busyRatio == nil ? .unavailable : (busyStale ? .stale : .measured)
        map[QWTelemetryMetric.prefillFromCache.rawValue] = prefillFromCache ? .derived : .unavailable
        return map
    }

    /// When each metric in the snapshot was taken.
    static func sampledAtMap(liveDecode: Double?, lastPrefill: Double?, lastDecode: Double?,
                                     emaPrefill: Double?, emaDecode: Double?,
                                     avgPrefill: Double?, avgDecode: Double?,
                                     displayPrefill: Double?, displayDecode: Double?,
                                     ttft: Bool, hw: QWSystemHardwareUtilization,
                                     busy: QWTelemetryValue?,
                                     lastPrefillAt: Date?, lastDecodeAt: Date?,
                                     lastTurnAt: Date?, lastTtftAt: Date?,
                                     prefillFromCache: Bool) -> [String: Date] {
        var map: [String: Date] = [:]
        if liveDecode != nil { map[QWTelemetryMetric.liveDecodeTokensPerSec.rawValue] = lastDecodeAt }
        if lastPrefill != nil { map[QWTelemetryMetric.lastPrefillTokensPerSec.rawValue] = lastPrefillAt }
        if lastDecode != nil { map[QWTelemetryMetric.lastDecodeTokensPerSec.rawValue] = lastDecodeAt }
        if emaPrefill != nil { map[QWTelemetryMetric.ema1mPrefillTokensPerSec.rawValue] = lastTurnAt }
        if emaDecode != nil { map[QWTelemetryMetric.ema1mDecodeTokensPerSec.rawValue] = lastTurnAt }
        if avgPrefill != nil { map[QWTelemetryMetric.avgPrefillTokensPerSec.rawValue] = lastTurnAt }
        if avgDecode != nil { map[QWTelemetryMetric.avgDecodeTokensPerSec.rawValue] = lastTurnAt }
        if displayPrefill != nil { map[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue] = lastPrefillAt }
        if displayDecode != nil { map[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue] = lastDecodeAt }
        if ttft { map[QWTelemetryMetric.ttftMs.rawValue] = lastTtftAt }
        map[QWTelemetryMetric.cpuPercent.rawValue] = hw.sampledAt
        map[QWTelemetryMetric.gpuPercent.rawValue] = hw.sampledAt
        map[QWTelemetryMetric.memoryPercent.rawValue] = hw.sampledAt
        map[QWTelemetryMetric.memoryPressure.rawValue] = hw.sampledAt
        if let at = busy?.sampledAt { map[QWTelemetryMetric.gpuBusyRatio.rawValue] = at }
        if prefillFromCache { map[QWTelemetryMetric.prefillFromCache.rawValue] = lastTurnAt }
        return map
    }

    /// Why a metric in the snapshot is unavailable or stale, when it is.
    static func reasonMap(livePrefill: Double?, liveDecode: Double?,
                                  lastPrefill: Double?, lastDecode: Double?,
                                  emaPrefill: Double?, emaDecode: Double?,
                                  avgPrefill: Double?, avgDecode: Double?,
                                  displayPrefill: Double?, displayDecode: Double?,
                                  prefillMode: QWRateDisplayMode, decodeMode: QWRateDisplayMode,
                                  ttft: Bool, hw: QWSystemHardwareUtilization, hwStale: Bool,
                                  busyRatio: Double?, busyStale: Bool,
                                  busyReason: String?,
                                  prefillFromCache: Bool) -> [String: String] {
        var map: [String: String] = [:]
        if livePrefill == nil {
            map[QWTelemetryMetric.livePrefillTokensPerSec.rawValue] =
                "the prefill phase has not reported a duration yet"
        }
        if liveDecode == nil {
            map[QWTelemetryMetric.liveDecodeTokensPerSec.rawValue] =
                "the engine is not decoding, or the turn has not produced two tokens yet"
        }
        if lastPrefill == nil {
            map[QWTelemetryMetric.lastPrefillTokensPerSec.rawValue] =
                "no turn has completed a prefill yet"
        }
        if lastDecode == nil {
            map[QWTelemetryMetric.lastDecodeTokensPerSec.rawValue] =
                "no turn has completed a decode yet"
        }
        if emaPrefill == nil {
            map[QWTelemetryMetric.ema1mPrefillTokensPerSec.rawValue] = "no prefill rate has been measured yet"
        }
        if emaDecode == nil {
            map[QWTelemetryMetric.ema1mDecodeTokensPerSec.rawValue] = "no decode rate has been measured yet"
        }
        if avgPrefill == nil {
            map[QWTelemetryMetric.avgPrefillTokensPerSec.rawValue] = "no turn has completed a prefill yet"
        }
        if avgDecode == nil {
            map[QWTelemetryMetric.avgDecodeTokensPerSec.rawValue] = "no turn has completed a decode yet"
        }
        if displayPrefill == nil {
            map[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue] = "no prefill rate has been measured yet"
        } else if prefillMode == .stale {
            map[QWTelemetryMetric.displayPrefillTokensPerSec.rawValue] =
                "the last prefill rate is older than the freshness bound"
        }
        if displayDecode == nil {
            map[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue] = "no decode rate has been measured yet"
        } else if decodeMode == .stale {
            map[QWTelemetryMetric.displayDecodeTokensPerSec.rawValue] =
                "the last decode rate is older than the freshness bound"
        }
        if !ttft { map[QWTelemetryMetric.ttftMs.rawValue] = "no request has completed yet" }
        if let reason = hw.cpuUsageUnavailableReason {
            map[QWTelemetryMetric.cpuPercent.rawValue] = reason
        } else if hwStale {
            map[QWTelemetryMetric.cpuPercent.rawValue] = "the reading is older than the freshness bound"
        }
        if let reason = hw.gpuUsageUnavailableReason {
            map[QWTelemetryMetric.gpuPercent.rawValue] = reason
        } else if hwStale {
            map[QWTelemetryMetric.gpuPercent.rawValue] = "the reading is older than the freshness bound"
        }
        if let reason = hw.memoryUnavailableReason {
            map[QWTelemetryMetric.memoryPercent.rawValue] = reason
        } else if hwStale {
            map[QWTelemetryMetric.memoryPercent.rawValue] = "the reading is older than the freshness bound"
        }
        if let reason = hw.memoryPressureUnavailableReason {
            map[QWTelemetryMetric.memoryPressure.rawValue] = reason
        } else if hwStale {
            map[QWTelemetryMetric.memoryPressure.rawValue] = "the reading is older than the freshness bound"
        }
        if busyRatio == nil {
            map[QWTelemetryMetric.gpuBusyRatio.rawValue] =
                busyReason ?? "the engine has not run a measured step yet"
        } else if busyStale {
            map[QWTelemetryMetric.gpuBusyRatio.rawValue] = "the engine's last measured step is too old to describe the present"
        }
        // Why the prefill card shows a word instead of a rate, and why it
        // does not: a prompt the cache served entirely had no prefill
        // phase, and a turn that computed a prefill has a rate to show.
        map[QWTelemetryMetric.prefillFromCache.rawValue] = prefillFromCache
            ? "the last turn's prompt came from the prefix cache: no prefill ran"
            : "no turn has completed since the last reset, or the last turn computed a prefill"
        return map
    }
}

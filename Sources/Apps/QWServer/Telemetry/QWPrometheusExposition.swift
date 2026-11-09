// QWPrometheusExposition.swift — The server's Prometheus surface.
// A series that was not measured is absent. A zero would say "idle", and
// that is a claim the server cannot make.

import Foundation

extension QWEngineTelemetryRegistry {
    /// Formats metrics in standard Prometheus exposition format (text/plain; version=0.0.4).
    public func formatPrometheus() -> String {
        let s = snapshot()
        var out = ""

        out += "# HELP qwegine_cache_hits_total Total prefix cache hits by tier\n"
        out += "# TYPE qwegine_cache_hits_total counter\n"
        out += "qwegine_cache_hits_total{tier=\"warm_ram\"} \(s.cache.hitsWarmTotal)\n"
        out += "qwegine_cache_hits_total{tier=\"cold_ssd\"} \(s.cache.hitsSSDTotal)\n"
        // The unclassified tier is left out when the server does not report
        // it: a series that says zero would claim no hit ever went
        // unnamed, which a server without the counter cannot know.
        if let unclassified = s.cache.hitsUnclassifiedTotal {
            out += "qwegine_cache_hits_total{tier=\"unclassified\"} \(unclassified)\n"
        }
        out += "qwegine_cache_hits_total{tier=\"miss\"} \(s.cache.missesTotal)\n\n"

        out += "# HELP qwegine_cache_tokens_reused_total Cumulative tokens reused from cache\n"
        out += "# TYPE qwegine_cache_tokens_reused_total counter\n"
        out += "qwegine_cache_tokens_reused_total \(s.cache.tokensReusedTotal)\n\n"

        out += "# HELP qwegine_cache_bytes Current cache storage bytes used\n"
        out += "# TYPE qwegine_cache_bytes gauge\n"
        // A tier the engine cannot measure is left out: a gauge with no
        // value says nothing, and a zero would say it is empty.
        if let warmBytes = s.cache.warmBytes {
            out += "qwegine_cache_bytes{tier=\"warm_ram\",limit=\"\(s.cache.warmLimitBytes)\") \(warmBytes)\n"
        }
        if let ssdBytes = s.cache.ssdBytes {
            out += "qwegine_cache_bytes{tier=\"cold_ssd\",limit=\"\(s.cache.ssdLimitBytes)\") \(ssdBytes)\n"
        }
        out += "\n"

        out += "# HELP qwegine_cache_evictions_total Total entries evicted from cache tiers\n"
        out += "# TYPE qwegine_cache_evictions_total counter\n"
        out += "qwegine_cache_evictions_total{tier=\"warm_ram\"} \(s.cache.evictionsWarmTotal)\n"
        out += "qwegine_cache_evictions_total{tier=\"cold_ssd\"} \(s.cache.evictionsSSDTotal)\n\n"

        out += "# HELP qwegine_tokens_total Cumulative token throughput\n"
        out += "# TYPE qwegine_tokens_total counter\n"
        out += "qwegine_tokens_total{direction=\"in\"} \(s.throughput.tokensInTotal)\n"
        out += "qwegine_tokens_total{direction=\"out\"} \(s.throughput.tokensOutTotal)\n\n"

        out += "# HELP qwegine_engine_phase Current engine inference phase (0=idle, 1=prefill, 2=decode)\n"
        out += "# TYPE qwegine_engine_phase gauge\n"
        let phaseCode: Int = (s.throughput.phase == "prefill") ? 1 : ((s.throughput.phase == "decode") ? 2 : 0)
        out += "qwegine_engine_phase{phase=\"\(s.throughput.phase)\"} \(phaseCode)\n\n"

        out += Self.block(s, "qwegine_speed_tokens_per_second",
                          "Processing speed in tokens per second",
                          [("phase", "prefill", "window", "instant", s.throughput.livePrefillTokensPerSec,
                            QWTelemetryMetric.livePrefillTokensPerSec),
                           ("phase", "decode", "window", "instant", s.throughput.liveDecodeTokensPerSec,
                            QWTelemetryMetric.liveDecodeTokensPerSec),
                           ("phase", "prefill", "window", "last", s.throughput.lastPrefillTokensPerSec,
                            QWTelemetryMetric.lastPrefillTokensPerSec),
                           ("phase", "decode", "window", "last", s.throughput.lastDecodeTokensPerSec,
                            QWTelemetryMetric.lastDecodeTokensPerSec),
                           ("phase", "prefill", "window", "ema_1m", s.throughput.ema1mPrefillTokensPerSec,
                            QWTelemetryMetric.ema1mPrefillTokensPerSec),
                           ("phase", "decode", "window", "ema_1m", s.throughput.ema1mDecodeTokensPerSec,
                            QWTelemetryMetric.ema1mDecodeTokensPerSec),
                           ("phase", "prefill", "window", "avg", s.throughput.avgPrefillTokensPerSec,
                            QWTelemetryMetric.avgPrefillTokensPerSec),
                           ("phase", "decode", "window", "avg", s.throughput.avgDecodeTokensPerSec,
                            QWTelemetryMetric.avgDecodeTokensPerSec)])

        out += Self.block(s, "qwegine_latency_ttft_ms",
                          "Time to first token in milliseconds",
                          [("type", "last", nil, nil, s.latency.lastTtftMs, QWTelemetryMetric.ttftMs),
                           ("type", "avg", nil, nil, s.latency.avgTtftMs, QWTelemetryMetric.ttftMs)])

        out += "# HELP qwegine_context_tokens Active and maximum context tokens\n"
        out += "# TYPE qwegine_context_tokens gauge\n"
        // The active context is left out when no turn has run: a gauge
        // with no value says nothing, and a zero would say it is empty.
        if let activeTokens = s.context.activeTokens {
            out += "qwegine_context_tokens{state=\"active\"} \(activeTokens)\n"
        }
        out += "qwegine_context_tokens{state=\"capacity\"} \(s.context.maxTokens)\n\n"

        out += "# HELP qwegine_memory_bytes Process memory footprint and arena allocation\n"
        out += "# TYPE qwegine_memory_bytes gauge\n"
        out += "qwegine_memory_bytes{kind=\"resident\"} \(s.memory.residentBytes)\n"
        out += "qwegine_memory_bytes{kind=\"arena_reserved\"} \(s.memory.arenaReservedBytes)\n"

        // A reading the host cannot give is absent from the exposition:
        // a missing series says "not measured", a zero says "idle".
        if let hw = s.hardware {
            out += Self.block(s, "qwegine_hardware_cpu_percent",
                              "Host CPU utilization percent",
                              [("kind", "cpu", nil, nil, hw.cpuUsagePercent, QWTelemetryMetric.cpuPercent)])

            if let gpu = hw.gpuUsagePercent {
                out += "# HELP qwegine_hardware_gpu_percent Apple Silicon GPU utilization percent\n"
                out += "# TYPE qwegine_hardware_gpu_percent gauge\n"
                out += "qwegine_hardware_gpu_percent{chip=\"\(hw.chipIdentifier)\"} \(String(format: "%.2f", gpu))\n\n"
            }

            out += "# HELP qwegine_hardware_memory_bytes Host unified memory used bytes\n"
            out += "# TYPE qwegine_hardware_memory_bytes gauge\n"
            if let used = hw.memoryUsedBytes {
                out += "qwegine_hardware_memory_bytes{state=\"used\"} \(used)\n"
            }
            out += "qwegine_hardware_memory_bytes{state=\"total\"} \(hw.memoryTotalBytes)\n"
        }

        out += Self.block(s, "qwegine_gpu_busy_ratio",
                          "Engine GPU busy ratio over the last measured step",
                          [("kind", "engine", nil, nil, s.gpuBusyRatio, QWTelemetryMetric.gpuBusyRatio)])

        return out
    }

    /// A metric family with the series that were measured. A metric the
    /// snapshot marks unavailable or stale has no series at all: the
    /// scraper sees it gone, and never reads "idle" where there is "none".
    private static func block(_ s: MetricsSnapshot, _ name: String, _ help: String,
                              _ entries: [(String, String, String?, String?, Double?, QWTelemetryMetric)]) -> String {
        var samples = ""
        for (label, value, label2, value2, reading, metric) in entries {
            guard let reading else { continue }
            guard let provenance = s.provenance[metric.rawValue],
                  provenance == .measured || provenance == .derived else { continue }
            var labels = "\(label)=\"\(value)\""
            if let label2, let value2 {
                labels += ",\(label2)=\"\(value2)\""
            }
            samples += "\(name){\(labels)} \(String(format: "%.2f", reading))\n"
        }
        guard !samples.isEmpty else { return "" }
        return "# HELP \(name) \(help)\n# TYPE \(name) gauge\n\(samples)\n"
    }
}

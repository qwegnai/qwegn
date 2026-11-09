import Foundation

/// `qwg bench m1 --check` — the CI merge-gate entry point (spec-22: "qwg
/// bench full --check green in CI"; BENCH_PROTOCOL rule 5 ratchet). PURE
/// file math over the m1 soak artifact (m1_soak_*.jsonl) and its windows.
public extension QWCli {

    struct SoakVerdict: Equatable {
        public var lines: [String]
        public var allPass: Bool
        public var metrics: [String: Double]   // derived ratchet metrics
    }

    static func verifySoak(samples: [SoakSample], minSteps: Int = 10_000,
                           jitterCeil: Double = 1.10, tpsFloor: Double = 30,
                           rssBudgetKb: Int64 = 65_536, fdBudget: Int = 2) -> SoakVerdict {
        var lines: [String] = []
        var pass = true
        func gate(_ ok: Bool, _ line: String) {
            lines.append("[bench] \(ok ? "PASS" : "FAIL") \(line)")
            if !ok { pass = false }
        }
        func skip(_ line: String) { lines.append("[bench] SKIP \(line)") }
        guard !samples.isEmpty else {
            lines.append("[bench] FAIL no soak samples (run: make m1 — the artifact is the gate)")
            return SoakVerdict(lines: lines, allPass: false, metrics: [:])
        }
        // What produced the artifact decides which laws apply. An artifact
        // with no source is a legacy engine soak: the engine laws apply.
        let isEngine = samples.last?.isEngineSoak ?? true
        gate(samples.last!.steps >= minSteps,
             "soak coverage \(samples.last!.steps) steps ≥ \(minSteps) (\(samples.count) windows)")
        // ascending, no duplicated windows (log integrity)
        let ascending = zip(samples, samples.dropFirst()).allSatisfy { $0.steps < $1.steps }
        gate(ascending, "window steps strictly ascending (log integrity)")
        // allocs: EVERY window must carry zero (the counter is cumulative —
        // one nonzero anywhere is a nonzero forever)
        gate(samples.allSatisfy { $0.steadyAllocs == 0 },
             "steady allocs == 0 in every window (\(samples.count) checked)")
        // command-buffer errors: the engine counts them. A harness that ran
        // no command buffer has NOT measured them — that is absent, not 0.
        if let cbErrors = samples.last!.cbErrors {
            gate(cbErrors == 0, "command-buffer errors == 0 at soak end")
        } else {
            skip("command-buffer errors not measured (this artifact ran no engine command buffer)")
        }
        // pooled decode t/s: window wall = n/tps, n = step delta
        var totalSteps = 0, totalWall = 0.0
        var prevSteps = 0
        for s in samples {
            let n = s.steps - prevSteps
            totalSteps += n
            if s.windowTps > 0 { totalWall += Double(n) / s.windowTps }
            prevSteps = s.steps
        }
        let pooledTps = totalWall > 0 ? Double(totalSteps) / totalWall : 0
        // The ≥30 t/s floor and the jitter bound are the ENGINE's decode
        // laws. A CPU-only harness measured its own step rate: reporting
        // that against the decode floor would compare two different things.
        if isEngine {
            gate(pooledTps >= tpsFloor,
                 String(format: "pooled decode %.1f t/s ≥ %.0f", pooledTps, tpsFloor))
        } else {
            skip(String(format: "pooled decode floor (%.0f t/s) is the engine's law; this artifact measured a CPU-only harness",
                        tpsFloor))
        }
        // jitter: conservative bound (step-weighted p50 across windows)
        var weightedP50 = 0.0
        prevSteps = 0
        for s in samples {
            let n = s.steps - prevSteps
            weightedP50 += s.windowP50Ms * Double(n)
            prevSteps = s.steps
        }
        weightedP50 /= Double(max(totalSteps, 1))
        let maxP95 = samples.map(\.windowP95Ms).max() ?? 0
        let jitter = weightedP50 > 0 ? maxP95 / weightedP50 : .greatestFiniteMagnitude
        if isEngine {
            gate(jitter <= jitterCeil,
                 String(format: "jitter bound max(p95) %.1f ms / weighted-p50 %.1f ms = %.3f ≤ %.2f (conservative; m1 ledger is authority)",
                        maxP95, weightedP50, jitter, jitterCeil))
        } else {
            skip("decode-jitter bound is the engine's law; this artifact measured a CPU-only harness")
        }
        // RSS / fd trend
        let rssMin = samples.map(\.rssKb).min() ?? 0
        let rssMax = samples.map(\.rssKb).max() ?? 0
        gate(rssMax - rssMin <= rssBudgetKb, "RSS trend \(rssMax - rssMin) KiB ≤ \(rssBudgetKb) KiB across windows")
        let fdMin = samples.map(\.fds).min() ?? 0
        let fdMax = samples.map(\.fds).max() ?? 0
        gate(fdMax - fdMin <= fdBudget, "fd trend \(fdMax - fdMin) ≤ \(fdBudget)")

        let metrics: [String: Double] = ratchetMetrics(samples: samples, pooledTps: pooledTps,
                                                        maxP95: maxP95, isEngine: isEngine)
        // encode_pct / encode_us_p50 / ttft4k are NOT recoverable from a
        // window-sampled log — m1's live ledger owns those ratchet keys;
        // verifySoak owns the soak-class three above.
        return SoakVerdict(lines: lines, allPass: pass, metrics: metrics)
    }

    /// The ratchet keys the gate publishes (spec 40 §4 T6 item 3):
    /// throughput, itl_p95, steady_allocs, kv_free_pages. A key this run
    /// did not measure is absent — never a measured zero.
    static func ratchetMetrics(samples: [SoakSample], pooledTps: Double,
                                maxP95: Double, isEngine: Bool) -> [String: Double] {
        var metrics: [String: Double] = [:]
        metrics[isEngine ? "m1_decode_tps" : "m1_step_rate_per_second"] = pooledTps
        metrics["m1_itl_p95_ms"] = maxP95
        metrics["m1_steady_allocs"] = Double(samples.map(\.steadyAllocs).max() ?? 0)
        if let pages = samples.last?.kvFreePages { metrics["m1_kv_free_pages"] = Double(pages) }
        return metrics
    }

    /// newest soak JSONL in a directory (name pattern m1_soak_*.jsonl)
    static func newestSoakLog(in dir: URL) -> URL? {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        return items.filter { $0.lastPathComponent.hasPrefix("m1_soak_") && $0.pathExtension == "jsonl" }
            .max { (a, b) in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return da < db
            }
    }

    // MARK: - Baselines and Ratchet

    enum MetricDirection: String, Codable, Sendable { case higher, lower }

    struct Metric: Codable, Equatable {
        public var value: Double
        public var direction: MetricDirection
        public var unit: String
        public init(_ value: Double, _ direction: MetricDirection, _ unit: String) {
            self.value = value; self.direction = direction; self.unit = unit
        }
    }

    struct Baselines: Codable, Equatable {
        public var schema: Int
        public var date: String
        public var machine: String
        public var metrics: [String: Metric]
        public init(schema: Int = 1, date: String, machine: String, metrics: [String: Metric]) {
            self.schema = schema; self.date = date; self.machine = machine; self.metrics = metrics
        }

        public static func load(_ url: URL) throws -> Baselines {
            try JSONDecoder().decode(Baselines.self, from: try Data(contentsOf: url))
        }
        public func save(_ url: URL) throws {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(self).write(to: url)
        }
    }

    struct RatchetRow: Equatable {
        public var metric: String; public var baseline: Double; public var current: Double
        public var deltaPct: Double     // signed regression magnitude (+ = worse)
        public var ok: Bool
    }

    /// Which way each ratchet key must move to be an improvement: a
    /// higher number is better, or a lower one. A key that is not listed
    /// is treated as "lower is better" (a latency, a count).
    static let ratchetDirections: [String: MetricDirection] = [
        "m1_decode_tps": .higher,
        "m1_step_rate_per_second": .higher,
        "m1_itl_p95_ms": .lower,
        "m1_steady_allocs": .lower,
        "m1_kv_free_pages": .higher,
    ]

    static func ratchetCheck(current: [String: Double], baseline: Baselines,
                             maxRegressionPct: Double = 3) -> [RatchetRow] {
        baseline.metrics.keys.sorted().map { k in
            let m = baseline.metrics[k]!
            guard let c = current[k] else {
                return RatchetRow(metric: k, baseline: m.value, current: .nan,
                                  deltaPct: .nan, ok: true)
            }
            // A baseline of zero has no percentage: the same value is no
            // change, a larger one is a regression the ratio cannot express.
            let delta: Double
            if m.value == 0 {
                delta = c == 0 ? 0 : .infinity
            } else {
                delta = m.direction == .higher
                    ? (m.value - c) / m.value * 100
                    : (c - m.value) / m.value * 100
            }
            return RatchetRow(metric: k, baseline: m.value, current: c,
                              deltaPct: delta, ok: delta <= maxRegressionPct)
        }
    }

    static func ratchetLines(_ rows: [RatchetRow]) -> [String] {
        rows.map { r in
            if r.current.isNaN { return "[ratchet] SKIP \(r.metric) (baseline \(r.baseline) not measured this run)" }
            return String(format: "[ratchet] %@ %@ %.3f → %.3f (%+.2f%% vs baseline, limit +3%%)",
                          r.ok ? "PASS" : "FAIL", r.metric, r.baseline, r.current, r.deltaPct)
        }
    }

    // MARK: - soak log (JSONL trend)

    /// One measured window of the soak. Every field is what the run
    /// measured; a field that was not measured is nil, never zero.
    struct SoakSample: Codable, Equatable {
        /// The source that produced this window: "engine" (a real decode
        /// soak) or "harness" (the CPU-only reference soak). Nil is a
        /// legacy engine soak.
        public static let engineSource = "engine"
        public static let harnessSource = "harness"
        /// The line kind in the telemetry export. Nil is a plain window.
        public static let windowKind = "window"

        public var kind: String?
        public var steps: Int
        public var windowTps: Double
        public var windowP50Ms: Double
        public var windowP95Ms: Double
        public var steadyAllocs: Int
        public var rssKb: Int64
        public var fds: Int
        /// The engine's command-buffer errors. Nil when the run issued no
        /// command buffer: not measured is not zero.
        public var cbErrors: Int?
        /// The engine's free KV pages at the end of the window. Nil when
        /// no engine ran.
        public var kvFreePages: Int?
        /// What measured this window.
        public var source: String?
        /// When the window was measured.
        public var sampledAt: Date?
        /// What the window is, in words.
        public var label: String?

        public init(steps: Int, windowTps: Double, windowP50Ms: Double, windowP95Ms: Double,
                    steadyAllocs: Int, rssKb: Int64, fds: Int, cbErrors: Int? = nil,
                    kvFreePages: Int? = nil, source: String? = nil, sampledAt: Date? = nil,
                    label: String? = nil, kind: String? = nil) {
            self.steps = steps; self.windowTps = windowTps; self.windowP50Ms = windowP50Ms
            self.windowP95Ms = windowP95Ms; self.steadyAllocs = steadyAllocs
            self.rssKb = rssKb; self.fds = fds; self.cbErrors = cbErrors
            self.kvFreePages = kvFreePages; self.source = source; self.sampledAt = sampledAt
            self.label = label; self.kind = kind
        }

        /// True when the engine's decode laws apply to this window.
        public var isEngineSoak: Bool { source != SoakSample.harnessSource }

        /// Reads the windows out of a soak artifact or a telemetry export.
        /// Lines that are not windows (a metric line, a request record)
        /// are skipped; a window line that does not decode is an error.
        public static func load(_ url: URL) throws -> [SoakSample] {
            let text = try String(contentsOf: url, encoding: .utf8)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            var samples: [SoakSample] = []
            for line in text.split(separator: "\n") where !line.isEmpty {
                guard let data = try JSONSerialization.jsonObject(with: Data(line.utf8))
                        as? [String: Any] else {
                    throw SoakLogError.malformedLine(String(line))
                }
                if let kind = data["kind"] as? String, kind != windowKind { continue }
                guard data["steps"] != nil else { continue }
                samples.append(try decoder.decode(SoakSample.self, from: Data(line.utf8)))
            }
            return samples
        }

        /// Appends this window to a JSONL file. This is the ONE place the
        /// soak artifact is written: the producer owns the schema.
        public func append(to url: URL) throws {
            let enc = JSONEncoder()
            enc.outputFormatting = .sortedKeys
            enc.dateEncodingStrategy = .iso8601
            var line = String(data: try enc.encode(self), encoding: .utf8) ?? "{}"
            line += "\n"
            if FileManager.default.fileExists(atPath: url.path) {
                // An existing file is appended to. It is never replaced:
                // dropping the windows already measured would be a quiet
                // loss of evidence.
                guard let handle = try? FileHandle(forWritingTo: url) else {
                    throw SoakLogError.unwritable(url.path)
                }
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: Data(line.utf8))
            } else {
                try Data(line.utf8).write(to: url)
            }
        }
    }

    /// Why a soak artifact could not be read or written.
    enum SoakLogError: Error, CustomStringConvertible {
        case malformedLine(String)
        case unwritable(String)
        public var description: String {
            switch self {
            case .malformedLine(let line): return "malformed soak line: \(line)"
            case .unwritable(let path): return "the soak log is not writable: \(path)"
            }
        }
    }
}

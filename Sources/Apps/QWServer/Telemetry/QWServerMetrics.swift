import Foundation
import Darwin
import QWEngine

/// The names the snapshot's provenance map uses.
public enum QWTelemetryMetric: String, Sendable, CaseIterable {
    case livePrefillTokensPerSec = "throughput.live_prefill_tokens_per_sec"
    case liveDecodeTokensPerSec = "throughput.live_decode_tokens_per_sec"
    case lastPrefillTokensPerSec = "throughput.last_prefill_tokens_per_sec"
    case lastDecodeTokensPerSec = "throughput.last_decode_tokens_per_sec"
    case ema1mPrefillTokensPerSec = "throughput.ema_1m_prefill_tokens_per_sec"
    case ema1mDecodeTokensPerSec = "throughput.ema_1m_decode_tokens_per_sec"
    case avgPrefillTokensPerSec = "throughput.avg_prefill_tokens_per_sec"
    case avgDecodeTokensPerSec = "throughput.avg_decode_tokens_per_sec"
    case displayPrefillTokensPerSec = "throughput.display_prefill_tokens_per_sec"
    case displayDecodeTokensPerSec = "throughput.display_decode_tokens_per_sec"
    case ttftMs = "latency.ttft_ms"
    case cpuPercent = "hardware.cpu_percent"
    case gpuPercent = "hardware.gpu_percent"
    case memoryPercent = "hardware.memory_percent"
    case memoryPressure = "hardware.memory_pressure"
    case gpuBusyRatio = "gpu.busy_ratio"
    /// The last turn's prompt was served entirely from the prefix
    /// cache: the engine ran no prefill phase, so it has no prefill speed.
    case prefillFromCache = "throughput.prefill_from_cache"
}

/// How long a reading stays current. Past its bound a reading is stale: it
/// was measured, but it no longer describes the present.
public enum QWTelemetryFreshness {
    /// A host reading is refreshed every couple of seconds.
    public static let hardwareStaleAfter: TimeInterval = 10
    /// A rate measured this long ago is no longer the engine's speed.
    public static let rateStaleAfter: TimeInterval = 30
    /// The engine's busy ratio describes its last measured step.
    public static let busyRatioStaleAfter: TimeInterval = 30
    /// The cache changes only when a cached request runs, so it stays
    /// current far longer than a host reading.
    public static let cacheStaleAfter: TimeInterval = 300
    /// A latency describes the turn it came from.
    public static let latencyStaleAfter: TimeInterval = 60
}

/// What a displayed rate is, so the UI can label it.
public enum QWRateDisplayMode: String, Sendable, Codable, CaseIterable {
    /// The current phase's live rate.
    case live
    /// The average over every completed turn.
    case average = "avg"
    /// The most recent measurement, still current.
    case last
    /// The most recent measurement, too old to present as current.
    case stale

    /// A payload may name a mode this build does not know. The honest
    /// default is the least committal one.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = QWRateDisplayMode(rawValue: raw) ?? .last
    }
}

/// Thread-safe registry for engine performance, cache telemetry, and throughput metrics.
public final class QWEngineTelemetryRegistry: @unchecked Sendable {
    public static let shared = QWEngineTelemetryRegistry()

    public enum EngineInferencePhase: String, Codable, Sendable {
        case idle = "idle"
        case prefill = "prefill"
        case decode = "decode"
    }

    public struct MetricsSnapshot: Codable, Sendable {
        public let cache: CacheMetrics
        public let throughput: ThroughputMetrics
        public let latency: LatencyMetrics
        public let context: ContextMetrics
        public let memory: MemoryMetrics
        public let hardware: QWSystemHardwareUtilization?
        /// The engine's own GPU busy ratio for its last measured step.
        public let gpuBusyRatio: Double?
        /// How each metric was obtained. A metric that is not listed was
        /// never measured, and the UI says so instead of showing a zero.
        public let provenance: [String: QWTelemetryProvenance]
        /// When each metric was taken. A cached metric says how old it is.
        public let sampledAt: [String: Date]
        /// Why a metric is unavailable or stale, when it is.
        public let unavailableReasons: [String: String]

        public struct CacheMetrics: Codable, Sendable {
            /// The warm cache's bytes. Nil means the engine cannot
            /// measure them — absent, never zero.
            public let warmBytes: Int?
            public let warmLimitBytes: Int
            /// The SSD cache's bytes. Nil means the engine cannot measure
            /// them — absent, never zero.
            public let ssdBytes: Int?
            public let ssdLimitBytes: Int
            public let lookupsTotal: Int
            public let hitsWarmTotal: Int
            public let hitsSSDTotal: Int
            /// Hits whose tier this build does not name. The tokens were
            /// reused; the tier is not in the vocabulary. Nil means this
            /// server does not report it — absent, never zero.
            public let hitsUnclassifiedTotal: Int?
            public let missesTotal: Int
            public let tokensReusedTotal: Int
            public let evictionsWarmTotal: Int
            public let evictionsSSDTotal: Int
            /// When the cache was last written. Nil means it never was.
            public var sampledAt: Date? = nil
        }

        public struct ThroughputMetrics: Codable, Sendable {
            public let phase: String
            public let tokensInTotal: Int
            public let tokensOutTotal: Int
            /// The live rates. Nil means "not measured", never zero.
            public let livePrefillTokensPerSec: Double?
            public let liveDecodeTokensPerSec: Double?
            public let lastPrefillTokensPerSec: Double?
            public let lastDecodeTokensPerSec: Double?
            public let ema1mPrefillTokensPerSec: Double?
            public let ema1mDecodeTokensPerSec: Double?
            /// Overall averages across every completed turn since start (token-weighted:
            /// computed tokens / time spent computing them). Prefill counts ONLY tokens the
            /// engine actually computed — tokens served from the prefix cache are excluded.
            public let avgPrefillTokensPerSec: Double?
            public let avgDecodeTokensPerSec: Double?
            /// What a UI should show right now. The mode says which number it is.
            public let displayPrefillTokensPerSec: Double?
            public let displayDecodeTokensPerSec: Double?
            public let prefillDisplayMode: QWRateDisplayMode
            public let decodeDisplayMode: QWRateDisplayMode
            /// The published trace of the prefill rate. A nil is a gap.
            public var prefillRateSeries: [Double?] = []
            /// The published trace of the decode rate. A nil is a gap.
            public var decodeRateSeries: [Double?] = []
            /// Progress of the turn in flight: the tokens the engine had to compute.
            public let prefillProgressTokens: Int
            public let prefillTotalTokens: Int
            /// True when the last turn's prompt was served entirely from
            /// the prefix cache: the engine computed no prefill tokens,
            /// so that turn had no prefill speed at all.
            public let prefillFromCache: Bool

            public init(phase: String, tokensInTotal: Int, tokensOutTotal: Int,
                        livePrefillTokensPerSec: Double?, liveDecodeTokensPerSec: Double?,
                        lastPrefillTokensPerSec: Double?, lastDecodeTokensPerSec: Double?,
                        ema1mPrefillTokensPerSec: Double?, ema1mDecodeTokensPerSec: Double?,
                        avgPrefillTokensPerSec: Double?, avgDecodeTokensPerSec: Double?,
                        displayPrefillTokensPerSec: Double?, displayDecodeTokensPerSec: Double?,
                        prefillDisplayMode: QWRateDisplayMode, decodeDisplayMode: QWRateDisplayMode,
                        prefillProgressTokens: Int, prefillTotalTokens: Int,
                        prefillRateSeries: [Double?] = [], decodeRateSeries: [Double?] = [],
                        prefillFromCache: Bool = false) {
                self.phase = phase
                self.tokensInTotal = tokensInTotal
                self.tokensOutTotal = tokensOutTotal
                self.livePrefillTokensPerSec = livePrefillTokensPerSec
                self.liveDecodeTokensPerSec = liveDecodeTokensPerSec
                self.lastPrefillTokensPerSec = lastPrefillTokensPerSec
                self.lastDecodeTokensPerSec = lastDecodeTokensPerSec
                self.ema1mPrefillTokensPerSec = ema1mPrefillTokensPerSec
                self.ema1mDecodeTokensPerSec = ema1mDecodeTokensPerSec
                self.avgPrefillTokensPerSec = avgPrefillTokensPerSec
                self.avgDecodeTokensPerSec = avgDecodeTokensPerSec
                self.displayPrefillTokensPerSec = displayPrefillTokensPerSec
                self.displayDecodeTokensPerSec = displayDecodeTokensPerSec
                self.prefillDisplayMode = prefillDisplayMode
                self.decodeDisplayMode = decodeDisplayMode
                self.prefillProgressTokens = prefillProgressTokens
                self.prefillTotalTokens = prefillTotalTokens
                self.prefillFromCache = prefillFromCache
                self.prefillRateSeries = prefillRateSeries
                self.decodeRateSeries = decodeRateSeries
            }

            /// Decodes a payload from an older server: a field that is not
            /// there was never measured, not zero.
            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                self.init(
                    phase: try c.decodeIfPresent(String.self, forKey: .phase) ?? "idle",
                    tokensInTotal: try c.decodeIfPresent(Int.self, forKey: .tokensInTotal) ?? 0,
                    tokensOutTotal: try c.decodeIfPresent(Int.self, forKey: .tokensOutTotal) ?? 0,
                    livePrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .livePrefillTokensPerSec),
                    liveDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .liveDecodeTokensPerSec),
                    lastPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .lastPrefillTokensPerSec),
                    lastDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .lastDecodeTokensPerSec),
                    ema1mPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .ema1mPrefillTokensPerSec),
                    ema1mDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .ema1mDecodeTokensPerSec),
                    avgPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .avgPrefillTokensPerSec),
                    avgDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .avgDecodeTokensPerSec),
                    displayPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .displayPrefillTokensPerSec),
                    displayDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .displayDecodeTokensPerSec),
                    prefillDisplayMode: try c.decodeIfPresent(QWRateDisplayMode.self, forKey: .prefillDisplayMode) ?? .last,
                    decodeDisplayMode: try c.decodeIfPresent(QWRateDisplayMode.self, forKey: .decodeDisplayMode) ?? .last,
                    prefillProgressTokens: try c.decodeIfPresent(Int.self, forKey: .prefillProgressTokens) ?? 0,
                    prefillTotalTokens: try c.decodeIfPresent(Int.self, forKey: .prefillTotalTokens) ?? 0,
                    prefillRateSeries: try c.decodeIfPresent([Double?].self, forKey: .prefillRateSeries) ?? [],
                    decodeRateSeries: try c.decodeIfPresent([Double?].self, forKey: .decodeRateSeries) ?? [],
                    prefillFromCache: try c.decodeIfPresent(Bool.self, forKey: .prefillFromCache) ?? false)
            }
        }

        public struct LatencyMetrics: Codable, Sendable {
            public let lastTtftMs: Double?
            public let avgTtftMs: Double?
        }

        public struct ContextMetrics: Codable, Sendable {
            /// The tokens the last turn used. Nil means no turn has run:
            /// absent, never zero.
            public var activeTokens: Int?
            public let maxTokens: Int
            /// When the context was last updated. Nil means never.
            public var sampledAt: Date? = nil
        }

        public struct MemoryMetrics: Codable, Sendable {
            public let residentBytes: Int
            public let arenaReservedBytes: Int
        }

        public init(cache: CacheMetrics, throughput: ThroughputMetrics, latency: LatencyMetrics,
                    context: ContextMetrics, memory: MemoryMetrics,
                    hardware: QWSystemHardwareUtilization?, gpuBusyRatio: Double?,
                    provenance: [String: QWTelemetryProvenance], sampledAt: [String: Date],
                    unavailableReasons: [String: String]) {
            self.cache = cache
            self.throughput = throughput
            self.latency = latency
            self.context = context
            self.memory = memory
            self.hardware = hardware
            self.gpuBusyRatio = gpuBusyRatio
            self.provenance = provenance
            self.sampledAt = sampledAt
            self.unavailableReasons = unavailableReasons
        }

        /// Decodes a payload from an older server: a field that is not there
        /// is unavailable, not a zero that reads like an idle engine.
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                cache: try c.decodeIfPresent(CacheMetrics.self, forKey: .cache)
                    ?? .init(warmBytes: 0, warmLimitBytes: 0, ssdBytes: 0, ssdLimitBytes: 0,
                             lookupsTotal: 0, hitsWarmTotal: 0, hitsSSDTotal: 0,
                             hitsUnclassifiedTotal: nil, missesTotal: 0,
                             tokensReusedTotal: 0, evictionsWarmTotal: 0, evictionsSSDTotal: 0),
                throughput: try c.decodeIfPresent(ThroughputMetrics.self, forKey: .throughput)
                    ?? .init(phase: "idle", tokensInTotal: 0, tokensOutTotal: 0,
                             livePrefillTokensPerSec: nil, liveDecodeTokensPerSec: nil,
                             lastPrefillTokensPerSec: nil, lastDecodeTokensPerSec: nil,
                             ema1mPrefillTokensPerSec: nil, ema1mDecodeTokensPerSec: nil,
                             avgPrefillTokensPerSec: nil, avgDecodeTokensPerSec: nil,
                             displayPrefillTokensPerSec: nil, displayDecodeTokensPerSec: nil,
                             prefillDisplayMode: .last, decodeDisplayMode: .last,
                             prefillProgressTokens: 0, prefillTotalTokens: 0),
                latency: try c.decodeIfPresent(LatencyMetrics.self, forKey: .latency)
                    ?? .init(lastTtftMs: nil, avgTtftMs: nil),
                context: try c.decodeIfPresent(ContextMetrics.self, forKey: .context)
                    ?? .init(activeTokens: 0, maxTokens: 0),
                memory: try c.decodeIfPresent(MemoryMetrics.self, forKey: .memory)
                    ?? .init(residentBytes: 0, arenaReservedBytes: 0),
                hardware: try c.decodeIfPresent(QWSystemHardwareUtilization.self, forKey: .hardware),
                gpuBusyRatio: try c.decodeIfPresent(Double.self, forKey: .gpuBusyRatio),
                provenance: try c.decodeIfPresent([String: QWTelemetryProvenance].self, forKey: .provenance) ?? [:],
                sampledAt: try c.decodeIfPresent([String: Date].self, forKey: .sampledAt) ?? [:],
                unavailableReasons: try c.decodeIfPresent([String: String].self, forKey: .unavailableReasons) ?? [:])
        }
    }

    private let lock = NSLock()

    // Cache Counters
    /// The cache bytes the engine reported. Nil means the engine cannot
    /// measure them: absent, never zero.
    private var warmCacheBytes: Int?
    private var warmCacheLimitBytes: Int = 0
    private var ssdCacheBytes: Int?
    private var ssdCacheLimitBytes: Int = 0
    private var cacheLookupsTotal: Int = 0
    private var cacheHitsWarmTotal: Int = 0
    private var cacheHitsSSDTotal: Int = 0
    private var cacheHitsUnclassifiedTotal: Int = 0
    private var cacheMissesTotal: Int = 0
    private var cacheTokensReusedTotal: Int = 0
    private var cacheEvictionsWarmTotal: Int = 0
    private var cacheEvictionsSSDTotal: Int = 0
    /// When the cache was last written.
    private var cacheSampledAt: Date?

    // Token Throughput & Phase Trackers
    private var phase: EngineInferencePhase = .idle
    /// The decode rate the engine's own inter-token latencies give:
    /// 1000 / p50(ITL) over the last `itlWindow` tokens, coalescing burst emissions
    /// into their round's per-token latency in a bounded FIFO window.
    private var itlDecodeTokPerSec: Double = 0.0
    /// FIFO ring buffer of the last `itlWindow` per-token latencies in milliseconds.
    private var itlFifo: [Double] = []
    private var itlFifoHead: Int = 0
    private var currentRoundMs: Double = 0.0
    private var currentRoundTokens: Int = 0
    private static let speculativeBurstThresholdMs: Double = 0.5
    /// The last `itlWindow` inter-token latencies in milliseconds, kept sorted so the
    /// median needs no copy.
    private var itlSorted: [Double] = []
    /// When the last token arrived, for the next inter-token latency.
    private var lastTokenAt: Date?

    private var tokensInTotal: Int = 0
    private var tokensOutTotal: Int = 0
    private var lastPrefillTokPerSec: Double = 0
    private var lastDecodeTokPerSec: Double = 0
    /// When the last prefill rate was measured.
    private var lastPrefillSampledAt: Date?
    /// The published trace of the two rates. A renderer draws this; it
    /// does not keep its own.
    private var prefillRateSeries: [Double?] = []
    private var decodeRateSeries: [Double?] = []
    /// When the last decode rate was measured.
    private var lastDecodeSampledAt: Date?
    private var ema1mPrefillTokPerSec: Double = 0
    private var ema1mDecodeTokPerSec: Double = 0

    // Per-turn prefill progress: the tokens the engine must compute this turn,
    // excluding tokens served from the prefix cache.
    private var prefillComputedDone = 0
    private var prefillComputedTotal = 0
    /// Whether the last completed turn's prompt came entirely from the
    /// prefix cache. Such a turn ran no prefill phase at all.
    private var lastTurnPrefillFromCache = false

    // Cumulative counters for the idle-time overall averages.
    private var totalPrefillComputedTokens: Double = 0
    private var totalPrefillSeconds: Double = 0
    private var totalDecodeTokens: Double = 0
    private var totalDecodeSeconds: Double = 0
    /// When the last completed turn finished; the averages describe the work
    /// done up to then.
    private var lastTurnSampledAt: Date?

    // Latency Trackers
    private var lastTtftMs: Double = 0
    private var totalTtftMs: Double = 0
    private var totalRequestsWithTtft: Int = 0
    private var lastTtftSampledAt: Date?

    // Context & Memory
    /// The tokens the last turn used. Nil means no turn has run: absent,
    /// never zero.
    private var contextTokensActive: Int?
    private var contextTokensMax: Int = 4096
    private var contextSampledAt: Date?
    private var arenaReservedBytes: Int = 0

    /// How long a rate stays current. Past this the display says the
    /// measurement is stale instead of presenting it as the engine's speed.
    private let rateStaleAfter: TimeInterval

    public init(rateStaleAfter: TimeInterval = QWTelemetryFreshness.rateStaleAfter) {
        self.rateStaleAfter = rateStaleAfter
        itlFifo.reserveCapacity(Self.itlWindow)
        itlSorted.reserveCapacity(Self.itlWindow)
    }

    public func recordCacheLookup(hitTier: String?, matchedTokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        cacheLookupsTotal += 1
        guard let tier = hitTier else {
            cacheMissesTotal += 1
            return
        }
        // A tier this build does not name is still a hit: the cache
        // served the prompt. It is counted as a hit of an unnamed
        // tier — never as a miss, which would be a claim the
        // measurement does not support.
        if tier == "warm_ram" {
            cacheHitsWarmTotal += 1
        } else if tier == "cold_ssd" || tier == "ssd" {
            cacheHitsSSDTotal += 1
        } else {
            cacheHitsUnclassifiedTotal += 1
        }
        cacheTokensReusedTotal += matchedTokens
    }

    public func recordWarmEviction() {
        lock.lock()
        defer { lock.unlock() }
        cacheEvictionsWarmTotal += 1
    }

    public func recordSSDEviction() {
        lock.lock()
        defer { lock.unlock() }
        cacheEvictionsSSDTotal += 1
    }

    public func updateCacheUsage(warmBytes: Int?, warmLimit: Int, ssdBytes: Int?, ssdLimit: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.warmCacheBytes = warmBytes
        self.warmCacheLimitBytes = warmLimit
        self.ssdCacheBytes = ssdBytes
        self.ssdCacheLimitBytes = ssdLimit
        // When the cache was last measured: a card that shows a size must
        // say how old the size is. A report with no bytes in it measured
        // nothing, so there is no moment to remember.
        self.cacheSampledAt = (warmBytes == nil && ssdBytes == nil) ? nil : Date()
    }

    // MARK: - Live Phase Transitions

    public func beginPrefill(promptTokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        phase = .prefill
        itlDecodeTokPerSec = 0.0
        itlFifo.removeAll(keepingCapacity: true)
        itlFifoHead = 0
        currentRoundMs = 0.0
        currentRoundTokens = 0
        itlSorted.removeAll(keepingCapacity: true)
        lastTokenAt = nil
        prefillComputedDone = 0
        prefillComputedTotal = promptTokens
    }

    /// The decode phase has begun. The prefill's own duration is not known
    /// here: the time to the first token is not the prefill's duration, and
    /// a fully cached prompt has no prefill phase at all.
    public func beginDecode(promptTokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        phase = .decode
        prefillComputedDone = promptTokens
        prefillComputedTotal = promptTokens
        itlDecodeTokPerSec = 0.0
        itlFifo.removeAll(keepingCapacity: true)
        itlFifoHead = 0
        currentRoundMs = 0.0
        currentRoundTokens = 0
        itlSorted.removeAll(keepingCapacity: true)
        // The first token has arrived, so the gap to the next one is a
        // measurement: the rate exists once the turn has two tokens.
        lastTokenAt = Date()
    }

    /// Resets the inter-token clock after synchronous post-prefill work (such as saving
    /// the prefix cache after emitting the first token) so cache I/O is not billed to decode ITL.
    public func resetDecodeStepClock() {
        lock.lock()
        defer { lock.unlock() }
        if phase == .decode {
            lastTokenAt = Date()
        }
    }

    /// `at` exists for tests only: macOS Thread.sleep overshoots by 3-5x under
    /// scheduler load, so timing behavior is verified with injected timestamps
    /// instead of wall-clock sleeps. Production always passes nil.
    public func recordDecodeToken(at date: Date? = nil) {
        lock.lock()
        defer { lock.unlock() }
        let now = date ?? Date()
        if let last = lastTokenAt {
            let latency = now.timeIntervalSince(last) * 1000.0
            if latency > 0 {
                let perTokenMs: Double
                if latency < Self.speculativeBurstThresholdMs && currentRoundTokens > 0 && !itlFifo.isEmpty {
                    currentRoundMs += latency
                    currentRoundTokens += 1
                    perTokenMs = currentRoundMs / Double(currentRoundTokens)
                    let updateCount = min(currentRoundTokens - 1, itlFifo.count)
                    for offset in 1...updateCount {
                        let idx = (itlFifoHead - offset + itlFifo.count) % itlFifo.count
                        let oldVal = itlFifo[idx]
                        itlFifo[idx] = perTokenMs
                        Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow, evicting: oldVal)
                    }
                } else {
                    currentRoundMs = latency
                    currentRoundTokens = 1
                    perTokenMs = latency
                }
                if itlFifo.count < Self.itlWindow {
                    itlFifo.append(perTokenMs)
                    itlFifoHead = itlFifo.count % Self.itlWindow
                    Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow)
                } else {
                    let evicted = itlFifo[itlFifoHead]
                    itlFifo[itlFifoHead] = perTokenMs
                    itlFifoHead = (itlFifoHead + 1) % Self.itlWindow
                    Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow, evicting: evicted)
                }
                if !itlSorted.isEmpty {
                    itlDecodeTokPerSec = 1000.0 / itlSorted[itlSorted.count / 2]
                }
            }
        }
        lastTokenAt = now
        if itlDecodeTokPerSec > 0 {
            lastDecodeSampledAt = now
        }
    }

    /// How many inter-token latencies the live rate looks at.
    public static let itlWindow = 32

    static func removeSorted(_ value: Double, from samples: inout [Double]) {
        guard !samples.isEmpty else { return }
        var low = 0
        var high = samples.count
        while low < high {
            let mid = (low + high) / 2
            if samples[mid] < value { low = mid + 1 } else { high = mid }
        }
        if low < samples.count {
            samples.remove(at: low)
        } else {
            samples.removeLast()
        }
    }

    /// Inserts a sample keeping the array sorted, and drops the oldest when
    /// the window is full. The buffer never grows past its reserved
    /// capacity, so this allocates nothing in the steady state.
    static func insertSorted(_ value: Double, into samples: inout [Double], capacity: Int, evicting evicted: Double? = nil) {
        if let evicted {
            removeSorted(evicted, from: &samples)
        } else if samples.count >= capacity {
            samples.removeFirst()
        }
        var low = 0
        var high = samples.count
        while low < high {
            let mid = (low + high) / 2
            if samples[mid] <= value { low = mid + 1 } else { high = mid }
        }
        samples.insert(value, at: low)
    }

    public func endRequest(tokensIn: Int, tokensOut: Int, cachedTokens: Int = 0,
                           prefillSec: Double?, decodeSec: Double?, ttftSec: Double?) {
        lock.lock()
        defer { lock.unlock() }
        phase = .idle
        itlDecodeTokPerSec = 0.0
        itlFifo.removeAll(keepingCapacity: true)
        itlFifoHead = 0
        currentRoundMs = 0.0
        currentRoundTokens = 0
        itlSorted.removeAll(keepingCapacity: true)
        lastTokenAt = nil
        prefillComputedDone = 0
        prefillComputedTotal = 0

        tokensInTotal += tokensIn
        tokensOutTotal += tokensOut

        // Prefill speed is computed tokens / time; tokens served from the cache cost
        // nothing and would otherwise inflate the rate by 10-50x on a warm turn.
        let computed = max(0, tokensIn - cachedTokens)
        // A prompt the cache served entirely had no prefill phase at all:
        // the engine computed nothing, so it has no prefill speed. The
        // card says where the prompt came from instead of a dash.
        lastTurnPrefillFromCache = tokensIn > 0 && computed == 0
        if let prefillSec, prefillSec > 0 && computed > 0 {
            let instPrefill = Double(computed) / prefillSec
            lastPrefillTokPerSec = instPrefill
            totalPrefillComputedTokens += Double(computed)
            totalPrefillSeconds += prefillSec
            if ema1mPrefillTokPerSec == 0 {
                ema1mPrefillTokPerSec = instPrefill
            } else {
                ema1mPrefillTokPerSec = 0.2 * instPrefill + 0.8 * ema1mPrefillTokPerSec
            }
            lastPrefillSampledAt = Date()
            Self.append(lastPrefillTokPerSec, to: &prefillRateSeries)
        } else {
            // The turn measured no prefill rate. The trace breaks here:
            // a gap, never a flat line at the last known value.
            Self.append(nil, to: &prefillRateSeries)
        }
        // The first generated token is produced by prefill, so decode throughput covers
        // the remaining tokens only.
        let decoded = tokensOut > 1 ? tokensOut - 1 : tokensOut
        if let decodeSec, decodeSec > 0 && decoded > 0 {
            let instDecode = Double(decoded) / decodeSec
            lastDecodeTokPerSec = instDecode
            totalDecodeTokens += Double(decoded)
            totalDecodeSeconds += decodeSec
            if ema1mDecodeTokPerSec == 0 {
                ema1mDecodeTokPerSec = instDecode
            } else {
                ema1mDecodeTokPerSec = 0.2 * instDecode + 0.8 * ema1mDecodeTokPerSec
            }
            lastDecodeSampledAt = Date()
            Self.append(lastDecodeTokPerSec, to: &decodeRateSeries)
        } else {
            // The turn measured no decode rate: the trace breaks here.
            Self.append(nil, to: &decodeRateSeries)
        }
        // The time to the first token, when the engine reported one. A
        // request that produced no token has no time to first token.
        if let ttftSec {
            let ttft = ttftSec * 1000.0
            lastTtftMs = ttft
            totalTtftMs += ttft
            totalRequestsWithTtft += 1
            lastTtftSampledAt = Date()
        }
        lastTurnSampledAt = Date()
    }

    public func recordGeneration(tokensIn: Int, tokensOut: Int, cachedTokens: Int = 0,
                                 prefillSec: Double, decodeSec: Double, ttftSec: Double?) {
        endRequest(tokensIn: tokensIn, tokensOut: tokensOut, cachedTokens: cachedTokens,
                   prefillSec: prefillSec, decodeSec: decodeSec, ttftSec: ttftSec)
    }

    /// Keeps a bounded trace: a gap is a nil, never a zero.
    private static func append(_ value: Double?, to series: inout [Double?]) {
        series.append(value)
        if series.count > 60 { series.removeFirst(series.count - 60) }
    }

    public func resetToIdle() {
        lock.lock()
        defer { lock.unlock() }
        phase = .idle
        // The cache counters belong to the model that filled the cache: a
        // new model starts with its own counters.
        cacheLookupsTotal = 0
        cacheHitsWarmTotal = 0
        cacheHitsSSDTotal = 0
        cacheHitsUnclassifiedTotal = 0
        cacheMissesTotal = 0
        cacheTokensReusedTotal = 0
        cacheEvictionsWarmTotal = 0
        cacheEvictionsSSDTotal = 0
        itlDecodeTokPerSec = 0.0
        itlFifo.removeAll(keepingCapacity: true)
        itlFifoHead = 0
        currentRoundMs = 0.0
        currentRoundTokens = 0
        itlSorted.removeAll(keepingCapacity: true)
        lastTokenAt = nil
        prefillComputedDone = 0
        prefillComputedTotal = 0
        // The engine is gone: the trace described a model that is no
        // longer resident, so it goes with it. The last rate stays: it
        // is a reading, and a reading says how old it is.
        prefillRateSeries.removeAll()
        decodeRateSeries.removeAll()
        // The state belongs to the model that ran the turn: a new model
        // starts with its own.
        lastTurnPrefillFromCache = false
    }

    public func updateContext(active: Int?, capacity: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.contextTokensActive = active
        self.contextTokensMax = capacity
        // A session with no turn in it measured nothing: there is no
        // moment to remember.
        self.contextSampledAt = active == nil ? nil : Date()
    }

    public func updateArena(reserved: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.arenaReservedBytes = reserved
    }

    public static func getResidentMemoryBytes() -> Int {
        var vmInfo = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmErr = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, mach_task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }
        if vmErr == KERN_SUCCESS && vmInfo.phys_footprint > 0 {
            return Int(vmInfo.phys_footprint)
        }
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, mach_task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    public func snapshot() -> MetricsSnapshot {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        let hw = QWSystemMetricsSampler.shared.current()
        let busy = QWTelemetryRegistry.shared.value("gpu.busy_ratio")

        // The live rates. The prefill's own rate is only known when the
        // engine reports the phase's duration, so there is none to show.
        let livePrefill: Double? = nil
        let liveDecode: Double? = phase == .decode && itlDecodeTokPerSec > 0 ? itlDecodeTokPerSec : nil
        let lastPrefill: Double? = lastPrefillTokPerSec > 0 ? lastPrefillTokPerSec : nil
        let lastDecode: Double? = lastDecodeTokPerSec > 0 ? lastDecodeTokPerSec : nil
        let emaPrefill: Double? = ema1mPrefillTokPerSec > 0 ? ema1mPrefillTokPerSec : nil
        let emaDecode: Double? = ema1mDecodeTokPerSec > 0 ? ema1mDecodeTokPerSec : nil
        let avgPrefill: Double? = totalPrefillSeconds > 0 ? totalPrefillComputedTokens / totalPrefillSeconds : nil
        let avgDecode: Double? = totalDecodeSeconds > 0 ? totalDecodeTokens / totalDecodeSeconds : nil

        // What the UI shows: the live rate when there is one, the overall
        // average when the engine is idle, and the last measurement — marked
        // stale once it is too old to describe the present — otherwise.
        let (displayPrefill, prefillMode) = Self.displayRate(
            live: livePrefill, average: phase == .idle ? avgPrefill : nil, last: lastPrefill,
            sampledAt: lastPrefillSampledAt, now: now, staleAfter: rateStaleAfter)
        let (displayDecode, decodeMode) = Self.displayRate(
            live: liveDecode, average: phase == .idle ? avgDecode : nil, last: lastDecode,
            sampledAt: lastDecodeSampledAt, now: now, staleAfter: rateStaleAfter)

        let hwAge = now.timeIntervalSince(hw.sampledAt)
        let hwStale = hwAge > QWTelemetryFreshness.hardwareStaleAfter
        let busyRatio = busy?.value
        let busyStale = busy.map { $0.isStale(now: now, bound: QWTelemetryFreshness.busyRatioStaleAfter) } ?? false

        return MetricsSnapshot(
            cache: .init(
                warmBytes: warmCacheBytes, warmLimitBytes: warmCacheLimitBytes,
                ssdBytes: ssdCacheBytes, ssdLimitBytes: ssdCacheLimitBytes,
                lookupsTotal: cacheLookupsTotal, hitsWarmTotal: cacheHitsWarmTotal,
                hitsSSDTotal: cacheHitsSSDTotal, hitsUnclassifiedTotal: cacheHitsUnclassifiedTotal,
                missesTotal: cacheMissesTotal, tokensReusedTotal: cacheTokensReusedTotal,
                evictionsWarmTotal: cacheEvictionsWarmTotal, evictionsSSDTotal: cacheEvictionsSSDTotal,
                sampledAt: cacheSampledAt),
            throughput: .init(
                phase: phase.rawValue, tokensInTotal: tokensInTotal, tokensOutTotal: tokensOutTotal,
                livePrefillTokensPerSec: livePrefill, liveDecodeTokensPerSec: liveDecode,
                lastPrefillTokensPerSec: lastPrefill, lastDecodeTokensPerSec: lastDecode,
                ema1mPrefillTokensPerSec: emaPrefill, ema1mDecodeTokensPerSec: emaDecode,
                avgPrefillTokensPerSec: avgPrefill, avgDecodeTokensPerSec: avgDecode,
                displayPrefillTokensPerSec: displayPrefill, displayDecodeTokensPerSec: displayDecode,
                prefillDisplayMode: prefillMode, decodeDisplayMode: decodeMode,
                prefillProgressTokens: prefillComputedDone, prefillTotalTokens: prefillComputedTotal,
                prefillRateSeries: prefillRateSeries, decodeRateSeries: decodeRateSeries,
                prefillFromCache: lastTurnPrefillFromCache),
            latency: .init(
                lastTtftMs: totalRequestsWithTtft > 0 ? lastTtftMs : nil,
                avgTtftMs: totalRequestsWithTtft > 0 ? totalTtftMs / Double(totalRequestsWithTtft) : nil),
            context: .init(
                activeTokens: contextTokensActive, maxTokens: contextTokensMax, sampledAt: contextSampledAt),
            memory: .init(
                residentBytes: Self.getResidentMemoryBytes(), arenaReservedBytes: arenaReservedBytes),
            hardware: hw, gpuBusyRatio: busyRatio,
            provenance: Self.provenanceMap(
                livePrefill: livePrefill, liveDecode: liveDecode,
                lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                prefillMode: prefillMode, decodeMode: decodeMode,
                ttft: totalRequestsWithTtft > 0, hw: hw, hwStale: hwStale,
                busyRatio: busyRatio, busyStale: busyStale,
                prefillFromCache: lastTurnPrefillFromCache),
            sampledAt: Self.sampledAtMap(
                liveDecode: liveDecode, lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                ttft: totalRequestsWithTtft > 0, hw: hw, busy: busy,
                lastPrefillAt: lastPrefillSampledAt, lastDecodeAt: lastDecodeSampledAt,
                lastTurnAt: lastTurnSampledAt, lastTtftAt: lastTtftSampledAt,
                prefillFromCache: lastTurnPrefillFromCache),
            unavailableReasons: Self.reasonMap(
                livePrefill: livePrefill, liveDecode: liveDecode,
                lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                prefillMode: prefillMode, decodeMode: decodeMode,
                ttft: totalRequestsWithTtft > 0, hw: hw, hwStale: hwStale,
                busyRatio: busyRatio, busyStale: busyStale, busyReason: busy?.reason,
                prefillFromCache: lastTurnPrefillFromCache)
        )
    }

}

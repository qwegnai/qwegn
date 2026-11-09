// QWRequestRecord.swift — What one request cost.
//
// The record says what happened: the sizes, the timings, the stage
// breakdown, the outcome. It never says what was asked — no prompt text,
// no completion text, no token ids. A request record is a measurement,
// not a transcript.

import Foundation

/// What one request cost. The timings nobody measured are absent.
public struct QWRequestRecord: Sendable, Codable, Equatable {
    /// The request's own id. It names a turn, not a person.
    public var id: String
    /// When the request ended.
    public var time: Date
    /// The model that served the request.
    public var modelId: String
    /// How many prompt tokens the request carried.
    public var promptTokens: Int
    /// How many tokens came back.
    public var generatedTokens: Int
    /// How many of the prompt tokens the cache served.
    public var cachedTokens: Int
    /// Which tier served the prompt: "warm_ram", "cold_ssd". Absent
    /// when the cache served nothing.
    public var cacheHitTier: String?
    /// The time to the first token, in milliseconds.
    public var ttftMs: Double?
    /// The prefill wall time, in milliseconds.
    public var prefillMs: Double?
    /// The decode wall time, in milliseconds.
    public var decodeMs: Double?
    /// The decode rate the turn reached, in tokens per second.
    public var decodeTps: Double?
    /// The GPU time each stage spent, in milliseconds, by stage name.
    public var stageGPU: [String: Double]
    /// The configuration the turn ran under, echoed as names and values:
    /// the knobs that shaped it, not the request that asked.
    public var config: [String: String]
    /// How the request ended: "ok", or the reason it did not.
    public var outcome: String

    public init(id: String, time: Date, modelId: String, promptTokens: Int,
                generatedTokens: Int, cachedTokens: Int = 0,
                cacheHitTier: String? = nil, ttftMs: Double? = nil,
                prefillMs: Double? = nil, decodeMs: Double? = nil,
                decodeTps: Double? = nil, stageGPU: [String: Double] = [:],
                config: [String: String] = [:], outcome: String = "ok") {
        self.id = id
        self.time = time
        self.modelId = modelId
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.cachedTokens = cachedTokens
        self.cacheHitTier = cacheHitTier
        self.ttftMs = ttftMs
        self.prefillMs = prefillMs
        self.decodeMs = decodeMs
        self.decodeTps = decodeTps
        self.stageGPU = stageGPU
        self.config = config
        self.outcome = outcome
    }
}

/// The last requests, in a bounded ring. The ring holds the recent
/// turns and forgets the rest: a server that remembered every request
/// would be a transcript, not a measurement.
public final class QWRequestRecordStore: @unchecked Sendable {
    public static let shared = QWRequestRecordStore()

    /// How many requests the ring keeps.
    public static let capacity = 64

    private let lock = NSLock()
    private var records: [QWRequestRecord] = []
    private var counter: Int = 0

    public init() {}

    /// Records a request. The ring drops the oldest when it is full.
    public func append(_ record: QWRequestRecord) {
        lock.lock()
        defer { lock.unlock() }
        records.append(record)
        if records.count > QWRequestRecordStore.capacity {
            records.removeFirst(records.count - QWRequestRecordStore.capacity)
        }
    }

    /// The most recent records, oldest first.
    public func recent(_ count: Int) -> [QWRequestRecord] {
        lock.lock()
        defer { lock.unlock() }
        guard count > 0 else { return [] }
        return Array(records.suffix(count))
    }

    /// The next request id. The ids are a sequence, not a secret.
    public func nextId() -> String {
        lock.lock()
        defer { lock.unlock() }
        counter += 1
        return "req-\(counter)"
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        records.removeAll()
        counter = 0
    }
}

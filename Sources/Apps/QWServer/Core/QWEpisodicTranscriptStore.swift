// QWEpisodicTranscriptStore.swift — 0o600 JSONL episodic archive and BM25-style turn recovery (G3).

import Darwin
import Foundation
import QWCore

public struct QWEpisodicTurn: Codable, Sendable, Equatable {
    public let index: Int
    public let role: String
    public let text: String
    public let timestampEpochMs: Int64

    enum CodingKeys: String, CodingKey {
        case index, role, text
        case timestampEpochMs = "timestamp_epoch_ms"
    }

    public init(index: Int, role: String, text: String, timestampEpochMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)) {
        self.index = index
        self.role = role
        self.text = text
        self.timestampEpochMs = timestampEpochMs
    }
}

public struct QWEpisodicRecoveredTurn: Codable, Sendable, Equatable {
    public let index: Int
    public let role: String
    public let text: String
    public let timestampEpochMs: Int64
    public let score: Double

    enum CodingKeys: String, CodingKey {
        case index, role, text, score
        case timestampEpochMs = "timestamp_epoch_ms"
    }
}

public enum QWEpisodicStoreError: Error, Sendable, CustomStringConvertible {
    case invalidSessionId(String)
    case ioFailed(String)

    public var description: String {
        switch self {
        case .invalidSessionId(let id):
            return "invalid episodic session_id '\(id)': only ASCII alphanumeric, '-', and '_' are permitted"
        case .ioFailed(let msg):
            return "episodic store I/O failed: \(msg)"
        }
    }
}

/// Thread-safe JSONL episodic transcript archive with strict `0o700`/`0o600` file permissions.
public final class QWEpisodicTranscriptStore: @unchecked Sendable {
    public let directoryURL: URL
    private let lock = NSLock()
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(directoryURL: URL = QWPaths.rootDirectoryURL.appendingPathComponent("episodic", isDirectory: true)) {
        self.directoryURL = directoryURL.standardizedFileURL
    }

    public static func validateSessionId(_ raw: String) throws -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 128 else {
            throw QWEpisodicStoreError.invalidSessionId(raw)
        }
        let ok = trimmed.unicodeScalars.allSatisfy { sc in
            (sc.value >= 48 && sc.value <= 57) ||
            (sc.value >= 65 && sc.value <= 90) ||
            (sc.value >= 97 && sc.value <= 122) ||
            sc.value == 45 || sc.value == 95
        }
        guard ok else { throw QWEpisodicStoreError.invalidSessionId(raw) }
        return trimmed
    }

    public func fileURL(for sessionId: String) throws -> URL {
        let safeId = try Self.validateSessionId(sessionId)
        return directoryURL.appendingPathComponent("\(safeId).jsonl", isDirectory: false)
    }

    @discardableResult
    public func archive(sessionId: String, turns: [QWEpisodicTurn]) throws -> Int {
        guard !turns.isEmpty else { return 0 }
        let target = try fileURL(for: sessionId)
        return try lock.withLock {
            let fm = FileManager.default
            if !fm.fileExists(atPath: directoryURL.path) {
                try fm.createDirectory(
                    at: directoryURL,
                    withIntermediateDirectories: true,
                    attributes: [.posixPermissions: NSNumber(value: Int16(0o700))]
                )
            }
            chmod(directoryURL.path, mode_t(0o700))

            var payload = Data()
            for turn in turns {
                payload.append(try encoder.encode(turn))
                payload.append(0x0A)
            }
            if fm.fileExists(atPath: target.path) {
                let handle = try FileHandle(forWritingTo: target)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: payload)
            } else {
                try payload.write(to: target, options: .atomic)
            }
            chmod(target.path, mode_t(0o600))
            return turns.count
        }
    }

    public func recover(sessionId: String, query: String, limit: Int = 5) throws -> [QWEpisodicRecoveredTurn] {
        let target = try fileURL(for: sessionId)
        let rawData: Data? = lock.withLock {
            try? Data(contentsOf: target)
        }
        guard let rawData, !rawData.isEmpty else { return [] }
        let text = String(decoding: rawData, as: UTF8.self)
        var turns: [QWEpisodicTurn] = []
        for line in text.split(whereSeparator: \.isNewline) {
            if let turn = try? decoder.decode(QWEpisodicTurn.self, from: Data(line.utf8)) {
                turns.append(turn)
            }
        }
        guard !turns.isEmpty else { return [] }

        let qTrim = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let qTokens = Self.tokenize(qTrim)
        if qTokens.isEmpty && qTrim.isEmpty {
            return Array(turns.suffix(max(1, limit)).reversed().map {
                QWEpisodicRecoveredTurn(index: $0.index, role: $0.role, text: $0.text, timestampEpochMs: $0.timestampEpochMs, score: 1.0)
            })
        }

        // BM25-inspired TF-IDF scoring over archived turns + substring bonus.
        let docsTokens = turns.map { Self.tokenize($0.text) }
        let avgLen = max(1.0, Double(docsTokens.reduce(0) { $0 + $1.count }) / Double(docsTokens.count))
        var docFreq: [String: Int] = [:]
        for dt in docsTokens {
            for term in Set(dt) { docFreq[term, default: 0] += 1 }
        }
        let nDocs = Double(turns.count)
        let k1 = 1.2
        let b = 0.75

        var scored: [QWEpisodicRecoveredTurn] = []
        for (idx, turn) in turns.enumerated() {
            let dt = docsTokens[idx]
            var tf: [String: Int] = [:]
            for t in dt { tf[t, default: 0] += 1 }
            var bm25 = 0.0
            for qt in Set(qTokens) {
                let f = Double(tf[qt] ?? 0)
                guard f > 0 else { continue }
                let df = Double(docFreq[qt] ?? 0)
                let idf = log(1.0 + (nDocs - df + 0.5) / (df + 0.5))
                let denom = f + k1 * (1.0 - b + b * Double(dt.count) / avgLen)
                bm25 += idf * (f * (k1 + 1.0)) / max(1e-6, denom)
            }
            if !qTrim.isEmpty && turn.text.lowercased().contains(qTrim) {
                bm25 += 2.0
            }
            guard bm25 > 0 else { continue }
            scored.append(QWEpisodicRecoveredTurn(
                index: turn.index, role: turn.role, text: turn.text,
                timestampEpochMs: turn.timestampEpochMs, score: bm25
            ))
        }
        return Array(scored.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.index > $1.index
        }.prefix(max(1, limit)))
    }

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .filter { $0.count >= 2 }
            .map(String.init)
    }
}

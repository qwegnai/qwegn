// QwgTelemetry.swift — `qwg telemetry`: what the engine measured.
//
// The server owns the process that measured, so the report and the export
// come from the server over HTTP (GET /v1/dashboard): the per-request
// records live in the server's ring and nowhere else. A server that cannot
// be reached is an error, not an empty report — an empty report would read
// as "the engine measured nothing" when the truth is "nobody asked it".

import ArgumentParser
import Foundation
import QWEngine
import QWServer

/// The telemetry the server publishes, as one JSON object per line.
struct TelemetryLine: Codable {
    /// "metric" (a reading) or "request" (a per-request record).
    var kind: String
    var name: String?
    var value: Double?
    /// A reading that is a word rather than a number: "from cache".
    var text: String?
    var unit: String?
    var provenance: String?
    var sampledAt: Date?
    var age: Double?
    var isStale: Bool?
    var reason: String?
    var series: [Double?]?
    var id: String?
    var time: Date?
    var modelId: String?
    var promptTokens: Int?
    var generatedTokens: Int?
    var cachedTokens: Int?
    var cacheHitTier: String?
    var ttftMs: Double?
    var prefillMs: Double?
    var decodeMs: Double?
    var decodeTps: Double?
    var stageGPU: [String: Double]?
    var config: [String: String]?
    var outcome: String?
}

/// Why the server's payload could not be read.
enum TelemetryError: Error, CustomStringConvertible {
    case notPresent
    var description: String {
        switch self {
        case .notPresent: return "the payload carried no such array"
        }
    }
}

/// What the server published: the cards it measured and the requests it kept.
struct TelemetrySnapshot {
    var cards: [QWDashboardCard]
    var records: [QWRequestRecord]
    var sampledAt: Date?
    /// False when the dashboard did not decode and only the records were read.
    var cardsDecoded: Bool
}

struct Telemetry: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Report the telemetry the engine measured; --jsonl dumps it, --last N dumps the per-request records.")

    @Flag(name: .customLong("jsonl"), help: "Dump the telemetry as JSONL: one JSON object per line.") var jsonl: Bool = false
    @Option(name: .customLong("last"), help: "Dump the last N per-request records (the server's bounded ring).") var last: Int?
    @Option(name: .customLong("host"), help: "The server host (default: 127.0.0.1).") var host: String = "127.0.0.1"
    @Option(name: .customLong("port"), help: "The server port (default: 8080).") var port: Int = 8080
    @Option(name: .customLong("out"), help: "Write the dump to this file instead of stdout.") var out: String?

    func run() async throws {
        let snapshot = try await fetch()
        if let last {
            let kept = Array(snapshot.records.suffix(max(0, last)))
            try emit(kept.map { TelemetryLine.request($0) })
            return
        }
        if jsonl {
            var lines = snapshot.cards.map { TelemetryLine.metric($0) }
            for record in snapshot.records { lines.append(TelemetryLine.request(record)) }
            try emit(lines)
            return
        }
        try write(report(snapshot: snapshot))
    }

    /// What the server published. The records are the requirement: if they
    /// cannot be read the command fails, rather than reporting none.
    private func fetch() async throws -> TelemetrySnapshot {
        guard let url = URL(string: "http://\(host):\(port)/v1/dashboard") else {
            throw ExitCode(2)
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        let session = URLSession(configuration: configuration)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(from: url)
        } catch {
            FileHandle.standardError.write(Data(
                "qwg telemetry: the server at \(host):\(port) did not answer: \(error.localizedDescription)\n".utf8))
            throw ExitCode(1)
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            FileHandle.standardError.write(Data(
                "qwg telemetry: the server at \(host):\(port) answered \(code)\n".utf8))
            throw ExitCode(1)
        }
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let records: [QWRequestRecord]
        do {
            records = try decodeArray([QWRequestRecord].self, from: object["requests"])
        } catch {
            FileHandle.standardError.write(Data(
                "qwg telemetry: the server's request records did not decode: \(error)\n".utf8))
            throw ExitCode(1)
        }
        // The cards are the report; if the dashboard moved on, the records
        // still stand and the report says the metrics were not read.
        let cards = (try? decodeArray([QWDashboardCard].self, from: object["cards"])) ?? []
        let sampledAt = Self.decodeDate(object["sampledAt"])
        return TelemetrySnapshot(cards: cards, records: records, sampledAt: sampledAt,
                                 cardsDecoded: !cards.isEmpty)
    }

    /// Decodes one array out of the dashboard payload.
    private func decodeArray<T: Decodable>(_ type: [T].Type, from object: Any?) throws -> [T] {
        guard let object, JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object) else {
            throw TelemetryError.notPresent
        }
        return try Self.payloadDecoder().decode(type, from: data)
    }

    /// The decoder the server's own payload needs: the server writes its
    /// dates as ISO 8601 strings, and the default decoder reads a number.
    private static func payloadDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    /// When the server took the reading, when it said so.
    private static func decodeDate(_ object: Any?) -> Date? {
        guard let text = object as? String else { return nil }
        return ISO8601DateFormatter().date(from: text)
    }

    /// The human report: one line per metric, a dash where there is none.
    private func report(snapshot: TelemetrySnapshot) -> String {
        var out = "telemetry — \(host):\(port)"
        out += snapshot.sampledAt.map { " — sampled \($0)" } ?? ""
        out += "\n"
        if !snapshot.cardsDecoded {
            out += "the dashboard carried no readable metrics; the per-request records are shown below\n"
        }
        for card in snapshot.cards {
            let value = card.value ?? "—"
            let unit = card.unit.map { " \($0)" } ?? ""
            let age = card.age.map { String(format: "  age %.1fs", $0) } ?? ""
            let stale = card.isStale ? "  [stale]" : ""
            // A reading that is a word needs its reason: "from cache"
            // says nothing without the sentence that explains it. A
            // card with no reading at all says that it has none.
            var reason = ""
            if card.number == nil {
                if let why = card.reason { reason = "  (\(why))" }
                else if card.value == nil { reason = "  (not measured)" }
            }
            out += "\(card.id.padding(toLength: 46, withPad: " ", startingAt: 0)) "
            out += "\(value)\(unit)  [\(card.provenance.rawValue)]\(age)\(stale)\(reason)\n"
        }
        out += "requests: \(snapshot.records.count) recorded (ring capacity \(QWRequestRecordStore.capacity))\n"
        for line in snapshot.records.suffix(10) { out += recordText(TelemetryLine.request(line)) }
        return out
    }

    /// One request record as a line: what it cost, never what it said.
    private func recordText(_ line: TelemetryLine) -> String {
        let stages = (line.stageGPU ?? [:])
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\(String(format: "%.1f", $0.value))ms" }
            .joined(separator: " ")
        var text = "  \(line.id ?? "?") \(line.modelId ?? "?") "
        text += "in \(line.promptTokens ?? 0) tok out \(line.generatedTokens ?? 0) tok "
        text += "ttft \(line.ttftMs.map { String(format: "%.1f", $0) } ?? "—")ms "
        text += "prefill \(line.prefillMs.map { String(format: "%.1f", $0) } ?? "—")ms "
        text += "decode \(line.decodeMs.map { String(format: "%.1f", $0) } ?? "—")ms "
        text += "→ \(line.outcome ?? "?")"
        if !stages.isEmpty { text += "  gpu[\(stages)]" }
        text += "\n"
        return text
    }

    /// Writes the dump: JSONL when asked, the text report otherwise.
    private func emit(_ lines: [TelemetryLine]) throws {
        if jsonl {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            encoder.dateEncodingStrategy = .iso8601
            var text = ""
            for line in lines {
                text += String(data: try encoder.encode(line), encoding: .utf8) ?? "{}"
                text += "\n"
            }
            try write(text)
            return
        }
        var text = ""
        for line in lines { text += recordText(line) }
        try write(text)
    }

    /// Writes to the file the caller asked for, or to stdout.
    private func write(_ text: String) throws {
        guard let out else {
            FileHandle.standardOutput.write(Data(text.utf8))
            return
        }
        try Data(text.utf8).write(to: URL(fileURLWithPath: out))
        FileHandle.standardError.write(Data("qwg telemetry: wrote \(out)\n".utf8))
    }
}

extension TelemetryLine {
    /// The line for one dashboard card. A gap in the series stays a gap.
    static func metric(_ card: QWDashboardCard) -> TelemetryLine {
        TelemetryLine(kind: "metric", name: card.id, value: card.number,
                      // A reading that is a word is not a number: it goes
                      // in its own field, never as a zero.
                      text: card.number == nil ? card.value : nil, unit: card.unit,
                      provenance: card.provenance.rawValue, sampledAt: card.sampledAt,
                      age: card.age, isStale: card.isStale, reason: card.reason,
                      series: card.series.isEmpty ? nil : card.series)
    }

    /// The line for one per-request record. It carries no prompt text and
    /// no completion text: it is a measurement, not a transcript.
    static func request(_ record: QWRequestRecord) -> TelemetryLine {
        TelemetryLine(kind: "request", id: record.id, time: record.time, modelId: record.modelId,
                      promptTokens: record.promptTokens, generatedTokens: record.generatedTokens,
                      cachedTokens: record.cachedTokens, cacheHitTier: record.cacheHitTier,
                      ttftMs: record.ttftMs, prefillMs: record.prefillMs, decodeMs: record.decodeMs,
                      decodeTps: record.decodeTps, stageGPU: record.stageGPU,
                      config: record.config, outcome: record.outcome)
    }
}

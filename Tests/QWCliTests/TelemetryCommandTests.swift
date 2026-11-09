// TelemetryCommandTests.swift — `qwg telemetry`: the report and the dump.
//
// The rules this suite binds:
//  1. the subcommand exists with the flags the spec names,
//  2. the per-request record carries measurements and no prompt content,
//  3. a server that cannot be reached is an error, never an empty report.

import ArgumentParser
import Foundation
import Network
import Testing
@testable import qwg
import QWEngine
import QWServer

/// A stub of the server's /v1/dashboard, so the fetch path is exercised
/// without an engine: no model, no GPU.
final class StubDashboardServer {
    private let listener: NWListener
    let port: UInt16

    init(payload: String) throws {
        listener = try NWListener(using: .tcp, on: .any)
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state { ready.signal() }
        }
        listener.newConnectionHandler = { connection in
            connection.start(queue: DispatchQueue.global(qos: .userInitiated))
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { _, _, _, _ in
                let body = Data(payload.utf8)
                var response = Data("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    .utf8)
                response.append(Data("Content-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8))
                response.append(body)
                connection.send(content: response, completion: .contentProcessed { _ in
                    connection.cancel()
                })
            }
        }
        listener.start(queue: DispatchQueue.global(qos: .userInitiated))
        ready.wait()
        port = listener.port?.rawValue ?? 0
    }

    func stop() { listener.cancel() }
}

/// The dashboard the stub publishes: one metric and two request records.
/// The dates are ISO 8601 strings, as the server writes them: the stub
/// is the server's payload, not a shape of its own.
let stubDashboard = """
{"schemaVersion":1,"sampledAt":"2026-10-25T14:20:00Z",
 "cards":[{"id":"latency.ttft_ms","title":"Time to first token","value":"123","unit":"ms",
           "number":123.0,"provenance":"measured","sampledAt":"2026-10-25T14:20:00Z","age":0.4,
           "isStale":false,"reason":null,"series":[120.0,null,130.0]},
          {"id":"gpu.busy_ratio","title":"Engine GPU","value":null,"unit":"%",
           "number":null,"provenance":"unavailable","sampledAt":null,"age":null,
           "isStale":false,"reason":"the engine has not measured this yet","series":[]}],
 "requests":[{"id":"req-1","time":"2026-10-25T14:20:00Z","modelId":"m","promptTokens":128,
              "generatedTokens":64,"cachedTokens":32,"cacheHitTier":"warm_ram","ttftMs":123.0,
              "prefillMs":210.0,"decodeMs":1200.0,"decodeTps":53.3,
              "stageGPU":{"attention":12.5,"gdn":3.25},"config":{"kvMode":"f16"},"outcome":"ok"},
             {"id":"req-2","time":"2026-10-25T14:21:40Z","modelId":"m","promptTokens":64,
              "generatedTokens":8,"cachedTokens":0,"ttftMs":90.0,"prefillMs":100.0,
              "decodeMs":900.0,"decodeTps":8.9,"stageGPU":{"attention":40.0},"config":{},"outcome":"ok"}]}
"""

@Test("qwg telemetry reads the server's dashboard and its records")
func telemetryReadsTheServer() async throws {
    let server = try StubDashboardServer(payload: stubDashboard)
    defer { server.stop() }
    let out = FileManager.default.temporaryDirectory
        .appendingPathComponent("telemetry-\(UUID().uuidString).jsonl")
    defer { try? FileManager.default.removeItem(at: out) }
    let command = try Telemetry.parse(["--jsonl", "--out", out.path, "--port", String(server.port)])
    try await command.run()
    let text = try String(contentsOf: out, encoding: .utf8)
    let lines = text.split(separator: "\n").map(String.init)
    #expect(lines.count == 4)                    // two metrics, two records
    #expect(lines.contains { $0.contains("\"kind\":\"metric\"") })
    #expect(lines.contains { $0.contains("\"id\":\"req-1\"") })
    // a gap in the series stays a gap
    #expect(lines.contains { $0.contains("null") && $0.contains("series") })
    // the record carries the stage breakdown, and no prompt text
    #expect(lines.contains { $0.contains("\"attention\":12.5") })
    #expect(!text.contains("promptText"))
}

@Test("qwg telemetry --last N dumps only the last N records")
func telemetryLastN() async throws {
    let server = try StubDashboardServer(payload: stubDashboard)
    defer { server.stop() }
    let out = FileManager.default.temporaryDirectory
        .appendingPathComponent("telemetry-last-\(UUID().uuidString).jsonl")
    defer { try? FileManager.default.removeItem(at: out) }
    let command = try Telemetry.parse(
        ["--jsonl", "--last", "1", "--out", out.path, "--port", String(server.port)])
    try await command.run()
    let text = try String(contentsOf: out, encoding: .utf8)
    #expect(text.contains("req-2"))
    #expect(!text.contains("req-1"))
}

@Test("the telemetry subcommand carries the flags the spec names")
func telemetryFlags() throws {
    let jsonl = try Telemetry.parse(["--jsonl"])
    #expect(jsonl.jsonl)
    let last = try Telemetry.parse(["--last", "5"])
    #expect(last.last == 5)
    let both = try Telemetry.parse(["--jsonl", "--last", "3", "--host", "127.0.0.1", "--port", "9999"])
    #expect(both.jsonl && both.last == 3 && both.port == 9999)
}

@Test("the per-request record carries no prompt content")
func recordCarriesNoPromptContent() throws {
    let record = QWRequestRecord(id: "req-1", time: Date(), modelId: "m",
                                 promptTokens: 12, generatedTokens: 34, cachedTokens: 4,
                                 cacheHitTier: "warm_ram", ttftMs: 90, prefillMs: 120,
                                 decodeMs: 800, decodeTps: 42, stageGPU: ["attention": 3.5],
                                 config: ["kvMode": "f16"], outcome: "ok")
    let data = try JSONEncoder().encode(TelemetryLine.request(record))
    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let allowed: Set<String> = ["kind", "id", "time", "modelId", "promptTokens", "generatedTokens",
                                "cachedTokens", "cacheHitTier", "ttftMs", "prefillMs", "decodeMs",
                                "decodeTps", "stageGPU", "config", "outcome"]
    #expect(Set(object.keys).isSubset(of: allowed))
    #expect(object["stageGPU"] != nil)
    // the stage breakdown is what explains a slow request
    #expect((object["stageGPU"] as? [String: Double])?["attention"] == 3.5)
}

@Test("a metric line keeps its gap a gap")
func metricLineKeepsGaps() throws {
    let card = QWDashboardCard(id: "latency.ttft_ms", title: "Time to first token", value: "—",
                               unit: "ms", number: nil, provenance: .unavailable, sampledAt: nil,
                               age: nil, isStale: false, reason: "not measured",
                               series: [1.0, nil, 3.0])
    let data = try JSONEncoder().encode(TelemetryLine.metric(card))
    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let series = try #require(object["series"] as? [Double?])
    #expect(series.count == 3)
    #expect(series[1] == nil)
    #expect(object["value"] == nil)
    #expect(object["reason"] as? String == "not measured")
}

@Test("a server that cannot be reached is an error, not an empty report")
func unreachableServerFails() async throws {
    let command = try Telemetry.parse(["--host", "127.0.0.1", "--port", "9"])
    await #expect(throws: ExitCode.self) {
        try await command.run()
    }
}

// QwgMCP.swift — `qwg mcp` stdio JSON-RPC 2.0 MCP server (in-process or --server proxy).

import ArgumentParser
import Darwin
import Foundation
import QWCli
import QWCore
import QWEngine
import QWServer

struct Mcp: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mcp",
        abstract: "Run a stdio JSON-RPC 2.0 MCP server exposing decision, cognitive rail, episodic recovery, and memory vault tools."
    )

    @Option(name: [.customLong("model"), .customShort("m")], help: "Optional decision model path or ~/.qwg/models name to load into the sidecar slot.")
    var model: String?

    @Option(name: .customLong("compute-unit"), help: "Decision compute unit: auto, metal, or ane.")
    var computeUnit: String = "auto"

    @Option(name: .customLong("server"), help: "Optional running QWServer base URL to proxy /mcp requests over stdio.")
    var serverURL: String?

    func run() async throws {
        if let serverURL, !serverURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try await runStdioProxy(baseURL: serverURL)
            return
        }

        let settings = QWSettingsManager.shared.loadSettings()
        let actorConfig = QWEngineLifecycleActor.Config(
            packPath: settings.modelPackPath,
            modelName: settings.modelName,
            decisionModelPackPath: model ?? settings.decisionModelPackPath,
            decisionComputeUnit: computeUnit,
            decisionLoadVision: settings.decisionLoadVision
        )
        let actor = QWEngineLifecycleActor(config: actorConfig, logger: QWServerLogger(level: .none))
        let resolvedSidecarPath = model ?? settings.decisionModelPackPath
        if !resolvedSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let ref = settings.resolveModelReference(resolvedSidecarPath, allowArbitraryFilesystemPaths: true)
            let path = ref.packPath.isEmpty ? resolvedSidecarPath.expandingTildePath : ref.packPath
            let unit = QWDecisionComputeUnit(rawValue: computeUnit.lowercased()) ?? .auto
            _ = try await actor.loadDecisionSidecar(
                packPath: path,
                computeUnit: unit,
                loadVision: settings.decisionLoadVision
            )
        }

        let dispatcher = QWMCPDispatcher(actor: actor)
        for try await line in FileHandle.standardInput.bytes.lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            switch await dispatcher.dispatch(body: Data(trimmed.utf8)) {
            case .notificationAccepted:
                continue
            case .jsonRPCResponse(let data):
                writeResponseLine(data)
            }
        }
    }

    private func runStdioProxy(baseURL: String) async throws {
        guard let url = URL(string: baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/mcp") else {
            throw ValidationError("Invalid --server URL: '\(baseURL)'")
        }
        for try await line in FileHandle.standardInput.bytes.lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = Data(trimmed.utf8)
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode == 202 {
                continue
            }
            if !data.isEmpty {
                writeResponseLine(data)
            }
        }
    }

    private func writeResponseLine(_ data: Data) {
        var lineData = data
        if lineData.last != 0x0A {
            lineData.append(0x0A)
        }
        FileHandle.standardOutput.write(lineData)
        fflush(stdout)
    }
}

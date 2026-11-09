import Foundation
import Network

/// Thread-safe gate that stops SSE keepalive comments once the first token arrives.
public final class QWStreamKeepaliveGate: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    public init() {}

    public func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
    }

    public func isStopped() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }
}

/// The streaming endpoints, shared wire-protocol helpers, and the static UI.
extension QWHttpServer {

    /// Resolves a client-requested model identifier against the active engine model and configured aliases.
    /// Returns `nil` when `strictNotFound` is `true` and the requested model is not recognized.
    public static func resolveReplyModelName(
        requested rawRequested: String?,
        serverModel: String,
        settings: QWServerSettings,
        strictNotFound: Bool,
        discovered: [DiscoveredModel] = QWServerSettings.scanModelsDirectory()
    ) -> String? {
        let requested = (rawRequested ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if requested.isEmpty {
            let active = settings.alias(for: serverModel)
            return active.isEmpty ? serverModel : active
        }
        let ref = settings.resolveModelReference(requested, discovered: discovered)
        let isKnownModel = requested.caseInsensitiveCompare(serverModel) == .orderedSame
            || requested.caseInsensitiveCompare(settings.modelName) == .orderedSame
            || discovered.contains(where: {
                $0.name.caseInsensitiveCompare(ref.modelName) == .orderedSame || $0.packPath == ref.packPath
            })
            || (!ref.packPath.isEmpty && FileManager.default.fileExists(atPath: ref.packPath))
        if !ref.alias.isEmpty {
            return ref.alias
        } else if isKnownModel {
            return ref.modelName
        } else if strictNotFound {
            return nil
        } else {
            let active = settings.alias(for: serverModel)
            return active.isEmpty ? serverModel : active
        }
    }

    func resolveReplyModelName(requested: String?, strictNotFound: Bool) -> String? {
        Self.resolveReplyModelName(
            requested: requested,
            serverModel: actor.getModelName(),
            settings: QWSettingsManager.shared.loadSettings(),
            strictNotFound: strictNotFound
        )
    }

    func sendSSEHeaders(_ connection: NWConnection) {
        let sseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        connection.send(content: sseHeaders.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))
    }

    func sendNDJSONHeaders(_ connection: NWConnection) {
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        connection.send(content: headers.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))
    }

    func startSSEKeepalive(connection: NWConnection) -> (gate: QWStreamKeepaliveGate, task: Task<Void, Never>) {
        let gate = QWStreamKeepaliveGate()
        let keepaliveComment = Data(": keepalive\n\n".utf8)
        let task = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled, !gate.isStopped() else { break }
                if self?.isPeerGone(connection) == true { break }
                connection.send(content: keepaliveComment, isComplete: false, completion: .idempotent)
            }
        }
        return (gate, task)
    }

    func handleMetricsStream(connection: NWConnection) {
        sendSSEHeaders(connection)

        Task {
            while !Task.isCancelled {
                // The peer can leave at any time. When it does, the loop
                // stops: a stream that keeps building a payload for a
                // client that is gone is work nobody asked for.
                if self.isPeerGone(connection) { break }
                let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                if let data = try? JSONEncoder().encode(snapshot),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    let msg = "data: \(jsonStr)\n\n"
                    guard let msgData = msg.data(using: .utf8) else { break }
                    connection.send(content: msgData, isComplete: false, completion: .contentProcessed({ error in
                        if error != nil {
                            connection.cancel()
                        }
                    }))
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            connection.cancel()
        }
    }

    /// The peer is gone: the connection is cancelled or it failed. A
    /// stream must stop building a payload for nobody.
    func isPeerGone(_ connection: NWConnection) -> Bool {
        if connection.state == .cancelled { return true }
        if case .failed = connection.state { return true }
        return false
    }

    /// The published view model, pushed as it changes. The renderer
    /// receives the same model the dashboard endpoint serves.
    /// `/v1/metrics/stream` still serves the raw metrics.
    func handleDashboardStream(connection: NWConnection) {
        let sseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        connection.send(content: sseHeaders.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))

        Task {
            while !Task.isCancelled {
                // The peer can leave at any time. When it does, the loop
                // stops: a stream that keeps building a model for a
                // client that is gone is work nobody asked for.
                if self.isPeerGone(connection) { break }
                let model = self.dashboardModel()
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                if let data = try? encoder.encode(model),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    let msg = "data: \(jsonStr)\n\n"
                    guard let msgData = msg.data(using: .utf8) else { break }
                    connection.send(content: msgData, isComplete: false, completion: .contentProcessed({ error in
                        if error != nil {
                            connection.cancel()
                        }
                    }))
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            connection.cancel()
        }
    }

    func handleLogsStream(connection: NWConnection) {
        let sseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        connection.send(content: sseHeaders.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))

        // Send recent logs first
        let recent = self.logger.getRecentLogs()
        for line in recent {
            if let json = try? JSONEncoder().encode(["line": line]),
               let str = String(data: json, encoding: .utf8) {
                let msg = "data: \(str)\n\n"
                connection.send(content: msg.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))
            }
        }

        let subId = self.logger.subscribe { [weak connection] line in
            guard let connection else { return }
            if let json = try? JSONEncoder().encode(["line": line]),
               let str = String(data: json, encoding: .utf8) {
                let msg = "data: \(str)\n\n"
                connection.send(content: msg.data(using: .utf8), isComplete: false, completion: .contentProcessed({ _ in }))
            }
        }

        connection.stateUpdateHandler = { [weak self] state in
            if state == .cancelled {
                self?.logger.unsubscribe(subId)
            }
        }
    }

    func handleStaticUI(connection: NWConnection, path: String) {
        let clean = path.components(separatedBy: "?").first ?? path
        let relative: String
        if clean == "/ui" || clean == "/ui/" || clean == "/ui/controlbar" || clean == "/ui/controlbar.html" {
            relative = "controlbar.html"
        } else if clean == "/ui/logo.svg" {
            relative = "logo.svg"
        } else if clean.hasPrefix("/ui/") {
            relative = String(clean.dropFirst(4))
        } else {
            relative = "controlbar.html"
        }

        let fm = FileManager.default
        // Resolved from the package's resources (see Sources/Apps/QWServer/Resources),
        // never from the caller's working directory.
        let localPath = Bundle.module.url(forResource: (relative as NSString).deletingPathExtension,
                                          withExtension: (relative as NSString).pathExtension)?.path
            ?? (fm.currentDirectoryPath as NSString).appendingPathComponent(relative)

        if fm.fileExists(atPath: localPath), let data = try? Data(contentsOf: URL(fileURLWithPath: localPath)) {
            let contentType = relative.hasSuffix(".svg") ? "image/svg+xml" : (relative.hasSuffix(".js") ? "application/javascript" : "text/html; charset=utf-8")
            sendRawData(connection, status: 200, contentType: contentType, data: data)
        } else {
            // The file is not there. A page that says "not found" with
            // status 200 reports a success the server did not have;
            // the UI is unavailable until it is built.
            let body = "<html><body style='background:#111318;color:#e2e2e9;font-family:sans-serif;padding:24px;'><h2>QWEGN Controlbar</h2><p>The control bar UI is not installed at <code>\(localPath)</code>.</p><p>Build it: <code>make build</code></p></body></html>"
            sendRawData(connection, status: 503, contentType: "text/html; charset=utf-8", data: Data(body.utf8))
        }
    }
}

// TelemetryObservable+Streams.swift — SSE streams, UTF-8 pipe buffering, and memory pack calls

import Foundation
import QWServer

/// Thread-safe UTF-8 byte accumulator splitting complete newline-delimited lines.
final class ChildPipeLineBuffer: @unchecked Sendable {
    static let defaultMaxBufferBytes = 64 * 1024

    private let lock = NSLock()
    private let maxBufferBytes: Int
    private var buffer = Data()

    init(maxBufferBytes: Int = ChildPipeLineBuffer.defaultMaxBufferBytes) {
        self.maxBufferBytes = max(1, maxBufferBytes)
    }

    var bufferedByteCount: Int {
        lock.withLock { buffer.count }
    }

    func append(_ chunk: Data, emit: (String) -> Void) {
        let ready: [String] = lock.withLock {
            var lines: [String] = []
            buffer.append(chunk)
            while let newlineIndex = buffer.firstIndex(of: 0x0A) {
                var lineData = buffer.subdata(in: buffer.startIndex..<newlineIndex)
                if lineData.last == 0x0D {
                    lineData.removeLast()
                }
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                let decoded = String(decoding: lineData, as: UTF8.self)
                if !decoded.isEmpty { lines.append(decoded) }
            }
            while buffer.count > maxBufferBytes {
                var splitIdx = buffer.index(buffer.startIndex, offsetBy: maxBufferBytes)
                var backedUp = 0
                while backedUp < 3
                    && splitIdx > buffer.startIndex
                    && splitIdx < buffer.endIndex
                    && (buffer[splitIdx] & 0xC0) == 0x80 {
                    splitIdx = buffer.index(before: splitIdx)
                    backedUp += 1
                }
                if splitIdx == buffer.startIndex {
                    splitIdx = buffer.index(buffer.startIndex, offsetBy: maxBufferBytes)
                }
                var lineData = buffer.subdata(in: buffer.startIndex..<splitIdx)
                if lineData.last == 0x0D {
                    lineData.removeLast()
                }
                buffer.removeSubrange(buffer.startIndex..<splitIdx)
                let decoded = String(decoding: lineData, as: UTF8.self)
                if !decoded.isEmpty { lines.append(decoded) }
            }
            if buffer.isEmpty {
                buffer = Data()
            }
            return lines
        }
        for line in ready { emit(line) }
    }

    func flush(emit: (String) -> Void) {
        let trailing: String? = lock.withLock {
            guard !buffer.isEmpty else { return nil }
            var remainder = buffer
            if remainder.last == 0x0D { remainder.removeLast() }
            let decoded = String(decoding: remainder, as: UTF8.self)
            buffer.removeAll(keepingCapacity: false)
            return decoded.isEmpty ? nil : decoded
        }
        if let trailing { emit(trailing) }
    }
}

private struct SSELogLinePayload: Decodable {
    let line: String
}

extension TelemetryObservable {
    static func urlHost(for rawHost: String) -> String {
        let trimmed = rawHost.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "0.0.0.0" || trimmed == "::" || trimmed == "*" {
            return "127.0.0.1"
        }
        if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") {
            return trimmed
        }
        if trimmed.contains(":") {
            let escaped = trimmed.replacingOccurrences(of: "%", with: "%25")
            return "[\(escaped)]"
        }
        return trimmed
    }

    static func serverURL(host: String, port: Int, path: String) -> URL? {
        guard (1...65535).contains(port) else { return nil }
        let normalizedPath = path.hasPrefix("/") ? path : "/\(path)"
        return URL(string: "http://\(urlHost(for: host)):\(port)\(normalizedPath)")
    }

    func serverURL(path: String) -> URL? {
        Self.serverURL(host: settings.host, port: settings.port, path: path)
    }

    static func authorizedRequest(
        url: URL,
        apiKey: String,
        method: String = "GET",
        timeoutInterval: TimeInterval? = nil
    ) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("menubar", forHTTPHeaderField: "X-Qwegn-Client")
        if let timeoutInterval {
            req.timeoutInterval = timeoutInterval
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedKey.isEmpty {
            req.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")
        }
        return req
    }

    func authorizedRequest(
        path: String,
        method: String = "GET",
        timeoutInterval: TimeInterval? = nil
    ) -> URLRequest? {
        guard let url = serverURL(path: path) else { return nil }
        return Self.authorizedRequest(
            url: url,
            apiKey: settings.apiKey,
            method: method,
            timeoutInterval: timeoutInterval
        )
    }

    public func setPopoverVisible(_ visible: Bool) {
        guard isPopoverVisible != visible else { return }
        isPopoverVisible = visible
        syncStreamsToVisibility()
        startPolling()
    }

    public func setLogWindowVisible(_ visible: Bool) {
        guard isLogWindowVisible != visible else { return }
        isLogWindowVisible = visible
        syncStreamsToVisibility()
        startPolling()
    }

    public func clearLogs() {
        recentLogs.removeAll()
        logEntries.removeAll()
    }

    func appendLogLine(_ rawLine: String) {
        let trimmed = rawLine.trimmingCharacters(in: .newlines)
        guard !trimmed.isEmpty else { return }
        // Avoid duplicate tail replay when re-subscribing to /v1/logs/stream.
        if logEntries.suffix(24).contains(where: { $0.line == trimmed }) {
            return
        }
        recentLogs.append(trimmed)
        if recentLogs.count > 2000 {
            recentLogs.removeFirst(recentLogs.count - 2000)
        }
        let seq = nextLogSequence
        nextLogSequence &+= 1
        logEntries.append(QWLogLineParser.parse(id: seq, line: trimmed))
        if logEntries.count > 2000 {
            logEntries.removeFirst(logEntries.count - 2000)
        }
    }

    func attachChildPipe(_ pipe: Pipe, buffer: ChildPipeLineBuffer) {
        pipe.fileHandleForReading.readabilityHandler = { fh in
            let data = fh.availableData
            if data.isEmpty {
                fh.readabilityHandler = nil
                buffer.flush { line in
                    QWServerLogger.shared.logChildOutputLine(line)
                }
                return
            }
            buffer.append(data) { line in
                QWServerLogger.shared.logChildOutputLine(line)
            }
        }
    }

    func syncStreamsToVisibility() {
        if isConnected && isPopoverVisible {
            startDashboardStreamIfNeeded()
        } else {
            stopDashboardStream()
        }
        if isConnected && serverProcess == nil && isLogWindowVisible {
            startRemoteLogStreamIfNeeded()
        } else {
            stopRemoteLogStream()
        }
    }

    func stopStreamsOnDisconnect() {
        stopDashboardStream()
        stopRemoteLogStream()
    }

    func stopDashboardStream() {
        dashboardStreamTask?.cancel()
        dashboardStreamTask = nil
        lastDashboardStreamAt = nil
    }

    func stopRemoteLogStream() {
        logStreamTask?.cancel()
        logStreamTask = nil
        isLogStreamConnected = false
    }

    private func startDashboardStreamIfNeeded() {
        guard dashboardStreamTask == nil,
              var req = authorizedRequest(path: "/v1/dashboard/stream", timeoutInterval: 3600) else {
            return
        }
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let request = req

        dashboardStreamTask = Task { @MainActor [weak self] in
            defer {
                self?.dashboardStreamTask = nil
                self?.lastDashboardStreamAt = nil
            }
            do {
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
                for try await rawLine in bytes.lines {
                    if Task.isCancelled { break }
                    guard rawLine.hasPrefix("data: ") else { continue }
                    let payload = Data(rawLine.dropFirst(6).utf8)
                    if let model = try? decoder.decode(QWDashboardViewModel.self, from: payload) {
                        self?.applyDashboardModel(model, fromStream: true)
                    }
                }
            } catch {
                // Fallback polling handles reconnects cleanly.
            }
        }
    }

    private func startRemoteLogStreamIfNeeded() {
        guard logStreamTask == nil,
              var req = authorizedRequest(path: "/v1/logs/stream", timeoutInterval: 3600) else {
            return
        }
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let request = req

        logStreamTask = Task { @MainActor [weak self] in
            defer {
                self?.logStreamTask = nil
                self?.isLogStreamConnected = false
            }
            do {
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
                self?.isLogStreamConnected = true
                for try await rawLine in bytes.lines {
                    if Task.isCancelled { break }
                    guard rawLine.hasPrefix("data: ") else { continue }
                    let payload = Data(rawLine.dropFirst(6).utf8)
                    if let decoded = try? JSONDecoder().decode(SSELogLinePayload.self, from: payload) {
                        self?.appendLogLine(decoded.line)
                    }
                }
            } catch {
                // Reconnects on next poll/window reactivation.
            }
        }
    }

    public func mountMemoryPack(
        path: String,
        onComplete: (@MainActor (Result<QWEngineInfoResponse.MountedMemoryPackInfo, Error>) -> Void)? = nil
    ) {
        guard var req = authorizedRequest(path: "/v1/memory/mount", method: "POST", timeoutInterval: 120.0) else { return }
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["path": path])
        let request = req
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: request)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let body = String(data: data, encoding: .utf8) ?? "mount failed"
                    throw NSError(domain: "QWMemoryPack", code: 1, userInfo: [NSLocalizedDescriptionKey: body])
                }
                let info = try JSONDecoder().decode(QWEngineInfoResponse.MountedMemoryPackInfo.self, from: data)
                self.settings.kvMode = info.enforcedKVMode
                self.fetchHttpEngineInfo()
                self.fetchDashboard()
                onComplete?(.success(info))
            } catch {
                onComplete?(.failure(error))
            }
        }
    }

    public func unmountMemoryPack(onComplete: (@MainActor () -> Void)? = nil) {
        guard var req = authorizedRequest(path: "/v1/memory/unmount", method: "POST", timeoutInterval: 30.0) else { return }
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let request = req
        Task {
            _ = try? await URLSession.shared.data(for: request)
            self.fetchHttpEngineInfo()
            self.fetchDashboard()
            onComplete?()
        }
    }

    func resolveConfiguredPackFallbackIfMissing() {
        let configuredPack = settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !configuredPack.isEmpty,
              !FileManager.default.fileExists(atPath: configuredPack.expandingTildePath),
              let fallback = QWServerSettings.scanModelsDirectory().first(where: { $0.supported }) else {
            return
        }
        QWServerLogger.shared.log(
            .warn,
            "Configured model pack '\(configuredPack)' not found; falling back to '\(fallback.name)'."
        )
        self.settings.modelPackPath = fallback.packPath
        self.settings.modelName = fallback.name
    }

    func checkServerHealth() async -> Bool {
        guard let req = authorizedRequest(path: "/health", timeoutInterval: 1.0) else { return false }
        do {
            let (_, res) = try await URLSession.shared.data(for: req)
            return (res as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    func findQWGBinary() -> String? {
        let fm = FileManager.default
        var candidates: [String] = []
        if let execDir = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(execDir.appendingPathComponent("qwg").path)
        }
        #if DEBUG
        let currentDir = fm.currentDirectoryPath
        candidates.append((currentDir as NSString).appendingPathComponent(".build/release/qwg"))
        candidates.append((currentDir as NSString).appendingPathComponent(".build/out/Products/Debug/qwg"))
        candidates.append((currentDir as NSString).appendingPathComponent(".build/debug/qwg"))
        #endif
        candidates.append("~/.qwg/bin/qwg".expandingTildePath)
        candidates.append("/opt/homebrew/bin/qwg")
        candidates.append("/usr/local/bin/qwg")
        #if DEBUG
        candidates.append("~/bin/qwg".expandingTildePath)
        candidates.append("~/.cargo/bin/qwg".expandingTildePath)
        #endif

        for path in candidates where isTrustedExecutable(atPath: path, fm: fm) {
            return path
        }
        #if DEBUG
        if let pathEnv = ProcessInfo.processInfo.environment["PATH"] {
            for dir in pathEnv.split(separator: ":") {
                let candidate = (String(dir) as NSString).appendingPathComponent("qwg")
                if isTrustedExecutable(atPath: candidate, fm: fm) { return candidate }
            }
        }
        #endif
        return nil
    }

    private func isTrustedExecutable(atPath path: String, fm: FileManager) -> Bool {
        guard fm.isExecutableFile(atPath: path),
              let attrs = try? fm.attributesOfItem(atPath: path),
              (attrs[.type] as? FileAttributeType) != .typeDirectory else {
            return false
        }
        if let perms = attrs[.posixPermissions] as? NSNumber {
            return (perms.intValue & 0o002) == 0
        }
        return true
    }
}

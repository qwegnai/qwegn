import Foundation
import Combine
import AppKit
import QWCore
import QWServer

/// Whether the client is talking to the server. The connection and the
/// engine's phase are two different things: a server that is up but has
/// no model loaded is connected and idle.
public enum ConnectionState: String, Sendable, CaseIterable {
    /// Nothing has been heard from the server yet.
    case unknown
    /// The server is not answering.
    case disconnected
    /// The server answered.
    case connected
}

public enum EngineState: Equatable, Sendable {
    case stopped
    case starting
    case serverUp(isLoaded: Bool)
    case stopping
    case error(String)

    public var isRunningOrUp: Bool {
        switch self {
        case .serverUp: return true
        default: return false
        }
    }

    public var isTransitioning: Bool {
        switch self {
        case .starting, .stopping: return true
        default: return false
        }
    }
}

@MainActor
public final class TelemetryObservable: ObservableObject {
    @Published public var engineState: EngineState = .stopped
    @Published public var engineInfo: QWEngineInfoResponse?
    /// The published dashboard. The MenuBar draws this and nothing else.
    @Published public var dashboard: QWDashboardViewModel?
    /// Whether the server is answering.
    @Published public var connection: ConnectionState = .unknown
    /// What went wrong, in words. A failure is shown, not logged and
    /// forgotten; a successful poll clears it.
    @Published public var lastError: String?
    @Published public var engineVersion: String = QWVersion.current
    @Published public var isStartingProcess: Bool = false
    @Published public var modelLoadedOnStart: Bool = false
    @Published public var recentLogs: [String] = []
    @Published public var settings: QWServerSettings
    @Published public var discoveredModels: [DiscoveredModel] = []

    public var serverPortString: String {
        String(settings.port)
    }

    /// Whether the server is answering. One fact, one place: the
    /// connection is what the dashboard fetch established.
    public var isConnected: Bool {
        connection == .connected
    }

    /// The engine's phase, as the server published it.
    public var enginePhase: QWEngineLifecyclePhase? {
        dashboard?.lifecycle.phase
    }

    /// Whether a model is resident. The server's phase says so; the
    /// client does not guess from a stale counter.
    public var isModelLoaded: Bool {
        dashboard?.lifecycle.phase == .loaded
    }

    /// Whether the engine is serving right now: reachable, and holding
    /// a model. This is what the halo pulses for.
    public var isServing: Bool {
        isConnected && isModelLoaded
    }

    /// The context window the engine runs with, in tokens. The knob the
    /// engine published is its own; nothing is guessed from the
    /// settings file.
    public var contextBudget: Int? {
        dashboard?.knobs["contextTokens"]?.intValue
    }

    /// The failure the user is shown. The server's reason wins when the
    /// engine failed; otherwise it is the last thing the client's own
    /// request did.
    public var displayedError: String? {
        if let detail = dashboard?.lifecycle.detail { return detail }
        return lastError
    }

    /// What the client knows, in words. The connection and the engine's
    /// phase are two different things: a server that is up but has no
    /// model loaded is connected and idle.
    public var serverStatusText: String {
        switch connection {
        case .unknown: return "Not contacted yet"
        case .disconnected: return "No response"
        case .connected:
            switch enginePhase {
            case .some(.loaded): return "Loaded"
            case .some(.loading): return "Loading"
            case .some(.unloading): return "Unloading"
            case .some(.failed): return "Failed"
            case .some(.unloaded): return "Unloaded"
            // The phase was never observed. "Idle" would assert a phase
            // nobody reported.
            case .none: return "Unknown"
            }
        }
    }

    public var primaryActionTitle: String {
        (!isConnected || !isModelLoaded) ? "Start" : "Stop"
    }

    public var primaryActionIcon: String {
        (!isConnected || !isModelLoaded) ? "play.fill" : "stop.fill"
    }

    private var pollTimer: Timer?
    private var loggerSubscriptionId: UUID?
    private var serverProcess: Process?
    /// The pid of the server this app last heard from. It is kept because
    /// the answer may stop coming while the process is still there: a
    /// server that is alive is not gone.
    private var lastKnownServerPid: Int32?
    /// Whether the running server was started with a parent-pid watchdog.
    @Published public var parentPidWatchdogArmed: Bool = false
    /// The settings the running server took, as they were when it started.
    @Published public var serverAppliedSettings: QWServerSettings?
    /// The newest request wins, per endpoint: a response that arrives late
    /// must not overwrite a newer one. Each endpoint keeps its own
    /// counter; a shared one made every request but the last one stale.
    private var dashboardGeneration: UInt64 = 0
    private var engineInfoGeneration: UInt64 = 0
    /// The phase the engine was in when the client's own request failed.
    private var lastErrorPhase: QWEngineLifecyclePhase?

    public init() {
        let defaultSettings = QWSettingsManager.shared.loadSettings()
        self.settings = defaultSettings
        self.discoveredModels = QWServerSettings.scanModelsDirectory().filter(\.supported)
        self.engineVersion = QWVersion.current
        self.engineState = .stopped

        setupLocalLoggerSubscription()
        startPolling()

        // The start decision is made from the settings as they are now.
        applyAutoStartDecision()
    }

    isolated deinit {
        pollTimer?.invalidate()
        if let id = loggerSubscriptionId {
            QWServerLogger.shared.unsubscribe(id)
        }
        if let proc = serverProcess, proc.isRunning {
            proc.terminate()
        }
    }

    private func setupLocalLoggerSubscription() {
        self.recentLogs = QWServerLogger.shared.getRecentLogs()
        self.loggerSubscriptionId = QWServerLogger.shared.subscribe { [weak self] line in
            Task { @MainActor in
                guard let self else { return }
                self.recentLogs.append(line)
                if self.recentLogs.count > 2000 {
                    self.recentLogs.removeFirst(self.recentLogs.count - 2000)
                }
            }
        }
    }

    public func startPolling() {
        pollTimer?.invalidate()
        fetchDirectOrHttp()

        // The user's interval, honoured. It used to be clamped to 250 ms
        // whatever the user asked for; a poll that runs faster than the
        // user asked for is a poll the user cannot turn off.
        let interval = Double(max(200, settings.telemetryPollIntervalMs)) / 1000.0
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.fetchDirectOrHttp()
            }
        }
    }

    public func fetchDirectOrHttp() {
        let snap = QWEngineTelemetryRegistry.shared.snapshot()
        let isLocalActive = snap.cache.lookupsTotal > 0 || snap.throughput.tokensInTotal > 0 || snap.throughput.phase != "idle"

        if isLocalActive {
            let loaded = (engineInfo?.parameters.isLoaded ?? false)
            if !self.engineState.isTransitioning {
                self.engineState = .serverUp(isLoaded: loaded)
            }
        }

        fetchHttpEngineInfo()
        fetchDashboard()
    }

    /// Fetches the published dashboard. The MenuBar draws this list and
    /// computes nothing of its own: no rate, no formatted value, no
    /// offline copy of what the server would have said.
    public func fetchDashboard() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/dashboard") else {
            connection = .disconnected
            dashboard = nil
            return
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.5
        dashboardGeneration &+= 1
        let generation = dashboardGeneration
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    // The server is not answering: the old cards are not
                    // shown as if they were live.
                    guard generation == dashboardGeneration else { return }
                    self.dashboard = nil
                    self.connection = .disconnected
                    return
                }
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let model = try decoder.decode(QWDashboardViewModel.self, from: data)
                // A response that arrives after a newer one is dropped:
                // it describes a moment that has already passed.
                guard generation == dashboardGeneration else { return }
                self.dashboard = model
                self.connection = .connected
                // A failure of the client's own is over when the engine
                // moves on. A failed unload leaves the phase where it
                // was, so its message stays.
                if let failed = self.lastErrorPhase, failed != model.lifecycle.phase {
                    self.lastError = nil
                    self.lastErrorPhase = nil
                }
            } catch {
                guard generation == dashboardGeneration else { return }
                self.dashboard = nil
                self.connection = .disconnected
            }
        }
    }

    public func fetchHttpEngineInfo() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/engine/info") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.0
        engineInfoGeneration &+= 1
        let generation = engineInfoGeneration
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    guard generation == engineInfoGeneration else { return }
                    self.applyUnreachableState()
                    return
                }
                let info = try JSONDecoder().decode(QWEngineInfoResponse.self, from: data)
                guard generation == engineInfoGeneration else { return }
                self.engineInfo = info
                self.engineVersion = info.version
                if let pid = info.pid, pid > 0 {
                    self.lastKnownServerPid = Int32(pid)
                }
                if !self.engineState.isTransitioning {
                    self.engineState = .serverUp(isLoaded: info.parameters.isLoaded)
                }
            } catch {
                guard generation == engineInfoGeneration else { return }
                self.applyUnreachableState()
                // The server is not answering, so there is nothing to
                // report. The client does not write the server's answers
                // for it: an invented GPU and an invented quantization
                // are facts the user would believe.
                self.engineInfo = nil
            }
        }
    }

    // MARK: - Server Process Management & Controls

    /// Whether the server's process is still there. A process that is
    /// alive is not gone, however quiet it has become.
    private var serverProcessIsAlive: Bool {
        if let proc = serverProcess, proc.isRunning { return true }
        if let lockPid = QWEngineInstanceLock.runningEnginePID(), lockPid > 0 {
            lastKnownServerPid = lockPid
            return true
        }
        guard let pid = lastKnownServerPid, pid > 0 else { return false }
        return kill(pid, 0) == 0
    }

    /// The server did not answer. A server whose process is alive is
    /// unreachable, not gone: it is reported as such so that a live
    /// server is never announced as stopped.
    private func applyUnreachableState() {
        guard !self.engineState.isTransitioning else { return }
        let liveness = QWServerLifecycleRules.liveness(
            answered: false,
            processAlive: self.serverProcessIsAlive
        )
        self.engineState = liveness == .gone ? .stopped : .error("Server not responding")
    }

    /// The settings the running server has not taken yet. A setting the
    /// server reads when it starts is not in force until it restarts.
    public func pendingServerRestartKeys(for candidate: QWServerSettings) -> [String] {
        guard let applied = serverAppliedSettings else { return [] }
        return QWSettingsCoherence.pendingServerRestartKeys(from: applied, to: candidate)
    }

    /// The server's fate follows the setting as it is now, not the copy
    /// the app was launched with: a change to autoStartServer takes
    /// effect at the next start decision.
    private func applyAutoStartDecision() {
        let decision = QWSettingsCoherence.startDecision(
            autoStartServer: settings.autoStartServer,
            serverRunning: isAnyServerRunning
        )
        if decision == .start {
            startEngine()
        }
    }

    public func startEngine() {
        if engineState.isRunningOrUp || engineState.isTransitioning { return }
        engineState = .starting
        isStartingProcess = true

        Task { [weak self] in
            guard let self else { return }
            if await self.checkServerHealth() {
                self.fetchHttpEngineInfo()
                self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                self.connection = .connected
                self.isStartingProcess = false
                // A server this app did not start has no watchdog tied to
                // this app: whatever parent it watches is not us.
                self.parentPidWatchdogArmed = false
                self.serverAppliedSettings = self.settings
                return
            }

            if let existingPid = QWEngineInstanceLock.runningEnginePID(), existingPid > 0 {
                self.lastKnownServerPid = existingPid
                QWServerLogger.shared.log(.info, "Detected existing qwg engine process (PID \(existingPid)); attaching instead of spawning a second instance.")
                for _ in 0..<20 {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if await self.checkServerHealth() {
                        self.fetchHttpEngineInfo()
                        self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                        self.connection = .connected
                        break
                    }
                    if QWEngineInstanceLock.runningEnginePID() == nil {
                        break
                    }
                }
                self.isStartingProcess = false
                self.parentPidWatchdogArmed = false
                self.serverAppliedSettings = self.settings
                return
            }

            guard let binaryPath = self.findQWGBinary() else {
                QWServerLogger.shared.log(.error, "Cannot start engine: 'qwg' binary not found. Build the project first.")
                self.engineState = .error("Binary not found")
                self.isStartingProcess = false
                return
            }

            let configuredPack = self.settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if !configuredPack.isEmpty && !FileManager.default.fileExists(atPath: configuredPack.expandingTildePath) {
                if let fallback = QWServerSettings.scanModelsDirectory().first(where: { $0.supported }) {
                    QWServerLogger.shared.log(.warn, "Configured model pack '\(configuredPack)' not found; falling back to '\(fallback.name)'.")
                    self.settings.modelPackPath = fallback.packPath
                    self.settings.modelName = fallback.name
                }
            }

            do {
                try self.settings.validateBeforeLaunch()
            } catch {
                let msg = "\(error)"
                QWServerLogger.shared.log(.error, "Pre-launch validation failed: \(msg)")
                self.engineState = .error(msg)
                self.fail(msg)
                self.isStartingProcess = false
                return
            }

            self.modelLoadedOnStart = !self.settings.lazyLoad

            // Ensure settings file is flushed before starting process
            try? QWSettingsManager.shared.saveSettings(self.settings)
            let configPath = QWSettingsManager.defaultSettingsFileURL.path

            QWServerLogger.shared.log(.info, "Starting engine via \(binaryPath) --config \(configPath)...")

            // The parent-pid watchdog is armed only when the server is
            // meant to die with this app: a watchdog armed against a
            // setting that says "keep the server running" is a lie.
            let watchdogArguments = QWServerLifecycleRules.parentPidArguments(
                keepServerRunningAfterQuit: self.settings.keepServerRunningAfterQuit,
                parentPid: ProcessInfo.processInfo.processIdentifier
            )
            self.parentPidWatchdogArmed = !watchdogArguments.isEmpty
            self.serverAppliedSettings = self.settings

            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: binaryPath)
            proc.arguments = ["serve", "--config", configPath] + watchdogArguments

            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { fh in
                let data = fh.availableData
                guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
                for line in str.split(separator: "\n") {
                    let cleaned = String(line)
                    QWServerLogger.shared.logChildOutputLine(cleaned)
                }
            }

            proc.terminationHandler = { [weak self] p in
                Task { @MainActor in
                    guard let self else { return }
                    self.serverProcess = nil
                    self.isStartingProcess = false
                    self.modelLoadedOnStart = false
                    if self.lastKnownServerPid == p.processIdentifier {
                        self.lastKnownServerPid = nil
                    }
                    if self.engineState != .stopping {
                        if p.terminationStatus != 0 {
                            self.engineState = .error("Exited with code \(p.terminationStatus)")
                        } else {
                            self.engineState = .stopped
                        }
                    } else {
                        self.engineState = .stopped
                    }
                    self.connection = .disconnected
                    QWServerLogger.shared.log(.warn, "Engine process exited with code \(p.terminationStatus)")
                }
            }

            do {
                try proc.run()
                self.serverProcess = proc

                var running = false
                for _ in 0..<30 {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if await self.checkServerHealth() {
                        running = true
                        break
                    }
                    if let p = self.serverProcess, !p.isRunning {
                        break
                    }
                }

                if running {
                    self.fetchHttpEngineInfo()
                    self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                    self.connection = .connected
                    self.isStartingProcess = false
                    if !self.settings.lazyLoad { self.modelLoadedOnStart = true }
                    QWServerLogger.shared.log(.info, "Engine successfully running at \(self.settings.host):\(self.settings.port)")
                } else {
                    if self.engineState == .starting {
                        self.engineState = .error("Health check timed out")
                    }
                    self.isStartingProcess = false
                }
            } catch {
                self.engineState = .error(error.localizedDescription)
                self.isStartingProcess = false
                QWServerLogger.shared.log(.error, "Failed to launch qwg: \(error.localizedDescription)")
            }
        }
    }

    public func performPrimaryAction() {
        if !isConnected { startEngine() }
        else if !isModelLoaded { reloadModel() }
        else if modelLoadedOnStart { stopEngine() }
        else { unloadModel() }
    }

    public func openWebDashboard() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/ui") else { return }
        NSWorkspace.shared.open(url)
    }

    public func stopEngineAsync() async {
        engineState = .stopping
        isStartingProcess = false
        if let proc = serverProcess, proc.isRunning {
            QWServerLogger.shared.log(.info, "Terminating child engine process...")
            proc.terminate()
        }
        let pidPath = QWPaths.pidFilePath
        var candidatePids: [Int32] = []
        if let pidVal = engineInfo?.pid, pidVal > 0 { candidatePids.append(Int32(pidVal)) }
        if let fileData = try? String(contentsOfFile: pidPath, encoding: .utf8),
           let filePid = Int32(fileData.trimmingCharacters(in: .whitespacesAndNewlines)), filePid > 0 {
            candidatePids.append(filePid)
        }
        let currentAppPid = ProcessInfo.processInfo.processIdentifier
        for pid in Set(candidatePids) where pid != currentAppPid && kill(pid, 0) == 0 {
            QWServerLogger.shared.log(.info, "Sending SIGTERM to engine process PID \(pid)...")
            kill(pid, SIGTERM)
        }
        let deadline = Date().addingTimeInterval(5.0)
        while Date() < deadline {
            let isHealthy = await checkServerHealth()
            let anyPidAlive = Set(candidatePids).contains { pid in pid != currentAppPid && kill(pid, 0) == 0 }
            let childAlive = serverProcess?.isRunning ?? false
            if !isHealthy && !anyPidAlive && !childAlive { break }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        for pid in Set(candidatePids) where pid != currentAppPid && kill(pid, 0) == 0 {
            QWServerLogger.shared.log(.warn, "Process \(pid) did not exit cleanly; sending SIGKILL...")
            kill(pid, SIGKILL)
        }
        if let proc = serverProcess, proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        serverProcess = nil
        try? FileManager.default.removeItem(atPath: pidPath)
        self.serverAppliedSettings = nil
        self.parentPidWatchdogArmed = false
        self.engineState = .stopped
        self.connection = .disconnected
        QWServerLogger.shared.log(.info, "Engine successfully stopped.")
    }

    public func stopEngine() {
        modelLoadedOnStart = false
        Task { @MainActor in await self.stopEngineAsync() }
    }

    /// Restarts the model inside the running server: the weights are
    /// unloaded and loaded again in the same process. This is the one
    /// thing "restart" means; the process restart has its own name.
    public func restartEngine() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/engine/restart") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 120.0
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Restart failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Engine restarted in place via API.")
            } catch {
                self.fail("Restart failed: \(error.localizedDescription)")
            }
        }
    }

    /// Restarts the server process: the old one is stopped and a new one
    /// is started. This is what brings a changed setting into force, and
    /// it is called by that name so the two restarts cannot be mixed up.
    public func restartServerProcess() {
        Task { @MainActor in
            let configuredPack = self.settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if !configuredPack.isEmpty && !FileManager.default.fileExists(atPath: configuredPack.expandingTildePath) {
                if let fallback = QWServerSettings.scanModelsDirectory().first(where: { $0.supported }) {
                    QWServerLogger.shared.log(.warn, "Configured model pack '\(configuredPack)' not found; falling back to '\(fallback.name)'.")
                    self.settings.modelPackPath = fallback.packPath
                    self.settings.modelName = fallback.name
                }
            }

            do {
                try self.settings.validateBeforeLaunch()
            } catch {
                self.fail("Model / Drafter validation failed: \(error)")
                return
            }
            await self.stopEngineAsync()
            try? await Task.sleep(nanoseconds: 300_000_000)
            self.startEngine()
        }
    }

    public var isAnyServerRunning: Bool {
        engineState.isRunningOrUp || engineState == .starting || serverProcessIsAlive || isConnected
    }

    public func unloadModel() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/unload") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 30.0
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                // The status is the answer. A rejected unload leaves the
                // model loaded, and the popover says so.
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Unload failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Model unloaded via API. Web server remains active.")
            } catch {
                self.fail("Unload failed: \(error.localizedDescription)")
            }
        }
    }

    public func reloadModel() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/load") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        // The load endpoint answers when the load answers. A cold load
        // has been measured past 10 s, so a 10 s timeout reports a live
        // load as a failure — and the UI says "failed" while the engine
        // is loading. The timeout only needs to catch a dead server.
        req.timeoutInterval = 120.0
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Load failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Model reloaded via API.")
            } catch {
                self.fail("Load failed: \(error.localizedDescription)")
            }
        }
    }

    public func refreshDiscoveredModels() {
        discoveredModels = QWServerSettings.scanModelsDirectory().filter(\.supported)
    }

    public func revealModelsFolder() {
        let url = QWPaths.modelsDirectoryURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    public func switchModel(to model: DiscoveredModel) {
        var updated = settings
        updated.modelName = model.name
        updated.modelPackPath = model.packPath
        let (effective, _) = QWSettingsManager.shared.effectiveSettings(
            forModel: model.name,
            packPath: model.packPath,
            base: updated
        )
        try? saveSettings(effective)
        refreshDiscoveredModels()

        guard isConnected,
              let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/load") else {
            return
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: [
            "model": model.packPath.isEmpty ? model.name : model.packPath
        ])
        req.timeoutInterval = 120.0
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Switch model failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                self.fetchHttpEngineInfo()
                QWServerLogger.shared.log(.info, "Switched active model to '\(model.name)' via API.")
            } catch {
                self.fail("Switch model failed: \(error.localizedDescription)")
            }
        }
    }

    /// A failure the user can see. The engine's phase is not guessed:
    /// the server will publish the truth on the next poll.
    private func fail(_ message: String) {
        self.lastError = message
        self.lastErrorPhase = enginePhase
        QWServerLogger.shared.log(.warn, message)
    }

    /// The server's failure reason, surfaced. A bare status code is not
    /// a diagnosis; the body carries the reason, and the log loses it
    /// no more.
    private static func responseDetail(_ data: Data) -> String {
        let body = String(data: data, encoding: .utf8) ?? ""
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "no detail in response body" }
        return trimmed.count > 300 ? String(trimmed.prefix(300)) + "…" : trimmed
    }

    /// Saves the settings and makes the change take effect. The settings
    /// are read back from the file, so the copy the app was launched with
    /// does not outlive the file it came from.
    public func saveSettings(_ newSettings: QWServerSettings) throws {
        try QWSettingsManager.shared.saveSettings(newSettings)
        self.settings = QWSettingsManager.shared.loadSettings()
        self.serverAppliedSettings?.loadVision = self.settings.loadVision
        QWServerLogger.shared.configureFileSink(path: self.settings.expandedLogFilePath)
        if isConnected, let url = URL(string: "http://\(settings.host):\(settings.port)/v1/settings"),
           let body = try? JSONEncoder().encode(self.settings) {
            var req = URLRequest(url: url); req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = body; req.timeoutInterval = 30.0
            Task {
                _ = try? await URLSession.shared.data(for: req)
                self.fetchDashboard(); self.fetchHttpEngineInfo()
            }
        }
        startPolling()
        applyAutoStartDecision()
    }

    private func checkServerHealth() async -> Bool {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/health") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.0
        do {
            let (_, res) = try await URLSession.shared.data(for: req)
            return (res as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func findQWGBinary() -> String? {
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath

        var candidates: [String] = []
        if let execDir = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(execDir.appendingPathComponent("qwg").path)
        }
        candidates.append((currentDir as NSString).appendingPathComponent(".build/out/Products/Debug/qwg"))
        candidates.append((currentDir as NSString).appendingPathComponent(".build/debug/qwg"))
        candidates.append((currentDir as NSString).appendingPathComponent(".build/release/qwg"))
        candidates.append("/usr/local/bin/qwg")
        candidates.append("~/bin/qwg".expandingTildePath)
        candidates.append("~/.cargo/bin/qwg".expandingTildePath)

        for path in candidates where fm.isExecutableFile(atPath: path) { return path }
        if let pathEnv = ProcessInfo.processInfo.environment["PATH"] {
            for dir in pathEnv.split(separator: ":") {
                let candidate = (String(dir) as NSString).appendingPathComponent("qwg")
                if fm.isExecutableFile(atPath: candidate) { return candidate }
            }
        }
        return nil
    }
}

import Foundation
import Network
import QWCore
import QWEngine
import QWHAL
import QWVision
import Metal

public final class QWHttpServer: @unchecked Sendable {
    // Internal, not private: the endpoints live in extensions across
    // several files, and a private member is invisible from them.
    let actor: QWEngineLifecycleActor
    private let host: String
    private let port: UInt16
    let logger: QWServerLogger
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "qwg.server.nw", qos: .userInitiated)

    public init(actor: QWEngineLifecycleActor, host: String = "127.0.0.1", port: Int = 8080, logger: QWServerLogger = .shared) {
        self.actor = actor
        self.host = host
        self.port = UInt16(port)
        self.logger = logger
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let nwPort = NWEndpoint.Port(rawValue: port) ?? NWEndpoint.Port(integerLiteral: 8080)

        let listener = try NWListener(using: params, on: nwPort)
        self.listener = listener

        let readySem = DispatchSemaphore(value: 0)
        let stateLock = NSLock()
        nonisolated(unsafe) var startupError: NWError? = nil
        nonisolated(unsafe) var signaled = false

        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.logger.log(.info, "HTTP Server listening on http://\(self.host):\(self.port)")
                stateLock.lock()
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            case .failed(let error):
                self.logger.log(.error, "HTTP Server failed: \(error)")
                stateLock.lock()
                startupError = error
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            case .cancelled:
                self.logger.log(.info, "HTTP Server cancelled.")
                stateLock.lock()
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            default:
                break
            }
        }

        listener.newConnectionHandler = { [weak self] connection in
            self?.handleConnection(connection)
        }

        listener.start(queue: queue)
        _ = readySem.wait(timeout: .now() + 3.0)
        stateLock.lock()
        let err = startupError
        stateLock.unlock()
        if let err {
            listener.cancel()
            self.listener = nil
            throw err
        }
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        readHttpRequest(connection: connection)
    }

    private func readHttpRequest(connection: NWConnection, accumulatedData: Data = Data()) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, context, isComplete, error in
            guard let self else { return }
            if error != nil {
                connection.cancel()
                return
            }
            guard let data, !data.isEmpty else {
                if isComplete { connection.cancel() }
                return
            }

            var totalData = accumulatedData
            totalData.append(data)

            // Look for header boundary \r\n\r\n
            guard let headerEndRange = totalData.range(of: Data([13, 10, 13, 10])) else {
                if totalData.count > 1_000_000 {
                    self.sendSimpleResponse(connection, status: 413, body: "Headers too large\n")
                    return
                }
                self.readHttpRequest(connection: connection, accumulatedData: totalData)
                return
            }

            let headerData = totalData.subdata(in: 0..<headerEndRange.lowerBound)
            guard let headerStr = String(data: headerData, encoding: .utf8) else {
                self.sendSimpleResponse(connection, status: 400, body: "Bad request encoding\n")
                return
            }

            let lines = headerStr.components(separatedBy: "\r\n")
            guard let requestLine = lines.first else {
                self.sendSimpleResponse(connection, status: 400, body: "Missing request line\n")
                return
            }

            let parts = requestLine.components(separatedBy: " ")
            guard parts.count >= 2 else {
                self.sendSimpleResponse(connection, status: 400, body: "Malformed request line\n")
                return
            }

            let method = parts[0].uppercased()
            let path = parts[1]

            // Parse Content-Length
            var contentLength = 0
            for line in lines.dropFirst() {
                let kv = line.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
                if kv.count == 2 && kv[0].lowercased() == "content-length" {
                    contentLength = Int(kv[1]) ?? 0
                }
            }

            let bodyStart = headerEndRange.upperBound
            let bodyBytesNeeded = contentLength - (totalData.count - bodyStart)

            if bodyBytesNeeded > 0 {
                self.readRemainingBody(connection: connection, totalData: totalData, needed: bodyBytesNeeded, method: method, path: path)
            } else {
                let bodyData = totalData.subdata(in: bodyStart..<(bodyStart + contentLength))
                self.dispatchRoute(connection: connection, method: method, path: path, body: bodyData)
            }
        }
    }

    private func readRemainingBody(connection: NWConnection, totalData: Data, needed: Int, method: String, path: String) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: max(needed, 65536)) { [weak self] data, context, isComplete, error in
            guard let self else { return }
            if error != nil {
                connection.cancel()
                return
            }
            guard let data, !data.isEmpty else {
                connection.cancel()
                return
            }
            var updated = totalData
            updated.append(data)
            let headerEndRange = updated.range(of: Data([13, 10, 13, 10]))!
            let bodyStart = headerEndRange.upperBound
            let remaining = needed - data.count
            if remaining > 0 && !isComplete {
                self.readRemainingBody(connection: connection, totalData: updated, needed: remaining, method: method, path: path)
            } else {
                let bodyData = updated.subdata(in: bodyStart..<updated.count)
                self.dispatchRoute(connection: connection, method: method, path: path, body: bodyData)
            }
        }
    }

    private func dispatchRoute(connection: NWConnection, method: String, path: String, body: Data) {
        // Handle CORS OPTIONS preflight
        if method == "OPTIONS" {
            let corsHeader = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 86400\r\nContent-Length: 0\r\n\r\n"
            connection.send(content: corsHeader.data(using: .utf8), completion: .contentProcessed({ _ in connection.cancel() }))
            return
        }

        // Clean path (strip query params)
        let cleanPath = path.components(separatedBy: "?").first ?? path

        Task {
            do {
                switch (method, cleanPath) {
                case ("POST", "/v1/chat/completions"):
                    try await self.handleChatCompletions(connection: connection, body: body)

                case ("POST", "/api/chat"):
                    try await self.handleOllamaChat(connection: connection, body: body)

                case ("POST", "/api/generate"):
                    try await self.handleOllamaGenerate(connection: connection, body: body)

                case ("POST", "/v1/messages"):
                    try await self.handleMessages(connection: connection, body: body)

                case ("POST", "/v1/models/load"), ("POST", "/api/load"):
                    if !body.isEmpty,
                       let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                       let requestedModel = payload["model"] as? String {
                        try await self.actor.selectModelIfNeeded(requestedModel)
                    }
                    _ = try await self.actor.loadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("POST", "/v1/models/unload"), ("POST", "/api/unload"):
                    let reclaimed = try await self.actor.unloadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: false, memoryReclaimedMB: reclaimed)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/models"), ("GET", "/models"):
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    var seenIds = Set<String>()
                    var models: [ModelObject] = []
                    // Each entry carries the one fact the model really
                    // has: when its bytes last changed on disk.
                    func addModel(id: String, loaded: Bool, path: String?) {
                        guard !id.isEmpty, seenIds.insert(id).inserted else { return }
                        models.append(ModelObject(id: id, loaded: loaded,
                                                  created: QWModelDiskStats.modifiedUnix(of: path)))
                    }
                    let activeAlias = settings.alias(for: modelName)
                    if !activeAlias.isEmpty {
                        addModel(id: activeAlias, loaded: isLoaded, path: settings.modelPackPath)
                    }
                    addModel(id: modelName, loaded: isLoaded, path: settings.modelPackPath)
                    for disc in QWServerSettings.scanModelsDirectory() where !disc.packPath.isEmpty && disc.supported {
                        let loadedHere = isLoaded && disc.name == modelName
                        let alias = settings.alias(for: disc.name)
                        if !alias.isEmpty {
                            addModel(id: alias, loaded: loadedHere, path: disc.packPath)
                        }
                        addModel(id: disc.name, loaded: loadedHere, path: disc.packPath)
                    }
                    let resp = ModelListResponse(models: models)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/models/discovered"):
                    let models = QWServerSettings.scanModelsDirectory()
                    self.sendJsonResponse(connection, status: 200, object: ["models": models])

                case ("GET", let path) where path.hasPrefix("/v1/models/") || path.hasPrefix("/models/"):
                    let prefix = path.hasPrefix("/v1/models/") ? "/v1/models/" : "/models/"
                    let requestedId = String(path.dropFirst(prefix.count))
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    // The model is known when it is the resident one,
                    // or a supported model the scan found. Anything
                    // else is a 404 naming it (OpenAI's shape).
                    let activeAlias = settings.alias(for: modelName)
                    let displayModel = activeAlias.isEmpty ? modelName : activeAlias
                    var path2: String?
                    var loaded = false
                    if requestedId.isEmpty {
                        path2 = settings.modelPackPath
                        loaded = isLoaded
                    } else if requestedId == modelName || requestedId == activeAlias {
                        path2 = settings.modelPackPath
                        loaded = isLoaded
                    } else if let disc = QWServerSettings.scanModelsDirectory().first(where: {
                        $0.name == requestedId || settings.alias(for: $0.name) == requestedId
                    }), disc.supported {
                        path2 = disc.packPath
                        loaded = isLoaded && disc.name == modelName
                    } else {
                        let err = ServerErrorPayload(message: "The model '\(requestedId)' does not exist.",
                                                      type: "invalid_request_error", code: 404)
                        self.sendJsonResponse(connection, status: 404, object: err)
                        return
                    }
                    let resp = ModelObject(id: requestedId.isEmpty ? displayModel : requestedId,
                                           loaded: loaded,
                                           created: QWModelDiskStats.modifiedUnix(of: path2))
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/api/version"):
                    self.sendJsonResponse(connection, status: 200, object: ["engine": "qwegine", "version": QWVersion.current])

                case ("POST", "/api/show"):
                    let modelName = self.actor.getModelName()
                    let params = self.actor.getLoadedParamsInfo()
                    let details: [String: Any] = [
                        "format": "qwg",
                        "family": "qwen3",
                        "parameter_size": "27B",
                        "quantization_level": params?.headStorage ?? "Q4g"
                    ]
                    let resp: [String: Any] = [
                        "modelfile": "# Modelfile for Qwegine\nFROM \(modelName)",
                        "parameters": "context_length \(params?.contextBudget ?? 131072)",
                        "details": details
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: resp) {
                        self.sendRawData(connection, status: 200, contentType: "application/json", data: data)
                    } else {
                        self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])
                    }

                case ("GET", "/api/tags"), ("GET", "/api/ps"):
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    let activeAlias = settings.alias(for: modelName)
                    let displayModel = activeAlias.isEmpty ? modelName : activeAlias
                    if cleanPath == "/api/ps" {
                        // The size of a resident model is the bytes its
                        // files occupy on disk, read now; the date is
                        // when those bytes last changed.
                        let models = isLoaded ? [OllamaPsResponse.RunningModel(
                            name: displayModel, model: displayModel,
                            size: QWModelDiskStats.sizeBytes(of: settings.modelPackPath),
                            expiresAt: "never")] : []
                        self.sendJsonResponse(connection, status: 200, object: OllamaPsResponse(models: models))
                    } else {
                        var seen = Set<String>()
                        var tags: [OllamaModelTag] = []
                        func addTag(_ name: String, path: String?) {
                            guard !name.isEmpty, seen.insert(name).inserted else { return }
                            tags.append(OllamaModelTag(name: name, model: name,
                                                       modifiedAt: QWModelDiskStats.modifiedRFC3339(of: path),
                                                       size: QWModelDiskStats.sizeBytes(of: path)))
                        }
                        if !activeAlias.isEmpty { addTag(activeAlias, path: settings.modelPackPath) }
                        addTag(modelName, path: settings.modelPackPath)
                        for disc in QWServerSettings.scanModelsDirectory() where !disc.packPath.isEmpty && disc.supported {
                            let alias = settings.alias(for: disc.name)
                            if !alias.isEmpty { addTag(alias, path: disc.packPath) }
                            addTag(disc.name, path: disc.packPath)
                        }
                        self.sendJsonResponse(connection, status: 200, object: OllamaTagsResponse(models: tags))
                    }

                case ("GET", "/metrics"):
                    let body = QWEngineTelemetryRegistry.shared.formatPrometheus()
                    self.sendRawData(connection, status: 200, contentType: "text/plain; version=0.0.4; charset=utf-8", data: Data(body.utf8))

                case ("GET", "/v1/metrics"), ("GET", "/api/metrics"):
                    let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                    self.sendJsonResponse(connection, status: 200, object: snapshot)

                case ("GET", "/v1/settings"), ("GET", "/api/settings"):
                    let settings = QWSettingsManager.shared.loadSettings()
                    self.sendJsonResponse(connection, status: 200, object: settings)

                case ("POST", "/v1/settings"), ("POST", "/api/settings"):
                    let decoder = JSONDecoder()
                    do {
                        let oldSettings = QWSettingsManager.shared.loadSettings()
                        var newSettings = try decoder.decode(QWServerSettings.self, from: body)
                        let modelChanged = (newSettings.modelPackPath != oldSettings.modelPackPath)
                            || (newSettings.modelName != oldSettings.modelName)
                        if modelChanged {
                            let ref = newSettings.resolveModelReference(
                                newSettings.modelPackPath.isEmpty ? newSettings.modelName : newSettings.modelPackPath
                            )
                            let targetModelName = ref.modelName.isEmpty ? newSettings.modelName : ref.modelName
                            if let profile = QWSettingsManager.shared.loadProfile(forModel: targetModelName) {
                                newSettings.applyModelConfigDefaults(packPath: ref.packPath)
                                newSettings.apply(knobProfile: profile)
                            } else if newSettings.maxContext == oldSettings.maxContext {
                                newSettings.applyModelConfigDefaults(packPath: ref.packPath)
                            }
                        }
                        if let mounted = self.actor.getMountedMemoryPackInfo(), mounted.enforcesKVMode {
                            newSettings.kvMode = mounted.enforcedKVMode
                        }
                        try newSettings.validateBeforeLaunch()
                        try QWSettingsManager.shared.saveSettings(newSettings)
                        if !modelChanged {
                            try await self.actor.setVisionLoaded(newSettings.loadVision)
                        }
                        self.sendJsonResponse(connection, status: 200, object: ["status": "ok", "message": "Settings saved successfully"])
                    } catch {
                        let err = ServerErrorPayload(message: "Failed to parse or save settings: \(error.localizedDescription)", code: 400)
                        self.sendJsonResponse(connection, status: 400, object: err)
                    }

                case ("POST", "/v1/memory/mount"), ("POST", "/api/memory/mount"):
                    guard !body.isEmpty,
                          let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                          let repoPath = payload["path"] as? String,
                          !repoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        let err = ServerErrorPayload(message: "Missing 'path' in memory mount request.", code: 400)
                        self.sendJsonResponse(connection, status: 400, object: err)
                        return
                    }
                    let mounted = try await self.actor.mountMemoryPack(at: repoPath)
                    self.sendJsonResponse(connection, status: 200, object: mounted)

                case ("POST", "/v1/memory/unmount"), ("POST", "/api/memory/unmount"):
                    await self.actor.unmountMemoryPack()
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/settings/reveal"), ("POST", "/api/settings/reveal"):
                    let path = QWPaths.settingsFilePath
                    let p = Process()
                    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                    p.arguments = ["-R", path]
                    // The reveal either happened or it did not. An
                    // unconditional 200 would report success the
                    // process never established.
                    do {
                        try p.run()
                    } catch {
                        let err = ServerErrorPayload(message: "Could not open the Finder window: \(error.localizedDescription)", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    p.waitUntilExit()
                    if p.terminationStatus != 0 {
                        let err = ServerErrorPayload(message: "Finder rejected the request (exit status \(p.terminationStatus)).", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/logs/reveal"), ("POST", "/api/logs/reveal"):
                    let settings = QWSettingsManager.shared.loadSettings()
                    let logDir = settings.logDir.expandingTildePath
                    try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
                    let p = Process()
                    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                    p.arguments = [logDir]
                    do {
                        try p.run()
                    } catch {
                        let err = ServerErrorPayload(message: "Could not open the Finder window: \(error.localizedDescription)", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    p.waitUntilExit()
                    if p.terminationStatus != 0 {
                        let err = ServerErrorPayload(message: "Finder rejected the request (exit status \(p.terminationStatus)).", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/engine/restart"), ("POST", "/api/restart"):
                    _ = try await self.actor.restartExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/engine/info"), ("GET", "/api/engine/info"):
                    let hw = QWMachineHardwareInfo.current()
                    let caps = QWDeviceCaps.current
                    let config = self.actor.getConfig()
                    let engineState = self.actor.engineState()
                    let isLoaded = engineState.isLoaded
                    let params = self.actor.getLoadedParamsInfo()

                    let metalDevName = caps?.name ?? MTLCreateSystemDefaultDevice()?.name
                    let isTensor = caps?.isMetal4TensorCapable ?? false
                    // The Metal family the device really supports, from
                    // the probes the HAL made. A device that supports
                    // none of them has no version to report.
                    let metalVer: String? = if caps?.familyApple11 == true || caps?.familyApple10 == true {
                        "Metal 4"
                    } else if caps?.familyApple9 == true {
                        "Metal 3"
                    } else {
                        nil
                    }
                    // The HAL's DRAM bandwidth is a constant from one
                    // machine, not a measurement of this one. It is not
                    // published as a fact about this host.
                    let dramBw: Double? = nil

                    let hwFacts = QWEngineInfoResponse.HardwareFacts(
                        modelIdentifier: hw.modelIdentifier,
                        cpuBrand: hw.cpuBrand,
                        physicalCores: hw.physicalCores,
                        logicalCores: hw.logicalCores,
                        physicalMemoryBytes: hw.physicalMemoryBytes,
                        physicalMemoryGB: hw.physicalMemoryGB,
                        metalDeviceName: metalDevName,
                        metalVersion: metalVer,
                        isTensorCapable: isTensor,
                        dramBwGBs: dramBw
                    )

                    let srvParams = QWEngineInfoResponse.ServerParameters(
                        modelName: self.actor.getModelName(),
                        modelPackPath: config.packPath,
                        // The head storage of the resident model. With no
                        // model resident there is no head, and a
                        // quantization that describes nothing is a claim
                        // about nothing.
                        headStorage: params?.headStorage,
                        contextBudget: config.maxContext,
                        kvMode: String(describing: config.kvMode),
                        mtpDrafts: config.mtpDrafts,
                        mtpBackend: config.mtpBackend,
                        gdnPrefillMode: config.gdnPrefill,
                        prefillScratchEviction: config.evictPrefillScratch,
                        hostLogicalCores: hw.logicalCores,
                        isLoaded: isLoaded
                    )

                    let cacheInfo = QWEngineInfoResponse.CacheConfigInfo(
                        warmLimitBytes: config.cacheConfig.warmByteLimit,
                        ssdLimitBytes: config.cacheConfig.ssdByteLimit,
                        cacheDir: config.cacheConfig.cacheDirectory.path
                    )

                    let snap = QWEngineTelemetryRegistry.shared.snapshot()
                    let hwUtil = snap.hardware ?? QWSystemMetricsSampler.shared.sample()
                    let info = QWEngineInfoResponse(
                        status: engineState.lifecycle.phase.rawValue,
                        version: QWVersion.current,
                        pid: Int(ProcessInfo.processInfo.processIdentifier),
                        hardware: hwFacts,
                        parameters: srvParams,
                        cacheConfig: cacheInfo,
                        hardwareUtilization: hwUtil,
                        lifecycle: engineState.lifecycle,
                        sources: config.sourceSummary.mapValues { QWKnobSource.normalise($0).rawValue },
                        capabilities: engineState.capabilities.map(\.rawValue).sorted(),
                        mountedMemoryPack: self.actor.getMountedMemoryPackInfo(),
                        endpoints: [
                            "chat": "POST /v1/chat/completions",
                            "models": "GET /v1/models",
                            "metrics": "GET /v1/metrics",
                            "metrics_stream": "GET /v1/metrics/stream",
                            "logs_stream": "GET /v1/logs/stream",
                            "engine_info": "GET /v1/engine/info",
                            "memory_mount": "POST /v1/memory/mount",
                            "memory_unmount": "POST /v1/memory/unmount",
                            "restart": "POST /v1/engine/restart",
                            "ui": "GET /ui"
                        ]
                    )
                    self.sendJsonResponse(connection, status: 200, object: info)

                case ("GET", "/v1/metrics/stream"):
                    self.handleMetricsStream(connection: connection)

                case ("GET", "/v1/dashboard/stream"):
                    self.handleDashboardStream(connection: connection)

                case ("GET", "/v1/logs/stream"):
                    self.handleLogsStream(connection: connection)

                case ("GET", let path) where path.hasPrefix("/ui"):
                    self.handleStaticUI(connection: connection, path: path)

                case ("GET", "/v1/dashboard"):
                    let dashboard = self.dashboardModel()
                    self.sendIsoJsonResponse(connection, status: 200, object: dashboard)

                case ("GET", "/health"), ("GET", "/"):
                    let engineState = self.actor.engineState()
                    let modelName = self.actor.getModelName()
                    // The server is alive and answering, so the status
                    // code says so. What the engine is doing is the
                    // phase — a failed engine is not an unreachable one.
                    let health = QWHealthResponse(status: engineState.lifecycle.phase.rawValue,
                                                  engine: "qwegine",
                                                  model: modelName, loaded: engineState.isLoaded,
                                                  lifecycle: engineState.lifecycle)
                    let data = try JSONEncoder().encode(health)
                    self.sendRawData(connection, status: 200, contentType: "application/json", data: data)

                default:
                    let err = ServerErrorPayload(message: "Not found: \(method) \(cleanPath)", type: "invalid_request_error", code: 404)
                    self.sendJsonResponse(connection, status: 404, object: err)
                }
            } catch let mmError as QWMultimodalError {
                let err = ServerErrorPayload(message: "\(mmError)", type: "invalid_request_error", code: 400)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) bad multimodal request: \(mmError)")
                self.sendJsonResponse(connection, status: 400, object: err)
            } catch {
                let err = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
                // The reason goes to the log, not only to the response body:
                // a 500 with no recorded reason is undebuggable after the fact.
                self.logger.log(.error, "HTTP \(method) \(cleanPath) failed: \(error)")
                self.sendJsonResponse(connection, status: 500, object: err)
            }
        }
    }

    // The stream endpoints and the static UI live in QWHttpServer+Streams.swift.
    // POST /v1/chat/completions lives in QWHttpServer+ChatCompletions.swift.

    /// The published model. The route and the stream publish the same
    /// thing, so a renderer cannot see two truths.
    func dashboardModel() -> QWDashboardViewModel {
        let engineState = self.actor.engineState()
        let config = self.actor.getConfig()
        // A knob the model cannot honour is not offered: the renderer
        // must not be asked to hide what it cannot show.
        let offered = Set(engineState.capabilities)
        let knobs = config.knobValues.filter { id, _ in
            guard let descriptor = QWKnobRegistry.descriptor(id) else { return true }
            guard let capability = descriptor.requiredCapability else { return true }
            return offered.contains(capability)
        }
        return QWDashboardViewModel.build(
            lifecycle: engineState.lifecycle,
            modelId: engineState.loadedModelName,
            capabilities: engineState.capabilities,
            knobs: knobs,
            knobSources: QWKnobSource.sources(from: config.sourceSummary,
                                              for: Array(knobs.keys)),
            facts: self.dashboardFacts(),
            requests: QWRequestRecordStore.shared.recent(QWRequestRecordStore.capacity))
    }

    /// The facts about the machine and the resident model, for the
    /// renderers. A fact nobody could establish is absent.
    private func dashboardFacts() -> QWDashboardFacts {
        let params = self.actor.getLoadedParamsInfo()
        let config = self.actor.getConfig()
        let caps = QWDeviceCaps.current
        let hw = QWMachineHardwareInfo.current()
        let metalFamily: String? = if caps?.familyApple11 == true || caps?.familyApple10 == true {
            "Metal 4"
        } else if caps?.familyApple9 == true {
            "Metal 3"
        } else {
            nil
        }
        return QWDashboardFacts(
            quantization: params?.headStorage,
            hostLogicalCores: hw.logicalCores,
            gdnPrefillMode: config.gdnPrefill,
            gpuName: caps?.name ?? MTLCreateSystemDefaultDevice()?.name,
            metalFamily: metalFamily)
    }

    /// A response whose dates are ISO-8601 instants, so a client in any
    /// language can read them.
    func sendIsoJsonResponse<T: Encodable>(_ connection: NWConnection, status: Int, object: T) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(object)
            sendRawData(connection, status: status, contentType: "application/json", data: data)
        } catch {
            sendSimpleResponse(connection, status: 500, body: "{\"error\": \"JSON encoding error\"}\n")
        }
    }

    func sendJsonResponse<T: Encodable>(_ connection: NWConnection, status: Int, object: T) {
        do {
            let data = try JSONEncoder().encode(object)
            sendRawData(connection, status: status, contentType: "application/json", data: data)
        } catch {
            sendSimpleResponse(connection, status: 500, body: "{\"error\": \"JSON encoding error\"}\n")
        }
    }

    func sendRawData(_ connection: NWConnection, status: Int, contentType: String, data: Data) {
        var header = "HTTP/1.1 \(status) \(status == 200 ? "OK" : "Error")\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(data.count)\r\n"
        header += "Access-Control-Allow-Origin: *\r\n"
        header += "Connection: close\r\n\r\n"

        var packet = header.data(using: .utf8)!
        packet.append(data)
        connection.send(content: packet, completion: .contentProcessed({ _ in
            connection.cancel()
        }))
    }

    func sendSimpleResponse(_ connection: NWConnection, status: Int, body: String) {
        let data = body.data(using: .utf8)!
        sendRawData(connection, status: status, contentType: "text/plain", data: data)
    }
}

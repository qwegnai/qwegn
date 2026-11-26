import Foundation
import Network
import QWCore
import QWEngine
import QWHAL
import QWVision
import Metal

public final class QWHttpServer: @unchecked Sendable {
    // Internal, not private: the endpoints live in extensions across
    // several files, and a member is invisible from them.
    let actor: QWEngineLifecycleActor
    let host: String
    let port: UInt16
    let apiKey: String?
    let corsAllowedOrigins: [String]
    let allowUnauthenticatedPublicBind: Bool
    let allowedWorkspaceRoots: [String]
    let maxConcurrentConnections: Int
    let requestReadTimeoutSeconds: TimeInterval
    let finderRevealRunner: (@Sendable ([String]) async throws -> Int32)?
    let modelCatalogScanner: @Sendable () -> [QWServerSettings.DiscoveredModel]
    let settingsFileURL: URL
    let updateService: QWServerUpdateService
    let enableAutoUpdateChecks: Bool
    let logger: QWServerLogger
    var listener: NWListener?
    let queue = DispatchQueue(label: "qwg.server.nw", qos: .userInitiated)
    let connectionStateLock = NSLock()
    var cancellationHubs: [ObjectIdentifier: QWConnectionCancellationHub] = [:]
    var connectionOrigins: [ObjectIdentifier: String] = [:]
    let responsesStore = QWResponsesStore()
    let episodicStore: QWEpisodicTranscriptStore
    let sessionTree: QWSessionTree

    var activeConnectionCount: Int {
        connectionStateLock.withLock { cancellationHubs.count }
    }

    public convenience init(
        actor: QWEngineLifecycleActor,
        host: String = "127.0.0.1",
        port: Int = 8080,
        logger: QWServerLogger = .shared,
        apiKey: String? = nil,
        corsAllowedOrigins: [String] = [],
        allowUnauthenticatedPublicBind: Bool = false,
        allowedWorkspaceRoots: [String] = [],
        maxConcurrentConnections: Int = 64,
        requestReadTimeoutSeconds: TimeInterval = 30.0,
        enableAutoUpdateChecks: Bool = false,
        updateService: QWServerUpdateService? = nil
    ) {
        self.init(
            actor: actor,
            host: host,
            port: port,
            logger: logger,
            apiKey: apiKey,
            corsAllowedOrigins: corsAllowedOrigins,
            allowUnauthenticatedPublicBind: allowUnauthenticatedPublicBind,
            allowedWorkspaceRoots: allowedWorkspaceRoots,
            maxConcurrentConnections: maxConcurrentConnections,
            requestReadTimeoutSeconds: requestReadTimeoutSeconds,
            finderRevealRunner: nil,
            enableAutoUpdateChecks: enableAutoUpdateChecks,
            updateService: updateService
        )
    }

    init(
        actor: QWEngineLifecycleActor,
        host: String = "127.0.0.1",
        port: Int = 8080,
        logger: QWServerLogger = .shared,
        apiKey: String? = nil,
        corsAllowedOrigins: [String] = [],
        allowUnauthenticatedPublicBind: Bool = false,
        allowedWorkspaceRoots: [String] = [],
        maxConcurrentConnections: Int = 64,
        requestReadTimeoutSeconds: TimeInterval = 30.0,
        finderRevealRunner: (@Sendable ([String]) async throws -> Int32)?,
        modelCatalogScanner: (@Sendable () -> [QWServerSettings.DiscoveredModel])? = nil,
        settingsFileURL: URL? = nil,
        enableAutoUpdateChecks: Bool = false,
        updateService: QWServerUpdateService? = nil,
        episodicStore: QWEpisodicTranscriptStore = QWEpisodicTranscriptStore(),
        sessionTree: QWSessionTree = QWSessionTree()
    ) {
        self.actor = actor
        self.host = host
        self.port = UInt16(clamping: max(0, port))
        self.logger = logger
        let trimmedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.apiKey = (trimmedKey?.isEmpty == false) ? trimmedKey : nil
        self.corsAllowedOrigins = corsAllowedOrigins
        self.allowUnauthenticatedPublicBind = allowUnauthenticatedPublicBind
        self.allowedWorkspaceRoots = allowedWorkspaceRoots
        self.maxConcurrentConnections = max(1, maxConcurrentConnections)
        self.requestReadTimeoutSeconds = max(0.05, requestReadTimeoutSeconds)
        self.finderRevealRunner = finderRevealRunner
        self.modelCatalogScanner = modelCatalogScanner ?? { QWServerSettings.scanModelsDirectory() }
        self.settingsFileURL = settingsFileURL
            ?? actor.configPath.map { $0.canonicalFileURL }
            ?? QWSettingsManager.defaultSettingsFileURL
        self.enableAutoUpdateChecks = enableAutoUpdateChecks
        self.updateService = updateService ?? QWServerUpdateService()
        self.episodicStore = episodicStore
        self.sessionTree = sessionTree
        if !allowedWorkspaceRoots.isEmpty {
            actor.configureAllowedWorkspaceRoots(allowedWorkspaceRoots)
        }
    }

    public func start() throws {
        let startupPolicy = QWCoolingPolicy(rawValue: QWSettingsManager.shared.loadSettings().coolingPolicy) ?? .auto
        QWFanController.shared.setPolicy(startupPolicy)
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty, !trimmedHost.contains(where: \.isWhitespace) else {
            throw QWHttpServerError.invalidListenHost(host)
        }
        if !QWHttpSecurity.isLoopbackHost(trimmedHost) && !allowUnauthenticatedPublicBind && apiKey == nil {
            throw QWHttpServerError.unauthenticatedNonLoopbackBind(host: trimmedHost)
        }

        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.noDelay = true
        let params = NWParameters(tls: nil, tcp: tcpOptions)
        params.allowLocalEndpointReuse = true
        let nwPort: NWEndpoint.Port = port == 0 ? .any : (NWEndpoint.Port(rawValue: port) ?? .any)
        let listener: NWListener
        if trimmedHost == "0.0.0.0" || trimmedHost == "::" || trimmedHost == "*" {
            listener = try NWListener(using: params, on: nwPort)
        } else {
            params.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(trimmedHost), port: nwPort)
            listener = try NWListener(using: params)
        }
        self.listener = listener

        let readySem = DispatchSemaphore(value: 0)
        let stateLock = NSLock()
        nonisolated(unsafe) var startupError: NWError? = nil
        nonisolated(unsafe) var signaled = false

        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                let activePort = listener.port?.rawValue ?? self.port
                self.logger.log(.info, "HTTP Server listening on http://\(trimmedHost):\(activePort)")
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
        if enableAutoUpdateChecks {
            let svc = updateService
            Task { await svc.startScheduler() }
        }
    }

    public func stop() {
        listener?.cancel()
        listener = nil
        let svc = updateService
        Task { await svc.stopScheduler() }
    }

    private func handleConnection(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        let hub = QWConnectionCancellationHub()
        let accepted = connectionStateLock.withLock { () -> Bool in
            guard cancellationHubs.count < maxConcurrentConnections else {
                return false
            }
            cancellationHubs[id] = hub
            return true
        }
        guard accepted else {
            rejectOverCapacity(connection)
            return
        }
        let readTimeout = QWRequestReadTimeout(seconds: requestReadTimeoutSeconds, queue: queue) { [weak self, weak connection] in
            guard let self, let connection, !self.isConnectionTerminated(connection) else { return }
            self.sendSimpleResponse(connection, status: 408, body: "Request Timeout\n")
            self.queue.asyncAfter(deadline: .now() + 1.0) {
                connection.cancel()
            }
        }
        hub.attach(to: connection)
        hub.register { [weak self] in
            readTimeout.cancel()
            self?.cleanupConnectionState(for: id)
        }
        connection.start(queue: queue)
        receiveNextChunk(connection: connection, parser: QWHttpIncrementalParser(), readTimeout: readTimeout, emptyFrames: 0)
    }

    func cleanupConnectionState(for id: ObjectIdentifier) {
        connectionStateLock.withLock {
            cancellationHubs.removeValue(forKey: id)
            connectionOrigins.removeValue(forKey: id)
        }
    }

    private func rejectOverCapacity(_ connection: NWConnection) {
        logger.log(.warn, "Rejecting connection: max concurrent connections (\(maxConcurrentConnections)) reached")
        connection.start(queue: queue)
        let body = Data("Service Unavailable\n".utf8)
        let header = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(
            content: Data(header.utf8) + body,
            contentContext: .finalMessage,
            isComplete: true,
            completion: .contentProcessed { error in
                if error != nil {
                    connection.cancel()
                }
            }
        )
        drainRejectedConnection(connection, remainingBytes: 65_536)
        queue.asyncAfter(deadline: .now() + 2.0) {
            connection.cancel()
        }
    }

    private func drainRejectedConnection(_ connection: NWConnection, remainingBytes: Int) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, error in
            guard let self else {
                connection.cancel()
                return
            }
            let nextRemaining = remainingBytes - (data?.count ?? 0)
            if isComplete || error != nil || nextRemaining <= 0 {
                connection.cancel()
                return
            }
            self.drainRejectedConnection(connection, remainingBytes: nextRemaining)
        }
    }

    func registerCancellationHandler(for connection: NWConnection, _ handler: @escaping @Sendable () -> Void) {
        let id = ObjectIdentifier(connection)
        let hub = connectionStateLock.withLock { cancellationHubs[id] }
        if let hub {
            hub.register(handler)
            return
        }
        switch connection.state {
        case .cancelled, .failed:
            handler()
        default:
            break
        }
    }

    func isConnectionTerminated(_ connection: NWConnection) -> Bool {
        let id = ObjectIdentifier(connection)
        let hub = connectionStateLock.withLock { cancellationHubs[id] }
        if let hub {
            return hub.isTerminated
        }
        switch connection.state {
        case .cancelled, .failed:
            return true
        default:
            return false
        }
    }

    func generationShouldStop(for connection: NWConnection) -> (@Sendable () -> Bool) {
        let id = ObjectIdentifier(connection)
        if let flag = connectionStateLock.withLock({ cancellationHubs[id]?.cancelFlag }) {
            return { Task.isCancelled || flag.isCancelled }
        }
        return { [weak self] in
            Task.isCancelled || self?.isPeerGone(connection) == true
        }
    }

    func corsHeaderLines(for connection: NWConnection) -> String {
        let id = ObjectIdentifier(connection)
        let origin = connectionStateLock.withLock { connectionOrigins[id] }
        guard let allowed = QWHttpSecurity.resolvedAllowedOrigin(origin, allowlist: corsAllowedOrigins) else {
            return ""
        }
        return "Access-Control-Allow-Origin: \(allowed)\r\nAccess-Control-Expose-Headers: x-request-id, x-session-id, x-qwegn-decision-route\r\nVary: Origin\r\n"
    }

    private func receiveNextChunk(
        connection: NWConnection,
        parser: QWHttpIncrementalParser,
        readTimeout: QWRequestReadTimeout,
        emptyFrames: Int
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil {
                readTimeout.cancel()
                connection.cancel()
                return
            }
            guard let data, !data.isEmpty else {
                if isComplete || emptyFrames >= 4 {
                    readTimeout.cancel()
                    connection.cancel()
                    return
                }
                self.receiveNextChunk(
                    connection: connection,
                    parser: parser,
                    readTimeout: readTimeout,
                    emptyFrames: emptyFrames + 1
                )
                return
            }
            var nextParser = parser
            do {
                switch try nextParser.append(data) {
                case .needMoreData:
                    if isComplete {
                        readTimeout.cancel()
                        self.sendSimpleResponse(connection, status: 400, body: "Incomplete HTTP request\n")
                        return
                    }
                    self.receiveNextChunk(connection: connection, parser: nextParser, readTimeout: readTimeout, emptyFrames: 0)
                case .complete(let request):
                    readTimeout.cancel()
                    let id = ObjectIdentifier(connection)
                    if let origin = request.headers["origin"] {
                        self.connectionStateLock.withLock {
                            self.connectionOrigins[id] = origin
                        }
                    }
                    self.armEOFWatcher(on: connection, id: id)
                    self.dispatchRoute(
                        connection: connection,
                        method: request.method.uppercased(),
                        path: request.path,
                        headers: request.headers,
                        body: request.body
                    )
                }
            } catch let parseError as QWHttpParseError {
                readTimeout.cancel()
                self.sendSimpleResponse(connection, status: parseError.httpStatus, body: "\(parseError.message)\n")
            } catch {
                readTimeout.cancel()
                self.sendSimpleResponse(connection, status: 400, body: "Malformed HTTP request\n")
            }
        }
    }

    private func dispatchRoute(
        connection: NWConnection,
        method: String,
        path: String,
        headers: [String: String],
        body: Data
    ) {
        if method == "OPTIONS" {
            var corsHeader = "HTTP/1.1 204 No Content\r\n"
            corsHeader += corsHeaderLines(for: connection)
            corsHeader += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
            corsHeader += "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, x-request-id, x-session-id, x-qwegn-client, x-client-name, x-title\r\n"
            corsHeader += "Access-Control-Expose-Headers: x-request-id, x-session-id, x-qwegn-decision-route\r\n"
            corsHeader += "Access-Control-Max-Age: 86400\r\n"
            corsHeader += "Content-Length: 0\r\nConnection: close\r\n\r\n"
            let id = ObjectIdentifier(connection)
            connection.send(
                content: Data(corsHeader.utf8),
                contentContext: .finalMessage,
                isComplete: true,
                completion: .contentProcessed { [weak self] _ in
                    self?.cleanupConnectionState(for: id)
                    connection.cancel()
                }
            )
            return
        }

        // Clean path (strip query params)
        let cleanPath = path.components(separatedBy: "?").first ?? path
        let trace = QWClientIdentity.makeTraceContext(
            method: method, route: cleanPath, headers: headers, peerEndpoint: connection.endpoint
        )

        if let expectedKey = apiKey {
            let presented = QWHttpSecurity.extractPresentedToken(headers: headers)
            guard let presented, QWHttpSecurity.constantTimeEquals(presented, expectedKey) else {
                logger.log(.warn, "HTTP \(method) \(cleanPath) unauthorized | \(trace.logTag)")
                sendSimpleResponse(connection, status: 401, body: "Unauthorized\n")
                return
            }
        }

        if logger.getLevel() <= .debug && !cleanPath.hasPrefix("/ui") {
            let hdrs = QWClientIdentity.debugHeaderSummary(headers: headers)
            logger.log(.debug, "HTTP \(method) \(cleanPath) incoming | \(trace.logTag)\(hdrs.isEmpty ? "" : " | \(hdrs)")")
        }

        guard validateMutationRequest(
            connection: connection,
            method: method,
            cleanPath: cleanPath,
            headers: headers,
            body: body
        ) else {
            return
        }

        let routeTask = Task {
            do {
                switch (method, cleanPath) {
                case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
                    try await self.handleChatCompletions(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/completions"), ("POST", "/completions"):
                    try await self.handleCompletions(connection: connection, headers: headers, body: body, trace: trace)

                case ("GET", "/tokenizer_info"), ("GET", "/v1/tokenizer_info"):
                    let model = URLComponents(string: path)?.queryItems?.first(where: { $0.name == "model" })?.value
                    try await self.handleTokenizerInfo(connection: connection, model: model)

                case ("POST", "/tokenize"), ("POST", "/v1/tokenize"):
                    try await self.handleTokenize(connection: connection, body: body)

                case ("POST", "/detokenize"), ("POST", "/v1/detokenize"):
                    try await self.handleDetokenize(connection: connection, body: body)

                case ("POST", "/v1/responses"), ("POST", "/responses"):
                    try await self.handleCreateResponse(connection: connection, headers: headers, body: body, trace: trace)

                case ("GET", let path) where path.hasPrefix("/v1/responses/") || path.hasPrefix("/responses/"):
                    let prefix = path.hasPrefix("/v1/responses/") ? "/v1/responses/" : "/responses/"
                    let responseId = String(path.dropFirst(prefix.count))
                    if responseId.isEmpty || responseId.contains("/") {
                        self.sendOpenAIError(connection, status: 404, message: "Not found: \(method) \(cleanPath)")
                    } else {
                        self.handleGetResponse(connection: connection, responseId: responseId)
                    }

                case ("DELETE", let path) where path.hasPrefix("/v1/responses/") || path.hasPrefix("/responses/"):
                    let prefix = path.hasPrefix("/v1/responses/") ? "/v1/responses/" : "/responses/"
                    let responseId = String(path.dropFirst(prefix.count))
                    if responseId.isEmpty || responseId.contains("/") {
                        self.sendOpenAIError(connection, status: 404, message: "Not found: \(method) \(cleanPath)")
                    } else {
                        self.handleDeleteResponse(connection: connection, responseId: responseId)
                    }

                case ("POST", "/api/chat"):
                    try await self.handleOllamaChat(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/api/generate"):
                    try await self.handleOllamaGenerate(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/messages/count_tokens"), ("POST", "/messages/count_tokens"):
                    try await self.handleMessagesCountTokens(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/messages"), ("POST", "/messages"):
                    try await self.handleMessages(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/models/load"), ("POST", "/api/load"):
                    let startedAt = Date()
                    if !body.isEmpty,
                       let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                       let requestedModel = payload["model"] as? String {
                        try await self.actor.selectModelIfNeeded(requestedModel)
                    }
                    _ = try await self.actor.loadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)
                    self.logger.logSimpleRequest(
                        method: method, path: cleanPath, statusCode: 200,
                        client: trace.client, requestId: trace.requestId,
                        sessionId: trace.sessionId, peerHost: trace.peerHost,
                        detail: modelName, totalSec: Date().timeIntervalSince(startedAt)
                    )

                case ("POST", "/v1/models/unload"), ("POST", "/api/unload"):
                    let startedAt = Date()
                    let reclaimed = try await self.actor.unloadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: false, memoryReclaimedMB: reclaimed)
                    self.sendJsonResponse(connection, status: 200, object: resp)
                    self.logger.logSimpleRequest(
                        method: method, path: cleanPath, statusCode: 200,
                        client: trace.client, requestId: trace.requestId,
                        sessionId: trace.sessionId, peerHost: trace.peerHost,
                        detail: modelName, totalSec: Date().timeIntervalSince(startedAt)
                    )

                case ("GET", "/v1/models"), ("GET", "/models"):
                    self.handleModelsListRoute(connection: connection)

                case ("GET", "/v1/models/discovered"):
                    self.handleDiscoveredModelsRoute(connection: connection)

                case ("GET", let path) where path.hasPrefix("/v1/models/") || path.hasPrefix("/models/"):
                    let prefix = path.hasPrefix("/v1/models/") ? "/v1/models/" : "/models/"
                    self.handleGetModelRoute(connection: connection, requestedId: String(path.dropFirst(prefix.count)))

                case ("GET", "/api/version"):
                    self.sendJsonResponse(connection, status: 200, object: ["engine": "qwegine", "version": QWVersion.current])

                case ("POST", "/api/show"):
                    self.handleOllamaShow(connection: connection, body: body)

                case ("GET", "/api/tags"), ("GET", "/api/ps"):
                    self.handleOllamaTagsOrPsRoute(connection: connection, isPs: cleanPath == "/api/ps")

                case ("GET", "/metrics"):
                    let body = QWEngineTelemetryRegistry.shared.formatPrometheus()
                    self.sendRawData(connection, status: 200, contentType: "text/plain; version=0.0.4; charset=utf-8", data: Data(body.utf8))

                case ("GET", "/v1/metrics"), ("GET", "/api/metrics"):
                    let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                    self.sendJsonResponse(connection, status: 200, object: snapshot)

                case ("GET", "/v1/settings"), ("GET", "/api/settings"):
                    var settings = QWSettingsManager.shared.loadSettings()
                    if !settings.apiKey.isEmpty {
                        settings.apiKey = "********"
                    }
                    self.sendJsonResponse(connection, status: 200, object: settings)

                case ("POST", "/v1/settings"), ("POST", "/api/settings"):
                    await self.handleSaveSettingsRoute(connection: connection, body: body)

                case ("POST", "/v1/memory/inspect"), ("POST", "/api/memory/inspect"):
                    try await self.handleMemoryInspectRoute(connection: connection, body: body)

                case ("POST", "/v1/memory/index"), ("POST", "/api/memory/index"):
                    try await self.handleMemoryIndexRoute(connection: connection, body: body)

                case ("POST", "/v1/memory/mount"), ("POST", "/api/memory/mount"):
                    try await self.handleMemoryMountRoute(connection: connection, body: body)

                case ("POST", "/v1/memory/unmount"), ("POST", "/api/memory/unmount"):
                    await self.actor.unmountMemoryPack()
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/settings/reveal"), ("POST", "/api/settings/reveal"):
                    await self.revealDirectoryInFinder(
                        connection: connection,
                        rawDirectoryPath: QWPaths.rootDirectoryURL.path,
                        createIfInQwgRoot: true
                    )

                case ("POST", "/v1/logs/reveal"), ("POST", "/api/logs/reveal"):
                    let settings = QWSettingsManager.shared.loadSettings()
                    await self.revealDirectoryInFinder(
                        connection: connection,
                        rawDirectoryPath: settings.logDir,
                        createIfInQwgRoot: true
                    )

                case ("POST", "/v1/vision/screen-capture"), ("POST", "/api/vision/screen-capture"):
                    await self.handleScreenCaptureRoute(connection: connection, body: body)

                case ("POST", "/v1/engine/restart"), ("POST", "/api/restart"):
                    let startedAt = Date()
                    _ = try await self.actor.restartExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)
                    self.logger.logSimpleRequest(
                        method: method, path: cleanPath, statusCode: 200,
                        client: trace.client, requestId: trace.requestId,
                        sessionId: trace.sessionId, peerHost: trace.peerHost,
                        detail: modelName, totalSec: Date().timeIntervalSince(startedAt)
                    )

                case ("GET", "/v1/engine/info"), ("GET", "/api/engine/info"):
                    self.handleEngineInfoRoute(connection: connection)

                case ("POST", "/v1/decisions"):
                    await self.handleDecisionRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/decisions/batch"):
                    await self.handleDecisionBatchRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/systemone"):
                    await self.handleSystemOneRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/systemone/batch"):
                    await self.handleSystemOneBatchRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", "/v1/qwegn"):
                    await self.handleQwegnAskRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("POST", let path) where path.hasPrefix("/v1/cognitive/"):
                    await self.handleCognitiveRoute(connection: connection, path: path, body: body)

                case ("GET", let path) where path == "/v1/sessions" || path.hasPrefix("/v1/sessions/"):
                    self.handleSessionsGetRoute(connection: connection, path: path)

                case ("POST", let path) where path.hasPrefix("/v1/sessions/"):
                    await self.handleSessionsPostRoute(connection: connection, path: path, body: body)

                case ("POST", "/v1/embeddings"), ("POST", "/embeddings"):
                    await self.handleEmbeddingsRoute(connection: connection, body: body)

                case ("POST", "/v1/rerank"), ("POST", "/rerank"):
                    await self.handleRerankRoute(connection: connection, body: body)

                case ("POST", "/v1/decisions/model/load"):
                    await self.handleDecisionLoadRoute(connection: connection, body: body)

                case ("POST", "/v1/decisions/model/unload"):
                    await self.handleDecisionUnloadRoute(connection: connection)

                case ("GET", "/v1/decisions/status"):
                    self.handleDecisionStatusRoute(connection: connection)

                case ("POST", "/mcp"):
                    await self.handleMCPRoute(connection: connection, headers: headers, body: body, trace: trace)

                case ("GET", "/v1/update"), ("GET", "/api/update"):
                    await self.handleUpdateStatusRoute(connection: connection)

                case ("POST", "/v1/update/check"), ("POST", "/api/update/check"):
                    await self.handleUpdateCheckRoute(connection: connection, body: body)

                case ("POST", "/v1/update/skip"), ("POST", "/api/update/skip"):
                    await self.handleUpdateSkipRoute(connection: connection, body: body)

                case ("POST", "/v1/update/download"), ("POST", "/api/update/download"):
                    await self.handleUpdateDownloadRoute(connection: connection)

                case ("POST", "/v1/update/download/cancel"), ("POST", "/api/update/download/cancel"):
                    await self.handleUpdateCancelDownloadRoute(connection: connection)

                case ("POST", "/v1/update/install"), ("POST", "/api/update/install"):
                    await self.handleUpdateInstallRoute(connection: connection, body: body)

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
                    try self.handleHealthRoute(connection: connection)

                default:
                    let err = ServerErrorPayload(message: "Not found: \(method) \(cleanPath)", type: "invalid_request_error", code: 404)
                    self.sendJsonResponse(connection, status: 404, object: err)
                }
            } catch let accessErr as QWMemoryWorkspaceAccessError {
                let err = ServerErrorPayload(message: accessErr.message, type: "invalid_request_error", code: 403)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) workspace forbidden | \(trace.logTag): \(accessErr.message)")
                self.sendJsonResponse(connection, status: 403, object: err)
            } catch let valErr as QWMetadataValidationError {
                let err = ServerErrorPayload(message: valErr.description, type: "invalid_request_error", code: 400)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) invalid request | \(trace.logTag): \(valErr.description)")
                self.sendJsonResponse(connection, status: 400, object: err)
            } catch let memError as QWMemoryPackError {
                let code: Int
                switch memError {
                case .directoryNotFound, .cacheFileMissing:
                    code = 404
                case .noMatchingCacheArtifact:
                    code = 400
                }
                let err = ServerErrorPayload(message: memError.localizedDescription, type: "invalid_request_error", code: code)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) memory pack error | \(trace.logTag): \(memError.localizedDescription)")
                self.sendJsonResponse(connection, status: code, object: err)
            } catch let mmError as QWMultimodalError {
                let err = ServerErrorPayload(message: "\(mmError)", type: "invalid_request_error", code: 400)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) bad multimodal request | \(trace.logTag): \(mmError)")
                self.sendJsonResponse(connection, status: 400, object: err)
            } catch {
                if Task.isCancelled || self.isPeerGone(connection) { return }
                let err = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
                // The reason goes to the log, not only to the response body:
                // a 500 with no recorded reason is undebuggable after the fact.
                self.logger.log(.error, "HTTP \(method) \(cleanPath) failed | \(trace.logTag): \(error)")
                self.sendJsonResponse(connection, status: 500, object: err)
            }
        }
        registerCancellationHandler(for: connection) {
            routeTask.cancel()
        }
    }

    // The stream endpoints and the static UI live in QWHttpServer+Streams.swift.
    // POST /v1/chat/completions lives in QWHttpServer+ChatCompletions.swift.

}

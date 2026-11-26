// QWEngineLifecycleActor+Decisions.swift — Dual-slot decision sidecar & routing (ADR-001/002/010).

import Foundation
import Metal
import QWCore
import QWDecision
import QWEngine
import QWToken
import QWVision

/// Public snapshot describing the resident or unloaded System-1 decision sidecar slot.
public struct QWDecisionSidecarStatus: Codable, Sendable, Equatable {
    public let status: String
    public let loaded: Bool
    public let modelName: String?
    public let packPath: String?
    public let backendKind: String?
    public let computeUnit: String?
    public let maxContextLength: Int?
    public let estimatedBytes: UInt64?
    public let totalRequests: Int
    public let meanLatencyMs: Double?
    public let lastLatencyMs: Double?
    public let supportedKinds: [String]
    public let allowedLabels: [String]?
    public let failureDetail: String?
    public let visionEnabled: Bool?
    public let archetype: String?
    public let totalPromptTokens: Int

    enum CodingKeys: String, CodingKey {
        case status, loaded
        case modelName = "model_name"
        case packPath = "pack_path"
        case backendKind = "backend_kind"
        case computeUnit = "compute_unit"
        case maxContextLength = "max_context_length"
        case estimatedBytes = "estimated_bytes"
        case totalRequests = "total_requests"
        case meanLatencyMs = "mean_latency_ms"
        case lastLatencyMs = "last_latency_ms"
        case supportedKinds = "supported_kinds"
        case allowedLabels = "allowed_labels"
        case failureDetail = "failure_detail"
        case visionEnabled = "vision_enabled"
        case archetype
        case totalPromptTokens = "total_prompt_tokens"
    }

    public init(
        status: String = "unloaded",
        loaded: Bool = false,
        modelName: String? = nil,
        packPath: String? = nil,
        backendKind: String? = nil,
        computeUnit: String? = nil,
        maxContextLength: Int? = nil,
        estimatedBytes: UInt64? = nil,
        totalRequests: Int = 0,
        meanLatencyMs: Double? = nil,
        lastLatencyMs: Double? = nil,
        supportedKinds: [String] = [],
        allowedLabels: [String]? = nil,
        failureDetail: String? = nil,
        visionEnabled: Bool? = nil,
        archetype: String? = nil,
        totalPromptTokens: Int = 0
    ) {
        self.status = status
        self.loaded = loaded
        self.modelName = modelName
        self.packPath = packPath
        self.backendKind = backendKind
        self.computeUnit = computeUnit
        self.maxContextLength = maxContextLength
        self.estimatedBytes = estimatedBytes
        self.totalRequests = totalRequests
        self.meanLatencyMs = meanLatencyMs
        self.lastLatencyMs = lastLatencyMs
        self.supportedKinds = supportedKinds
        self.allowedLabels = allowedLabels
        self.failureDetail = failureDetail
        self.visionEnabled = visionEnabled
        self.archetype = archetype
        self.totalPromptTokens = totalPromptTokens
    }
}

/// Loaded decision runner paired with provenance metadata.
public struct QWLoadedDecisionSidecar: Sendable {
    public let runner: QWDecisionRunner
    public let modelName: String
    public let packPath: String
    public let backendKind: String
    public let computeUnit: String
    public let estimatedBytes: UInt64
    public let supportedKinds: [String]
    public let allowedLabels: [String]?
    public let visionEnabled: Bool?
    public let archetype: String?

    public init(
        runner: QWDecisionRunner,
        modelName: String,
        packPath: String,
        backendKind: String,
        computeUnit: String,
        estimatedBytes: UInt64,
        supportedKinds: [String] = QWDecisionKind.allCases.map(\.rawValue),
        allowedLabels: [String]? = nil,
        visionEnabled: Bool? = nil,
        archetype: String? = nil
    ) {
        self.runner = runner
        self.modelName = modelName
        self.packPath = packPath
        self.backendKind = backendKind
        self.computeUnit = computeUnit
        self.estimatedBytes = estimatedBytes
        self.supportedKinds = supportedKinds
        self.allowedLabels = allowedLabels
        self.visionEnabled = visionEnabled
        self.archetype = archetype
    }
}

/// Decision execution result plus which slot/model served it.
public struct QWDecisionExecutionEnvelope: Sendable, Equatable {
    public let response: QWDecisionResponse
    public let modelName: String
    public let backendKind: String

    public init(response: QWDecisionResponse, modelName: String, backendKind: String) {
        self.response = response
        self.modelName = modelName
        self.backendKind = backendKind
    }
}

/// Thread-safe snapshot box for nonisolated `/v1/decisions/status` and UI polling.
final class QWDecisionSidecarStatusBox: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshot = QWDecisionSidecarStatus()
    private var cumulativeLatencyMs: Double = 0

    func get() -> QWDecisionSidecarStatus {
        lock.withLock { snapshot }
    }

    func isLoaded() -> Bool {
        lock.withLock { snapshot.loaded }
    }

    func setLoading(modelName: String, packPath: String, computeUnit: String) {
        lock.withLock {
            snapshot = QWDecisionSidecarStatus(
                status: "loading",
                loaded: false,
                modelName: modelName,
                packPath: packPath,
                computeUnit: computeUnit
            )
            cumulativeLatencyMs = 0
        }
    }

    func setLoaded(_ sidecar: QWLoadedDecisionSidecar) {
        lock.withLock {
            let resolvedArchetype = sidecar.archetype
                ?? QWModelCatalog.describeDecisionArchetype(
                    sidecar.runner.backend.decisionConfig,
                    isCoreAI: sidecar.backendKind.hasPrefix("coreai")
                )
            snapshot = QWDecisionSidecarStatus(
                status: "loaded",
                loaded: true,
                modelName: sidecar.modelName,
                packPath: sidecar.packPath,
                backendKind: sidecar.backendKind,
                computeUnit: sidecar.computeUnit,
                maxContextLength: sidecar.runner.backend.maxContextLength,
                estimatedBytes: sidecar.estimatedBytes,
                totalRequests: 0,
                meanLatencyMs: nil,
                lastLatencyMs: nil,
                supportedKinds: sidecar.supportedKinds,
                allowedLabels: sidecar.allowedLabels,
                failureDetail: nil,
                visionEnabled: sidecar.visionEnabled,
                archetype: resolvedArchetype,
                totalPromptTokens: 0
            )
            cumulativeLatencyMs = 0
        }
    }

    func setFailed(_ message: String, modelName: String?, packPath: String?) {
        lock.withLock {
            snapshot = QWDecisionSidecarStatus(
                status: "failed",
                loaded: false,
                modelName: modelName,
                packPath: packPath,
                failureDetail: message
            )
        }
    }

    func setUnloaded() {
        lock.withLock {
            snapshot = QWDecisionSidecarStatus()
            cumulativeLatencyMs = 0
        }
    }

    func recordSuccess(latencyMs: Double, promptTokens: Int = 0) {
        lock.withLock {
            guard snapshot.loaded else { return }
            let count = snapshot.totalRequests + 1
            cumulativeLatencyMs += max(0, latencyMs)
            let mean = (cumulativeLatencyMs / Double(count) * 100.0).rounded() / 100.0
            let roundedLast = (max(0, latencyMs) * 100.0).rounded() / 100.0
            snapshot = QWDecisionSidecarStatus(
                status: snapshot.status,
                loaded: true,
                modelName: snapshot.modelName,
                packPath: snapshot.packPath,
                backendKind: snapshot.backendKind,
                computeUnit: snapshot.computeUnit,
                maxContextLength: snapshot.maxContextLength,
                estimatedBytes: snapshot.estimatedBytes,
                totalRequests: count,
                meanLatencyMs: mean,
                lastLatencyMs: roundedLast,
                supportedKinds: snapshot.supportedKinds,
                allowedLabels: snapshot.allowedLabels,
                failureDetail: nil,
                visionEnabled: snapshot.visionEnabled,
                archetype: snapshot.archetype,
                totalPromptTokens: snapshot.totalPromptTokens + max(0, promptTokens)
            )
        }
    }
}

extension QWEngineLifecycleActor {
    public typealias DecisionRunnerLoader = @Sendable (
        String,
        QWDecisionComputeUnit,
        Bool,
        QWGPUTurnArbitrator,
        QWServerLogger
    ) async throws -> QWLoadedDecisionSidecar

    nonisolated public func decisionSidecarStatus() -> QWDecisionSidecarStatus {
        decisionStatusBox.get()
    }

    nonisolated public func isDecisionSidecarLoaded() -> Bool {
        decisionStatusBox.isLoaded()
    }

    nonisolated public func residencyLeases() -> [QWResidencyLease] {
        residencyPool.currentLeases()
    }

    /// Hot-loads a decision sidecar alongside the primary model after UMA admission checks.
    @discardableResult
    public func loadDecisionSidecar(
        packPath: String,
        computeUnit: QWDecisionComputeUnit = .auto,
        loadVision: Bool = true
    ) async throws -> QWDecisionSidecarStatus {
        let trimmed = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QWDecisionError.invalidQuestion("decision model path must not be empty")
        }
        let expanded = (trimmed as NSString).expandingTildeInPath
        let resolvedName = Config.derivePackName(from: expanded)

        let estimatedSidecarBytes = Self.estimateDecisionSidecarBytes(at: expanded, loadVision: loadVision)
        let primaryWiredBytes = Int(Self.deviceAllocatedBytes())
        let admission = await residencyPool.admit(
            slot: .decisionSidecar,
            modelName: resolvedName,
            packPath: expanded,
            estimatedBytes: estimatedSidecarBytes,
            computeUnit: computeUnit.rawValue,
            measuredPrimaryWiredBytes: primaryWiredBytes,
            requestedWarmCacheBytes: config.cacheConfig.warmByteLimit,
            requestedSessions: config.maxConcurrentSessions
        )
        guard admission.admitted else {
            let reason = admission.reason ?? "insufficient unified memory for decision sidecar"
            decisionStatusBox.setFailed(reason, modelName: resolvedName, packPath: expanded)
            throw QWDecisionError.backendFailed(reason)
        }

        decisionStatusBox.setLoading(modelName: resolvedName, packPath: expanded, computeUnit: computeUnit.rawValue)
        let loader = self.decisionLoader
        let arbitrator = self.gpuTurnArbitrator
        let logger = self.logger
        do {
            let loadedSidecar = try await loader(expanded, computeUnit, loadVision, arbitrator, logger)
            self.decisionSidecar = loadedSidecar
            await residencyPool.commitLease(QWResidencyLease(
                slot: .decisionSidecar,
                modelName: loadedSidecar.modelName,
                packPath: loadedSidecar.packPath,
                estimatedBytes: loadedSidecar.estimatedBytes,
                computeUnit: loadedSidecar.computeUnit,
                bypassesGPUArbitrator: loadedSidecar.backendKind.hasPrefix("coreai")
            ))
            decisionStatusBox.setLoaded(loadedSidecar)
            return decisionStatusBox.get()
        } catch {
            decisionStatusBox.setFailed(Self.detail(for: error), modelName: resolvedName, packPath: expanded)
            throw error
        }
    }

    /// Updates live decision request defaults on the actor config without restarting the engine.
    public func updateDecisionDefaults(
        scoreReadout: String,
        truncation: String,
        alwaysEmitBox: Bool
    ) {
        config.decisionDefaultScoreReadout = scoreReadout
        config.decisionDefaultTruncation = truncation
        config.decisionAlwaysEmitBox = alwaysEmitBox
    }

    /// Unloads the decision sidecar without restarting or resetting the primary LLM.
    @discardableResult
    public func unloadDecisionSidecar() async -> QWDecisionSidecarStatus {
        decisionSidecar = nil
        _ = await residencyPool.releaseLease(for: .decisionSidecar)
        decisionStatusBox.setUnloaded()
        return decisionStatusBox.get()
    }

    /// Routes `request` to the co-resident decision sidecar or falls back to the resident primary LLM.
    public func decide(
        _ request: QWDecisionRequest,
        preferredModel: String? = nil
    ) async throws -> QWDecisionExecutionEnvelope {
        if decisionSidecar == nil && !config.decisionModelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let unit = QWDecisionComputeUnit(rawValue: config.decisionComputeUnit.lowercased()) ?? .auto
            _ = try await loadDecisionSidecar(
                packPath: config.decisionModelPackPath,
                computeUnit: unit,
                loadVision: config.decisionLoadVision
            )
        }
        let requested = preferredModel?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let sidecar = decisionSidecar, Self.matchesSidecar(requested: requested, sidecar: sidecar) {
            let resp = try await sidecar.runner.decide(request)
            decisionStatusBox.recordSuccess(latencyMs: resp.usage.latencyMs, promptTokens: resp.usage.promptTokens)
            return QWDecisionExecutionEnvelope(
                response: resp,
                modelName: sidecar.modelName,
                backendKind: sidecar.backendKind
            )
        }

        if case .loaded(let primary) = state,
           let unified = (primary.modelBackend as? QWUnifiedModelBackend)?.engine as? QWEngine {
            let adapter = QWEngineDecisionAdapter(
                engine: unified,
                decisionConfig: primary.descriptor.decisionConfig,
                gpuTurnArbitrator: nil,
                isSidecar: false
            )
            let runner = QWDecisionRunner(tokenizer: primary.tokenizer, backend: adapter)
            let resp = try await runner.decide(request)
            return QWDecisionExecutionEnvelope(
                response: resp,
                modelName: config.modelName,
                backendKind: "metal4_primary_fallback"
            )
        }

        throw QWDecisionError.backendFailed("no decision sidecar or resident primary model available")
    }

    /// Evaluates multiple decision requests in a coalesced GPU batch when sharing the active runner.
    public func decideBatch(
        _ requests: [QWDecisionRequest],
        preferredModel: String? = nil
    ) async throws -> [QWDecisionExecutionEnvelope] {
        guard !requests.isEmpty else { return [] }
        if requests.count == 1 {
            return [try await decide(requests[0], preferredModel: preferredModel)]
        }
        if decisionSidecar == nil && !config.decisionModelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let unit = QWDecisionComputeUnit(rawValue: config.decisionComputeUnit.lowercased()) ?? .auto
            _ = try await loadDecisionSidecar(
                packPath: config.decisionModelPackPath,
                computeUnit: unit,
                loadVision: config.decisionLoadVision
            )
        }
        let requested = preferredModel?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let sidecar = decisionSidecar, Self.matchesSidecar(requested: requested, sidecar: sidecar) {
            let resps = try await sidecar.runner.decideBatch(requests)
            for resp in resps {
                decisionStatusBox.recordSuccess(latencyMs: resp.usage.latencyMs, promptTokens: resp.usage.promptTokens)
            }
            return resps.map { resp in
                QWDecisionExecutionEnvelope(
                    response: resp,
                    modelName: sidecar.modelName,
                    backendKind: sidecar.backendKind
                )
            }
        }
        if case .loaded(let primary) = state,
           let unified = (primary.modelBackend as? QWUnifiedModelBackend)?.engine as? QWEngine {
            let adapter = QWEngineDecisionAdapter(
                engine: unified,
                decisionConfig: primary.descriptor.decisionConfig,
                gpuTurnArbitrator: nil,
                isSidecar: false
            )
            let runner = QWDecisionRunner(tokenizer: primary.tokenizer, backend: adapter)
            let resps = try await runner.decideBatch(requests)
            return resps.map { resp in
                QWDecisionExecutionEnvelope(
                    response: resp,
                    modelName: config.modelName,
                    backendKind: "metal4_primary_fallback"
                )
            }
        }
        throw QWDecisionError.backendFailed("no decision sidecar or resident primary model available")
    }

    /// Computes dense embeddings over the loaded embedding-capable sidecar backend.
    public func embedTexts(
        _ inputs: [String],
        preferredModel: String? = nil,
        pooling: QWDecisionModelConfig.PoolingKind? = nil,
        normalize: Bool = true,
        dimensions: Int? = nil
    ) async throws -> (embeddings: [[Float]], modelName: String, promptTokens: Int) {
        if decisionSidecar == nil && !config.decisionModelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let unit = QWDecisionComputeUnit(rawValue: config.decisionComputeUnit.lowercased()) ?? .auto
            _ = try await loadDecisionSidecar(
                packPath: config.decisionModelPackPath,
                computeUnit: unit,
                loadVision: config.decisionLoadVision
            )
        }
        let requested = preferredModel?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let sidecar = decisionSidecar, Self.matchesSidecar(requested: requested, sidecar: sidecar),
              let embBackend = sidecar.runner.backend as? any QWEmbeddingExecutionBackend else {
            throw QWDecisionError.backendFailed("no embedding-capable decision sidecar loaded")
        }
        let resolvedPooling: QWDecisionModelConfig.PoolingKind
        if let pooling {
            resolvedPooling = pooling
        } else if case .pooledClassifier(let p, _, _) = sidecar.runner.backend.decisionConfig.head {
            resolvedPooling = p
        } else {
            resolvedPooling = .meanNonPad
        }
        let tokenBatches = try inputs.map { text -> [Int] in
            let ids = try sidecar.runner.tokenizer.encode(text, addSpecialTokens: true)
            guard !ids.isEmpty else {
                throw QWDecisionError.invalidQuestion("embedding input must tokenize to at least 1 token")
            }
            return Array(ids.prefix(embBackend.maxContextLength))
        }
        let out = try await embBackend.embedBatch(
            tokenBatches: tokenBatches,
            pooling: resolvedPooling,
            normalize: normalize,
            truncateDimensions: dimensions
        )
        decisionStatusBox.recordSuccess(latencyMs: out.latencyMs, promptTokens: out.promptTokens)
        return (out.embeddings, sidecar.modelName, out.promptTokens)
    }

    private static func matchesSidecar(requested: String?, sidecar: QWLoadedDecisionSidecar) -> Bool {
        guard let requested, !requested.isEmpty else { return true }
        let low = requested.lowercased()
        if low == "decision" || low == "sidecar" || low == "laya" || low == "systemone" {
            return true
        }
        return requested == sidecar.modelName || requested == sidecar.packPath
    }

    /// Default production sidecar loader supporting CoreAI `.aimodel` bundles and QWP1/Safetensors Metal packs.
    public static let defaultDecisionLoader: DecisionRunnerLoader = { packPath, computeUnit, loadVision, arbitrator, logger in
        let url = URL(fileURLWithPath: packPath)
        let root = QWModelRoot(url: url)
        let modelName = Config.derivePackName(from: packPath)
        let calibration = loadOptionalCalibration(root: root, bundleURL: url)
        let tokenizer = try loadDecisionTokenizer(root: root, bundleURL: url)

        var desc: QWModelDescriptor? = nil
        if let cfgURL = root.configURL, let cfgData = try? Data(contentsOf: cfgURL) {
            desc = try? QWModelDescriptor.decode(configJSON: cfgData)
        }
        let allowedLabels: [String]? = {
            if case .pooledClassifier(_, _, let id2label) = desc?.decisionConfig?.head, id2label.count >= 3 {
                return id2label.sorted { $0.key < $1.key }.map(\.value)
            }
            return nil
        }()

        let archRaw = QWModelCatalog.describeDecisionArchetype(desc?.decisionConfig, isCoreAI: root.kind == .coreAIModel || computeUnit == .ane)

        if root.kind == .coreAIModel || computeUnit == .ane {
            let adapter = try await QWCoreAIDecisionAdapter.load(
                bundleURL: url,
                decisionConfig: desc?.decisionConfig,
                computeUnit: computeUnit == .auto ? .ane : computeUnit
            )
            let runner = QWDecisionRunner(tokenizer: tokenizer, backend: adapter, calibration: calibration)
            let estBytes = UInt64(max(0, estimateDecisionSidecarBytes(at: packPath, loadVision: loadVision)))
            logger.log(.info, "Loaded CoreAI decision sidecar '\(modelName)' (unit=\(adapter.activeComputeUnit), L_max=\(adapter.maxContextLength))")
            return QWLoadedDecisionSidecar(
                runner: runner,
                modelName: modelName,
                packPath: packPath,
                backendKind: "coreai_\(adapter.activeComputeUnit)",
                computeUnit: adapter.activeComputeUnit,
                estimatedBytes: estBytes,
                allowedLabels: allowedLabels,
                visionEnabled: desc != nil ? ((desc?.visDepth ?? 0) > 0 || desc?.visionTowerConfig != nil ? loadVision : false) : loadVision,
                archetype: archRaw ?? "coreai"
            )
        }

        let hasVision = (desc?.visDepth ?? 0) > 0 || desc?.visionTowerConfig != nil
        var visionBackend: (any QWVisionBackend)? = nil
        if hasVision {
            visionBackend = try QWVisionOnDemandBackend(
                modelPath: packPath,
                keepResident: loadVision,
                allowOnDemand: loadVision
            )
        }
        if let desc, QWDecisionPackEngine.canLoad(descriptor: desc, root: root) {
            let backend = try QWDecisionPackEngine(
                packPath: packPath,
                descriptor: desc,
                gpuTurnArbitrator: arbitrator,
                isSidecar: true,
                visionBackend: visionBackend,
                loadVision: loadVision
            )
            let runner = QWDecisionRunner(tokenizer: tokenizer, backend: backend, calibration: calibration)
            let estBytes = UInt64(max(0, estimateDecisionSidecarBytes(at: packPath, loadVision: loadVision)))
            logger.log(.info, "Loaded Metal 4 native decision sidecar '\(modelName)' (L_max=\(backend.maxContextLength), layers=\(desc.numLayers))")
            return QWLoadedDecisionSidecar(
                runner: runner,
                modelName: modelName,
                packPath: packPath,
                backendKind: "metal4",
                computeUnit: "metal",
                estimatedBytes: estBytes,
                allowedLabels: allowedLabels,
                visionEnabled: hasVision ? loadVision : false,
                archetype: archRaw
            )
        }

        let maxPos = desc?.maxPositionEmbeddings ?? 4096
        let buildOpts = QWEngineBuildOptions(
            packPath: packPath,
            maxTokens: maxPos,
            kvMode: "f16",
            prefillChunkSize: QWPrefillLimits.chunkRows
        )
        let engine = try QWEngine(options: buildOpts, descriptor: desc)
        let adapter = QWEngineDecisionAdapter(
            engine: engine,
            decisionConfig: desc?.decisionConfig,
            gpuTurnArbitrator: arbitrator,
            isSidecar: true,
            visionBackend: visionBackend,
            loadVision: loadVision
        )
        let runner = QWDecisionRunner(tokenizer: tokenizer, backend: adapter, calibration: calibration)
        let estBytes = UInt64(max(0, estimateDecisionSidecarBytes(at: packPath, loadVision: loadVision)))
        logger.log(.info, "Loaded Metal 4 decision sidecar '\(modelName)' (L_max=\(adapter.maxContextLength))")
        return QWLoadedDecisionSidecar(
            runner: runner,
            modelName: modelName,
            packPath: packPath,
            backendKind: "metal4",
            computeUnit: "metal",
            estimatedBytes: estBytes,
            allowedLabels: allowedLabels,
            visionEnabled: hasVision ? loadVision : false,
            archetype: archRaw
        )
    }

    private static func loadDecisionTokenizer(root: QWModelRoot, bundleURL: URL) throws -> QWTokenizer {
        let fm = FileManager.default
        let candidates = [
            root.assets.appendingPathComponent("tokenizer.json"),
            root.directory.appendingPathComponent("tokenizer.json"),
            root.directory.appendingPathComponent("tokenizer/tokenizer.json"),
            bundleURL.deletingLastPathComponent().appendingPathComponent("tokenizer/tokenizer.json"),
            bundleURL.deletingLastPathComponent().appendingPathComponent("tokenizer.json"),
        ]
        guard let tokURL = candidates.first(where: { fm.fileExists(atPath: $0.path) }),
              let tokJSON = fm.contents(atPath: tokURL.path) else {
            throw QWDecisionError.backendFailed("tokenizer.json not found for decision pack at \(bundleURL.path)")
        }
        let dir = tokURL.deletingLastPathComponent()
        return try QWTokenizer(
            json: tokJSON,
            tokenizerConfigJSON: fm.contents(atPath: dir.appendingPathComponent("tokenizer_config.json").path),
            specialTokensMapJSON: fm.contents(atPath: dir.appendingPathComponent("special_tokens_map.json").path),
            generationConfigJSON: fm.contents(atPath: dir.appendingPathComponent("generation_config.json").path)
        )
    }

    private static func loadOptionalCalibration(root: QWModelRoot, bundleURL: URL) -> QWCalibrationProfile {
        let fm = FileManager.default
        let candidates = [
            root.directory.appendingPathComponent("calibration.json"),
            root.assets.appendingPathComponent("calibration.json"),
            bundleURL.deletingLastPathComponent().appendingPathComponent("calibration.json"),
        ]
        for cand in candidates where fm.fileExists(atPath: cand.path) {
            if let data = fm.contents(atPath: cand.path),
               let profile = try? QWCalibrationProfile.load(fromData: data) {
                return profile
            }
        }
        return QWCalibrationProfile()
    }

    static func estimateDecisionSidecarBytes(at packPath: String, loadVision: Bool = true) -> Int {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: packPath, isDirectory: &isDir) else {
            return 256 << 20
        }
        if isDir.boolValue,
           let exact = QWDecisionPackEngine.estimateResidentBytes(at: packPath, loadVision: loadVision),
           exact > 0 {
            return exact
        }
        let url = URL(fileURLWithPath: packPath)
        let root = QWModelRoot(url: url)
        if (isDir.boolValue || root.kind != nil),
           let cfgURL = root.configURL,
           let cfgData = try? Data(contentsOf: cfgURL),
           let desc = try? QWModelDescriptor.decode(configJSON: cfgData) {
            let est = QWEngine.estimateContextMemory(
                descriptor: desc,
                packPath: packPath,
                contextTokens: min(desc.maxPositionEmbeddings, 4096),
                kvMode: .f16
            )
            if est.requiredBytes > 0 { return est.requiredBytes }
        }
        if isDir.boolValue,
           let enumerator = fm.enumerator(at: root.directory, includingPropertiesForKeys: [.fileSizeKey]) {
            var total = 0
            for case let fileURL as URL in enumerator {
                if !loadVision && fileURL.lastPathComponent.lowercased().contains("vision") { continue }
                if let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                    total += size
                }
            }
            if total > 0 { return total }
        } else if !isDir.boolValue,
                  let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize,
                  size > 0 {
            return size
        }
        return 256 << 20
    }
}

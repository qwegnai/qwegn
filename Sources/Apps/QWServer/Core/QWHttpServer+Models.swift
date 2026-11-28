// QWHttpServer+Models.swift — OpenAI /v1/models, /v1/models/:id, and /v1/models/discovered handlers.

import Foundation
import Network
import QWCore
import QWEngine

extension QWHttpServer {
    struct CatalogModelEntry {
        let id: String
        let canonicalName: String
        let loaded: Bool
        let path: String?
        let isDecision: Bool
        let isActivePrimary: Bool
        let isActiveSidecar: Bool
        let discovered: QWServerSettings.DiscoveredModel?
    }

    struct CatalogSnapshot {
        let entries: [CatalogModelEntry]
        let config: QWEngineLifecycleActor.Config
        let settings: QWServerSettings
        let loadedParams: QWEngineParametersInfo?
        let sidecar: QWDecisionSidecarStatus
        let engineState: QWEngineState
    }

    func catalogModelEntries() -> [CatalogModelEntry] {
        buildCatalogSnapshot().entries
    }

    private func buildCatalogSnapshot() -> CatalogSnapshot {
        let config = actor.getConfig()
        let modelName = actor.getModelName()
        let isLoaded = actor.isLoaded()
        let sidecar = actor.decisionSidecarStatus()
        let engineState = actor.engineState()
        let loadedParams = actor.getLoadedParamsInfo()
        let settings = QWSettingsManager.shared.loadSettings(from: settingsFileURL)
        let scanned = modelCatalogScanner()
        var byName: [String: QWServerSettings.DiscoveredModel] = [:]
        var byPath: [String: QWServerSettings.DiscoveredModel] = [:]
        for item in scanned {
            if byName[item.name] == nil { byName[item.name] = item }
            if !item.packPath.isEmpty {
                let canon = canonicalPackPath(item.packPath)
                if byPath[canon] == nil { byPath[canon] = item }
            }
        }
        func findDiscovered(name: String, path: String?) -> QWServerSettings.DiscoveredModel? {
            if let match = byName[name] { return match }
            if let path, !path.isEmpty {
                return byPath[canonicalPackPath(path)]
            }
            return nil
        }

        let activePackPath = config.packPath.isEmpty ? settings.modelPackPath : config.packPath
        var seen = Set<String>()
        var entries: [CatalogModelEntry] = []
        func appendIfNew(
            id: String,
            canonicalName: String,
            loaded: Bool,
            path: String?,
            isDecision: Bool,
            isActivePrimary: Bool,
            isActiveSidecar: Bool,
            discovered: QWServerSettings.DiscoveredModel?
        ) {
            guard !id.isEmpty, seen.insert(id).inserted else { return }
            entries.append(CatalogModelEntry(
                id: id,
                canonicalName: canonicalName,
                loaded: loaded,
                path: path,
                isDecision: isDecision,
                isActivePrimary: isActivePrimary,
                isActiveSidecar: isActiveSidecar,
                discovered: discovered
            ))
        }

        let pinActiveCatalog = config.pinModel && !activePackPath.isEmpty
        if !pinActiveCatalog, sidecar.loaded, let sidecarName = sidecar.modelName {
            let scPath = sidecar.packPath
            let scDisc = findDiscovered(name: sidecarName, path: scPath)
            appendIfNew(
                id: sidecarName,
                canonicalName: sidecarName,
                loaded: true,
                path: scPath,
                isDecision: true,
                isActivePrimary: false,
                isActiveSidecar: true,
                discovered: scDisc
            )
        }
        let activeDisc = findDiscovered(name: modelName, path: activePackPath)
        let activeAlias = settings.alias(for: modelName)
        if !activeAlias.isEmpty {
            appendIfNew(
                id: activeAlias,
                canonicalName: modelName,
                loaded: isLoaded,
                path: activePackPath,
                isDecision: activeDisc?.isDecisionCapable ?? false,
                isActivePrimary: true,
                isActiveSidecar: false,
                discovered: activeDisc
            )
        }
        appendIfNew(
            id: modelName,
            canonicalName: modelName,
            loaded: isLoaded,
            path: activePackPath,
            isDecision: activeDisc?.isDecisionCapable ?? false,
            isActivePrimary: true,
            isActiveSidecar: false,
            discovered: activeDisc
        )
        for disc in scanned where !disc.packPath.isEmpty && (disc.supported || disc.isDecisionCapable) {
            let isPrimary = disc.name == modelName
            let isSidecar = sidecar.loaded && disc.name == sidecar.modelName
            if pinActiveCatalog && !isPrimary {
                continue
            }
            let loadedHere = (isLoaded && isPrimary) || isSidecar
            let alias = settings.alias(for: disc.name)
            if !alias.isEmpty {
                appendIfNew(
                    id: alias,
                    canonicalName: disc.name,
                    loaded: loadedHere,
                    path: disc.packPath,
                    isDecision: disc.isDecisionCapable,
                    isActivePrimary: isPrimary,
                    isActiveSidecar: isSidecar,
                    discovered: disc
                )
            }
            appendIfNew(
                id: disc.name,
                canonicalName: disc.name,
                loaded: loadedHere,
                path: disc.packPath,
                isDecision: disc.isDecisionCapable,
                isActivePrimary: isPrimary,
                isActiveSidecar: isSidecar,
                discovered: disc
            )
        }
        return CatalogSnapshot(
            entries: entries,
            config: config,
            settings: settings,
            loadedParams: loadedParams,
            sidecar: sidecar,
            engineState: engineState
        )
    }

    func handleModelsListRoute(connection: NWConnection) {
        let snapshot = buildCatalogSnapshot()
        let models = snapshot.entries.map { buildModelObject(for: $0, snapshot: snapshot) }
        let metadata = snapshot.entries.map { item in
            let created = QWModelDiskStats.modifiedUnix(of: item.path)
            let desc = item.isDecision
                ? "Single-pass decision model (\(item.id)) supporting /v1/systemone and /v1/decisions."
                : "Qwegn Engine model '\(item.id)'."
            return TypeSafeModelMetadata(
                name: item.id,
                description: desc,
                releaseDate: ModelListResponse.formatReleaseDate(unix: created)
            )
        }
        sendJsonResponse(connection, status: 200, object: ModelListResponse(models: models, metadata: metadata))
    }

    func handleDiscoveredModelsRoute(connection: NWConnection) {
        struct DiscoveredInventoryResponse: Encodable {
            let models: [QWServerSettings.DiscoveredModel]
            let drafters: [QWServerSettings.DiscoveredDrafter]
        }
        let models = modelCatalogScanner()
        let drafters = QWServerSettings.scanDrafterSummaries(primaryModels: models)
        sendJsonResponse(
            connection,
            status: 200,
            object: DiscoveredInventoryResponse(models: models, drafters: drafters)
        )
    }

    func handleGetModelRoute(connection: NWConnection, requestedId: String) {
        let snapshot = buildCatalogSnapshot()
        let activeAlias = snapshot.settings.alias(for: snapshot.config.modelName)
        let displayModel = activeAlias.isEmpty ? snapshot.config.modelName : activeAlias
        let resolvedEntry: CatalogModelEntry?
        if requestedId.isEmpty {
            resolvedEntry = snapshot.entries.first { $0.id == displayModel }
                ?? snapshot.entries.first { $0.isActivePrimary }
        } else {
            resolvedEntry = snapshot.entries.first { $0.id == requestedId }
        }
        guard let entry = resolvedEntry else {
            let err = ServerErrorPayload(
                message: "The model '\(requestedId)' does not exist.",
                type: "invalid_request_error",
                code: 404
            )
            sendJsonResponse(connection, status: 404, object: err)
            return
        }
        let resp = buildModelObject(for: entry, snapshot: snapshot)
        sendJsonResponse(connection, status: 200, object: resp)
    }

    private func buildModelObject(for entry: CatalogModelEntry, snapshot: CatalogSnapshot) -> ModelObject {
        let profile: QWKnobProfile? = entry.canonicalName.isEmpty
            ? nil
            : QWSettingsManager.shared.loadProfile(forModel: entry.canonicalName, from: settingsFileURL)
        let profileContext = profile?.values["contextTokens"]?.intValue.flatMap { $0 > 0 ? $0 : nil }
        let profileMaxOut = profile?.values["maxTokens"]?.intValue.flatMap { $0 > 0 ? $0 : nil }

        let activeMaxPos = (entry.isActivePrimary && entry.loaded)
            ? snapshot.loadedParams?.modelDescriptor?.maxPositionEmbeddings
            : nil
        let rawTrainCtx = activeMaxPos ?? entry.discovered?.nativeWindow
        let trainingContext = (rawTrainCtx ?? 0) > 0 ? rawTrainCtx : nil

        let runtimeContext: Int?
        if entry.isActivePrimary {
            let loadedBudget = entry.loaded ? snapshot.loadedParams?.contextBudget : nil
            let validLoaded = loadedBudget.flatMap { $0 > 0 ? $0 : nil }
            let validCfg = snapshot.config.maxContext > 0 ? snapshot.config.maxContext : nil
            runtimeContext = validLoaded ?? validCfg ?? profileContext ?? trainingContext
        } else if entry.isActiveSidecar {
            runtimeContext = trainingContext
        } else {
            runtimeContext = profileContext ?? trainingContext
        }
        let effectiveContext = runtimeContext ?? trainingContext

        let genMaxNew = entry.path.flatMap { Self.readGenerationMaxNewTokens(atPackPath: $0) }
        let explicitCap: Int?
        if entry.isActivePrimary {
            let defaultServerMaxTokens = QWServerSettings().maxTokens
            let maxTokensSource = snapshot.config.sourceSummary["maxTokens"] ?? "default"
            let hasExplicitActiveMax = (maxTokensSource != "default"
                || snapshot.config.maxTokens != defaultServerMaxTokens) && snapshot.config.maxTokens > 0
            let activeMax = hasExplicitActiveMax ? snapshot.config.maxTokens : nil
            explicitCap = profileMaxOut ?? genMaxNew ?? activeMax
        } else {
            explicitCap = profileMaxOut ?? genMaxNew
        }
        let maxOutputTokens: Int? = if let effectiveContext {
            explicitCap.map { min(effectiveContext, $0) } ?? effectiveContext
        } else {
            explicitCap
        }

        let loadedDescHasVision = (snapshot.loadedParams?.modelDescriptor?.visDepth ?? 0) > 0
            || snapshot.loadedParams?.modelDescriptor?.visionTowerConfig != nil
        let packHasVision = entry.discovered?.hasVision
            ?? (entry.isActivePrimary && entry.loaded ? loadedDescHasVision : false)
        let visionAvailable: Bool
        if entry.isActivePrimary && entry.loaded {
            visionAvailable = packHasVision && snapshot.config.loadVision
        } else if entry.isActiveSidecar && entry.loaded {
            visionAvailable = snapshot.sidecar.visionEnabled ?? packHasVision
        } else {
            visionAvailable = packHasVision
        }
        let inputModalities = visionAvailable ? ["text", "image"] : ["text"]
        let outputModalities = ["text"]
        let architecture = ModelObject.Architecture(
            inputModalities: inputModalities,
            outputModalities: outputModalities
        )

        let supportsThinking: Bool?
        if entry.isActivePrimary && entry.loaded {
            supportsThinking = actor.supportsThinkingTags()
        } else if let path = entry.path, !path.isEmpty {
            supportsThinking = QWChatTemplate.detectSupportsThinkingTags(atPackPath: path)
        } else {
            supportsThinking = nil
        }

        let reasoning: Bool?
        let supportedReasoningLevels: [String]?
        let defaultReasoningLevel: String?
        if supportsThinking == true {
            reasoning = true
            supportedReasoningLevels = ["none", "low", "medium", "high", "xhigh"]
            let rawEffort = profile?.values["reasoningEffort"]?.stringValue
                ?? (entry.isActivePrimary ? snapshot.config.reasoningEffort : snapshot.settings.reasoningEffort)
            defaultReasoningLevel = QWChatPolicyResolver.normalizeReasoningEffort(rawEffort)
        } else if supportsThinking == false {
            reasoning = false
            supportedReasoningLevels = nil
            defaultReasoningLevel = nil
        } else {
            reasoning = nil
            supportedReasoningLevels = nil
            defaultReasoningLevel = nil
        }

        let rawCaps: [String]?
        if entry.isActivePrimary && entry.loaded {
            let liveCaps = snapshot.engineState.capabilities.map(\.rawValue).sorted()
            rawCaps = liveCaps.isEmpty ? entry.discovered?.capabilities : liveCaps
        } else {
            rawCaps = entry.discovered?.capabilities
        }
        let capabilities = (rawCaps?.isEmpty == false) ? rawCaps : nil
        let meta: ModelObject.Meta? = (effectiveContext != nil || trainingContext != nil)
            ? ModelObject.Meta(nCtx: effectiveContext, nCtxTrain: trainingContext)
            : nil

        return ModelObject(
            id: entry.id,
            loaded: entry.loaded,
            created: QWModelDiskStats.modifiedUnix(of: entry.path),
            contextLength: effectiveContext,
            maxModelLen: effectiveContext,
            maxOutputTokens: maxOutputTokens,
            maxCompletionTokens: maxOutputTokens,
            inputModalities: inputModalities,
            outputModalities: outputModalities,
            architecture: architecture,
            meta: meta,
            capabilities: capabilities,
            reasoning: reasoning,
            supportedReasoningLevels: supportedReasoningLevels,
            defaultReasoningLevel: defaultReasoningLevel
        )
    }

    private func canonicalPackPath(_ path: String) -> String {
        URL(fileURLWithPath: path.expandingTildePath)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private static func readGenerationMaxNewTokens(atPackPath packPath: String) -> Int? {
        let expanded = packPath.trimmingCharacters(in: .whitespacesAndNewlines).expandingTildePath
        guard !expanded.isEmpty else { return nil }
        let fm = FileManager.default
        guard fm.fileExists(atPath: expanded) else { return nil }
        let assets = QWModelRoot(url: URL(fileURLWithPath: expanded)).assets
        let genURL = assets.appendingPathComponent("generation_config.json")
        guard fm.fileExists(atPath: genURL.path),
              let data = try? Data(contentsOf: genURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let num = obj["max_new_tokens"] as? NSNumber,
              CFGetTypeID(num) != CFBooleanGetTypeID() else {
            return nil
        }
        let value = num.intValue
        return value > 0 ? value : nil
    }
}

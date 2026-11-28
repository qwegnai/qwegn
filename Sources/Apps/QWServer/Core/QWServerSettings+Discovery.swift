// QWServerSettings+Discovery.swift — Model aliases, directory scanning, and pre-launch validation.

import Foundation
import QWCore
import QWEngine


extension QWServerSettings {
    /// Returns the persisted alias for `modelName` (or empty string if none).
    public func alias(for modelName: String) -> String {
        if let mapped = modelAliases[modelName], !mapped.isEmpty {
            return mapped
        }
        if modelName == self.modelName {
            return modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    /// Updates or removes the persisted alias for `modelName`.
    public mutating func setAlias(_ alias: String, for modelName: String) {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if trimmed.isEmpty {
                modelAliases.removeValue(forKey: modelName)
            } else {
                modelAliases[modelName] = trimmed
            }
        }
        if modelName == self.modelName {
            self.modelAlias = trimmed
        }
    }

    /// Synchronizes `modelAlias` and `modelAliases` for the currently active `modelName`.
    public mutating func syncActiveModelAlias() {
        let trimmed = modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if !trimmed.isEmpty {
                modelAliases[modelName] = trimmed
            } else if let mapped = modelAliases[modelName], !mapped.isEmpty {
                self.modelAlias = mapped
                return
            } else {
                modelAliases.removeValue(forKey: modelName)
            }
        }
        self.modelAlias = trimmed
    }

    /// Removes aliases of models that are no longer present in `~/.qwg/models`.
    /// Returns true when at least one orphaned alias was removed.
    @discardableResult
    public mutating func pruneRemovedModels(discovered: [DiscoveredModel] = Self.scanModelsDirectory()) -> Bool {
        let installed = Set(discovered.filter { !$0.packPath.isEmpty }.map(\.name))
        guard !installed.isEmpty else { return false }
        // The active model may be served under a name of the user's
        // choosing; the pack that backs it says whether it is still
        // there. An alias whose model is installed is not an orphan.
        let installedPacks = Set(discovered.filter { !$0.packPath.isEmpty }
            .map { $0.packPath.expandingTildePath })
        let activePack = modelPackPath.expandingTildePath
        let activeInstalled = installedPacks.contains(activePack)
        var changed = false
        for key in Array(modelAliases.keys)
        where !installed.contains(key) && !(key == modelName && activeInstalled) {
            modelAliases.removeValue(forKey: key)
            changed = true
        }
        if !installed.contains(modelName) {
            let modelsRoot = QWPaths.modelsDirectoryPath
            if !activeInstalled && (activePack.isEmpty || activePack.hasPrefix(modelsRoot)) {
                if !modelAlias.isEmpty {
                    modelAlias = ""
                    changed = true
                }
            }
        } else if let mapped = modelAliases[modelName], modelAlias != mapped {
            modelAlias = mapped
            changed = true
        }
        return changed
    }

    /// Returns true when `raw` uses path syntax (`/`, `~`, `.`, `..`, `\`, NUL) rather than a registered model name or alias.
    public static func isArbitraryFilesystemModelPath(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.hasPrefix("/") || trimmed.hasPrefix("~") || trimmed.hasPrefix("./")
            || trimmed.hasPrefix("../") || trimmed == "." || trimmed == ".."
            || trimmed.contains("\\") || trimmed.contains("\0") {
            return true
        }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        return parts.contains { $0.isEmpty || $0 == "." || $0 == ".." }
    }

    /// Resolves a model reference (CLI `--model` argument, model name in `~/.qwg/models`, alias, or pack path)
    /// to `(packPath, modelName, alias)`.
    public func resolveModelReference(
        _ ref: String,
        discovered: [DiscoveredModel] = Self.scanModelsDirectory(),
        allowArbitraryFilesystemPaths: Bool = true,
        registeredModelsRoot: URL = QWPaths.modelsDirectoryURL
    ) -> (packPath: String, modelName: String, alias: String) {
        let trimmed = ref.trimmingCharacters(in: .whitespacesAndNewlines)
        let activeAlias = alias(for: modelName)
        guard !trimmed.isEmpty else { return (modelPackPath.expandingTildePath, modelName, activeAlias) }
        if !allowArbitraryFilesystemPaths && Self.isArbitraryFilesystemModelPath(trimmed) {
            return ("", "", "")
        }
        func isAllowedRegisteredPack(_ path: String) -> Bool {
            guard !path.isEmpty else { return false }
            if allowArbitraryFilesystemPaths { return true }
            return QWHttpSecurity.canonicalDirectoryContainment(
                URL(fileURLWithPath: path.expandingTildePath),
                inside: registeredModelsRoot
            )
        }
        // 0. Prefer the active model when `ref` matches its name or alias case-insensitively.
        if !modelPackPath.isEmpty && (trimmed.caseInsensitiveCompare(modelName) == .orderedSame
            || (!activeAlias.isEmpty && trimmed.caseInsensitiveCompare(activeAlias) == .orderedSame)) {
            return (modelPackPath.expandingTildePath, modelName, activeAlias)
        }
        // 1. Match by persisted alias in modelAliases
        if let (targetModel, targetAlias) = modelAliases.first(where: { $0.value == trimmed })
            ?? modelAliases.first(where: { $0.value.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            if let disc = discovered.first(where: { $0.name == targetModel }),
               !disc.packPath.isEmpty,
               isAllowedRegisteredPack(disc.packPath) {
                return (disc.packPath, disc.name, targetAlias)
            }
            if !Self.isArbitraryFilesystemModelPath(targetModel) {
                let candidate = (registeredModelsRoot.path as NSString).appendingPathComponent(targetModel)
                if isAllowedRegisteredPack(candidate), FileManager.default.fileExists(atPath: candidate) {
                    return (candidate, targetModel, targetAlias)
                }
            }
            if targetModel == modelName && !modelPackPath.isEmpty {
                return (modelPackPath.expandingTildePath, modelName, targetAlias)
            }
        }
        if !modelAlias.isEmpty && modelAlias.caseInsensitiveCompare(trimmed) == .orderedSame {
            return (modelPackPath.expandingTildePath, modelName, modelAlias)
        }
        // 2. Match by discovered model name in ~/.qwg/models
        if let disc = discovered.first(where: { $0.name == trimmed })
            ?? discovered.first(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }),
           !disc.packPath.isEmpty,
           isAllowedRegisteredPack(disc.packPath) {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        // 3. Direct path or relative to ~/.qwg/models or HuggingFace / MLX cache
        let fm = FileManager.default
        func resolvedTuple(_ path: String) -> (String, String, String) {
            let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: path), fileManager: fm)
            let name = QWModelRoot.canonicalModelName(for: resolvedURL)
            return (resolvedURL.path, name, alias(for: name))
        }
        let inModelsDir = (registeredModelsRoot.path as NSString).appendingPathComponent(trimmed)
        if !trimmed.contains("/") && !trimmed.hasPrefix("~")
            && isAllowedRegisteredPack(inModelsDir) && fm.fileExists(atPath: inModelsDir) {
            return resolvedTuple(inModelsDir)
        }
        guard allowArbitraryFilesystemPaths else {
            return ("", "", "")
        }
        let expanded = trimmed.expandingTildePath
        if !fm.fileExists(atPath: expanded) {
            let repoLeaf = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
            for rootDir in ["./models", QWPaths.modelsDirectoryPath] {
                for sub in [trimmed, repoLeaf] {
                    let cand = (rootDir as NSString).appendingPathComponent(sub)
                    if fm.fileExists(atPath: cand) { return resolvedTuple(cand) }
                }
            }
            // Check HuggingFace hub cache (~/.cache/huggingface/hub/models--<org>--<repo>) or ~/.mlx-serve/models/<org>/<repo>
            let hfHub = QWPaths.hfHubCacheDirectoryURL.path
            let sanitized = trimmed.replacingOccurrences(of: "/", with: "--")
            let hfRepoCandidate = (hfHub as NSString).appendingPathComponent(
                sanitized.hasPrefix("models--") ? sanitized : "models--\(sanitized)"
            )
            if fm.fileExists(atPath: hfRepoCandidate) { return resolvedTuple(hfRepoCandidate) }
            let mlxRoot = ("~/.mlx-serve/models" as NSString).expandingTildeInPath
            let mlxCandidate = (mlxRoot as NSString).appendingPathComponent(trimmed)
            if fm.fileExists(atPath: mlxCandidate) { return resolvedTuple(mlxCandidate) }
        }
        let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: expanded), fileManager: fm)
        let resolvedPath = resolvedURL.path
        let canonical = QWModelRoot.canonicalModelName(for: resolvedURL)
        let derivedName = canonical.isEmpty ? modelName : canonical
        if let disc = discovered.first(where: { $0.packPath == resolvedPath || $0.packPath == expanded }) {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        return (resolvedPath, derivedName, alias(for: derivedName))
    }

    /// Scans ~/.qwg/models and the Hugging Face hub cache for installed primary model packs.
    public static func scanModelsDirectory() -> [DiscoveredModel] {
        let drafters = scanDrafterModelsDirectory()
        return QWModelCatalog.scanInstalledPrimaryModels().map { e in
            var caps = Set(e.capabilities.map(\.rawValue))
            if e.supported {
                let hasMatchedDrafter = loadPrimaryDescriptor(packPath: e.path)
                    .map { !compatibleDrafterModels(for: $0, in: drafters).isEmpty } ?? false
                if e.capabilities.contains(.mtp) || hasMatchedDrafter {
                    caps.insert(QWKnobCapability.dflash.rawValue)
                }
            }
            return DiscoveredModel(name: e.name, packPath: e.path, family: e.family?.rawValue,
                                   nativeWindow: e.nativeWindow, supported: e.supported, reason: e.reason,
                                   capabilities: caps.sorted(),
                                   maxSpeculativeDrafts: e.maxSpeculativeDrafts,
                                   hasLSASidecar: e.hasLSASidecar, lsaSidecarPath: e.lsaSidecarPath,
                                   isDecisionCapable: e.isDecisionCapable,
                                   isCoreAIBundle: e.isCoreAIBundle,
                                   decisionArchetype: e.decisionArchetype,
                                   hasVision: caps.contains(QWKnobCapability.vision.rawValue))
        }
    }

    /// Parsed `config.json` geometry and backend metadata for an installed DFlash 2 or DSpine drafter pack.
    public typealias DrafterMetadata = QWDrafterMetadata

    /// HTTP/WebUI summary of an installed DFlash 2 or DSpine drafter pack and compatible primary models.
    public struct DiscoveredDrafter: Codable, Sendable, Equatable {
        public let name: String
        public let packPath: String
        public let backend: String
        public let maxDrafts: Int
        public let compatibleModels: [String]
    }

    /// Scans `directory` (defaults to `~/.qwg/models`) for installed DFlash 2 and DSpine drafter packs.
    public static func scanDrafterModelsDirectory(
        in directory: String = QWPaths.modelsDirectoryPath
    ) -> [DiscoveredModel] {
        QWModelCatalog.scanDrafters(in: directory).map {
            DiscoveredModel(name: $0.name, packPath: $0.path)
        }
    }

    /// Reads drafter geometry and backend metadata from `packPath`'s `config.json`.
    public static func drafterMetadata(atPackPath packPath: String) -> DrafterMetadata? {
        QWModelCatalog.drafterMetadata(at: packPath)
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferDrafterBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        if let packPath, let meta = drafterMetadata(atPackPath: packPath) {
            return (meta.backend, meta.maxDrafts)
        }
        return QWDFlashConfig.inferBackendAndMaxDrafts(packPath: packPath)
    }

    /// Normalizes `"dflash"` / `"dflash2"` / `"dspine"` backend strings to canonical external drafter names.
    public static func normalizeExternalDrafterBackend(_ backend: String?) -> String? {
        guard let lower = backend?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !lower.isEmpty else {
            return nil
        }
        if lower == "dspine" { return "dspine" }
        if lower == "dflash2" || lower == "dflash" { return "dflash2" }
        return nil
    }

    /// Decodes a primary model's `QWModelDescriptor` from its `config.json` without reading tokenizer assets.
    public static func loadPrimaryDescriptor(packPath: String) -> QWModelDescriptor? {
        QWModelCatalog.loadPrimaryDescriptor(at: packPath)
    }

    /// Returns true when drafter metadata matches target `hidden`, `vocab`, and `target_layer_ids` range/count.
    public static func isDrafterCompatible(
        metadata: DrafterMetadata,
        with target: QWModelDescriptor
    ) -> Bool {
        metadata.isCompatible(with: target)
    }

    /// Returns true when the drafter pack at `packPath` has `config.json` geometry matching `target`.
    public static func isDrafterCompatible(packPath: String, with target: QWModelDescriptor) -> Bool {
        guard let meta = drafterMetadata(atPackPath: packPath) else { return false }
        return isDrafterCompatible(metadata: meta, with: target)
    }

    /// Filters `drafters` to packs whose `config.json` geometry matches `target`.
    public static func compatibleDrafterModels(
        for target: QWModelDescriptor,
        in drafters: [DiscoveredModel] = scanDrafterModelsDirectory()
    ) -> [DiscoveredModel] {
        drafters.filter { !$0.packPath.isEmpty && isDrafterCompatible(packPath: $0.packPath, with: target) }
    }

    /// Selects the best geometry-matched drafter pack for `target`, preferring `preferredBackend`.
    public static func resolveDefaultDrafterPack(
        for target: QWModelDescriptor?,
        preferredBackend: String? = nil,
        drafters: [DiscoveredModel] = scanDrafterModelsDirectory(),
        targetModelName: String? = nil
    ) -> DiscoveredModel? {
        let normalizedBackend = normalizeExternalDrafterBackend(preferredBackend)
        guard let target else {
            if let normalizedBackend,
               let match = drafters.first(where: {
                   !$0.packPath.isEmpty && inferDrafterBackendAndMaxDrafts(packPath: $0.packPath).backend == normalizedBackend
               }) {
                return match
            }
            return drafters.first(where: { !$0.packPath.isEmpty })
        }

        let affinityTokens = modelAffinityTokens(for: targetModelName)
        let candidates: [(model: DiscoveredModel, meta: DrafterMetadata)] = drafters.compactMap { model in
            guard !model.packPath.isEmpty,
                  let meta = drafterMetadata(atPackPath: model.packPath),
                  isDrafterCompatible(metadata: meta, with: target) else {
                return nil
            }
            return (model, meta)
        }
        return candidates.sorted { lhs, rhs in
            let lScore = drafterMatchScore(lhs, target: target, preferredBackend: normalizedBackend, affinityTokens: affinityTokens)
            let rScore = drafterMatchScore(rhs, target: target, preferredBackend: normalizedBackend, affinityTokens: affinityTokens)
            if lScore != rScore { return lScore > rScore }
            return lhs.model.name < rhs.model.name
        }.first?.model
    }

    /// Returns an explicit `configuredPath` or resolves a target-matched default drafter pack when empty.
    public static func effectiveDrafterPackPath(
        configuredPath: String?,
        for target: QWModelDescriptor?,
        preferredBackend: String?,
        drafters: [DiscoveredModel] = scanDrafterModelsDirectory(),
        targetModelName: String? = nil
    ) -> String? {
        let trimmed = (configuredPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return trimmed
        }
        return resolveDefaultDrafterPack(
            for: target,
            preferredBackend: preferredBackend,
            drafters: drafters,
            targetModelName: targetModelName
        )?.packPath
    }

    /// Returns true when the target model supports DFlash 2 / DSpine speculative decoding.
    public static func supportsDFlash2(
        descriptor: QWModelDescriptor?,
        discoveredModel: DiscoveredModel? = nil,
        drafters: [DiscoveredModel] = []
    ) -> Bool {
        if let disc = discoveredModel,
           disc.capabilities.contains(QWKnobCapability.dflash.rawValue)
            || disc.capabilities.contains("mtp") {
            return true
        }
        if let desc = descriptor {
            return desc.capabilities.contains(.mtp)
                || !compatibleDrafterModels(for: desc, in: drafters).isEmpty
        }
        if discoveredModel != nil {
            return false
        }
        return true
    }

    /// Returns true when the target model supports `backend` (`"dflash2"` or `"dspine"`) and has
    /// at least one matching drafter pack installed (or configured via `configuredPackPath`).
    public static func supportsExternalDrafter(
        backend: String,
        descriptor: QWModelDescriptor?,
        discoveredModel: DiscoveredModel? = nil,
        drafters: [DiscoveredModel] = scanDrafterModelsDirectory(),
        configuredPackPath: String? = nil
    ) -> Bool {
        guard let canonical = normalizeExternalDrafterBackend(backend) else { return false }
        guard supportsDFlash2(descriptor: descriptor, discoveredModel: discoveredModel, drafters: drafters) else {
            return false
        }
        var packPaths = drafters.map(\.packPath).filter { !$0.isEmpty }
        if let trimmed = configuredPackPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !trimmed.isEmpty, !packPaths.contains(trimmed) {
            packPaths.append(trimmed)
        }
        if let desc = descriptor {
            return packPaths.contains { path in
                guard let meta = drafterMetadata(atPackPath: path),
                      isDrafterCompatible(metadata: meta, with: desc) else {
                    return false
                }
                return meta.backend == canonical
            }
        }
        if !packPaths.isEmpty {
            return packPaths.contains { inferDrafterBackendAndMaxDrafts(packPath: $0).backend == canonical }
        }
        return false
    }

    /// Returns WebUI/HTTP drafter summaries paired with geometry-compatible installed primary model names.
    public static func scanDrafterSummaries(
        primaryModels: [DiscoveredModel] = scanModelsDirectory(),
        drafters: [DiscoveredModel] = scanDrafterModelsDirectory()
    ) -> [DiscoveredDrafter] {
        var descriptors: [String: QWModelDescriptor] = [:]
        for model in primaryModels where model.supported && !model.packPath.isEmpty {
            if let desc = loadPrimaryDescriptor(packPath: model.packPath) {
                descriptors[model.packPath] = desc
            }
        }
        return drafters.compactMap { drafter in
            guard !drafter.packPath.isEmpty,
                  let meta = drafterMetadata(atPackPath: drafter.packPath) else {
                return nil
            }
            let compat = primaryModels.compactMap { model -> String? in
                guard model.supported,
                      let desc = descriptors[model.packPath],
                      isDrafterCompatible(metadata: meta, with: desc) else {
                    return nil
                }
                return model.name
            }
            return DiscoveredDrafter(
                name: drafter.name,
                packPath: drafter.packPath,
                backend: meta.backend,
                maxDrafts: meta.maxDrafts,
                compatibleModels: compat
            )
        }
    }

    /// Heals stale or cross-model `dflashDraftPackPath` and `mtpBackend` selections when switching
    /// primary models or loading profiles whose referenced drafter pack is missing or incompatible.
    @discardableResult
    public mutating func reconcileStaleDrafterSelection(
        for targetDescriptor: QWModelDescriptor?,
        drafters: [DiscoveredModel] = scanDrafterModelsDirectory()
    ) -> Bool {
        let trimmedPack = dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let extBackend = Self.normalizeExternalDrafterBackend(mtpBackend)
        let packMeta = trimmedPack.isEmpty ? nil : Self.drafterMetadata(atPackPath: trimmedPack)
        let packValidForTarget: Bool
        if let packMeta {
            packValidForTarget = targetDescriptor.map { Self.isDrafterCompatible(metadata: packMeta, with: $0) } ?? true
        } else {
            packValidForTarget = false
        }
        let backendSatisfied = extBackend == nil || (packMeta?.backend == extBackend)
        if (trimmedPack.isEmpty && extBackend == nil) || (packValidForTarget && backendSatisfied) {
            return false
        }

        if let targetDescriptor,
           let replacement = Self.resolveDefaultDrafterPack(
               for: targetDescriptor,
               preferredBackend: extBackend,
               drafters: drafters,
               targetModelName: modelName
           ) {
            var changed = false
            if dflashDraftPackPath != replacement.packPath {
                dflashDraftPackPath = replacement.packPath
                changed = true
            }
            if extBackend != nil {
                let inferred = Self.inferDrafterBackendAndMaxDrafts(packPath: replacement.packPath)
                if mtpBackend != inferred.backend {
                    mtpBackend = inferred.backend
                    changed = true
                }
                if mtpDrafts > inferred.maxDrafts || mtpDrafts <= 0 {
                    mtpDrafts = inferred.maxDrafts
                    changed = true
                }
            }
            return changed
        }

        var changed = false
        if !trimmedPack.isEmpty {
            dflashDraftPackPath = ""
            changed = true
        }
        if extBackend != nil {
            mtpBackend = "variational"
            let nativeCap = max(1, targetDescriptor?.maxSpeculativeDrafts ?? 3)
            if mtpDrafts > nativeCap {
                mtpDrafts = min(3, nativeCap)
            }
            changed = true
        }
        return changed
    }

    private static func drafterMatchScore(
        _ entry: (model: DiscoveredModel, meta: DrafterMetadata),
        target: QWModelDescriptor,
        preferredBackend: String?,
        affinityTokens: Set<String>
    ) -> (Int, Int, Int, Int) {
        let backendMatch: Int = if let preferredBackend {
            entry.meta.backend == preferredBackend ? 1 : 0
        } else {
            entry.meta.backend == "dflash2" ? 1 : 0
        }
        let targetLayersMatch: Int = if let ntl = entry.meta.numTargetLayers {
            ntl == target.numLayers ? 2 : 0
        } else {
            1
        }
        let affinityScore = affinityTokens.isEmpty
            ? 0
            : modelAffinityTokens(for: entry.model.name).intersection(affinityTokens).count
        let quantRank = QWDFlashConfig.drafterQuantPreferenceRank(entry.model.packPath)
        return (backendMatch, targetLayersMatch, affinityScore, quantRank)
    }

    private static func modelAffinityTokens(for name: String?) -> Set<String> {
        guard let raw = name?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return []
        }
        let ignored: Set<String> = ["dflash", "dflash2", "dspine", "b16", "qwen3", "8", "mixed", "fast"]
        let parts = raw.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
        return Set(parts.filter { tok in
            tok.count >= 2
                && !ignored.contains(tok)
                && !tok.hasPrefix("qwp")
                && !tok.hasPrefix("hq")
                && !tok.hasPrefix("aq")
                && !tok.hasPrefix("shvq")
                && !tok.hasPrefix("vq")
        })
    }

    /// Validates the currently selected primary model pack (fail-fast check before server launch or upon UI selection).
    public func validatePrimaryModel() throws -> QWModelDescriptor {
        let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
        let candidatePath = ref.packPath.isEmpty ? modelName : ref.packPath
        let assets = try QWModelRuntimeAssets.load(packPath: candidatePath)
        return assets.descriptor
    }

    /// Validates the DFlash 2 / DSpine drafter model configuration against the selected primary model.
    public func validateDFlashConfiguration() throws {
        let descriptor = try validatePrimaryModel()
        let fallbackPath = mtpBackend.lowercased() == "dspine" ? QWDFlashConfig.defaultDSpinePackPath : dflashDraftPackPath
        let effPath = Self.effectiveDrafterPackPath(
            configuredPath: dflashDraftPackPath,
            for: descriptor,
            preferredBackend: mtpBackend,
            targetModelName: modelName
        ) ?? fallbackPath
        let maxD = Self.inferDrafterBackendAndMaxDrafts(packPath: effPath).maxDrafts
        let drafts = max(1, min(maxD, mtpDrafts))
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: descriptor,
            packPath: effPath,
            drafts: drafts,
            selectorTopK: dflashSelectorTopK
        )
    }

    /// Validates both the primary model and, when DFlash 2 / DSpine speculative decoding is enabled, the drafter pack.
    public func validateBeforeLaunch() throws {
        guard !modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
              !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        let descriptor = try validatePrimaryModel()
        let isExternalDrafter = ["dflash2", "dflash", "dspine"].contains(mtpBackend.lowercased())
        if speculativeDecodingEnabled && isExternalDrafter && mtpDrafts > 0 {
            let fallbackPath = mtpBackend.lowercased() == "dspine" ? QWDFlashConfig.defaultDSpinePackPath : dflashDraftPackPath
            let effPath = Self.effectiveDrafterPackPath(
                configuredPath: dflashDraftPackPath,
                for: descriptor,
                preferredBackend: mtpBackend,
                targetModelName: modelName
            ) ?? fallbackPath
            try QWDFlashConfig.validateDrafter(targetDescriptor: descriptor, packPath: effPath,
                                               drafts: mtpDrafts, selectorTopK: dflashSelectorTopK)
        }
        guard QWLSAMode.resolve(lsaMode) != nil else {
            throw QWEngineError.state("Unknown LSA mode '\(lsaMode)' (expected off, auto, sidecar, svd, or qsa-cluster)")
        }
        if lsaMode.lowercased() == "sidecar" {
            let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
            guard QWLSAConfig.resolveSidecarURL(packPath: ref.packPath,
                                                explicitSidecarPath: lsaSidecarPath.isEmpty ? nil : lsaSidecarPath) != nil else {
                throw QWEngineError.state("LSA mode 'sidecar' requires a .qlsa sidecar file in '\(ref.packPath)' or via lsaSidecarPath")
            }
        }
    }
}

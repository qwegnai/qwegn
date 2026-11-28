import Foundation

/// One installed model folder, classified from its own `assets/config.json` only.
public struct QWModelCatalogEntry: Equatable, Sendable {
    public let name: String
    public let path: String
    public let modelType: String?
    public let family: QWModelDescriptor.Family?
    public let nativeWindow: Int?
    public let capabilities: Set<QWModelDescriptor.Capability>
    public let maxSpeculativeDrafts: Int
    /// Whether a `.qlsa` (`model.qlsa`) Latent Sparse Attention sidecar is present in the model directory.
    public let hasLSASidecar: Bool
    /// Full path to `model.qlsa` when present in the model directory.
    public let lsaSidecarPath: String?
    /// True when the pack carries native `.decision` metadata or a CoreAI `.aimodel` bundle.
    public let isDecisionCapable: Bool
    /// True when the pack contains a compiled CoreAI `.aimodel` ANE bundle.
    public let isCoreAIBundle: Bool
    /// Optional human-readable decision archetype summary (`verbalizer`, `optionMarkers`, `pooledClassifier`).
    public let decisionArchetype: String?
    /// nil when the model is runnable; otherwise why it is not.
    public let reason: String?

    public init(name: String, path: String, modelType: String?, family: QWModelDescriptor.Family?,
                nativeWindow: Int?, reason: String?,
                capabilities: Set<QWModelDescriptor.Capability> = [],
                maxSpeculativeDrafts: Int = 0,
                hasLSASidecar: Bool = false,
                lsaSidecarPath: String? = nil,
                isDecisionCapable: Bool = false,
                isCoreAIBundle: Bool = false,
                decisionArchetype: String? = nil) {
        self.name = name
        self.path = path
        self.modelType = modelType
        self.family = family
        self.nativeWindow = nativeWindow
        self.reason = reason
        self.capabilities = capabilities
        self.maxSpeculativeDrafts = maxSpeculativeDrafts
        self.hasLSASidecar = hasLSASidecar
        self.lsaSidecarPath = lsaSidecarPath
        self.isDecisionCapable = isDecisionCapable
        self.isCoreAIBundle = isCoreAIBundle
        self.decisionArchetype = decisionArchetype
    }

    public var supported: Bool { reason == nil }
}

/// Parsed `config.json` geometry and backend metadata for a DFlash 2 or DSpine drafter pack.
public struct QWDrafterMetadata: Equatable, Sendable {
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numLayers: Int
    public let targetLayerIds: [Int]
    public let numTargetLayers: Int?
    public let backend: String
    public let maxDrafts: Int

    public init(
        hiddenSize: Int,
        vocabSize: Int,
        numLayers: Int,
        targetLayerIds: [Int],
        numTargetLayers: Int?,
        backend: String,
        maxDrafts: Int
    ) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.targetLayerIds = targetLayerIds
        self.numTargetLayers = numTargetLayers
        self.backend = backend
        self.maxDrafts = maxDrafts
    }

    /// Returns true when drafter metadata matches target `hidden`, `vocab`, and `target_layer_ids` range/count.
    public func isCompatible(with target: QWModelDescriptor) -> Bool {
        hiddenSize == target.hidden
            && vocabSize == target.vocab
            && !targetLayerIds.isEmpty
            && targetLayerIds.count == numLayers
            && targetLayerIds.allSatisfy { (0..<target.numLayers).contains($0) }
    }
}

/// One installed DFlash 2 or DSpine drafter folder discovered from its `config.json`.
public struct QWDrafterCatalogEntry: Equatable, Sendable {
    public let name: String
    public let path: String
    public let metadata: QWDrafterMetadata?

    public init(name: String, path: String, metadata: QWDrafterMetadata?) {
        self.name = name
        self.path = path
        self.metadata = metadata
    }
}

/// Scans model directories and probes primary and drafter pack descriptors from `config.json`.
public enum QWModelCatalog {
    public static func scan(directory: String) -> [QWModelCatalogEntry] {
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: directory) else { return [] }
        return items.filter { !$0.hasPrefix(".") }.sorted().map { item in
            classify(name: item, path: (directory as NSString).appendingPathComponent(item))
        }
    }

    /// Runnable models in a Hugging Face hub cache (`models--<org>--<repo>/snapshots/<rev>`).
    /// Repos without a snapshot, or whose model the engine does not serve, are left out:
    /// a hub cache holds many non-LLM repos that must not show up as unsupported models.
    public static func scanHuggingFaceHub(directory: String) -> [QWModelCatalogEntry] {
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: directory) else { return [] }
        return items.filter { $0.hasPrefix("models--") }.sorted().compactMap { item in
            let repoDir = URL(fileURLWithPath: (directory as NSString).appendingPathComponent(item), isDirectory: true)
            let snapshot = QWModelRoot.resolveSnapshotDirectory(repoDir)
            guard snapshot.path != repoDir.path else { return nil }
            let entry = classify(name: QWModelRoot.canonicalModelName(for: repoDir), path: snapshot.path)
            return entry.supported ? entry : nil
        }
    }

    /// Scans local `~/.qwg/models` and the Hugging Face hub cache, preferring local packs on name collisions.
    public static func scanInstalledPrimaryModels(
        localDirectory: String = QWPaths.modelsDirectoryPath,
        hfHubDirectory: String = QWPaths.hfHubCacheDirectoryURL.path,
        createLocalDirectoryIfMissing: Bool = true
    ) -> [QWModelCatalogEntry] {
        let fm = FileManager.default
        if createLocalDirectoryIfMissing && !fm.fileExists(atPath: localDirectory) {
            try? fm.createDirectory(atPath: localDirectory, withIntermediateDirectories: true)
        }
        let local = scan(directory: localDirectory)
        let hub = scanHuggingFaceHub(directory: hfHubDirectory)
        let localNames = Set(local.map(\.name))
        return local + hub.filter { !localNames.contains($0.name) }
    }

    /// Classifies one installed model folder or snapshot directory from its `config.json` or CoreAI bundle.
    public static func classify(name: String, path: String) -> QWModelCatalogEntry {
        let rootURL = URL(fileURLWithPath: path)
        let root = QWModelRoot(url: rootURL)
        let isCoreAI = (root.kind == .coreAIModel)
        guard let data = readConfigData(at: path) else {
            if isCoreAI {
                let coreWindow = readCoreAIWindow(at: path)
                return QWModelCatalogEntry(
                    name: name, path: path, modelType: "coreai_decision",
                    family: .lfm2Encoder, nativeWindow: coreWindow, reason: nil,
                    capabilities: [.decision], maxSpeculativeDrafts: 0,
                    isDecisionCapable: true, isCoreAIBundle: true,
                    decisionArchetype: "optionMarkers+CoreAI"
                )
            }
            return QWModelCatalogEntry(name: name, path: path, modelType: nil, family: nil,
                                       nativeWindow: nil, reason: "no config.json")
        }
        let modelType = readModelType(data)
        do {
            let desc = try QWModelDescriptor.decode(configJSON: data)
            let sidecarURL = QWLSAConfig.resolveSidecarURL(packPath: path)
            let isDecision = (desc.decisionConfig != nil) || desc.capabilities.contains(.decision) || isCoreAI
            let archetype = describeDecisionArchetype(desc.decisionConfig, isCoreAI: isCoreAI)
            guard QWModelFamilyRegistry.shared.isSupported(modelType: desc.modelType) else {
                return QWModelCatalogEntry(name: name, path: path, modelType: desc.modelType,
                                           family: desc.family, nativeWindow: desc.maxPositionEmbeddings,
                                           reason: "model_type '\(desc.modelType)' is not officially supported yet",
                                           capabilities: desc.capabilities,
                                           maxSpeculativeDrafts: desc.maxSpeculativeDrafts,
                                           hasLSASidecar: sidecarURL != nil,
                                           lsaSidecarPath: sidecarURL?.path,
                                           isDecisionCapable: isDecision,
                                           isCoreAIBundle: isCoreAI,
                                           decisionArchetype: archetype)
            }
            return QWModelCatalogEntry(name: name, path: path, modelType: desc.modelType,
                                       family: desc.family, nativeWindow: desc.maxPositionEmbeddings,
                                       reason: nil,
                                       capabilities: desc.capabilities,
                                       maxSpeculativeDrafts: desc.maxSpeculativeDrafts,
                                       hasLSASidecar: sidecarURL != nil,
                                       lsaSidecarPath: sidecarURL?.path,
                                       isDecisionCapable: isDecision,
                                       isCoreAIBundle: isCoreAI,
                                       decisionArchetype: archetype)
        } catch let error as QWModelDescriptor.DescriptorError {
            return QWModelCatalogEntry(name: name, path: path, modelType: modelType, family: nil,
                                       nativeWindow: nil, reason: error.description)
        } catch {
            return QWModelCatalogEntry(name: name, path: path, modelType: modelType, family: nil,
                                       nativeWindow: nil, reason: "not runnable: \(error)")
        }
    }

    /// Decodes a primary model's `QWModelDescriptor` from its `config.json` without reading tokenizer assets.
    public static func loadPrimaryDescriptor(at packPath: String) -> QWModelDescriptor? {
        guard let data = readConfigData(at: packPath) else { return nil }
        return try? QWModelDescriptor.decode(configJSON: data)
    }

    /// Returns true when `packPath` holds a DFlash 2 or DSpine drafter `config.json`.
    public static func isDrafterPack(at packPath: String) -> Bool {
        guard let root = readConfigJSONObject(at: packPath) else { return false }
        return isDrafterConfigJSON(root)
    }

    /// Reads drafter geometry and backend metadata from `packPath`'s `config.json`.
    public static func drafterMetadata(at packPath: String) -> QWDrafterMetadata? {
        guard let root = readConfigJSONObject(at: packPath),
              isDrafterConfigJSON(root) else {
            return nil
        }
        return parseDrafterMetadata(from: root)
    }

    /// Scans `directory` (defaults to `~/.qwg/models`) for installed DFlash 2 and DSpine drafter packs.
    public static func scanDrafters(
        in directory: String = QWPaths.modelsDirectoryPath
    ) -> [QWDrafterCatalogEntry] {
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: directory) else {
            return []
        }
        var out: [QWDrafterCatalogEntry] = []
        for item in items.sorted() where !item.hasPrefix(".") {
            let itemPath = (directory as NSString).appendingPathComponent(item)
            guard let root = readConfigJSONObject(at: itemPath),
                  isDrafterConfigJSON(root) else {
                continue
            }
            let name = QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: itemPath))
            out.append(QWDrafterCatalogEntry(
                name: name,
                path: itemPath,
                metadata: parseDrafterMetadata(from: root)
            ))
        }
        return out
    }

    private static func readConfigData(at packPath: String) -> Data? {
        let trimmed = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let root = QWModelRoot(url: URL(fileURLWithPath: trimmed.expandingTildePath))
        guard let cfgURL = root.configURL else { return nil }
        return try? Data(contentsOf: cfgURL)
    }

    private static func readConfigJSONObject(at packPath: String) -> [String: Any]? {
        guard let data = readConfigData(at: packPath) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func isDrafterConfigJSON(_ root: [String: Any]) -> Bool {
        if QWModelDescriptor.isDrafterJSON(root) { return true }
        let tc = (root["text_config"] as? [String: Any]) ?? root
        if tc["dflash_config"] != nil || tc["selected_prefix_code_bridge"] != nil { return true }
        if let mt = (tc["model_type"] as? String)?.lowercased(),
           mt.contains("dflash") || mt.contains("dspine") {
            return true
        }
        return false
    }

    private static func parseDrafterMetadata(from root: [String: Any]) -> QWDrafterMetadata? {
        let tc = (root["text_config"] as? [String: Any]) ?? root
        let dflash = (root["dflash_config"] as? [String: Any])
            ?? (tc["dflash_config"] as? [String: Any])
            ?? [:]
        guard let hiddenSize = tc["hidden_size"] as? Int, hiddenSize > 0,
              let vocabSize = tc["vocab_size"] as? Int, vocabSize > 0,
              let numLayers = tc["num_hidden_layers"] as? Int, numLayers > 0 else {
            return nil
        }
        let targetLayerIds = (dflash["target_layer_ids"] as? [Int])
            ?? (tc["target_layer_ids"] as? [Int])
            ?? (root["target_layer_ids"] as? [Int])
            ?? []
        let numTargetLayers = (tc["num_target_layers"] as? Int)
            ?? (dflash["num_target_layers"] as? Int)
            ?? (root["num_target_layers"] as? Int)
        let bridgeObj = dflash["selected_prefix_code_bridge"]
            ?? tc["selected_prefix_code_bridge"]
            ?? root["selected_prefix_code_bridge"]
        let modelTypeLower = ((tc["model_type"] as? String) ?? (root["model_type"] as? String) ?? "").lowercased()
        let isDSpine = (bridgeObj != nil && !(bridgeObj is NSNull) && (bridgeObj as? Bool) != false)
            || modelTypeLower.contains("dspine")
        let blockSize = (dflash["block_size"] as? Int)
            ?? (tc["block_size"] as? Int)
            ?? (root["block_size"] as? Int)
            ?? (isDSpine ? 16 : 8)
        return QWDrafterMetadata(
            hiddenSize: hiddenSize,
            vocabSize: vocabSize,
            numLayers: numLayers,
            targetLayerIds: targetLayerIds,
            numTargetLayers: numTargetLayers,
            backend: isDSpine ? "dspine" : "dflash2",
            maxDrafts: max(7, min(15, blockSize - 1))
        )
    }

    private static func readModelType(_ data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let top = obj["model_type"] as? String { return top }
        return (obj["text_config"] as? [String: Any])?["model_type"] as? String
    }

    private static func readCoreAIWindow(at path: String) -> Int? {
        let dir = URL(fileURLWithPath: path)
        for name in ["combined_provenance.json", "rl_agent_config.json"] {
            let url = dir.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let len = obj["sequence_length"] as? Int ?? obj["max_length"] as? Int, len > 0 { return len }
        }
        return nil
    }

    public static func describeDecisionArchetype(
        _ config: QWDecisionModelConfig?,
        isCoreAI: Bool
    ) -> String? {
        guard let config else { return isCoreAI ? "coreai" : nil }
        switch config.framing {
        case .turnVerbalizer: return isCoreAI ? "verbalizer+CoreAI" : "verbalizer"
        case .optionMarkers: return isCoreAI ? "optionMarkers+CoreAI" : "optionMarkers"
        case .sequencePair: return isCoreAI ? "sequencePair+CoreAI" : "sequencePair"
        }
    }
}

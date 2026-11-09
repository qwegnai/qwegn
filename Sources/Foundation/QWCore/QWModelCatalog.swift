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
    /// nil when the model is runnable; otherwise why it is not.
    public let reason: String?

    public init(name: String, path: String, modelType: String?, family: QWModelDescriptor.Family?,
                nativeWindow: Int?, reason: String?,
                capabilities: Set<QWModelDescriptor.Capability> = [],
                maxSpeculativeDrafts: Int = 0,
                hasLSASidecar: Bool = false,
                lsaSidecarPath: String? = nil) {
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
    }

    public var supported: Bool { reason == nil }
}

/// Scans a models directory. Classification uses config.json alone; no folder-name rules.
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

    public static func classify(name: String, path: String) -> QWModelCatalogEntry {
        let root = QWModelRoot(url: URL(fileURLWithPath: path))
        guard let configURL = root.configURL,
              let data = try? Data(contentsOf: configURL) else {
            return QWModelCatalogEntry(name: name, path: path, modelType: nil, family: nil,
                                       nativeWindow: nil, reason: "no config.json")
        }
        let modelType = readModelType(data)
        do {
            let desc = try QWModelDescriptor.decode(configJSON: data)
            let sidecarURL = QWLSAConfig.resolveSidecarURL(packPath: path)
            // A model the engine does not officially serve is not runnable,
            // and the entry must say so rather than advertise it.
            guard QWModelFamilyRegistry.shared.isSupported(modelType: desc.modelType) else {
                return QWModelCatalogEntry(name: name, path: path, modelType: desc.modelType,
                                           family: desc.family, nativeWindow: desc.maxPositionEmbeddings,
                                           reason: "model_type '\(desc.modelType)' is not officially supported yet",
                                           capabilities: desc.capabilities,
                                           maxSpeculativeDrafts: desc.maxSpeculativeDrafts,
                                           hasLSASidecar: sidecarURL != nil,
                                           lsaSidecarPath: sidecarURL?.path)
            }
            return QWModelCatalogEntry(name: name, path: path, modelType: desc.modelType,
                                       family: desc.family, nativeWindow: desc.maxPositionEmbeddings,
                                       reason: nil,
                                       capabilities: desc.capabilities,
                                       maxSpeculativeDrafts: desc.maxSpeculativeDrafts,
                                       hasLSASidecar: sidecarURL != nil,
                                       lsaSidecarPath: sidecarURL?.path)
        } catch let error as QWModelDescriptor.DescriptorError {
            return QWModelCatalogEntry(name: name, path: path, modelType: modelType, family: nil,
                                       nativeWindow: nil, reason: error.description)
        } catch {
            return QWModelCatalogEntry(name: name, path: path, modelType: modelType, family: nil,
                                       nativeWindow: nil, reason: "not runnable: \(error)")
        }
    }

    private static func readModelType(_ data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let top = obj["model_type"] as? String { return top }
        return (obj["text_config"] as? [String: Any])?["model_type"] as? String
    }
}

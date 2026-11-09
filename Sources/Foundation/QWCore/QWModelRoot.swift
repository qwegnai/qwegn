import Foundation

/// Which weight container a model root holds.
public enum QWWeightSourceKind: String, Sendable, CaseIterable, Codable {
    /// A packed `model.qwp` (QWP1).
    case qwp1
    /// A single `model.safetensors`.
    case safetensorsSingle
    /// A sharded checkpoint described by `model.safetensors.index.json`.
    case safetensorsSharded
}

/// Which weight container the caller wants.
public enum QWWeightSourcePreference: String, Sendable, CaseIterable, Codable {
    /// Prefer the packed `.qwp1` (instant startup), else the safetensors.
    case auto
    /// Require `model.qwp`.
    case qwp
    /// Require a safetensors container.
    case safetensors
}

public enum QWModelRootError: Error, Equatable {
    case missingConfig(String)
}

/// Resolves a model root — a directory *or* a weight file — into its weight
/// container, its assets directory and its `config.json`. The single authority
/// for where a model's files live; the full table is in specs/01-qwcore.md.
public struct QWModelRoot: Sendable {
    public let directory: URL
    /// The weight container, or the path it would have when there is none.
    public let weights: URL
    /// The container kind, or `nil` when the root holds no container.
    public let kind: QWWeightSourceKind?
    public let assets: URL
    /// The `config.json`, or `nil` when the root holds none.
    public let configURL: URL?
    public let preference: QWWeightSourcePreference

    public init(url: URL, preference: QWWeightSourcePreference = .auto) {
        let fm = FileManager.default
        let resolvedURL = Self.resolveSnapshotDirectory(url, fileManager: fm)
        var isDirectory: ObjCBool = false
        let exists = fm.fileExists(atPath: resolvedURL.path, isDirectory: &isDirectory)
        let isDir = exists && isDirectory.boolValue
        self.preference = preference
        self.directory = isDir ? resolvedURL : resolvedURL.deletingLastPathComponent()

        let resolved = Self.resolveWeights(root: directory, isDirectory: isDir,
                                           url: resolvedURL, preference: preference)
        self.weights = resolved.0
        self.kind = resolved.1
        self.assets = Self.resolveAssets(root: directory, isDirectory: isDir, url: resolvedURL)

        let flatConfig = directory.appendingPathComponent("config.json")
        if fm.fileExists(atPath: flatConfig.path) {
            self.configURL = flatConfig
        } else {
            let legacyConfig = assets.appendingPathComponent("config.json")
            self.configURL = fm.fileExists(atPath: legacyConfig.path) ? legacyConfig : nil
        }
    }

    /// Resolves a HuggingFace cache repo directory (`models--<org>--<repo>`) or
    /// any parent directory containing `refs/` + `snapshots/` to its active
    /// snapshot directory so callers do not need to pass `/snapshots/<sha>`.
    public static func resolveSnapshotDirectory(_ url: URL,
                                                fileManager fm: FileManager = .default) -> URL {
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return url
        }
        let directMarkers = [
            "config.json",
            "assets/config.json",
            "model.qwp",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
        for marker in directMarkers {
            if fm.fileExists(atPath: url.appendingPathComponent(marker).path) {
                return url
            }
        }
        let snapshotsDir = url.appendingPathComponent("snapshots", isDirectory: true)
        guard fm.fileExists(atPath: snapshotsDir.path, isDirectory: &isDir), isDir.boolValue else {
            return url
        }
        let refsMain = url.appendingPathComponent("refs/main")
        if let rev = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !rev.isEmpty {
            let snap = snapshotsDir.appendingPathComponent(rev, isDirectory: true)
            if fm.fileExists(atPath: snap.path, isDirectory: &isDir), isDir.boolValue {
                return snap
            }
        }
        let refsDir = url.appendingPathComponent("refs", isDirectory: true)
        if let refFiles = try? fm.contentsOfDirectory(at: refsDir, includingPropertiesForKeys: nil) {
            for refFile in refFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if let rev = try? String(contentsOf: refFile, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !rev.isEmpty {
                    let snap = snapshotsDir.appendingPathComponent(rev, isDirectory: true)
                    if fm.fileExists(atPath: snap.path, isDirectory: &isDir), isDir.boolValue {
                        return snap
                    }
                }
            }
        }
        if let snaps = try? fm.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) {
            let candidates = snaps.filter { snap in
                var subDir: ObjCBool = false
                guard fm.fileExists(atPath: snap.path, isDirectory: &subDir), subDir.boolValue else {
                    return false
                }
                return fm.fileExists(atPath: snap.appendingPathComponent("config.json").path)
                    || fm.fileExists(atPath: snap.appendingPathComponent("model.safetensors").path)
                    || fm.fileExists(atPath: snap.appendingPathComponent("model.safetensors.index.json").path)
            }
            if let best = candidates.max(by: { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return da < db
            }) {
                return best
            }
        }
        return url
    }

    /// Derives a human-readable canonical model name from a model root URL or
    /// HuggingFace snapshot path (avoiding 40-char snapshot commit hashes).
    public static func canonicalModelName(for url: URL) -> String {
        for component in url.pathComponents.reversed() {
            if component.hasPrefix("models--") {
                let parts = component.components(separatedBy: "--")
                if let repo = parts.last, !repo.isEmpty {
                    return repo
                }
            }
        }
        let parent = url.deletingLastPathComponent()
        if parent.lastPathComponent == "snapshots" {
            let grandParent = parent.deletingLastPathComponent().lastPathComponent
            if !grandParent.isEmpty && grandParent != "/" {
                return grandParent
            }
        }
        let last = url.lastPathComponent
        let knownFileExtensions: Set<String> = ["qwp", "safetensors", "bin", "gguf", "json", "qlsa"]
        let ext = url.pathExtension.lowercased()
        if knownFileExtensions.contains(ext) {
            let stem = url.deletingPathExtension().lastPathComponent
            let genericStems: Set<String> = ["model", "weights", "model.safetensors.index", "config"]
            if genericStems.contains(stem.lowercased()) {
                let parentName = url.deletingLastPathComponent().lastPathComponent
                if !parentName.isEmpty && parentName != "/" {
                    return parentName
                }
            }
            return stem
        }
        return last
    }

    private static func resolveWeights(root: URL,
                                       isDirectory: Bool,
                                       url: URL,
                                       preference: QWWeightSourcePreference) -> (URL, QWWeightSourceKind?) {
        let fm = FileManager.default
        // A file argument IS the weights.
        if !isDirectory {
            var isFileDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isFileDir), !isFileDir.boolValue else {
                return (url, nil)
            }
            let kind: QWWeightSourceKind =
                url.lastPathComponent == "model.safetensors.index.json" ? .safetensorsSharded
                : (url.lastPathComponent.hasSuffix(".safetensors") ? .safetensorsSingle : .qwp1)
            let acceptable: Bool
            switch preference {
            case .auto: acceptable = true
            case .qwp: acceptable = (kind == .qwp1)
            case .safetensors: acceptable = (kind != .qwp1)
            }
            return (url, acceptable ? kind : nil)
        }
        // A directory argument: look for a container inside it.
        let qwp = root.appendingPathComponent("model.qwp")
        let single = root.appendingPathComponent("model.safetensors")
        let index = root.appendingPathComponent("model.safetensors.index.json")
        let hasQWP = fm.fileExists(atPath: qwp.path)
        let hasSingle = fm.fileExists(atPath: single.path)
        let hasSharded = fm.fileExists(atPath: index.path)

        func pickSafetensors() -> (URL, QWWeightSourceKind)? {
            if hasSingle { return (single, .safetensorsSingle) }
            if hasSharded { return (index, .safetensorsSharded) }
            for alt in ["weights.00.safetensors", "weights.safetensors"] {
                let altURL = root.appendingPathComponent(alt)
                if fm.fileExists(atPath: altURL.path) { return (altURL, .safetensorsSingle) }
            }
            return nil
        }

        switch preference {
        case .qwp:
            return (qwp, hasQWP ? .qwp1 : nil)
        case .safetensors:
            return pickSafetensors() ?? (single, nil)
        case .auto:
            if hasQWP { return (qwp, .qwp1) }
            return pickSafetensors() ?? (qwp, nil)
        }
    }

    private static func resolveAssets(root: URL, isDirectory: Bool, url: URL) -> URL {
        let fm = FileManager.default
        // Flat root: the config and the n-gram table sit beside the weights.
        for name in ["config.json", "ngram.bin", "ple_ngram_locations.json"] {
            if fm.fileExists(atPath: root.appendingPathComponent(name).path) { return root }
        }
        if isDirectory {
            // A directory root: the legacy layout is <root>/assets, whether
            // or not it exists yet.
            return root.appendingPathComponent("assets", isDirectory: true)
        }
        // A file root: a sibling assets/ folder, else <file>.assets.
        let sibling = root.appendingPathComponent("assets", isDirectory: true)
        if fm.fileExists(atPath: sibling.path) { return sibling }
        return URL(fileURLWithPath: url.path + ".assets", isDirectory: true)
    }
}

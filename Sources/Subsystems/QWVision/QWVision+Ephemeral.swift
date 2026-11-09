// QWVision+Ephemeral.swift — Vision backend protocol, reference adapter, and
// prompt-scoped on-demand residency controller (0 B steady-state text RAM).

import Foundation
import QWCore
import QWQuant

/// Unified vision-tower contract: C19 patch slab in, merged visual tokens out.
public protocol QWVisionBackend: Sendable {
    func encode(patches: [Float], grid: QWVisionGridPlan,
                bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float]
    /// Prepares vision residency for the upcoming prompt: loads on demand when
    /// `hasImages == true` and unloads resident weights when `hasImages == false`.
    func prepareForPrompt(hasImages: Bool) throws
    /// Dynamically loads or unloads the vision tower weights in memory without restarting the LLM.
    func setVisionLoaded(_ loaded: Bool) throws
    /// True when vision tower weights are currently resident in memory.
    var isLoaded: Bool { get }
    /// True when vision tower weights are configured to stay resident across prompts.
    var keepResident: Bool { get }
}

extension QWVisionBackend {
    public func prepareForPrompt(hasImages: Bool) throws {}
    public func setVisionLoaded(_ loaded: Bool) throws {}
    public var isLoaded: Bool { true }
    public var keepResident: Bool { isLoaded }
}

public enum QWVisionBackendMode: String, Sendable { case metal, coreai, auto, ref }

/// Reference backend holding resident `QWVisionWeights` for the lifetime of the struct.
public struct QWVisionRefBackend: QWVisionBackend {
    public let ref: QWVisionRef
    public init(_ ref: QWVisionRef) { self.ref = ref }
    public func encode(patches: [Float], grid: QWVisionGridPlan,
                       bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float] {
        ref.encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
    }
}

extension QWVisionWeights {
    /// Total bytes occupied by the widened fp32 vision tower arrays in RAM.
    public var residentBytes: Int {
        let blockFloats = blocks.reduce(0) { acc, b in
            acc + b.norm1G.count + b.norm1B.count + b.norm2G.count + b.norm2B.count
                + b.qkvW.count + b.qkvB.count + b.projW.count + b.projB.count
                + b.l1W.count + b.l1B.count + b.l2W.count + b.l2B.count
        }
        let topFloats = posEmbed.count + patchW.count + patchB.count
            + mergerGamma.count + mergerBeta.count
            + fc1W.count + fc1B.count + fc2W.count + fc2B.count + invFreq.count
        return (blockFloats + topFloats) * MemoryLayout<Float>.stride
    }
}

/// Prompt-scoped on-demand vision backend: unloaded by default (0 B resident),
/// loaded once when a prompt has images, and unloaded when a subsequent prompt has no images.
public final class QWVisionOnDemandBackend: QWVisionBackend, @unchecked Sendable {
    public enum Source: Sendable {
        case pack(String)
        case directory(String)
        case custom(@Sendable () throws -> QWVisionWeights)
    }

    public let source: Source
    private let lock = NSLock()
    private var residentWeights: QWVisionWeights?
    private var keepResidentFlag: Bool = false
    private var allowOnDemandFlag: Bool = true
    private var loads = 0
    private var unloads = 0

    public var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return residentWeights != nil
    }

    public var keepResident: Bool {
        lock.lock()
        defer { lock.unlock() }
        return keepResidentFlag
    }

    public var residentBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return residentWeights?.residentBytes ?? 0
    }

    public var loadCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return loads
    }

    public var unloadCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return unloads
    }

    public init(source: Source, keepResident: Bool = false, allowOnDemand: Bool = true) throws {
        self.source = source
        self.keepResidentFlag = keepResident
        self.allowOnDemandFlag = allowOnDemand
        switch source {
        case .pack(let packPath):
            try Self.validate(checkpoint: try QWVisionCheckpoint(pack: packPath))
        case .directory(let dir):
            try Self.validate(checkpoint: try QWVisionCheckpoint(dir: dir))
        case .custom:
            break
        }
        if keepResident {
            _ = try loadWeightsLocked()
        }
    }

    /// Resolves a model path (`.qwp` pack or `.safetensors` directory/file) into an on-demand backend.
    public convenience init(modelPath: String, keepResident: Bool = false, allowOnDemand: Bool = true) throws {
        let url = URL(fileURLWithPath: modelPath)
        let root = QWModelRoot(url: url)
        switch root.kind {
        case .qwp1:
            try self.init(source: .pack(root.weights.path), keepResident: keepResident, allowOnDemand: allowOnDemand)
        case .safetensorsSingle, .safetensorsSharded:
            let dir = root.weights.hasDirectoryPath
                ? root.weights.path
                : root.weights.deletingLastPathComponent().path
            try self.init(source: .directory(dir), keepResident: keepResident, allowOnDemand: allowOnDemand)
        case nil:
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir), isDir.boolValue {
                try self.init(source: .directory(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand)
            } else {
                try self.init(source: .pack(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand)
            }
        }
    }

    /// True when the checkpoint at `modelPath` carries valid vision tower weights.
    public static func hasVisionWeights(modelPath: String) -> Bool {
        (try? QWVisionOnDemandBackend(modelPath: modelPath)) != nil
    }

    private static func validate(checkpoint ck: QWVisionCheckpoint) throws {
        defer {
            ck.shards.removeAll()
            ck.pack?.adviseDontNeed()
            ck.pack = nil
        }
        guard ck.shape("pos_embed.weight") == [2304, QWVisionShape.dim] else {
            throw QWVisionError.missingTensor("pos_embed.weight")
        }
        let pw = ck.shape("patch_embed.proj.weight")
        guard pw == [1152, 3, 2, 16, 16] || pw == [1152, 2, 16, 16, 3] else {
            throw QWVisionError.badLayout("patch_embed.proj.weight \(pw.debugDescription) invalid")
        }
        guard ck.has("blocks.0.norm1.weight"),
              ck.has("merger.norm.weight"),
              ck.has("merger.linear_fc2.weight") else {
            throw QWVisionError.missingTensor("vision blocks/merger missing")
        }
    }

    private func loadWeightsLocked() throws -> QWVisionWeights {
        if let existing = residentWeights {
            return existing
        }
        let loaded: QWVisionWeights
        switch source {
        case .pack(let packPath):
            loaded = try QWVisionWeights(pack: packPath)
        case .directory(let dir):
            loaded = try QWVisionWeights(dir: dir)
        case .custom(let builder):
            loaded = try builder()
        }
        residentWeights = loaded
        loads += 1
        return loaded
    }

    public func setVisionLoaded(_ loaded: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        keepResidentFlag = loaded
        if loaded {
            _ = try loadWeightsLocked()
        } else if residentWeights != nil {
            residentWeights = nil
            unloads += 1
        }
    }

    public func prepareForPrompt(hasImages: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        if hasImages {
            guard keepResidentFlag || allowOnDemandFlag || residentWeights != nil else {
                throw QWVisionError.badLayout("Vision tower is disabled in Model Settings (enable 'Load Vision Tower' in Settings → Model)")
            }
            _ = try loadWeightsLocked()
        } else if !keepResidentFlag, residentWeights != nil {
            residentWeights = nil
            unloads += 1
        }
    }

    public func unload() {
        lock.lock()
        keepResidentFlag = false
        if residentWeights != nil {
            residentWeights = nil
            unloads += 1
        }
        lock.unlock()
    }

    public func encodeSync(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) throws -> [Float] {
        let weights: QWVisionWeights = try {
            lock.lock()
            defer { lock.unlock() }
            return try loadWeightsLocked()
        }()
        let ref = QWVisionRef(weights: weights)
        return ref.encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
    }

    public func encode(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) async throws -> [Float] {
        try encodeSync(patches: patches, grid: grid, bins: bins, posIds: posIds)
    }
}

/// Backwards-compatible alias for ephemeral/on-demand vision backend.
public typealias QWVisionEphemeralBackend = QWVisionOnDemandBackend

public enum QWVision {
    public static func make(mode: QWVisionBackendMode, weightsDir: String) throws -> any QWVisionBackend {
        let weights = try QWVisionWeights(dir: weightsDir)
        switch mode {
        case .ref:
            return QWVisionRefBackend(QWVisionRef(weights: weights))
        case .metal, .coreai, .auto:
            throw QWVisionError.badLayout("backend \(mode.rawValue) not landed (Metal/C-AI race pending; DECISIONS gates the default)")
        }
    }

    public static func make(mode: QWVisionBackendMode, packPath: String) throws -> any QWVisionBackend {
        let weights = try QWVisionWeights(pack: packPath)
        switch mode {
        case .ref:
            return QWVisionRefBackend(QWVisionRef(weights: weights))
        case .metal, .coreai, .auto:
            throw QWVisionError.badLayout("backend \(mode.rawValue) not landed (Metal/C-AI race pending; DECISIONS gates the default)")
        }
    }

    /// Creates a prompt-scoped on-demand vision backend from a `.qwp` pack (0 B startup RAM).
    public static func makeEphemeral(mode: QWVisionBackendMode = .ref, packPath: String) throws -> any QWVisionBackend {
        switch mode {
        case .ref:
            return try QWVisionOnDemandBackend(source: .pack(packPath))
        case .metal, .coreai, .auto:
            throw QWVisionError.badLayout("backend \(mode.rawValue) not landed")
        }
    }

    /// Creates a prompt-scoped on-demand vision backend from a checkpoint directory (0 B startup RAM).
    public static func makeEphemeral(mode: QWVisionBackendMode = .ref, weightsDir: String) throws -> any QWVisionBackend {
        switch mode {
        case .ref:
            return try QWVisionOnDemandBackend(source: .directory(weightsDir))
        case .metal, .coreai, .auto:
            throw QWVisionError.badLayout("backend \(mode.rawValue) not landed")
        }
    }

    /// Creates a prompt-scoped or resident vision backend from any model path (`.qwp` or `.safetensors`).
    public static func makeOnDemand(
        mode: QWVisionBackendMode = .ref,
        modelPath: String,
        keepResident: Bool = false,
        allowOnDemand: Bool = true
    ) throws -> any QWVisionBackend {
        switch mode {
        case .ref:
            return try QWVisionOnDemandBackend(
                modelPath: modelPath,
                keepResident: keepResident,
                allowOnDemand: allowOnDemand
            )
        case .metal, .coreai, .auto:
            throw QWVisionError.badLayout("backend \(mode.rawValue) not landed")
        }
    }
}

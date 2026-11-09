import Foundation
import QWCore

public struct ModelObject: Codable, Sendable {
    public let id: String
    public let object: String
    /// When the model's bytes last changed on disk, as unix seconds.
    /// Absent when the model has no file to measure — an absent field
    /// is honest, a date the server chose is not.
    public let created: Int?
    public let ownedBy: String
    public let loaded: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
        case loaded
    }

    public init(id: String, loaded: Bool = true, created: Int? = nil) {
        self.id = id
        self.object = "model"
        self.created = created
        self.ownedBy = "qwegine"
        self.loaded = loaded
    }
}

public struct ModelListResponse: Codable, Sendable {
    public let object: String
    public let data: [ModelObject]

    public init(models: [ModelObject]) {
        self.object = "list"
        self.data = models
    }
}

public struct ModelLifecycleResponse: Codable, Sendable {
    public let status: String
    public let model: String
    public let loaded: Bool
    public let memoryReclaimedMB: Double?

    public init(status: String, model: String, loaded: Bool, memoryReclaimedMB: Double? = nil) {
        self.status = status
        self.model = model
        self.loaded = loaded
        self.memoryReclaimedMB = memoryReclaimedMB
    }
}

/// The Ollama-compatible payloads live in QWServerSchema+Ollama.swift.
public struct ServerErrorPayload: Codable, Sendable {
    public struct Detail: Codable, Sendable {
        public let message: String
        public let type: String
        public let code: Int?
    }
    public let error: Detail

    public init(message: String, type: String = "invalid_request_error", code: Int? = nil) {
        self.error = Detail(message: message, type: type, code: code)
    }
}

public struct QWEngineInfoResponse: Codable, Sendable {
    public struct HardwareFacts: Codable, Sendable {
        public let modelIdentifier: String
        public let cpuBrand: String
        public let physicalCores: Int
        public let logicalCores: Int
        public let physicalMemoryBytes: UInt64
        public let physicalMemoryGB: Double
        /// The GPU's name, as the device reports it. Absent when the
        /// device could not be read.
        public let metalDeviceName: String?
        /// The Metal family the device supports, from the probes the HAL
        /// made. Absent when no family could be established — never a
        /// guess.
        public let metalVersion: String?
        public let isTensorCapable: Bool
        /// The host's DRAM read bandwidth. Absent unless it was measured
        /// on this host: the HAL's number is a constant from one machine.
        public let dramBwGBs: Double?

        public init(
            modelIdentifier: String,
            cpuBrand: String,
            physicalCores: Int,
            logicalCores: Int,
            physicalMemoryBytes: UInt64,
            physicalMemoryGB: Double,
            metalDeviceName: String?,
            metalVersion: String?,
            isTensorCapable: Bool,
            dramBwGBs: Double?
        ) {
            self.modelIdentifier = modelIdentifier
            self.cpuBrand = cpuBrand
            self.physicalCores = physicalCores
            self.logicalCores = logicalCores
            self.physicalMemoryBytes = physicalMemoryBytes
            self.physicalMemoryGB = physicalMemoryGB
            self.metalDeviceName = metalDeviceName
            self.metalVersion = metalVersion
            self.isTensorCapable = isTensorCapable
            self.dramBwGBs = dramBwGBs
        }
    }

    public struct ServerParameters: Codable, Sendable {
        public let modelName: String
        public let modelPackPath: String
        /// The head storage of the resident model. Absent when no model is
        /// resident: a quantization that describes nothing is a claim
        /// about nothing.
        public let headStorage: String?
        public let contextBudget: Int
        public let kvMode: String
        public let mtpDrafts: Int
        /// The speculative backend in use. Absent when none is configured.
        public let mtpBackend: String?
        public let gdnPrefillMode: String
        public let prefillScratchEviction: String
        /// The host's logical cores. The engine's own thread count is not
        /// published, so this is not called "threads". Absent when the
        /// host could not say.
        public let hostLogicalCores: Int?
        public let isLoaded: Bool

        public init(
            modelName: String,
            modelPackPath: String,
            headStorage: String?,
            contextBudget: Int,
            kvMode: String,
            mtpDrafts: Int,
            mtpBackend: String?,
            gdnPrefillMode: String,
            prefillScratchEviction: String,
            hostLogicalCores: Int?,
            isLoaded: Bool
        ) {
            self.modelName = modelName
            self.modelPackPath = modelPackPath
            self.headStorage = headStorage
            self.contextBudget = contextBudget
            self.kvMode = kvMode
            self.mtpDrafts = mtpDrafts
            self.mtpBackend = mtpBackend
            self.gdnPrefillMode = gdnPrefillMode
            self.prefillScratchEviction = prefillScratchEviction
            self.hostLogicalCores = hostLogicalCores
            self.isLoaded = isLoaded
        }
    }

    public struct CacheConfigInfo: Codable, Sendable {
        public let warmLimitBytes: Int
        public let ssdLimitBytes: Int
        public let cacheDir: String

        public init(warmLimitBytes: Int, ssdLimitBytes: Int, cacheDir: String) {
            self.warmLimitBytes = warmLimitBytes
            self.ssdLimitBytes = ssdLimitBytes
            self.cacheDir = cacheDir
        }
    }

    public struct MountedMemoryPackInfo: Codable, Sendable, Equatable {
        public let packName: String
        public let rootPath: String
        public let enforcedKVMode: String
        public let lsaMode: String
        public let tokenCount: Int
        public let fileCount: Int
        public let modifiedFileCount: Int
        public let cacheBytes: Int
        public let enforcesKVMode: Bool

        public init(
            packName: String,
            rootPath: String,
            enforcedKVMode: String,
            lsaMode: String,
            tokenCount: Int,
            fileCount: Int,
            modifiedFileCount: Int = 0,
            cacheBytes: Int,
            enforcesKVMode: Bool = true
        ) {
            self.packName = packName
            self.rootPath = rootPath
            self.enforcedKVMode = enforcedKVMode
            self.lsaMode = lsaMode
            self.tokenCount = tokenCount
            self.fileCount = fileCount
            self.modifiedFileCount = modifiedFileCount
            self.cacheBytes = cacheBytes
            self.enforcesKVMode = enforcesKVMode
        }
    }

    public let status: String
    public let version: String
    public let pid: Int?
    public let hardware: HardwareFacts
    public let parameters: ServerParameters
    public let cacheConfig: CacheConfigInfo
    public let hardwareUtilization: QWSystemHardwareUtilization?
    /// The engine's phase. Absent only when the producer cannot know it —
    /// never a guessed phase.
    public let lifecycle: QWEngineLifecycleInfo?
    /// Where each setting came from, by setting key: `explicit`,
    /// `profile`, `default` or `deprecatedEnv`.
    public var sources: [String: String] = [:]
    /// What the loaded model can do, by name.
    public var capabilities: [String] = []
    /// Active codebase/knowledge memory pack mounted as a shared base prefix.
    public var mountedMemoryPack: MountedMemoryPackInfo? = nil
    public let endpoints: [String: String]

    public init(
        status: String,
        version: String = QWVersion.current,
        pid: Int? = nil,
        hardware: HardwareFacts,
        parameters: ServerParameters,
        cacheConfig: CacheConfigInfo,
        hardwareUtilization: QWSystemHardwareUtilization? = nil,
        lifecycle: QWEngineLifecycleInfo?,
        sources: [String: String] = [:],
        capabilities: [String] = [],
        mountedMemoryPack: MountedMemoryPackInfo? = nil,
        endpoints: [String: String]
    ) {
        self.status = status
        self.version = version
        self.pid = pid
        self.hardware = hardware
        self.parameters = parameters
        self.cacheConfig = cacheConfig
        self.hardwareUtilization = hardwareUtilization
        self.lifecycle = lifecycle
        self.sources = sources
        self.capabilities = capabilities
        self.mountedMemoryPack = mountedMemoryPack
        self.endpoints = endpoints
    }
}

/// The server's health, and what the engine is doing.
public struct QWHealthResponse: Codable, Sendable {
    /// The engine's phase — the truth, not a fixed "ok".
    public let status: String
    /// The server's name.
    public let engine: String
    /// The model the server would load.
    public let model: String
    /// Whether a model is loaded right now.
    public let loaded: Bool
    /// The engine's phase, when it entered it, and why it failed.
    public let lifecycle: QWEngineLifecycleInfo

    public init(status: String, engine: String, model: String, loaded: Bool,
                lifecycle: QWEngineLifecycleInfo) {
        self.status = status
        self.engine = engine
        self.model = model
        self.loaded = loaded
        self.lifecycle = lifecycle
    }
}

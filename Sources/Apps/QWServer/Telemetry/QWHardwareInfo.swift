import Foundation
import QWCore
import QWHAL
import QWKV

public struct QWMachineHardwareInfo: Sendable, Equatable {
    public let modelIdentifier: String
    public let cpuBrand: String
    public let physicalCores: Int
    public let logicalCores: Int
    public let physicalMemoryBytes: UInt64
    public let osProductVersion: String
    public let kernelRelease: String

    public var physicalMemoryGB: Double {
        Double(physicalMemoryBytes) / (1024.0 * 1024.0 * 1024.0)
    }

    public static func current() -> QWMachineHardwareInfo {
        let model = sysctlString("hw.model") ?? "Unknown Mac"
        let cpuBrand = sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
        let physCores = sysctlInt("hw.physicalcpu") ?? 1
        let logCores = sysctlInt("hw.ncpu") ?? 1
        let memBytes = sysctlUInt64("hw.memsize") ?? 0
        let osVersion = sysctlString("kern.osproductversion") ?? "macOS"
        let kernelRel = sysctlString("kern.osrelease") ?? "Darwin"

        return QWMachineHardwareInfo(
            modelIdentifier: model,
            cpuBrand: cpuBrand,
            physicalCores: physCores,
            logicalCores: logCores,
            physicalMemoryBytes: memBytes,
            osProductVersion: osVersion,
            kernelRelease: kernelRel
        )
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: Int = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return buffer.withUnsafeBufferPointer { ptr in
            ptr.baseAddress.map { String(cString: $0) }
        }
    }

    private static func sysctlInt(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return Int(value)
    }

    private static func sysctlUInt64(_ name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return value
    }
}

public struct QWEngineParametersInfo: Sendable {
    public let modelPackPath: String
    public let modelName: String
    /// The loaded model's own descriptor — the authority for every
    /// architecture line the banner prints. Absent means unknown, and the
    /// banner says so; it never substitutes a literal for a fact.
    public let modelDescriptor: QWModelDescriptor?
    public let tensorCount: Int?
    public let headStorage: String
    /// What the pack actually carries for vision, as the backend found it.
    public let visionStatus: String
    public let contextBudget: Int
    public let kvMode: QWKVMode
    public let mtpDrafts: Int
    public let mtpBackend: String
    public let gdnPrefillMode: String
    public let prefillScratchEviction: String
    /// The chunk rows the engine actually runs with — the resolved setting,
    /// not the default constant. The banner prints only facts it was given.
    public let prefillChunkRows: Int
    public let weightsSizeBytes: Int?
    public let kvCacheSizeBytes: Int?
    /// Why the model is NOT in the size-preserving container, when it is
    /// not. Absent when the model's own bytes are the container: there is
    /// nothing to explain.
    public let weightContainerNote: String?

    public init(
        modelPackPath: String,
        modelName: String,
        modelDescriptor: QWModelDescriptor? = nil,
        tensorCount: Int? = nil,
        headStorage: String = "Q4g",
        visionStatus: String = "not present in this pack",
        contextBudget: Int = 4096,
        kvMode: QWKVMode = .f16,
        mtpDrafts: Int = 0,
        mtpBackend: String = "standard",
        gdnPrefillMode: String = "chunked",
        prefillScratchEviction: String = "auto",
        prefillChunkRows: Int = QWPrefillLimits.chunkRows,
        weightsSizeBytes: Int? = nil,
        kvCacheSizeBytes: Int? = nil,
        weightContainerNote: String? = nil
    ) {
        self.modelPackPath = modelPackPath
        self.modelName = modelName
        self.modelDescriptor = modelDescriptor
        self.tensorCount = tensorCount
        self.headStorage = headStorage
        self.visionStatus = visionStatus
        self.contextBudget = contextBudget
        self.kvMode = kvMode
        self.mtpDrafts = mtpDrafts
        self.mtpBackend = mtpBackend
        self.gdnPrefillMode = gdnPrefillMode
        self.prefillScratchEviction = prefillScratchEviction
        self.prefillChunkRows = prefillChunkRows
        self.weightsSizeBytes = weightsSizeBytes
        self.weightContainerNote = weightContainerNote
        self.kvCacheSizeBytes = kvCacheSizeBytes
    }
}

public enum QWHardwareBanner {
    public static func renderBanner(
        hardware: QWMachineHardwareInfo,
        caps: QWDeviceCaps?,
        params: QWEngineParametersInfo,
        host: String,
        port: Int
    ) -> String {
        let line = String(repeating: "=", count: 80)
        let gpuName = caps?.name ?? hardware.cpuBrand
        let tensorCapable = caps?.isMetal4TensorCapable == true ? "Yes (Apple10+)" : "Standard"
        let maxVramGB = caps.map { String(format: "%.2f GB", Double($0.maxWorkingSetBytes) / (1024 * 1024 * 1024)) } ?? "Unified"
        // The HAL's bandwidth is a constant from the reference machine,
        // not a measurement of this one: it says so.
        let dramBw = caps.map { String(format: "%.1f GB/s (reference)", $0.measuredReadBWBps / 1e9) } ?? "not measured"
        let l2Bw = caps.map { String(format: "%.1f GB/s (reference)", $0.measuredL2BWBps / 1e9) } ?? "not measured"

        var out = ""
        out += "\(line)\n"
        out += "QWEGINE OPENAI/OLLAMA/ANTHROPIC INFERENCE SERVER — HARDWARE & ENGINE AUDIT\n"
        out += "\(line)\n"
        out += "Hardware & Platform:\n"
        out += String(format: "  • Machine Model      : %@ (%@)\n", hardware.modelIdentifier, hardware.cpuBrand)
        out += String(format: "  • CPU Cores          : %d physical / %d logical\n", hardware.physicalCores, hardware.logicalCores)
        out += String(format: "  • Unified RAM        : %.2f GB (%llu bytes)\n", hardware.physicalMemoryGB, hardware.physicalMemoryBytes)
        out += String(format: "  • Operating System   : macOS %@ (Kernel %@)\n", hardware.osProductVersion, hardware.kernelRelease)
        out += String(format: "  • Metal Device       : %@ (Tensor Acceleration: %@)\n", gpuName, tensorCapable)
        out += String(format: "  • Working Set Budget : %@\n", maxVramGB)
        out += String(format: "  • Hardware Bandwidth : %@ DRAM Read | %@ L2 Cache\n", dramBw, l2Bw)
        out += "Engine Parameters & Architecture:\n"
        out += String(format: "  • Model Pack         : %@\n", params.modelPackPath)
        // The architecture lines come from the descriptor the backend
        // loaded with — the model's own config.json. No geometry is
        // asserted here about a model the banner has no facts for.
        if let d = params.modelDescriptor {
            var kindCounts: [String: Int] = [:]
            for t in d.layerTypes { kindCounts[t, default: 0] += 1 }
            let kinds = kindCounts.keys.sorted()
                .map { "\(kindCounts[$0]!) \($0)" }
                .joined(separator: ", ")
            let loopSuffix: String
            if let loop = d.loopSettings, loop.evalCount > 1 {
                loopSuffix = "; \(d.executionPlan.count) effective via loop \(loop.start)..<\(loop.end) x\(loop.evalCount)"
            } else {
                loopSuffix = ""
            }
            out += String(format: "  • Architecture       : %@ (model_type %@)\n", d.architectureDisplayName, d.modelType)
            out += String(format: "  • Text Geometry      : hidden %d, %d layers (%@%@), Q %d / KV %d, head_dim %d\n",
                          d.hidden, d.numLayers, kinds, loopSuffix, d.qHeads, d.kvHeads, d.headDim)
            if let experts = d.numExperts {
                out += String(format: "  • Sparse MoE         : %d experts, %d/token, shared intermediate %d\n",
                              experts, d.expertsPerTok ?? 0, d.sharedExpertIntermediate ?? 0)
            }
            let linearLayers = kindCounts["linear_attention"] ?? 0
            if linearLayers > 0 {
                out += String(format: "  • Linear Recurrence  : Gated DeltaNet, %d layers, QK heads %d, V heads %d, Dk %d, Dv %d\n",
                              linearLayers, d.gdnQKHeads, d.gdnVHeads, d.gdnHeadDim, d.gdnHeadDim)
            }
            if let ple = d.pleLayerIds, !ple.isEmpty {
                out += String(format: "  • PLE                : %d layer(s) [%@], embed dim %d\n",
                              ple.count, ple.map(String.init).joined(separator: ", "), d.pleEmbedDim ?? 0)
            }
        } else {
            out += "  • Architecture       : (unknown — the backend passed no descriptor; not assumed)\n"
        }
        out += String(format: "  • Multimodal Vision  : %@\n", params.visionStatus)
        out += String(format: "  • LM Head Storage    : %@\n", params.headStorage)
        if let note = params.weightContainerNote {
            out += String(format: "  • Weights Container : %@\n", note)
        }
        out += String(format: "  • KV Cache Mode      : %@\n", String(describing: params.kvMode))
        out += String(format: "  • Context Budget     : %d tokens\n", params.contextBudget)
        out += String(format: "  • Speculative MTP    : %d drafts (Backend: %@)\n", params.mtpDrafts, params.mtpBackend)
        out += String(format: "  • GDN Prefill Mode   : %@\n", params.gdnPrefillMode)
        out += String(format: "  • Prefill Chunk Size : %d tokens (default 512)\n", params.prefillChunkRows)
        out += String(format: "  • Scratch Eviction   : %@\n", params.prefillScratchEviction)
        out += "Network & Endpoints:\n"
        out += String(format: "  • Listening Address  : http://%@:%d\n", host, port)
        out += "  • OpenAI Endpoints   : POST /v1/chat/completions (text + multimodal image_url)\n"
        out += "                         POST /v1/models/load | POST /v1/models/unload | GET /v1/models\n"
        out += "  • Anthropic Endpoints: POST /v1/messages\n"
        out += "  • Ollama Endpoints   : POST /api/chat | POST /api/generate | POST /api/show\n"
        out += "                         POST /api/load | POST /api/unload\n"
        out += "                         GET /api/tags | GET /api/ps | GET /api/version\n"
        out += "  • Health Check       : GET /health | GET /metrics | GET /v1/dashboard\n"
        out += "\(line)\n"
        return out
    }
}

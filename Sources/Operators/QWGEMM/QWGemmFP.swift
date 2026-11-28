// QWGemmFP.swift — Generalized BF16 NAX/GEMV, NVFP4_G16, and MXFP8_G32
// MoE grouped-NAX prefill, active-bucket compact decode/verify, and combine encoders.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

/// Element scalar representations supported by generalized FP GEMM/MoE kernels.
public enum QWFPElemDtype: UInt32, Sendable, Hashable {
    case f16 = 0
    case bf16 = 1
    case f32 = 2

    public var byteSize: Int {
        switch self {
        case .f16, .bf16: return 2
        case .f32: return 4
        }
    }

    var bf16KernelCode: UInt32 {
        switch self {
        case .bf16: return 0
        case .f32: return 1
        case .f16: return 2
        }
    }

    public init?(qwdType: QWDType) {
        switch qwdType {
        case .f16: self = .f16
        case .bf16: self = .bf16
        case .f32: self = .f32
        default: return nil
        }
    }
}

/// Zero-copy Metal buffer + byte offset descriptor shared by tensor-arena and standalone callers.
public struct QWFPBufferSlice: @unchecked Sendable {
    public let buffer: any MTLBuffer
    public let offset: Int
    public let bytes: Int

    public init(buffer: any MTLBuffer, offset: Int = 0, bytes: Int = 0) {
        self.buffer = buffer
        self.offset = offset
        self.bytes = bytes
    }

    public init(_ tensor: QWTensorValue) {
        self.buffer = tensor.region.buffer
        self.offset = tensor.region.base + tensor.offset
        self.bytes = tensor.bytes
    }

    @inline(__always)
    public func slice(byteOffset: Int, bytes: Int? = nil) -> QWFPBufferSlice {
        let rem = self.bytes > 0 ? max(0, self.bytes - byteOffset) : 0
        return QWFPBufferSlice(buffer: buffer, offset: offset + byteOffset, bytes: bytes ?? rem)
    }

    @inline(__always)
    public func bind(_ enc: any MTLComputeCommandEncoder, index: Int) {
        enc.setBuffer(buffer, offset: offset, index: index)
    }
}

/// Expert weight storage container spanning legacy quantized banks and FP containers.
public enum QWMoEExpertBank: Sendable {
    case weightView(QWWeightView)
    case nvfp4(QWGemmFP.NVFP4Bank)
    case mxfp8(QWGemmFP.MXFP8Bank)

    public var k: Int {
        switch self {
        case .weightView(let v): return v.k
        case .nvfp4(let b): return b.k
        case .mxfp8(let b): return b.k
        }
    }

    public var isFPContainer: Bool {
        switch self {
        case .weightView: return false
        case .nvfp4, .mxfp8: return true
        }
    }

    public var weightView: QWWeightView? {
        if case .weightView(let v) = self { return v }
        return nil
    }

    public func expertCount(rowsPerExpert: Int) -> Int {
        switch self {
        case .weightView(let v): return rowsPerExpert > 0 ? v.n / rowsPerExpert : 0
        case .nvfp4(let b): return b.experts
        case .mxfp8(let b): return b.experts
        }
    }
}

private struct QWGemmFPBF16Params {
    var m: UInt32
    var n: UInt32
    var k: UInt32
    var strideX: UInt32
    var strideY: UInt32
    var inIsF32: UInt32
    var outIsF32: UInt32
    var addResidual: UInt32
    var softcap: Float
}

internal struct QWGemmFPMoENaxParams {
    var experts: UInt32
    var n: UInt32
    var k: UInt32
    var yStride: UInt32
    var yColOff: UInt32
    var inDtype: UInt32
    var outDtype: UInt32
    var flags: UInt32
}

internal struct QWGemmFPCompactMoEParams {
    var n: UInt32
    var k: UInt32
    var topK: UInt32
    var numExperts: UInt32
    var wExpertRows: UInt32
    var upRowBias: UInt32
    var inDtype: UInt32
    var outDtype: UInt32
    var flags: UInt32
    var gateCap: Float
    var upCap: Float
}

internal struct QWGemmFPSwiGLUParams {
    var m: UInt32
    var i: UInt32
    var gateCap: Float
    var upCap: Float
    var flags: UInt32
}

internal struct QWGemmFPCombineParams {
    var hidden: UInt32
    var topK: UInt32
    var tokens: UInt32
    var inDtype: UInt32
    var outDtype: UInt32
    var sharedDtype: UInt32
    var flags: UInt32
}

/// Dispatchers for BF16, NVFP4_G16, and MXFP8_G32 linear and MoE kernels.
public enum QWGemmFP {
    /// Row-major `bfloat16 [n, k]` dense weight matrix.
    public struct BF16Bank: Sendable {
        public let weights: QWFPBufferSlice
        public let n: Int
        public let k: Int

        public init(weights: QWFPBufferSlice, n: Int, k: Int) {
            self.weights = weights
            self.n = n
            self.k = k
        }

        public init(weights: QWTensorValue, n: Int, k: Int) {
            self.init(weights: QWFPBufferSlice(weights), n: n, k: k)
        }
    }

    /// Block-16 FP4 (`NVFP4_G16`) weights with E4M3FN group scales and optional per-expert `s2`.
    public struct NVFP4Bank: Sendable {
        public let weights: QWFPBufferSlice
        public let scales: QWFPBufferSlice
        public let scales2: QWFPBufferSlice?
        public let experts: Int
        public let rowsPerExpert: Int
        public let k: Int
        public let splitGateUpScales2: Bool

        public init(weights: QWFPBufferSlice, scales: QWFPBufferSlice,
                    scales2: QWFPBufferSlice? = nil,
                    experts: Int, rowsPerExpert: Int, k: Int,
                    splitGateUpScales2: Bool = false) {
            self.weights = weights
            self.scales = scales
            self.scales2 = scales2
            self.experts = experts
            self.rowsPerExpert = rowsPerExpert
            self.k = k
            self.splitGateUpScales2 = splitGateUpScales2
        }
    }

    /// Block-32 FP8 E4M3FN (`MXFP8_G32`) weights with OCP UE8M0 group scales.
    public struct MXFP8Bank: Sendable {
        public let weights: QWFPBufferSlice
        public let scales: QWFPBufferSlice
        public let experts: Int
        public let rowsPerExpert: Int
        public let k: Int

        public init(weights: QWFPBufferSlice, scales: QWFPBufferSlice,
                    experts: Int, rowsPerExpert: Int, k: Int) {
            self.weights = weights
            self.scales = scales
            self.experts = experts
            self.rowsPerExpert = rowsPerExpert
            self.k = k
        }
    }

    /// Single source of truth for active-bucket indirect dispatch buffer layout (`48 B`).
    public struct MoEActiveBucketPlan: Sendable {
        public static let indirectBufferBytes: Int = 48
        public static let gateUpIndirectByteOffset: Int = 16
        public static let downIndirectByteOffset: Int = 32

        public let experts: Int
        public let gateUpGridX: Int
        public let downGridX: Int

        public init(experts: Int, intermediate: Int, hidden: Int, downTileRows: Int = 32) {
            self.experts = experts
            self.gateUpGridX = max(1, (intermediate + 31) / 32)
            self.downGridX = max(1, (hidden + downTileRows - 1) / downTileRows)
        }
    }

    /// Prewarms Metal compute pipelines used by BF16, NVFP4, and MXFP8 linear/MoE operators.
    public static func prewarm(hal: QWHal) throws {
        var names = (1...8).map { "qwg_gemv_bf16_rows2_m\($0)" }
        names.append(contentsOf: (1...4).map { "qwg_gemv_bf16_rows4_m\($0)" })
        names.append(contentsOf: [
            "qwg_moe_group_build_buckets_f32",
            "qwg_moe_build_active_buckets",
            "qwg_moe_compact_nvfp4_gate_up_swiglu",
            "qwg_moe_compact_nvfp4_gate_up_swiglu_t1",
            "qwg_moe_compact_nvfp4_down",
            "qwg_moe_compact_nvfp4_down_t1",
            "qwg_moe_compact_mxfp8_gate_up_swiglu",
            "qwg_moe_compact_mxfp8_gate_up_swiglu_t1",
            "qwg_moe_compact_mxfp8_down",
            "qwg_moe_compact_mxfp8_down_t1",
            "qwg_swiglu_softcap_f16",
            "qwg_moe_down_reduce_combine_bf16"
        ])
        if hal.caps.isMetal4TensorCapable {
            names.append(contentsOf: [
                "qwg_gemm_bf16_nax_bm32",
                "qwg_gemm_bf16_nax_bm64",
                "qwg_moe_gather_x",
                "qwg_moe_nax_gemm_nvfp4_g16",
                "qwg_moe_nax_gemm_nvfp4_g16_swiglu",
                "qwg_moe_nax_gemm_mxfp8_g32",
                "qwg_moe_nax_gemm_mxfp8_g32_swiglu"
            ])
        } else if hal.caps.supportsSimdgroupMMA {
            names.append(contentsOf: [
                "qwg_gemm_bf16_sg_bm32",
                "qwg_gemm_bf16_sg_bm64",
                "qwg_moe_gather_x",
                "qwg_moe_sg_gemm_nvfp4_g16",
                "qwg_moe_sg_gemm_nvfp4_g16_swiglu",
                "qwg_moe_sg_gemm_mxfp8_g32",
                "qwg_moe_sg_gemm_mxfp8_g32_swiglu"
            ])
        }
        for name in names {
            _ = try hal.pipeline(name)
        }
    }

    /// Auto-selects direct-register NAX / simdgroup MMA (`M > 8`) or row-paired GEMV.
    public static func encodeBF16(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        weight: BF16Bank, x: QWFPBufferSlice, y: QWFPBufferSlice,
        m: Int, inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        strideX: Int? = nil, strideY: Int? = nil,
        addResidual: Bool = false, softcap: Float = 0,
        minNaxRows: Int = 9
    ) throws {
        guard m > 0, weight.n > 0, weight.k > 0 else { return }
        if m >= minNaxRows && hal.caps.supportsSimdgroupMMA {
            try encodeBF16NAX(enc, hal: hal, weight: weight, x: x, y: y, m: m,
                              inDtype: inDtype, outDtype: outDtype,
                              strideX: strideX, strideY: strideY,
                              addResidual: addResidual, softcap: softcap)
        } else {
            try encodeBF16Gemv(enc, hal: hal, weight: weight, x: x, y: y, m: m,
                               inDtype: inDtype, outDtype: outDtype,
                               strideX: strideX, strideY: strideY,
                               addResidual: addResidual, softcap: softcap)
        }
    }

    /// Encodes `BF16` GEMM (`BM=32/64, BN=64`) using NAX on Apple10+ or `simdgroup_half8x8` on Apple7+.
    public static func encodeBF16NAX(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        weight: BF16Bank, x: QWFPBufferSlice, y: QWFPBufferSlice,
        m: Int, inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        strideX: Int? = nil, strideY: Int? = nil,
        addResidual: Bool = false, softcap: Float = 0
    ) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWGemmError.badShape("qwg_gemm_bf16_nax_bm32 requires Apple7+ simdgroup or Metal 4 NAX tensor support")
        }
        guard m > 0, weight.n > 0, weight.k > 0 else { return }
        let useBM64 = (m >= 48)
        let kName = useBM64
            ? (useNAX ? "qwg_gemm_bf16_nax_bm64" : "qwg_gemm_bf16_sg_bm64")
            : (useNAX ? "qwg_gemm_bf16_nax_bm32" : "qwg_gemm_bf16_sg_bm32")
        let bm = useBM64 ? 64 : 32
        let sx = strideX ?? weight.k
        let sy = strideY ?? weight.n
        let params = QWGemmFPBF16Params(
            m: UInt32(m), n: UInt32(weight.n), k: UInt32(weight.k),
            strideX: UInt32(sx), strideY: UInt32(sy),
            inIsF32: inDtype.bf16KernelCode, outIsF32: outDtype.bf16KernelCode,
            addResidual: addResidual ? 1 : 0, softcap: softcap)
        enc.setComputePipelineState(try hal.pipeline(kName))
        weight.weights.bind(enc, index: 0)
        x.bind(enc, index: 1)
        x.bind(enc, index: 2)
        y.bind(enc, index: 3)
        y.bind(enc, index: 4)
        enc.setPod(params, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (weight.n + 63) / 64, height: (m + bm - 1) / bm, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Encodes row-paired `BF16` GEMV across arbitrary `M` in chunks of `1..8`.
    public static func encodeBF16Gemv(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        weight: BF16Bank, x: QWFPBufferSlice, y: QWFPBufferSlice,
        m: Int, inDtype: QWFPElemDtype, outDtype: QWFPElemDtype,
        strideX: Int? = nil, strideY: Int? = nil,
        addResidual: Bool = false, softcap: Float = 0,
        rowsPerSimdgroup: Int? = nil
    ) throws {
        guard m > 0, weight.n > 0, weight.k > 0 else { return }
        let rowsSG: Int
        if let explicit = rowsPerSimdgroup {
            guard explicit == 2 || explicit == 4 else {
                throw QWGemmError.badShape("BF16 GEMV rowsPerSimdgroup must be 2 or 4")
            }
            rowsSG = explicit
        } else {
            rowsSG = 2
        }
        let sx = strideX ?? weight.k
        let sy = strideY ?? weight.n
        let rowsPerTG = 4 * rowsSG
        let gridX = (weight.n + rowsPerTG - 1) / rowsPerTG
        let maxChunk = (rowsSG == 4) ? 4 : 8
        var mOff = 0
        while mOff < m {
            let chunk = min(maxChunk, m - mOff)
            let kName = rowsSG == 4
                ? "qwg_gemv_bf16_rows4_m\(chunk)"
                : "qwg_gemv_bf16_rows2_m\(chunk)"
            let xSlice = x.slice(byteOffset: mOff * sx * inDtype.byteSize)
            let ySlice = y.slice(byteOffset: mOff * sy * outDtype.byteSize)
            let params = QWGemmFPBF16Params(
                m: UInt32(chunk), n: UInt32(weight.n), k: UInt32(weight.k),
                strideX: UInt32(sx), strideY: UInt32(sy),
                inIsF32: inDtype.bf16KernelCode, outIsF32: outDtype.bf16KernelCode,
                addResidual: addResidual ? 1 : 0, softcap: softcap)
            enc.setComputePipelineState(try hal.pipeline(kName))
            weight.weights.bind(enc, index: 0)
            xSlice.bind(enc, index: 1)
            xSlice.bind(enc, index: 2)
            ySlice.bind(enc, index: 3)
            ySlice.bind(enc, index: 4)
            enc.setPod(params, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: gridX, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            mOff += chunk
        }
    }
}

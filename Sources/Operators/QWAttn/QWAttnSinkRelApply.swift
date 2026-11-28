// QWAttnSinkRelApply.swift — L4.0 encoder for generalized NAX FlashAttention prefill
// and GQA-shared Split-KV decode with optional sink, banded relBias, and output gate.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV

public enum QWAttnSinkRelError: Error, CustomStringConvertible, Equatable {
    case badShape(String)
    case unsupportedHardware(String)

    public var description: String {
        switch self {
        case .badShape(let msg): return "QWAttnSinkRelError badShape: \(msg)"
        case .unsupportedHardware(let msg): return "QWAttnSinkRelError unsupportedHardware: \(msg)"
        }
    }
}

/// Configuration for one generalized sink + banded-relBias + gated attention layer.
public struct QWSinkRelConfig: Sendable, Equatable {
    public let nH: Int
    public let nKV: Int
    public let headDim: Int
    public let dRel: Int
    public let relExtent: Int
    public let slidingWindow: Int
    public let hasSink: Bool
    public let hasRelBias: Bool
    public let hasOutGate: Bool
    public let kvMode: QWKVMode

    public init(
        nH: Int,
        nKV: Int,
        headDim: Int = 128,
        dRel: Int = 0,
        relExtent: Int = 0,
        slidingWindow: Int = 0,
        hasSink: Bool = false,
        hasRelBias: Bool = false,
        hasOutGate: Bool = false,
        kvMode: QWKVMode = .f16
    ) {
        self.nH = nH
        self.nKV = nKV
        self.headDim = headDim
        self.dRel = dRel
        self.relExtent = relExtent
        self.slidingWindow = slidingWindow
        self.hasSink = hasSink
        self.hasRelBias = hasRelBias
        self.hasOutGate = hasOutGate
        self.kvMode = kvMode
    }

    public init(
        geometry: QWModelGeometry,
        isSliding: Bool,
        hasSink: Bool? = nil,
        hasRelBias: Bool? = nil,
        hasOutGate: Bool? = nil,
        kvMode: QWKVMode = .f16
    ) {
        let sr = geometry.descriptor.attentionSinkRelConfig
        let resolvedDRel = sr?.dRel ?? 0
        let resolvedRelExt = sr?.relExtent ?? 0
        let resolvedSliding = isSliding ? (sr?.slidingWindowSize ?? geometry.descriptor.slidingWindow ?? 0) : 0
        self.nH = (isSliding ? sr?.swaQHeads : nil) ?? geometry.qHeads
        self.nKV = (isSliding ? sr?.swaKVHeads : nil) ?? geometry.kvHeads
        self.headDim = (isSliding ? sr?.swaHeadDim : nil) ?? geometry.headDim
        self.dRel = resolvedDRel
        self.relExtent = resolvedRelExt
        self.slidingWindow = resolvedSliding
        self.hasSink = hasSink ?? (sr?.hasSink == true)
        self.hasRelBias = hasRelBias ?? (resolvedDRel > 0 && resolvedRelExt > 0)
        self.hasOutGate = hasOutGate ?? (sr != nil || geometry.attnOutputGate)
        self.kvMode = kvMode
    }
}

/// Non-paged ring/linear KV cache views for sink+rel attention.
public struct QWSinkRelKVCache: Sendable {
    public let kCache: QWMRegion
    public let vCache: QWMRegion
    public let kMeta: QWMRegion?
    public let vMeta: QWMRegion?
    public let ringCap: Int
    public let mode: QWKVMode

    public init(
        kCache: QWMRegion,
        vCache: QWMRegion,
        kMeta: QWMRegion? = nil,
        vMeta: QWMRegion? = nil,
        ringCap: Int,
        mode: QWKVMode
    ) {
        self.kCache = kCache
        self.vCache = vCache
        self.kMeta = kMeta
        self.vMeta = vMeta
        self.ringCap = ringCap
        self.mode = mode
    }
}

/// Pre-allocated FP32 split-KV scratch regions (`partO` + `partML`).
public struct QWSinkRelPartials: Sendable {
    public let partO: QWMRegion
    public let partML: QWMRegion

    public init(partO: QWMRegion, partML: QWMRegion) {
        self.partO = partO
        self.partML = partML
    }
}

/// Stateless encoder for `qwg_attn_prefill_sink_rel_nax_*` and `qwg_attn_decode_sink_rel_gqa_*`.
public final class QWAttnSinkRel: @unchecked Sendable {
    public static let maxSplits = QWAttnDecode.maxSplits
    public static let maxDecodeBatchTokens = 16

    public struct TreeMeta: Sendable, Equatable {
        public var ancMask: [UInt32]
        public var depths: [UInt32]

        public init(ancMask: [UInt32], depths: [UInt32]) {
            self.ancMask = ancMask
            self.depths = depths
        }
    }

    internal struct MetalTreeMeta {
        var isTree: UInt32
        var pad: (UInt32, UInt32, UInt32) = (0, 0, 0)
        var ancMask: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
        var depths: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                     UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }

    internal struct MetalParams {
        var T: UInt32
        var nH: UInt32
        var nKV: UInt32
        var headDim: UInt32
        var dRel: UInt32
        var sliding: UInt32
        var relExtent: UInt32
        var posOffset: UInt32
        var ringCap: UInt32
        var kvMode: UInt32
        var kvLen: UInt32
        var splits: UInt32
        var hasSink: UInt32
        var hasRelBias: UInt32
        var hasOutGate: UInt32
        var splitSpan: UInt32 = 0
    }

    internal let hal: QWHal
    internal let dummyRegion: QWMRegion
    private let psoPrefillF16: (any MTLComputePipelineState)?
    private let psoPrefillQ8: (any MTLComputePipelineState)?
    private let psoPrefillQ4: (any MTLComputePipelineState)?
    internal let psoPrefillSparseF16: (any MTLComputePipelineState)?
    internal let psoPrefillSparseQ8: (any MTLComputePipelineState)?
    internal let psoPrefillSparseQ4: (any MTLComputePipelineState)?
    private let psoDecodeF16: any MTLComputePipelineState
    private let psoDecodeQ8: any MTLComputePipelineState
    private let psoDecodeQ4: any MTLComputePipelineState
    private let psoDecodeBatchF16: any MTLComputePipelineState
    private let psoDecodeBatchQ8: any MTLComputePipelineState
    private let psoDecodeBatchQ4: any MTLComputePipelineState
    private let psoDecodeBatch8F16: any MTLComputePipelineState
    private let psoDecodeBatch8Q8: any MTLComputePipelineState
    private let psoDecodeBatch8Q4: any MTLComputePipelineState
    private let psoDecodeBatch4F16: any MTLComputePipelineState
    private let psoDecodeBatch4Q8: any MTLComputePipelineState
    private let psoDecodeBatch4Q4: any MTLComputePipelineState
    internal let psoDecodeSparseF16: any MTLComputePipelineState
    internal let psoDecodeSparseQ8: any MTLComputePipelineState
    internal let psoDecodeSparseQ4: any MTLComputePipelineState
    internal let psoCombine: any MTLComputePipelineState

    public init(hal: QWHal) throws {
        self.hal = hal
        guard let dummy = hal.device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw QWAttnSinkRelError.badShape("failed to allocate 256-byte dummy buffer")
        }
        memset(dummy.contents(), 0, 256)
        self.dummyRegion = QWMRegion(buffer: dummy, base: 0, length: 256)

        self.psoDecodeF16 = try hal.pipeline(Self.decodeKernelName(.f16))
        self.psoDecodeQ8 = try hal.pipeline(Self.decodeKernelName(.q8))
        self.psoDecodeQ4 = try hal.pipeline(Self.decodeKernelName(.q4))
        self.psoDecodeBatchF16 = try hal.pipeline(Self.decodeBatchKernelName(.f16))
        self.psoDecodeBatchQ8 = try hal.pipeline(Self.decodeBatchKernelName(.q8))
        self.psoDecodeBatchQ4 = try hal.pipeline(Self.decodeBatchKernelName(.q4))
        self.psoDecodeBatch8F16 = try hal.pipeline(Self.decodeBatch8KernelName(.f16))
        self.psoDecodeBatch8Q8 = try hal.pipeline(Self.decodeBatch8KernelName(.q8))
        self.psoDecodeBatch8Q4 = try hal.pipeline(Self.decodeBatch8KernelName(.q4))
        self.psoDecodeBatch4F16 = try hal.pipeline(Self.decodeBatch4KernelName(.f16))
        self.psoDecodeBatch4Q8 = try hal.pipeline(Self.decodeBatch4KernelName(.q8))
        self.psoDecodeBatch4Q4 = try hal.pipeline(Self.decodeBatch4KernelName(.q4))
        self.psoDecodeSparseF16 = try hal.pipeline(Self.decodeSparseKernelName(.f16))
        self.psoDecodeSparseQ8 = try hal.pipeline(Self.decodeSparseKernelName(.q8))
        self.psoDecodeSparseQ4 = try hal.pipeline(Self.decodeSparseKernelName(.q4))
        self.psoCombine = try hal.pipeline(Self.combineKernelName)

        let useNAX = hal.caps.isMetal4TensorCapable
        if useNAX || hal.caps.supportsSimdgroupMMA {
            self.psoPrefillF16 = try hal.pipeline(Self.prefillKernelName(.f16, useNAX: useNAX))
            self.psoPrefillQ8 = try hal.pipeline(Self.prefillKernelName(.q8, useNAX: useNAX))
            self.psoPrefillQ4 = try hal.pipeline(Self.prefillKernelName(.q4, useNAX: useNAX))
            self.psoPrefillSparseF16 = try hal.pipeline(Self.prefillSparseKernelName(.f16, useNAX: useNAX))
            self.psoPrefillSparseQ8 = try hal.pipeline(Self.prefillSparseKernelName(.q8, useNAX: useNAX))
            self.psoPrefillSparseQ4 = try hal.pipeline(Self.prefillSparseKernelName(.q4, useNAX: useNAX))
        } else {
            self.psoPrefillF16 = nil
            self.psoPrefillQ8 = nil
            self.psoPrefillQ4 = nil
            self.psoPrefillSparseF16 = nil
            self.psoPrefillSparseQ8 = nil
            self.psoPrefillSparseQ4 = nil
        }
    }

    static func prefillKernelName(_ mode: QWKVMode, useNAX: Bool = true) -> String {
        let kind = useNAX ? "nax" : "mma"
        switch mode {
        case .f16: return "qwg_attn_prefill_sink_rel_\(kind)_f16"
        case .q8:  return "qwg_attn_prefill_sink_rel_\(kind)_q8"
        case .q4:  return "qwg_attn_prefill_sink_rel_\(kind)_q4"
        }
    }

    static func decodeKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sink_rel_gqa_f16"
        case .q8:  return "qwg_attn_decode_sink_rel_gqa_q8"
        case .q4:  return "qwg_attn_decode_sink_rel_gqa_q4"
        }
    }

    static func decodeBatchKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sink_rel_gqa_batch_f16"
        case .q8:  return "qwg_attn_decode_sink_rel_gqa_batch_q8"
        case .q4:  return "qwg_attn_decode_sink_rel_gqa_batch_q4"
        }
    }

    static func decodeBatch8KernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sink_rel_gqa_batch8_f16"
        case .q8:  return "qwg_attn_decode_sink_rel_gqa_batch8_q8"
        case .q4:  return "qwg_attn_decode_sink_rel_gqa_batch8_q4"
        }
    }

    static func decodeBatch4KernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sink_rel_gqa_batch4_f16"
        case .q8:  return "qwg_attn_decode_sink_rel_gqa_batch4_q8"
        case .q4:  return "qwg_attn_decode_sink_rel_gqa_batch4_q4"
        }
    }

    static func prefillSparseKernelName(_ mode: QWKVMode, useNAX: Bool = true) -> String {
        let kind = useNAX ? "nax" : "mma"
        switch mode {
        case .f16: return "qwg_attn_prefill_sink_rel_sparse_\(kind)_f16"
        case .q8:  return "qwg_attn_prefill_sink_rel_sparse_\(kind)_q8"
        case .q4:  return "qwg_attn_prefill_sink_rel_sparse_\(kind)_q4"
        }
    }

    static func decodeSparseKernelName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "qwg_attn_decode_sink_rel_sparse_gqa_f16"
        case .q8:  return "qwg_attn_decode_sink_rel_sparse_gqa_q8"
        case .q4:  return "qwg_attn_decode_sink_rel_sparse_gqa_q4"
        }
    }

    static let combineKernelName = "qwg_attn_decode_sink_rel_combine"

    public static func kvModeRaw(_ mode: QWKVMode) -> UInt32 {
        switch mode {
        case .f16: return 0
        case .q8:  return 1
        case .q4:  return 2
        }
    }

    public static func kvRowBytes(mode: QWKVMode, nKV: Int, headDim: Int) -> (payload: Int, meta: Int) {
        let elems = nKV * headDim
        switch mode {
        case .f16: return (elems * 2, 0)
        case .q8:  return (elems, nKV * 2)
        case .q4:  return (elems / 2, nKV * 4)
        }
    }

    public static func partialsBytes(tokens: Int, qHeads: Int, headDim: Int, splits: Int)
        -> (partOBytes: Int, partMLBytes: Int) {
        let count = max(1, tokens) * max(1, qHeads) * max(1, splits)
        return (count * max(1, headDim) * 4, count * 8)
    }

    internal static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a), y = abs(b)
        while y != 0 { let t = x % y; x = y; y = t }
        return max(1, x)
    }

    /// Fixed key-block span (128 sliding / 256 global) keeping Split-KV reduction order identical across T=1 and T=2...16.
    public static func splitSpan(ringCap: Int, slidingWindow: Int, maxSplits: Int = maxSplits) -> Int {
        let spanCap = slidingWindow > 0 ? (slidingWindow + maxDecodeBatchTokens) : max(1, ringCap)
        let denom = slidingWindow > 0 ? max(1, maxSplits - 1) : max(1, maxSplits)
        let minSpan = (spanCap + denom - 1) / denom
        if slidingWindow > 0 { return max(128, ((minSpan + 127) / 128) * 128) }
        return max(256, ((minSpan + 255) / 256) * 256)
    }

    /// Wave-aligned Split-KV count across sliding (> 128) and global (> 256) windows.
    public static func splits(
        forKeys keys: Int,
        kvHeads: Int,
        batchTokens: Int = 1,
        slidingWindow: Int = 0,
        gpuCores: Int = 40,
        tokensPerSplit: Int = 256,
        maxSplits: Int = maxSplits,
        gqa: Int? = nil
    ) -> Int {
        guard keys > 0 else { return 1 }
        let activeKeys = slidingWindow > 0 ? min(keys, slidingWindow) : keys
        let minActive = slidingWindow > 0 ? 128 : 256
        if activeKeys <= minActive { return 1 }
        let tgZ = (batchTokens >= 2) ? ((batchTokens + 15) / 16) : max(1, batchTokens)
        let gChunks = (batchTokens >= 2) ? max(1, ((gqa ?? 4) + 3) / 4) : 1
        let groups = max(1, kvHeads * tgZ * gChunks)
        let effTps = (batchTokens >= 2) ? max(tokensPerSplit, 1024) : max(tokensPerSplit, 512)
        let raw = max(2, min(maxSplits, (activeKeys + effTps - 1) / effTps))
        let cores = max(1, gpuCores)
        let step = cores / Self.gcd(cores, groups)
        let aligned = ((raw + step - 1) / step) * step
        let candidate = (aligned <= maxSplits)
            ? aligned
            : (step <= maxSplits ? (maxSplits / step) * step : raw)
        return max(1, min(maxSplits, candidate))
    }

    private func prefillPipeline(_ mode: QWKVMode) throws -> any MTLComputePipelineState {
        let pso: (any MTLComputePipelineState)? = switch mode {
        case .f16: psoPrefillF16
        case .q8:  psoPrefillQ8
        case .q4:  psoPrefillQ4
        }
        guard let pso else {
            throw QWAttnSinkRelError.unsupportedHardware("Sink/rel prefill requires Apple7+ simdgroup or Apple10+ NAX hardware")
        }
        return pso
    }

    private func decodePipeline(_ mode: QWKVMode) -> any MTLComputePipelineState {
        switch mode {
        case .f16: return psoDecodeF16
        case .q8:  return psoDecodeQ8
        case .q4:  return psoDecodeQ4
        }
    }

    private func decodeBatchPipeline(_ mode: QWKVMode) -> any MTLComputePipelineState {
        switch mode {
        case .f16: return psoDecodeBatchF16
        case .q8:  return psoDecodeBatchQ8
        case .q4:  return psoDecodeBatchQ4
        }
    }

    private func decodeBatch8Pipeline(_ mode: QWKVMode) -> any MTLComputePipelineState {
        switch mode {
        case .f16: return psoDecodeBatch8F16
        case .q8:  return psoDecodeBatch8Q8
        case .q4:  return psoDecodeBatch8Q4
        }
    }

    private func decodeBatch4Pipeline(_ mode: QWKVMode) -> any MTLComputePipelineState {
        switch mode {
        case .f16: return psoDecodeBatch4F16
        case .q8:  return psoDecodeBatch4Q8
        case .q4:  return psoDecodeBatch4Q4
        }
    }

    private func validateTensor(
        _ t: QWTensorValue,
        minElements: Int,
        alignBytes: Int,
        name: String
    ) throws {
        guard t.shape.dtype == .bf16 else {
            throw QWAttnSinkRelError.badShape("\(name) must be .bf16, got \(t.shape.dtype)")
        }
        let reg = t.region
        guard reg.base >= 0, reg.length >= 0,
              reg.base <= reg.buffer.length, reg.length <= reg.buffer.length - reg.base,
              t.offset >= 0, t.offset <= reg.length, t.bytes <= reg.length - t.offset,
              (reg.base + t.offset) % alignBytes == 0,
              t.shape.elementCount >= minElements
        else {
            throw QWAttnSinkRelError.badShape(
                "\(name) invalid bounds/alignment: elems=\(t.shape.elementCount) need=\(minElements)")
        }
    }

    internal func validateRegion(_ r: QWMRegion, minBytes: Int, alignBytes: Int, name: String) throws {
        guard r.base >= 0, r.length >= minBytes,
              r.base % alignBytes == 0,
              r.base <= r.buffer.length, r.length <= r.buffer.length - r.base
        else {
            throw QWAttnSinkRelError.badShape("\(name) length \(r.length) < \(minBytes) or misaligned")
        }
    }

    internal func validateCommon(
        config: QWSinkRelConfig,
        q: QWTensorValue,
        r: QWTensorValue?,
        relProj: QWTensorValue?,
        sink: QWTensorValue?,
        gate: QWTensorValue?,
        kv: QWSinkRelKVCache,
        out: QWTensorValue,
        T: Int,
        posOffset: Int,
        kvLen: Int,
        isPrefill: Bool
    ) throws {
        guard config.nKV > 0, config.nH >= config.nKV, config.nH % config.nKV == 0 else {
            throw QWAttnSinkRelError.badShape("invalid heads nH=\(config.nH) nKV=\(config.nKV)")
        }
        let gqa = config.nH / config.nKV
        if isPrefill {
            guard config.headDim == 128, gqa <= 16 else {
                throw QWAttnSinkRelError.badShape("prefill requires headDim=128 and gqa<=16, got hd=\(config.headDim) gqa=\(gqa)")
            }
        } else {
            guard config.headDim >= 32, config.headDim <= 128, config.headDim % 32 == 0,
                  (1...Self.maxDecodeBatchTokens).contains(T) else {
                throw QWAttnSinkRelError.badShape("decode requires headDim in 32...128 (%32==0) and T in 1...\(Self.maxDecodeBatchTokens), got hd=\(config.headDim) T=\(T)")
            }
        }
        guard T >= 1, posOffset >= 0, kvLen >= posOffset + T, kv.ringCap >= 1,
              config.slidingWindow >= 0, kv.mode == config.kvMode else {
            throw QWAttnSinkRelError.badShape("T=\(T) pos=\(posOffset) kvLen=\(kvLen) ringCap=\(kv.ringCap)")
        }
        if config.slidingWindow == 0 {
            guard kv.ringCap >= kvLen else {
                throw QWAttnSinkRelError.badShape("global ringCap \(kv.ringCap) < kvLen \(kvLen)")
            }
        } else {
            let minRing = min(kvLen, config.slidingWindow + T - 1)
            guard kv.ringCap >= minRing else {
                throw QWAttnSinkRelError.badShape("sliding ringCap \(kv.ringCap) < minRequired \(minRing)")
            }
        }

        let qElems = T * config.nH * config.headDim
        try validateTensor(q, minElements: qElems, alignBytes: 8, name: "q")
        try validateTensor(out, minElements: qElems, alignBytes: 8, name: "out")
        if config.hasOutGate {
            guard let gate else { throw QWAttnSinkRelError.badShape("hasOutGate=true requires gate tensor") }
            try validateTensor(gate, minElements: qElems, alignBytes: 8, name: "gate")
        }
        if config.hasSink {
            guard let sink else { throw QWAttnSinkRelError.badShape("hasSink=true requires sink tensor") }
            try validateTensor(sink, minElements: T * config.nH, alignBytes: 2, name: "sink")
        }
        if config.hasRelBias {
            guard config.dRel > 0, config.relExtent > 0 else {
                throw QWAttnSinkRelError.badShape("hasRelBias=true requires positive dRel/relExtent")
            }
            if !isPrefill {
                guard config.dRel <= 32 else {
                    throw QWAttnSinkRelError.badShape("decode relBias requires dRel <= 32, got \(config.dRel)")
                }
            }
            guard let r, let relProj else {
                throw QWAttnSinkRelError.badShape("hasRelBias=true requires r and relProj tensors")
            }
            try validateTensor(r, minElements: T * config.nH * config.dRel, alignBytes: 2, name: "r")
            try validateTensor(relProj, minElements: config.dRel * config.relExtent, alignBytes: 2, name: "relProj")
        }

        let rowBytes = Self.kvRowBytes(mode: kv.mode, nKV: config.nKV, headDim: config.headDim)
        let payMin = kv.ringCap * rowBytes.payload
        try validateRegion(kv.kCache, minBytes: payMin, alignBytes: 4, name: "kCache")
        try validateRegion(kv.vCache, minBytes: payMin, alignBytes: 4, name: "vCache")
        if rowBytes.meta > 0 {
            let metaMin = kv.ringCap * rowBytes.meta
            guard let km = kv.kMeta, let vm = kv.vMeta else {
                throw QWAttnSinkRelError.badShape("kvMode \(kv.mode) requires kMeta and vMeta")
            }
            try validateRegion(km, minBytes: metaMin, alignBytes: 4, name: "kMeta")
            try validateRegion(vm, minBytes: metaMin, alignBytes: 4, name: "vMeta")
        }
    }

    internal func bindOptionalTensor(_ t: QWTensorValue?, active: Bool, enc: any MTLComputeCommandEncoder, index: Int) {
        if active, let t {
            t.bind(enc, index: index)
        } else {
            enc.setBuffer(dummyRegion.buffer, offset: dummyRegion.base, index: index)
        }
    }

    internal func bindMetaRegion(_ r: QWMRegion?, enc: any MTLComputeCommandEncoder, index: Int) {
        if let r {
            enc.setBuffer(r.buffer, offset: r.base, index: index)
        } else {
            enc.setBuffer(dummyRegion.buffer, offset: dummyRegion.base, index: index)
        }
    }

    internal static func packTreeMeta(_ tree: TreeMeta?) -> MetalTreeMeta {
        guard let tree else {
            let z16: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) =
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            return MetalTreeMeta(isTree: 0, ancMask: z16, depths: z16)
        }
        func m(_ i: Int) -> UInt32 { i < tree.ancMask.count ? tree.ancMask[i] : 0 }
        func d(_ i: Int) -> UInt32 { i < tree.depths.count ? tree.depths[i] : 0 }
        return MetalTreeMeta(
            isTree: 1,
            ancMask: (m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7),
                      m(8), m(9), m(10), m(11), m(12), m(13), m(14), m(15)),
            depths: (d(0), d(1), d(2), d(3), d(4), d(5), d(6), d(7),
                     d(8), d(9), d(10), d(11), d(12), d(13), d(14), d(15))
        )
    }

    internal func makeMetalParams(
        config: QWSinkRelConfig,
        T: Int,
        posOffset: Int,
        kvLen: Int,
        ringCap: Int,
        splits: Int,
        splitSpan: UInt32 = 0
    ) -> MetalParams {
        MetalParams(
            T: UInt32(T),
            nH: UInt32(config.nH),
            nKV: UInt32(config.nKV),
            headDim: UInt32(config.headDim),
            dRel: UInt32(config.hasRelBias ? config.dRel : 0),
            sliding: UInt32(config.slidingWindow),
            relExtent: UInt32(config.hasRelBias ? config.relExtent : 0),
            posOffset: UInt32(posOffset),
            ringCap: UInt32(ringCap),
            kvMode: Self.kvModeRaw(config.kvMode),
            kvLen: UInt32(kvLen),
            splits: UInt32(splits),
            hasSink: config.hasSink ? 1 : 0,
            hasRelBias: config.hasRelBias ? 1 : 0,
            hasOutGate: config.hasOutGate ? 1 : 0,
            splitSpan: splitSpan
        )
    }

    /// Encode NAX FlashAttention prefill (`BM=32, BK=32, headDim=128`, zero steady-state allocs).
    public func encodePrefill(
        _ enc: any MTLComputeCommandEncoder,
        config: QWSinkRelConfig,
        q: QWTensorValue,
        r: QWTensorValue? = nil,
        relProj: QWTensorValue? = nil,
        sink: QWTensorValue? = nil,
        gate: QWTensorValue? = nil,
        kv: QWSinkRelKVCache,
        out: QWTensorValue,
        T: Int,
        posOffset: Int = 0,
        kvLen: Int? = nil
    ) throws {
        let resolvedKVLen = kvLen ?? (posOffset + T)
        try validateCommon(
            config: config, q: q, r: r, relProj: relProj, sink: sink, gate: gate,
            kv: kv, out: out, T: T, posOffset: posOffset, kvLen: resolvedKVLen, isPrefill: true
        )
        let pso = try prefillPipeline(config.kvMode)
        var params = makeMetalParams(
            config: config, T: T, posOffset: posOffset, kvLen: resolvedKVLen,
            ringCap: kv.ringCap, splits: 1
        )

        enc.setComputePipelineState(pso)
        q.bind(enc, index: 0)
        bindOptionalTensor(r, active: config.hasRelBias, enc: enc, index: 1)
        bindOptionalTensor(relProj, active: config.hasRelBias, enc: enc, index: 2)
        bindOptionalTensor(sink, active: config.hasSink, enc: enc, index: 3)
        bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 4)
        enc.setBuffer(kv.kCache.buffer, offset: kv.kCache.base, index: 5)
        enc.setBuffer(kv.vCache.buffer, offset: kv.vCache.base, index: 6)
        bindMetaRegion(kv.kMeta, enc: enc, index: 7)
        bindMetaRegion(kv.vMeta, enc: enc, index: 8)
        out.bind(enc, index: 9)
        enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 10)

        let numTiles = (T + 31) / 32
        let gqa = config.nH / config.nKV
        enc.dispatchThreadgroups(
            MTLSize(width: numTiles, height: config.nKV, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64 * gqa, height: 1, depth: 1)
        )
    }

    /// Encode GQA-shared Split-KV decode & speculative verify (`T = 1...16`, optional tree mask, zero steady-state allocs).
    public func encodeDecode(
        _ enc: any MTLComputeCommandEncoder,
        config: QWSinkRelConfig,
        q: QWTensorValue,
        r: QWTensorValue? = nil,
        relProj: QWTensorValue? = nil,
        sink: QWTensorValue? = nil,
        gate: QWTensorValue? = nil,
        kv: QWSinkRelKVCache,
        partials: QWSinkRelPartials? = nil,
        out: QWTensorValue,
        T: Int,
        posOffset: Int,
        kvLen: Int? = nil,
        splits: Int? = nil,
        treeMeta: TreeMeta? = nil
    ) throws {
        let resolvedKVLen = kvLen ?? (posOffset + T)
        try validateCommon(
            config: config, q: q, r: r, relProj: relProj, sink: sink, gate: gate,
            kv: kv, out: out, T: T, posOffset: posOffset, kvLen: resolvedKVLen, isPrefill: false
        )
        let useBatch = (T >= 2 || treeMeta != nil)
        let useBatch4 = useBatch && T <= 4
        let useBatch8 = useBatch && !useBatch4 && T <= 8
        let gqa = config.nH / config.nKV
        let span = Self.splitSpan(ringCap: kv.ringCap, slidingWindow: config.slidingWindow)
        let globalMin = (config.slidingWindow > 0 && posOffset + 1 > config.slidingWindow)
            ? (posOffset + 1 - config.slidingWindow) : 0
        let clampedMin = min(globalMin, resolvedKVLen)
        let firstBlock = clampedMin / span
        let endBlock = resolvedKVLen > clampedMin ? ((resolvedKVLen + span - 1) / span) : (firstBlock + 1)
        let activeBlocks = max(1, endBlock - firstBlock)
        let resolvedSplits: Int
        let effectiveSplitSpan: UInt32
        if let explicitSplits = splits {
            resolvedSplits = explicitSplits
            effectiveSplitSpan = (explicitSplits >= activeBlocks) ? UInt32(span) : 0
        } else if activeBlocks <= 1 {
            resolvedSplits = 1
            effectiveSplitSpan = UInt32(span)
        } else {
            resolvedSplits = min(Self.maxSplits, activeBlocks)
            effectiveSplitSpan = UInt32(span)
        }
        guard (1...Self.maxSplits).contains(resolvedSplits) else {
            throw QWAttnSinkRelError.badShape("splits \(resolvedSplits) out of range 1...\(Self.maxSplits)")
        }

        let partO: QWMRegion
        let partML: QWMRegion
        if resolvedSplits > 1 {
            guard let partials else {
                throw QWAttnSinkRelError.badShape("splits=\(resolvedSplits) > 1 requires QWSinkRelPartials")
            }
            let need = Self.partialsBytes(
                tokens: T, qHeads: config.nH, headDim: config.headDim, splits: resolvedSplits
            )
            try validateRegion(partials.partO, minBytes: need.partOBytes, alignBytes: 4, name: "partO")
            try validateRegion(partials.partML, minBytes: need.partMLBytes, alignBytes: 8, name: "partML")
            partO = partials.partO
            partML = partials.partML
        } else {
            partO = partials?.partO ?? dummyRegion
            partML = partials?.partML ?? dummyRegion
        }

        var params = makeMetalParams(
            config: config, T: T, posOffset: posOffset, kvLen: resolvedKVLen,
            ringCap: kv.ringCap, splits: resolvedSplits, splitSpan: effectiveSplitSpan
        )
        let decodePSO = useBatch4
            ? decodeBatch4Pipeline(config.kvMode)
            : (useBatch8
                ? decodeBatch8Pipeline(config.kvMode)
                : (useBatch ? decodeBatchPipeline(config.kvMode) : decodePipeline(config.kvMode)))
        enc.setComputePipelineState(decodePSO)
        q.bind(enc, index: 0)
        bindOptionalTensor(r, active: config.hasRelBias, enc: enc, index: 1)
        bindOptionalTensor(relProj, active: config.hasRelBias, enc: enc, index: 2)
        bindOptionalTensor(sink, active: config.hasSink, enc: enc, index: 3)
        bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 4)
        enc.setBuffer(kv.kCache.buffer, offset: kv.kCache.base, index: 5)
        enc.setBuffer(kv.vCache.buffer, offset: kv.vCache.base, index: 6)
        bindMetaRegion(kv.kMeta, enc: enc, index: 7)
        bindMetaRegion(kv.vMeta, enc: enc, index: 8)
        out.bind(enc, index: 9)
        enc.setBuffer(partO.buffer, offset: partO.base, index: 10)
        enc.setBuffer(partML.buffer, offset: partML.base, index: 11)
        enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 12)
        if useBatch {
            let gChunks = (gqa + 3) / 4
            var tm = Self.packTreeMeta(treeMeta)
            enc.setBytes(&tm, length: MemoryLayout<MetalTreeMeta>.stride, index: 13)
            enc.dispatchThreadgroups(
                MTLSize(width: resolvedSplits, height: config.nKV * gChunks, depth: (useBatch4 || useBatch8) ? 1 : (T + 15) / 16),
                threadsPerThreadgroup: MTLSize(width: useBatch4 ? 128 : 256, height: 1, depth: 1)
            )
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: resolvedSplits, height: config.nKV, depth: T),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        if resolvedSplits > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoCombine)
            enc.setBuffer(partO.buffer, offset: partO.base, index: 0)
            enc.setBuffer(partML.buffer, offset: partML.base, index: 1)
            bindOptionalTensor(gate, active: config.hasOutGate, enc: enc, index: 2)
            out.bind(enc, index: 3)
            enc.setBytes(&params, length: MemoryLayout<MetalParams>.stride, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: config.nH, height: T, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
    }
}

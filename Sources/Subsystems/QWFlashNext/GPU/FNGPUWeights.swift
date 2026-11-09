import Darwin
import Foundation
import Metal
import QWCore
import QWQuant
import QWHAL
import QWAlloc
import QWGEMM
import QWGDN
import QWLoad

// FlashNext GPU layer weights: every tensor a decoder layer needs on the device, uploaded once from the HQ4 pack (stacked
// banks as HQ4 slabs; small projections as f32; the depthwise conv as f16). The CPU layer objects provide the small arrays.

/// A GatedDeltaNet mixer on the device, with its persistent recurrent state and ping-pong conv state.
public final class FNGPUGDNMixer: @unchecked Sendable {
    public let qkv: QWWeightView, z: QWWeightView, qkvZ: QWWeightView?, out: QWWeightView
    public let inProjBA: QWWeightView
    public let conv: QWTensorValue, normWeight: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue
    public let inProjB: QWTensorValue, inProjA: QWTensorValue
    public var states: [QWTensorValue]             // eight f32 [vHeads, kDim, vDim] ring buffers for O(1) verify rollback
    public var stateParity = 0                     // index of the active recurrent state buffer
    public var state: QWTensorValue { states[stateParity & QWFourSlotRollbackRing.ringMask] }
    public var altState: QWTensorValue { states[(stateParity + 1) & QWFourSlotRollbackRing.ringMask] }
    public var convState: [QWTensorValue]          // eight f32 [3, convDim] ring buffers, stepped each call
    public var convParity = 0                      // index of the conv state read by the next call
    public let vHeads: Int, kHeads: Int, kDim: Int, vDim: Int, convDim: Int

    init(qkv: QWWeightView, z: QWWeightView, qkvZ: QWWeightView?, out: QWWeightView, inProjBA: QWWeightView,
         conv: QWTensorValue, normWeight: QWTensorValue, aLog: QWTensorValue, dtBias: QWTensorValue,
         inProjB: QWTensorValue, inProjA: QWTensorValue, states: [QWTensorValue], convState: [QWTensorValue],
         vHeads: Int, kHeads: Int, kDim: Int, vDim: Int, convDim: Int) {
        self.qkv = qkv; self.z = z; self.qkvZ = qkvZ; self.out = out; self.inProjBA = inProjBA
        self.conv = conv; self.normWeight = normWeight; self.aLog = aLog; self.dtBias = dtBias
        self.inProjB = inProjB; self.inProjA = inProjA; self.states = states; self.convState = convState
        self.vHeads = vHeads; self.kHeads = kHeads; self.kDim = kDim; self.vDim = vDim; self.convDim = convDim
    }
}

/// Full-attention mixer on the device: stacked q/k/v/index_qk bank, o_proj, and the norms and RoPE of the CPU layer.
public struct FNGPUAttnMixer: Sendable {
    public let bank: QWWeightView, oProj: QWWeightView
    public let qNorm: QWTensorValue, kNorm: QWTensorValue, iqNorm: QWTensorValue, ikNorm: QWTensorValue
    public let heads: Int, kvHeads: Int, headDim: Int, indexHeads: Int, indexKVHeads: Int, indexHeadDim: Int
    public let rotaryDim: Int, compress: Int, blockTopK: Int, bankRows: Int
    public let rope: FNRope
    public var iqOff: Int { heads * 2 * headDim + 2 * kvHeads * headDim }
    public var ikOff: Int { iqOff + indexHeads * indexHeadDim }
}

public enum FNGPUMixer: Sendable {
    case gdn(FNGPUGDNMixer)
    case attention(FNGPUAttnMixer)
}

/// Hyper-connection mixer on the device (f32 activations, BF16 injection weight, HQ4 down/up).
public struct FNGPUHC: Sendable {
    public let norm: QWTensorValue, injectBF16: QWTensorValue
    /// down/up stay HQ4 on the device (rot domain): the activation is rotated instead of the weight.
    public let down: QWWeightView, up: QWWeightView
    public let hc: Int, hidden: Int, lowrank: Int
    public let eps: Float
}

/// Per-layer embedding on the device (HQ4 key/value bank for T=1 decode and sliced keyBank/valueBank for batched prefill) with its conv history:
/// the last (K-1)*dilation rows of the normed values, ping-pong buffers swapped each pass (persists across decode steps).
public final class FNGPUPLE: @unchecked Sendable {
    public let keyValueBank: QWWeightView
    public let keyBank: QWWeightView, valueBank: QWWeightView
    public let normKey: QWTensorValue, normQuery: QWTensorValue, normConv: QWTensorValue, convW: QWTensorValue
    public let embedDim: Int, kernel: Int, dilation: Int, eps: Float
    public var history: [QWTensorValue]          // two f32 [historyRows, width]
    public var parity = 0                        // index of the history read by the next pass
    public var historyRows: Int { (kernel - 1) * dilation }

    init(keyValueBank: QWWeightView, keyBank: QWWeightView, valueBank: QWWeightView,
         normKey: QWTensorValue, normQuery: QWTensorValue,
         normConv: QWTensorValue, convW: QWTensorValue, embedDim: Int, kernel: Int, dilation: Int, eps: Float, history: [QWTensorValue]) {
        self.keyValueBank = keyValueBank; self.keyBank = keyBank; self.valueBank = valueBank
        self.normKey = normKey; self.normQuery = normQuery
        self.normConv = normConv; self.convW = convW; self.embedDim = embedDim; self.kernel = kernel
        self.dilation = dilation; self.eps = eps; self.history = history
    }
}

/// Sparse MoE on the device: BF16 router + shared gate, HQ4 routed banks (stacked per expert), HQ4 shared expert.
public struct FNGPUMoE: Sendable {
    public let gateAndSharedBF16: QWTensorValue
    public let gateUp: QWWeightView, down: QWWeightView
    public let sharedGateUp: QWWeightView, sharedDown: QWWeightView
    public let sharedGateScalar: QWTensorValue
    public let experts: Int, topK: Int, intermediate: Int, sharedIntermediate: Int
}

public struct FNGPULayerWeights: Sendable {
    public let index: Int
    public let mixer: FNGPUMixer
    public let attnHC: FNGPUHC, mlpHC: FNGPUHC
    public let moe: FNGPUMoE
    public let ple: FNGPUPLE?
}

public enum FNGPUError: Error, Equatable {
    case notHQ4(String)
    case selectionLimitExceeded(positions: Int, limit: Int)
    case contextDoesNotFit(requested: Int, maxFitting: Int)
    case kvModeNotImplemented(String)
    case invalidToken(Int)
}

public typealias FNGPUEmbedTable = QWDeviceEmbedTable

extension QWDeviceEmbedTable {
    public static func load(arena: QWArena, loader: FNPackLoader, name: String,
                            vocab: Int, hidden: Int, hc: Int) throws -> QWDeviceEmbedTable {
        if let pack = loader.pack {
            return try load(arena: arena, pack: pack, name: name, vocab: vocab, hidden: hidden, hc: hc)
        }
        guard let st = loader.st else { throw FNGPUError.notHQ4(name) }
        return try load(arena: arena, container: st, name: name, vocab: vocab, hidden: hidden, hc: hc)
    }
}

/// Uploads FlashNext decoder layers from the pack or SafeTensors container into arena weight buffers.
public enum FNGPULoader {
    public typealias Span = QWCheckpointTensorLoader.PackSpan

    /// True when the SafeTensors checkpoint carries MLX affine quantized weights and
    /// `QW_WEIGHT_FORMAT` has not explicitly forced the incumbent rotated `HQ4_G32` path.
    public static func hasNativeAffineWeights(_ st: QWSafeTensorsContainer) -> Bool {
        QWCheckpointTensorLoader.hasNativeAffineWeights(st)
    }

    static func stackedSpan(_ pack: QWQuant.Pack, _ name: String) throws -> (span: Span, n: Int, k: Int) {
        try QWCheckpointTensorLoader.stackedSpan(pack, name)
    }

    static func span(_ pack: QWQuant.Pack, _ name: String) throws -> Span {
        try QWCheckpointTensorLoader.span(pack, name)
    }

    static func mappedBank(_ arena: QWArena, _ pack: QWQuant.Pack, _ s: Span, n: Int, k: Int) -> QWWeightView? {
        QWCheckpointTensorLoader.mappedBank(arena, pack, s, n: n, k: k)
    }

    public static func f32Buffer(_ arena: QWArena, _ a: [Float]) throws -> QWTensorValue {
        try f32(arena, a)
    }

    public static func hq4Bank(_ arena: QWArena, _ pack: QWQuant.Pack, _ name: String, n: Int, k: Int) throws -> QWWeightView {
        try QWCheckpointTensorLoader.bankFrom(arena, pack, try span(pack, name), n: n, k: k)
    }

    public static func hq4Bank(_ arena: QWArena, _ loader: FNPackLoader, _ name: String,
                               n: Int, k: Int, staging: (any MTLBuffer)? = nil) throws -> QWWeightView {
        try QWCheckpointTensorLoader.hqBank(
            arena: arena, pack: loader.pack, st: loader.st, name: name, n: n, k: k, staging: staging
        )
    }

    static func concatHQ4Bank(_ arena: QWArena, _ loader: FNPackLoader, _ names: [String],
                              k: Int, staging: (any MTLBuffer)? = nil) throws -> (view: QWWeightView, rows: Int) {
        try QWCheckpointTensorLoader.concatHQBank(
            arena: arena, pack: loader.pack, st: loader.st, names: names, k: k, staging: staging
        )
    }

    static func copySafeTensorsRowsAsBF16(st: QWSafeTensorsContainer, name: String,
                                          rowOffset: Int, rowCount: Int, cols: Int,
                                          into dst: UnsafeMutablePointer<UInt16>,
                                          parallel: Bool = true) throws {
        try QWHQ4Quantizer.copySafeTensorsRowsAsBF16(
            st: st, name: name, rowOffset: rowOffset, rowCount: rowCount,
            cols: cols, into: dst, parallel: parallel
        )
    }

    static func quantizeSafeTensorsInto(arena: QWArena, st: QWSafeTensorsContainer, name: String,
                                        srcRowOffset: Int, rowCount: Int, k: Int,
                                        dstBuffer: any MTLBuffer, dstByteOffset: Int,
                                        staging: (any MTLBuffer)? = nil) throws {
        try QWHQ4Quantizer.quantizeSafeTensorsInto(
            hal: arena.hal, st: st, name: name, srcRowOffset: srcRowOffset,
            rowCount: rowCount, k: k, dstBuffer: dstBuffer,
            dstByteOffset: dstByteOffset, staging: staging
        )
    }

    public static func releaseRepackStageBuffer() {
        QWCheckpointTensorLoader.releaseRepackStageBuffer()
    }

    static func stackedExpertBanks(_ arena: QWArena, _ loader: FNPackLoader, _ mp: String,
                                   experts: Int, inter: Int, hidden: Int,
                                   staging: (any MTLBuffer)? = nil) throws -> (gateUp: QWWeightView, down: QWWeightView) {
        try QWCheckpointTensorLoader.stackedExpertBanks(
            arena: arena, pack: loader.pack, st: loader.st, mp: mp,
            experts: experts, inter: inter, hidden: hidden, staging: staging
        )
    }

    static func bankFrom(_ arena: QWArena, _ pack: QWQuant.Pack, _ s: Span, n: Int, k: Int) throws -> QWWeightView {
        try QWCheckpointTensorLoader.bankFrom(arena, pack, s, n: n, k: k)
    }

    static func concatBank(_ arena: QWArena, _ pack: QWQuant.Pack, _ spans: [Span], n: Int, k: Int) throws -> QWWeightView {
        try QWCheckpointTensorLoader.concatBank(arena, pack, spans, n: n, k: k)
    }

    static func f32(_ arena: QWArena, _ a: [Float]) throws -> QWTensorValue {
        let t = try arena.weightsRegion().allocate(QWTensorShape([a.count], .f32))
        arena.write(t, from: a); return t
    }

    static func zerosF32(_ arena: QWArena, count: Int) throws -> QWTensorValue {
        let t = try arena.weightsRegion().allocate(QWTensorShape([count], .f32))
        memset(arena.cpuPointer(of: t), 0, t.bytes); return t
    }

    static func bf16FromF32(_ arena: QWArena, _ a: [Float]) throws -> QWTensorValue {
        let t = try arena.weightsRegion().allocate(QWTensorShape([a.count], .f16))
        arena.write(t, from: a.map(QWBF16.fromFloatRNE)); return t
    }

    /// HC weights: the norm is f32, injection stays BF16, and down/up are HQ4 and stay packed.
    static func hcWeights(_ arena: QWArena, _ loader: FNPackLoader, _ prefix: String,
                          staging: (any MTLBuffer)? = nil) throws -> FNGPUHC {
        let d = loader.descriptor, hc = d.hcCount!, hidden = d.hidden, lowrank = d.hcLowrank!, width = hc * hidden
        return FNGPUHC(norm: try f32(arena, try loader.tensor(prefix + "hc_norm.weight")),
                       injectBF16: try bf16FromF32(arena, try loader.tensor(prefix + "block_inject_weight.weight")),
                       down: try hq4Bank(arena, loader, prefix + "input_mix_weight_down.weight", n: lowrank, k: width, staging: staging),
                       up: try hq4Bank(arena, loader, prefix + "input_mix_weight_up.weight", n: width, k: lowrank, staging: staging),
                       hc: hc, hidden: hidden, lowrank: lowrank, eps: d.rmsEps)
    }

    /// Upload layer `index` directly from the pack or SafeTensors container into arena weight buffers.
    public static func layer(_ index: Int, loader: FNPackLoader, arena: QWArena,
                             prefix: String? = nil, staging: (any MTLBuffer)? = nil,
                             yarnFactor: Float? = nil) throws -> FNGPULayerWeights {
        let d = loader.descriptor, p = prefix ?? loader.layerPrefix(index)
        let isFull = (prefix != nil) || d.isFullAttention(index)
        let mixer: FNGPUMixer
        if !isFull {
            let gp = p + "linear_attn."
            let kHeads = d.gdnQKHeads, vHeads = d.gdnVHeads, kDim = d.gdnHeadDim, vDim = d.gdnHeadDim, hidden = d.hidden
            let convDim = 2 * kHeads * kDim + vHeads * vDim, valDim = vHeads * vDim
            let qkvView: QWWeightView, zView: QWWeightView, qkvZ: QWWeightView?, outView: QWWeightView
            if let pack = loader.pack {
                let qkv = try span(pack, gp + "in_proj_qkv.weight"), z = try span(pack, gp + "in_proj_z.weight")
                let out = try span(pack, gp + "out_proj.weight")
                if qkv.off + qkv.bytes == z.off && qkv.bits == z.bits && qkv.group == z.group && qkv.isIncumbent == z.isIncumbent {
                    let combined = try bankFrom(arena, pack,
                                                Span(off: qkv.off, bytes: qkv.bytes + z.bytes, rows: convDim + valDim, cols: hidden,
                                                     bits: qkv.bits, group: qkv.group, isIncumbent: qkv.isIncumbent),
                                                n: convDim + valDim, k: hidden)
                    let qkvBytes = convDim * combined.rowBytes, zBytes = valDim * combined.rowBytes
                    qkvView = combined.subview(blocks: combined.blocks.slice(byteOffset: 0, count: qkvBytes / 2, dtype: .f16), n: convDim, k: hidden)!
                    zView = combined.subview(blocks: combined.blocks.slice(byteOffset: qkvBytes, count: zBytes / 2, dtype: .f16), n: valDim, k: hidden)!
                    qkvZ = combined
                } else {
                    qkvView = try bankFrom(arena, pack, qkv, n: convDim, k: hidden)
                    zView = try bankFrom(arena, pack, z, n: valDim, k: hidden)
                    qkvZ = nil
                }
                outView = try bankFrom(arena, pack, out, n: hidden, k: valDim)
            } else {
                let combined = try concatHQ4Bank(arena, loader, [gp + "in_proj_qkv.weight", gp + "in_proj_z.weight"], k: hidden, staging: staging).view
                let qkvBytes = convDim * combined.rowBytes, zBytes = valDim * combined.rowBytes
                qkvView = combined.subview(blocks: combined.blocks.slice(byteOffset: 0, count: qkvBytes / 2, dtype: .f16), n: convDim, k: hidden)!
                zView = combined.subview(blocks: combined.blocks.slice(byteOffset: qkvBytes, count: zBytes / 2, dtype: .f16), n: valDim, k: hidden)!
                qkvZ = combined
                outView = try hq4Bank(arena, loader, gp + "out_proj.weight", n: hidden, k: valDim, staging: staging)
            }
            let inProjBA = try concatHQ4Bank(arena, loader, [gp + "in_proj_b.weight", gp + "in_proj_a.weight"], k: hidden, staging: staging).view
            let convSlotCount = 3 * convDim
            let convRing = try arena.activationsRegion().allocate(QWTensorShape([8 * convSlotCount], .f32))
            memset(arena.cpuPointer(of: convRing), 0, convRing.bytes)
            let convState = (0..<8).map { s in convRing.slice(byteOffset: s * convSlotCount * 4, count: convSlotCount, dtype: .f32) }
            let stateCount = vHeads * kDim * vDim
            let stateRing = try arena.activationsRegion().allocate(QWTensorShape([8 * stateCount], .f32))
            memset(arena.cpuPointer(of: stateRing), 0, stateCount * 4)
            let states = (0..<8).map { s in stateRing.slice(byteOffset: s * stateCount * 4, count: stateCount, dtype: .f32) }
            mixer = .gdn(FNGPUGDNMixer(
                qkv: qkvView, z: zView, qkvZ: qkvZ, out: outView, inProjBA: inProjBA,
                conv: try f32(arena, try loader.tensor(gp + "conv1d.weight")),
                normWeight: try f32(arena, try loader.tensor(gp + "norm.weight")),
                aLog: try f32(arena, try loader.tensor(gp + "A_log")),
                dtBias: try f32(arena, try loader.tensor(gp + "dt_bias")),
                inProjB: try f32(arena, try loader.tensor(gp + "in_proj_b.weight")),
                inProjA: try f32(arena, try loader.tensor(gp + "in_proj_a.weight")),
                states: states, convState: convState, vHeads: vHeads, kHeads: kHeads, kDim: kDim, vDim: vDim, convDim: convDim))
        } else {
            let ap = p + "self_attn.", ixp = ap + "indexer."
            let names = [ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight", ixp + "index_qk_proj.weight"]
            let (bankView, rows) = try concatHQ4Bank(arena, loader, names, k: d.hidden, staging: staging)
            let oView = try hq4Bank(arena, loader, ap + "o_proj.weight", n: d.hidden, k: d.qHeads * d.headDim, staging: staging)
            let ixHeads = d.indexerHeads!, ixKVHeads = d.indexerKvHeads!, ixHeadDim = d.indexerHeadDim!
            let ixCompress = d.indexerCompressRatio!, ixBudget = d.indexerBudget!
            let rope = FNRope(rotaryDim: d.rotaryDims, theta: d.ropeTheta, yarnFactor: yarnFactor, originalMaxPosition: d.maxPositionEmbeddings)
            mixer = .attention(FNGPUAttnMixer(
                bank: bankView, oProj: oView,
                qNorm: try f32(arena, try loader.tensor(ap + "q_norm.weight")),
                kNorm: try f32(arena, try loader.tensor(ap + "k_norm.weight")),
                iqNorm: try f32(arena, try loader.tensor(ixp + "q_layernorm.weight")),
                ikNorm: try f32(arena, try loader.tensor(ixp + "k_layernorm.weight")),
                heads: d.qHeads, kvHeads: d.kvHeads, headDim: d.headDim,
                indexHeads: ixHeads, indexKVHeads: ixKVHeads, indexHeadDim: ixHeadDim,
                rotaryDim: rope.rotaryDim, compress: ixCompress, blockTopK: ixBudget / ixCompress,
                bankRows: rows, rope: rope))
        }
        let mp = p + "mlp.", experts = d.numExperts!, inter = d.moeIntermediate!, sharedInter = d.sharedExpertIntermediate!
        let (guView, dnView) = try stackedExpertBanks(arena, loader, mp, experts: experts, inter: inter, hidden: d.hidden, staging: staging)
        let sharedGU = try concatHQ4Bank(arena, loader, [mp + "shared_expert.gate_proj.weight", mp + "shared_expert.up_proj.weight"], k: d.hidden, staging: staging).view
        let sharedDown = try hq4Bank(arena, loader, mp + "shared_expert.down_proj.weight", n: d.hidden, k: sharedInter, staging: staging)
        let gateArr = try loader.tensor(mp + "gate.weight"), sharedGateScalarArr = try loader.tensor(mp + "shared_expert_gate.weight")
        let moe = FNGPUMoE(
            gateAndSharedBF16: try bf16FromF32(arena, gateArr + sharedGateScalarArr),
            gateUp: guView, down: dnView, sharedGateUp: sharedGU, sharedDown: sharedDown,
            sharedGateScalar: try f32(arena, sharedGateScalarArr),
            experts: experts, topK: d.expertsPerTok!, intermediate: inter, sharedIntermediate: sharedInter)
        var ple: FNGPUPLE? = nil
        if prefix == nil && d.pleModuleIndices.contains(index) {
            let pp = p + "ple.", eDim = d.pleEmbedDim!, wDim = d.hcCount! * d.hidden, hDim = d.hidden
            let keyValueBank = try concatHQ4Bank(arena, loader, [pp + "key_proj.weight", pp + "value_proj.weight"], k: eDim, staging: staging).view
            let kBytes = wDim * keyValueBank.rowBytes, vBytes = hDim * keyValueBank.rowBytes
            let keyBank = keyValueBank.subview(blocks: keyValueBank.blocks.slice(byteOffset: 0, count: kBytes / 2, dtype: .f16), n: wDim, k: eDim)!
            let valueBank = keyValueBank.subview(blocks: keyValueBank.blocks.slice(byteOffset: kBytes, count: vBytes / 2, dtype: .f16), n: hDim, k: eDim)!
            let kernel = d.pleConvKernel!, dilation = d.ngramSize!, histCount = (kernel - 1) * dilation * wDim
            let histRing = try arena.activationsRegion().allocate(QWTensorShape([8 * histCount], .f32))
            memset(arena.cpuPointer(of: histRing), 0, histRing.bytes)
            ple = FNGPUPLE(keyValueBank: keyValueBank, keyBank: keyBank, valueBank: valueBank,
                           normKey: try f32(arena, try loader.tensor(pp + "norm_key.weight")), normQuery: try f32(arena, try loader.tensor(pp + "norm_query.weight")),
                           normConv: try f32(arena, try loader.tensor(pp + "norm_conv.weight")), convW: try f32(arena, try loader.tensor(pp + "conv1d.weight")),
                           embedDim: eDim, kernel: kernel, dilation: dilation, eps: d.rmsEps,
                           history: (0..<8).map { s in histRing.slice(byteOffset: s * histCount * 4, count: histCount, dtype: .f32) })
        }
        return FNGPULayerWeights(index: index, mixer: mixer, attnHC: try hcWeights(arena, loader, p + "attn_hyper_connection.", staging: staging),
                                 mlpHC: try hcWeights(arena, loader, p + "mlp_hyper_connection.", staging: staging), moe: moe, ple: ple)
    }
}

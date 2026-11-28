// QWDFlashDrafter+MLA.swift — Matrix-Absorbed MLA and GatedNorm execution
// extension for QWDFlashDrafter (context c_KV/k_pe prep and 8-token block proposal).
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWLoad
import QWQuant

struct DFlashGatedNormWeights {
    let weight: QWTensorValue
    let gateUp: QWTensorValue
    let gateDown: QWTensorValue
}

enum DFlashNormWeights {
    case rms(QWTensorValue)
    case gated(DFlashGatedNormWeights)
}

struct DFlashMLALayerWeights {
    let inputNorm: DFlashNormWeights
    let qANorm: DFlashGatedNormWeights
    let kvANorm: DFlashGatedNormWeights
    let postSelfAttnNorm: DFlashNormWeights?
    let postAttentionNorm: DFlashNormWeights
    let postFeedforwardNorm: DFlashNormWeights?
    let attnConvBase: QWTensorValue?
    let attnConvProj: QWTensorValue?
    let mlpConvBase: QWTensorValue?
    let mlpConvProj: QWTensorValue?
    let qkvGateProj: DFlashLinearWeight
    let kvAProj: DFlashLinearWeight
    let qbProj: DFlashLinearWeight
    let kvBProj: DFlashLinearWeight
    let oProj: DFlashLinearWeight
    let gateUpProj: DFlashLinearWeight
    let downProj: DFlashLinearWeight

    var quantViews: [QWWeightView] {
        [
            qkvGateProj.quantView, kvAProj.quantView, qbProj.quantView,
            kvBProj.quantView, oProj.quantView, gateUpProj.quantView, downProj.quantView,
        ].compactMap { $0 }
    }
}

struct DFlashMLAScratch {
    let qLat: QWTensorValue
    let qFull: QWTensorValue
    let propCKV: QWTensorValue
    let propKPe: QWTensorValue
    let qAbs: QWTensorValue
    let uLat: QWTensorValue
}

struct DFlashMLAPipelines {
    let psoGatedRMSNorm: any MTLComputePipelineState
    let psoGatedSandwich: any MTLComputePipelineState
    let psoKVCtxPrep: any MTLComputePipelineState
    let psoLatentPrep: any MTLComputePipelineState
    let psoQAbsorb: any MTLComputePipelineState
    let psoQAbsorbSub2: any MTLComputePipelineState
    let psoAttn: any MTLComputePipelineState
    let psoProjUVGate: any MTLComputePipelineState
    let psoProjUVGateSub2: any MTLComputePipelineState

    init(hal: QWHal, drafterRotated: Bool, headRotated: Bool) throws {
        self.psoGatedRMSNorm = try hal.pipeline(
            "qwg_dflash_gated_rmsnorm",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
        self.psoGatedSandwich = try hal.pipeline(
            "qwg_dflash_gated_rmsnorm_sandwich",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
        self.psoKVCtxPrep = try hal.pipeline(
            "qwg_dflash_mla_kv_ctx_prep",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
        self.psoLatentPrep = try hal.pipeline(
            "qwg_dflash_mla_latent_prep",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
        self.psoQAbsorb = try hal.pipeline("qwg_dflash_mla_q_absorb")
        self.psoQAbsorbSub2 = try hal.pipeline("qwg_dflash_mla_q_absorb_sub2")
        self.psoAttn = try hal.pipeline("qwg_dflash_mla_attn")
        self.psoProjUVGate = try hal.pipeline(
            "qwg_dflash_mla_proj_uv_gate",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
        self.psoProjUVGateSub2 = try hal.pipeline(
            "qwg_dflash_mla_proj_uv_gate_sub2",
            fcDrafterRotated: drafterRotated, fcHeadRotated: headRotated
        )
    }
}

struct DFlashMLAState {
    let hiddenNorm: DFlashNormWeights
    let finalNorm: DFlashNormWeights
    let layers: [DFlashMLALayerWeights]
    let scratch: DFlashMLAScratch
    let pso: DFlashMLAPipelines
}

private struct DFlashGatedNormMetalParams {
    var eps: Float
    var gatingWeight: Float
    var scale: Float
    var hidden: UInt32
    var rank: UInt32
    var zeroCentered: UInt32
    var mode: UInt32
    var hasNextPreNorm: UInt32
}

private struct DFlashMLALatentMetalParams {
    var eps: Float
    var gatingWeight: Float
    var alphaQ: Float
    var alphaKV: Float
    var ropeMscale: Float
    var qLoraRank: UInt32
    var kvLoraRank: UInt32
    var ropeDim: UInt32
    var gateDim: UInt32
    var normRank: UInt32
    var zeroCentered: UInt32
    var ropeInterleave: UInt32
    var posOffset: UInt32
    var maxCacheSize: UInt32
}

private struct DFlashMLAQBMetalParams {
    var qHeads: UInt32
    var nopeDim: UInt32
    var ropeDim: UInt32
    var vHeadDim: UInt32
    var kvLoraRank: UInt32
    var mRows: UInt32
    var posOffset: UInt32
    var ropeInterleave: UInt32
    var ropeMscale: Float
    var quantKind: UInt32
    var bits: UInt32
    var group: UInt32
    var qkvStride: UInt32
    var gateOffset: UInt32
    var gatedAttention: UInt32
    var pad: UInt32
}

private struct DFlashMLAAttnMetalParams {
    var qHeads: UInt32
    var nopeDim: UInt32
    var ropeDim: UInt32
    var kvLoraRank: UInt32
    var newOffset: UInt32
    var mRows: UInt32
    var maxCacheSize: UInt32
    var isCausal: UInt32
    var softmaxScale: Float
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var pad2: UInt32 = 0
}

extension QWDFlashDrafter {
    static func loadMLALayersAndNorms(
        config: QWDFlashConfig,
        stContainer: QWSafeTensorsContainer?,
        qwpPack: QWQuant.Pack?,
        wSub: QWSubAllocator,
        loadF16: (String, Int) throws -> QWTensorValue,
        loadRawBF16: (String, Int) throws -> QWTensorValue,
        loadLinear: (String, Int, Int) throws -> DFlashLinearWeight,
        loadQ4: (String, Int, Int) throws -> QWWeightView,
        concatQ4: ([(name: String, n: Int)], Int) throws -> (QWWeightView, [QWWeightView])
    ) throws -> (hiddenNorm: DFlashNormWeights, finalNorm: DFlashNormWeights, layers: [DFlashMLALayerWeights]) {
        let H = config.hiddenSize
        let I = config.intermediateSize
        let qHeads = config.numAttentionHeads
        let qLora = config.qLoraRank
        let kvLora = config.kvLoraRank
        let nopeDim = config.qkNopeHeadDim
        let ropeDim = config.qkRopeHeadDim
        let vDim = config.vHeadDim
        let kvADim = kvLora + ropeDim
        let gateDim = config.gatedAttention ? (qHeads * vDim) : 0
        let nRank = config.gatedNormRank
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / max(1, config.convGroupSize))) : 0

        func loadGated(_ prefix: String, dim: Int) throws -> DFlashGatedNormWeights {
            let r = max(1, nRank)
            let w = try loadRawBF16(prefix + ".weight", dim)
            let gu = try loadRawBF16(prefix + ".gate_up_projection.weight", r * dim)
            let gd = try loadRawBF16(prefix + ".gate_down_projection.weight", dim * r)
            return DFlashGatedNormWeights(weight: w, gateUp: gu, gateDown: gd)
        }

        func loadNorm(_ prefix: String, dim: Int) throws -> DFlashNormWeights {
            if nRank > 0 {
                return .gated(try loadGated(prefix, dim: dim))
            }
            return .rms(try loadF16(prefix + ".weight", dim))
        }

        let hNorm = try loadNorm("hidden_norm", dim: H)
        let fNorm = try loadNorm("norm", dim: H)
        var outLayers: [DFlashMLALayerWeights] = []
        outLayers.reserveCapacity(config.numLayers)

        for l in 0..<config.numLayers {
            let p = "layers.\(l)."
            let inNorm = try loadNorm(p + "input_layernorm", dim: H)
            let qaNorm = try loadGated(p + "self_attn.q_a_layernorm", dim: qLora)
            let kvaNorm = try loadGated(p + "self_attn.kv_a_layernorm", dim: kvLora)
            let postSelf = config.useSandwichNorm ? try loadNorm(p + "post_self_attn_layernorm", dim: H) : nil
            let postAttn = try loadNorm(p + "post_attention_layernorm", dim: H)
            let postFFN = config.useSandwichNorm ? try loadNorm(p + "post_feedforward_layernorm", dim: H) : nil

            let acBase = hasConv ? try loadF16(p + "attention_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let acProj = hasConv ? try loadF16(p + "attention_conv.kernel_projection.weight", convProjDim * H) : nil
            let mcBase = hasConv ? try loadF16(p + "mlp_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let mcProj = hasConv ? try loadF16(p + "mlp_conv.kernel_projection.weight", convProjDim * H) : nil

            var qkvSpecs: [(name: String, n: Int)] = [
                (p + "self_attn.q_a_proj.weight", qLora),
                (p + "self_attn.kv_a_proj_with_mqa.weight", kvADim),
            ]
            if gateDim > 0 {
                qkvSpecs.append((p + "self_attn.gate_proj.weight", gateDim))
            }
            let qkvGateW: DFlashLinearWeight
            let kvAW: DFlashLinearWeight
            let qbW: DFlashLinearWeight
            let guW: DFlashLinearWeight
            if let st = stContainer {
                let loaded = try Self.concatSafeTensorsBF16Banks(st: st, wSub: wSub, specs: qkvSpecs, k: H)
                guard case let .bf16(bank) = loaded else {
                    throw QWDFlashError.state("DFlash MLA SafeTensors expected BF16 bank")
                }
                qkvGateW = loaded
                let kvSlice = bank.weights.slice(byteOffset: qLora * H * 2, bytes: kvADim * H * 2)
                kvAW = .bf16(QWGemmFP.BF16Bank(weights: kvSlice, n: kvADim, k: H))
                qbW = try Self.loadDeinterleavedSafeTensorsQBProj(
                    st: st, wSub: wSub, prefix: p,
                    qHeads: qHeads, nopeDim: nopeDim, ropeDim: ropeDim, qLoraRank: qLora
                )
                guW = try Self.concatSafeTensorsBF16Banks(
                    st: st, wSub: wSub,
                    specs: [(p + "mlp.gate_proj.weight", I), (p + "mlp.up_proj.weight", I)],
                    k: H
                )
            } else {
                let (big, subs) = try concatQ4(qkvSpecs, H)
                qkvGateW = .quant(big)
                kvAW = .quant(subs[1])
                if let pack = qwpPack, (try? pack.entry(p + "self_attn.q_b_nope_proj.weight")) != nil {
                    let (qbBig, _) = try concatQ4([
                        (p + "self_attn.q_b_nope_proj.weight", qHeads * nopeDim),
                        (p + "self_attn.q_b_rope_proj.weight", qHeads * ropeDim),
                    ], qLora)
                    qbW = .quant(qbBig)
                } else {
                    qbW = .quant(try loadQ4(p + "self_attn.q_b_proj.weight", qHeads * (nopeDim + ropeDim), qLora))
                }
                let (guBig, _) = try concatQ4([
                    (p + "mlp.gate_proj.weight", I),
                    (p + "mlp.up_proj.weight", I),
                ], H)
                guW = .quant(guBig)
            }
            let kvBW = try loadLinear(p + "self_attn.kv_b_proj.weight", qHeads * (nopeDim + vDim), kvLora)
            let oW = try loadLinear(p + "self_attn.o_proj.weight", H, qHeads * vDim)
            let downW = try loadLinear(p + "mlp.down_proj.weight", H, I)

            outLayers.append(DFlashMLALayerWeights(
                inputNorm: inNorm, qANorm: qaNorm, kvANorm: kvaNorm,
                postSelfAttnNorm: postSelf, postAttentionNorm: postAttn, postFeedforwardNorm: postFFN,
                attnConvBase: acBase, attnConvProj: acProj, mlpConvBase: mcBase, mlpConvProj: mcProj,
                qkvGateProj: qkvGateW, kvAProj: kvAW, qbProj: qbW, kvBProj: kvBW,
                oProj: oW, gateUpProj: guW, downProj: downW
            ))
        }
        return (hNorm, fNorm, outLayers)
    }

    static func allocateMLAScratch(
        config: QWDFlashConfig,
        sSub: QWSubAllocator,
        maxB: Int
    ) throws -> DFlashMLAScratch {
        let qHeads = config.numAttentionHeads
        let qFullDim = qHeads * (config.qkNopeHeadDim + config.qkRopeHeadDim)
        return DFlashMLAScratch(
            qLat: try sSub.allocate(QWTensorShape([maxB * config.qLoraRank], .f16)),
            qFull: try sSub.allocate(QWTensorShape([maxB * qFullDim], .f16)),
            propCKV: try sSub.allocate(QWTensorShape([maxB * config.kvLoraRank], .f16)),
            propKPe: try sSub.allocate(QWTensorShape([maxB * config.qkRopeHeadDim], .f16)),
            qAbs: try sSub.allocate(QWTensorShape([maxB * qHeads * config.kvLoraRank], .f16)),
            uLat: try sSub.allocate(QWTensorShape([maxB * qHeads * config.kvLoraRank], .f16))
        )
    }

    func encodeDFlashNorm(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        norm: DFlashNormWeights,
        y: QWTensorValue,
        yRot: QWTensorValue,
        rows: Int,
        mode: UInt32,
        yRot32: QWTensorValue? = nil,
        mlaPSO: DFlashMLAPipelines
    ) {
        switch norm {
        case .rms(let gamma):
            encodeRMSNorm(enc, x: x, gamma: gamma, y: y, yRot: yRot, rows: rows, mode: mode, yRot32: yRot32)
        case .gated(let gw):
            var p = DFlashGatedNormMetalParams(
                eps: config.rmsNormEps,
                gatingWeight: config.layernormGatingWeight,
                scale: 1.0,
                hidden: UInt32(config.hiddenSize),
                rank: UInt32(config.gatedNormRank),
                zeroCentered: config.gatedNormZeroCentered ? 1 : 0,
                mode: mode,
                hasNextPreNorm: 0
            )
            let tptg = (config.gatedNormRank >= 16 && mlaPSO.psoGatedRMSNorm.maxTotalThreadsPerThreadgroup >= 512)
                ? 512 : 256
            enc.setComputePipelineState(mlaPSO.psoGatedRMSNorm)
            x.bind(enc, index: 0)
            gw.weight.bind(enc, index: 1)
            gw.gateUp.bind(enc, index: 2)
            gw.gateDown.bind(enc, index: 3)
            y.bind(enc, index: 4)
            yRot.bind(enc, index: 5)
            (yRot32 ?? propRot32).bind(enc, index: 6)
            enc.setBytes(&p, length: MemoryLayout<DFlashGatedNormMetalParams>.stride, index: 7)
            enc.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
            )
        }
    }

    private func encodeGatedSandwich(
        _ enc: any MTLComputeCommandEncoder,
        branch16: QWTensorValue,
        postNorm: DFlashGatedNormWeights,
        preNorm: DFlashGatedNormWeights?,
        outPre16: QWTensorValue,
        rotateOut: Bool,
        rows: Int,
        mlaPSO: DFlashMLAPipelines
    ) {
        let pre = preNorm ?? postNorm
        var p = DFlashGatedNormMetalParams(
            eps: config.rmsNormEps,
            gatingWeight: config.layernormGatingWeight,
            scale: 1.0,
            hidden: UInt32(config.hiddenSize),
            rank: UInt32(config.gatedNormRank),
            zeroCentered: config.gatedNormZeroCentered ? 1 : 0,
            mode: rotateOut ? 1 : 0,
            hasNextPreNorm: preNorm != nil ? 1 : 0
        )
        let tptg = (config.gatedNormRank >= 16 && mlaPSO.psoGatedSandwich.maxTotalThreadsPerThreadgroup >= 512)
            ? 512 : 256
        enc.setComputePipelineState(mlaPSO.psoGatedSandwich)
        propX.bind(enc, index: 0)
        branch16.bind(enc, index: 1)
        postNorm.weight.bind(enc, index: 2)
        postNorm.gateUp.bind(enc, index: 3)
        postNorm.gateDown.bind(enc, index: 4)
        pre.weight.bind(enc, index: 5)
        pre.gateUp.bind(enc, index: 6)
        pre.gateDown.bind(enc, index: 7)
        outPre16.bind(enc, index: 8)
        enc.setBytes(&p, length: MemoryLayout<DFlashGatedNormMetalParams>.stride, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
    }

    func encodeMLAContextUpdate(
        _ enc: any MTLComputeCommandEncoder,
        slot: QWDFlashSessionSlot,
        startPosition: Int,
        totalCount: Int,
        mla: DFlashMLAState
    ) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        let posOffset = startPosition + (totalCount - captureRows)
        let rowDrop = totalCount - captureRows
        let rowStrideBytes = config.targetLayerIds.count * config.hiddenSize * 2
        let fcInput: QWTensorValue
        if rowDrop > 0 && targetHidden.bytes >= totalCount * rowStrideBytes {
            fcInput = targetHidden.slice(
                byteOffset: rowDrop * rowStrideBytes,
                count: captureRows * config.targetLayerIds.count * config.hiddenSize,
                dtype: .f16
            )
        } else {
            fcInput = targetHidden
        }

        try encodeLinearMatmul(enc, weight: fc, input: fcInput, output: hCtx, rows: captureRows)
        enc.memoryBarrier(scope: .buffers)
        encodeDFlashNorm(
            enc, x: hCtx, norm: mla.hiddenNorm, y: hCtx, yRot: hCtx,
            rows: captureRows, mode: 1, mlaPSO: mla.pso
        )
        enc.memoryBarrier(scope: .buffers)

        var lp = DFlashMLALatentMetalParams(
            eps: config.rmsNormEps,
            gatingWeight: config.layernormGatingWeight,
            alphaQ: config.alphaQ,
            alphaKV: config.alphaKV,
            ropeMscale: ropeMscale,
            qLoraRank: UInt32(config.qLoraRank),
            kvLoraRank: UInt32(config.kvLoraRank),
            ropeDim: UInt32(config.qkRopeHeadDim),
            gateDim: UInt32(config.gatedAttention ? config.numAttentionHeads * config.vHeadDim : 0),
            normRank: UInt32(max(1, config.gatedNormRank)),
            zeroCentered: config.gatedNormZeroCentered ? 1 : 0,
            ropeInterleave: config.ropeInterleave ? 1 : 0,
            posOffset: UInt32(posOffset),
            maxCacheSize: UInt32(config.maxCacheSize)
        )
        for l in 0..<config.numLayers {
            let lw = mla.layers[l]
            try encodeLinearMatmul(enc, weight: lw.kvAProj, input: hCtx, output: kvCtx, rows: captureRows)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(mla.pso.psoKVCtxPrep)
            kvCtx.bind(enc, index: 0)
            lw.kvANorm.weight.bind(enc, index: 1)
            lw.kvANorm.gateUp.bind(enc, index: 2)
            lw.kvANorm.gateDown.bind(enc, index: 3)
            invFreq.bind(enc, index: 4)
            slot.kLayers[l].bind(enc, index: 5)
            slot.vLayers[l].bind(enc, index: 6)
            enc.setBytes(&lp, length: MemoryLayout<DFlashMLALatentMetalParams>.stride, index: 7)
            enc.dispatchThreadgroups(
                MTLSize(width: captureRows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.memoryBarrier(scope: .buffers)
        }
    }

    private static func kvBQuantSpec(
        _ w: DFlashLinearWeight
    ) -> (kind: UInt32, bits: UInt32, group: UInt32, sub2: Bool, slice: QWFPBufferSlice) {
        switch w {
        case .bf16(let bank):
            return (3, 16, 1, false, bank.weights)
        case .quant(let view):
            let slice = QWFPBufferSlice(view.blocks)
            if view.isSHVQ {
                let isSub2 = (view.bits == 1 || view.bits == 15)
                return (2, UInt32(view.bits), UInt32(view.groupSize), isSub2, slice)
            } else if view.isIncumbentContainer {
                return (1, UInt32(view.bits), UInt32(view.groupSize), false, slice)
            } else {
                return (0, UInt32(view.bits), UInt32(view.groupSize), false, slice)
            }
        }
    }

    func encodeMLAProposeLayersAndFinalNorm(
        _ enc: any MTLComputeCommandEncoder,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        rows M: Int,
        draftCount D: Int,
        propXDraftSlice: QWTensorValue,
        propNormSlice: QWTensorValue,
        propRotSlice: QWTensorValue,
        propRot32Slice: QWTensorValue,
        mla: DFlashMLAState
    ) throws {
        let H = config.hiddenSize
        let qHeads = config.numAttentionHeads
        let qLora = config.qLoraRank
        let kvLora = config.kvLoraRank
        let nopeDim = config.qkNopeHeadDim
        let ropeDim = config.qkRopeHeadDim
        let vDim = config.vHeadDim
        let gateDim = config.gatedAttention ? (qHeads * vDim) : 0
        let qkvStride = qLora + kvLora + ropeDim + gateDim
        let gateOffset = qLora + kvLora + ropeDim
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / max(1, config.convGroupSize))) : 0
        let sc = mla.scratch

        if config.numLayers > 0 {
            encodeDFlashNorm(
                enc, x: propX, norm: mla.layers[0].inputNorm,
                y: propNorm, yRot: propRot, rows: M,
                mode: hasConv ? 0 : 2, mlaPSO: mla.pso
            )
        }

        var lp = DFlashMLALatentMetalParams(
            eps: config.rmsNormEps,
            gatingWeight: config.layernormGatingWeight,
            alphaQ: config.alphaQ,
            alphaKV: config.alphaKV,
            ropeMscale: ropeMscale,
            qLoraRank: UInt32(qLora),
            kvLoraRank: UInt32(kvLora),
            ropeDim: UInt32(ropeDim),
            gateDim: UInt32(gateDim),
            normRank: UInt32(max(1, config.gatedNormRank)),
            zeroCentered: config.gatedNormZeroCentered ? 1 : 0,
            ropeInterleave: config.ropeInterleave ? 1 : 0,
            posOffset: UInt32(cacheOffset),
            maxCacheSize: UInt32(config.maxCacheSize)
        )
        var ap = DFlashMLAAttnMetalParams(
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            kvLoraRank: UInt32(kvLora),
            newOffset: UInt32(cacheOffset),
            mRows: UInt32(M),
            maxCacheSize: UInt32(config.maxCacheSize),
            isCausal: config.draftBlockCausal ? 1 : 0,
            softmaxScale: config.softmaxScale
        )

        for l in 0..<config.numLayers {
            let lw = mla.layers[l]
            if let acProj = lw.attnConvProj, let acBase = lw.attnConvBase, let pDyn = propDyn {
                encodeF16GemvBatched(enc, w: acProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propRot, rows: M, mode: 0)
            }

            try encodeLinearMatmul(enc, weight: lw.qkvGateProj, input: propRot, output: qkvProp, rows: M)

            enc.setComputePipelineState(mla.pso.psoLatentPrep)
            qkvProp.bind(enc, index: 0)
            lw.qANorm.weight.bind(enc, index: 1)
            lw.qANorm.gateUp.bind(enc, index: 2)
            lw.qANorm.gateDown.bind(enc, index: 3)
            lw.kvANorm.weight.bind(enc, index: 4)
            lw.kvANorm.gateUp.bind(enc, index: 5)
            lw.kvANorm.gateDown.bind(enc, index: 6)
            invFreq.bind(enc, index: 7)
            sc.qLat.bind(enc, index: 8)
            sc.propCKV.bind(enc, index: 9)
            sc.propKPe.bind(enc, index: 10)
            enc.setBytes(&lp, length: MemoryLayout<DFlashMLALatentMetalParams>.stride, index: 11)
            enc.dispatchThreadgroups(
                MTLSize(width: M, height: 2, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )

            try encodeLinearMatmul(enc, weight: lw.qbProj, input: sc.qLat, output: sc.qFull, rows: M)

            let kvBSpec = Self.kvBQuantSpec(lw.kvBProj)
            var qbp = DFlashMLAQBMetalParams(
                qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim), ropeDim: UInt32(ropeDim),
                vHeadDim: UInt32(vDim), kvLoraRank: UInt32(kvLora), mRows: UInt32(M),
                posOffset: UInt32(cacheOffset), ropeInterleave: config.ropeInterleave ? 1 : 0,
                ropeMscale: ropeMscale, quantKind: kvBSpec.kind, bits: kvBSpec.bits, group: kvBSpec.group,
                qkvStride: UInt32(qkvStride), gateOffset: UInt32(gateOffset),
                gatedAttention: config.gatedAttention ? 1 : 0, pad: 0
            )
            let qAbsPSO = kvBSpec.sub2 ? mla.pso.psoQAbsorbSub2 : mla.pso.psoQAbsorb
            let qAbsTptg = (qAbsPSO.maxTotalThreadsPerThreadgroup >= 128) ? 128 : 32
            let qAbsColsPerTG = (qAbsTptg / 32) * 128
            enc.setComputePipelineState(qAbsPSO)
            sc.qFull.bind(enc, index: 0)
            enc.setBuffer(kvBSpec.slice.buffer, offset: kvBSpec.slice.offset, index: 1)
            invFreq.bind(enc, index: 2)
            sc.qAbs.bind(enc, index: 3)
            enc.setBytes(&qbp, length: MemoryLayout<DFlashMLAQBMetalParams>.stride, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: (kvLora + qAbsColsPerTG - 1) / qAbsColsPerTG, height: qHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: qAbsTptg, height: 1, depth: 1)
            )

            let attnTptg = (mla.pso.psoAttn.maxTotalThreadsPerThreadgroup >= 256) ? 256 : 32
            enc.setComputePipelineState(mla.pso.psoAttn)
            sc.qAbs.bind(enc, index: 0)
            sc.qFull.bind(enc, index: 1)
            slot.kLayers[l].bind(enc, index: 2)
            slot.vLayers[l].bind(enc, index: 3)
            sc.propCKV.bind(enc, index: 4)
            sc.propKPe.bind(enc, index: 5)
            sc.uLat.bind(enc, index: 6)
            enc.setBytes(&ap, length: MemoryLayout<DFlashMLAAttnMetalParams>.stride, index: 7)
            enc.dispatchThreadgroups(
                MTLSize(width: qHeads, height: M, depth: 1),
                threadsPerThreadgroup: MTLSize(width: attnTptg, height: 1, depth: 1)
            )

            enc.setComputePipelineState(kvBSpec.sub2 ? mla.pso.psoProjUVGateSub2 : mla.pso.psoProjUVGate)
            sc.uLat.bind(enc, index: 0)
            enc.setBuffer(kvBSpec.slice.buffer, offset: kvBSpec.slice.offset, index: 1)
            qkvProp.bind(enc, index: 2)
            attnOut.bind(enc, index: 3)
            enc.setBytes(&qbp, length: MemoryLayout<DFlashMLAQBMetalParams>.stride, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: vDim / 32, height: qHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )

            try encodeLinearMatmul(enc, weight: lw.oProj, input: attnOut, output: propDown, rows: M)

            if case let .gated(postSelf)? = lw.postSelfAttnNorm,
               case let .gated(postAttn) = lw.postAttentionNorm {
                let branch: QWTensorValue
                if let acBase = lw.attnConvBase, let pDyn = propDyn {
                    encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: acBase, outOrResidual: propNorm, rows: M, mode: 2)
                    branch = propNorm
                } else {
                    branch = propDown
                }
                encodeGatedSandwich(
                    enc, branch16: branch, postNorm: postSelf, preNorm: postAttn,
                    outPre16: hasConv ? propNorm : propRot, rotateOut: !hasConv,
                    rows: M, mlaPSO: mla.pso
                )
            } else {
                if let acBase = lw.attnConvBase, let pDyn = propDyn {
                    encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: acBase, outOrResidual: propX, rows: M, mode: 1)
                }
                encodeDFlashNorm(
                    enc, x: propX, norm: lw.postAttentionNorm,
                    y: propNorm, yRot: propRot, rows: M,
                    mode: hasConv ? 0 : 2, mlaPSO: mla.pso
                )
            }

            if let mcProj = lw.mlpConvProj, let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeF16GemvBatched(enc, w: mcProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: mcBase, outOrResidual: propRot, rows: M, mode: 0)
            }

            try encodeLinearGateUpMatmul(enc, weight: lw.gateUpProj, input: propRot, output: mlpInter, rows: M)
            try encodeLinearMatmul(enc, weight: lw.downProj, input: mlpInter, output: propDown, rows: M)

            let isLast = (l == config.numLayers - 1)
            if case let .gated(postFFN)? = lw.postFeedforwardNorm {
                let branch: QWTensorValue
                if let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                    encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: mcBase, outOrResidual: propNorm, rows: M, mode: 2)
                    branch = propNorm
                } else {
                    branch = propDown
                }
                let nextPre: DFlashGatedNormWeights?
                if !isLast, case let .gated(nPre) = mla.layers[l + 1].inputNorm {
                    nextPre = nPre
                } else {
                    nextPre = nil
                }
                encodeGatedSandwich(
                    enc, branch16: branch, postNorm: postFFN, preNorm: nextPre,
                    outPre16: hasConv ? propNorm : propRot, rotateOut: !hasConv,
                    rows: M, mlaPSO: mla.pso
                )
            } else if let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: mcBase, outOrResidual: propX, rows: M, mode: 1)
                if !isLast {
                    encodeDFlashNorm(
                        enc, x: propX, norm: mla.layers[l + 1].inputNorm,
                        y: propNorm, yRot: propRot, rows: M,
                        mode: hasConv ? 0 : 2, mlaPSO: mla.pso
                    )
                }
            }
        }

        encodeDFlashNorm(
            enc, x: propXDraftSlice, norm: mla.finalNorm,
            y: propNormSlice, yRot: propRotSlice, rows: D,
            mode: 5, yRot32: propRot32Slice, mlaPSO: mla.pso
        )
    }
}

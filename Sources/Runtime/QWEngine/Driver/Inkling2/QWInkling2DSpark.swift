// QWInkling2DSpark.swift — Unified BF16/SHVQ/QWP1 DSpark speculative drafter state & pipelines.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWLoad
import QWQuant

final class QWInkling2DSpark: @unchecked Sendable {
    struct Slice: @unchecked Sendable {
        let buffer: any MTLBuffer
        let offset: Int
    }

    enum FusedQKVW: @unchecked Sendable {
        case fused(Ink2LinearW)
        case split(q: Ink2LinearW, k: Ink2LinearW, v: Ink2LinearW)
    }

    enum FusedKVCtxW: @unchecked Sendable {
        case fused(Ink2LinearW)
        case split(k: Ink2LinearW, v: Ink2LinearW)
    }

    struct DraftLayerW: @unchecked Sendable {
        let dflareNorm: Slice
        let inputNorm: Slice
        let postAttnNorm: Slice
        let qNorm: Slice
        let kNorm: Slice
        let kNormCtx: Slice
        let qkvW: FusedQKVW
        let oW: Ink2LinearW
        let kvCtxW: FusedKVCtxW
        let gateUpW: Ink2LinearW
        let downW: Ink2LinearW
        let isSliding: Bool
        let ringCap: Int
        let usesRingBuffer: Bool
        let effectiveWindow: Int
        let kCache: any MTLBuffer
        let vCache: any MTLBuffer
    }

    struct MixP { var T, H, tapPlaneStride: UInt32; var eps, a0, a1, a2, a3: Float }
    struct MixMapP {
        var T, H, tapPlaneStride, useRowMap: UInt32
        var eps, a0, a1, a2, a3: Float
        var rowMap: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                     UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct CtxKVP { var T, nKV, headDim, isSliding, posOff, ringCap: UInt32; var eps, ropeTheta: Float }
    struct AddNormP { var rows, dim, hasRes, writeRes: UInt32; var eps: Float }
    struct PropQKRP { var B, nH, nKV, headDim, posOff: UInt32; var eps, ropeTheta: Float }
    struct PropQKRStridedP { var B, nH, nKV, headDim, posOff, rowStride: UInt32; var eps, ropeTheta: Float }
    struct AttnP { var B, nH, nKV, headDim, ctxStart, ctxEnd, sliding, ringCap: UInt32; var scale: Float }
    struct AttnBatchP {
        var B, nH, nKV, headDim, ctxStart, ctxEnd, sliding, ringCap, qStride, kvPropStride: UInt32
        var scale: Float
    }
    struct AttnSplitP {
        var B, nH, nKV, headDim, ctxStart, ctxEnd, sliding, ringCap, qStride, kvPropStride: UInt32
        var qRowOffset, activeQRows, splits: UInt32
        var scale: Float
    }
    struct AttnCombineP { var nH, headDim, qRowOffset, activeQRows, splits: UInt32 }
    struct MarkovCtxP { var count, H, projRows, rank: UInt32 }
    struct MarkovStepP { var stepIdx, H, projRows, rank, resetState: UInt32 }
    struct MarkovStepBatchP {
        var P, H, projRows, rank, ctxStepIdx, resetState: UInt32
        var parentStateSlots: (UInt32, UInt32, UInt32, UInt32)
        var prevTokNodes: (UInt32, UInt32, UInt32, UInt32)
    }
    struct LogitsP { var stepIdx, V, rank, partCount: UInt32; var softcap: Float }
    struct TopKP {
        var stepIdx, V, rank, partCount, numParents: UInt32
        var softcap: Float
        var keepK: (UInt32, UInt32, UInt32, UInt32)
        var outNodeBase: (UInt32, UInt32, UInt32, UInt32)
    }
    struct SConvVerifyCommitP {
        var numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride: UInt32
    }
    struct SConvVerifyCommitTreeP {
        var numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride: UInt32
        var keptNodes: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                        UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct TreeKVCompactP {
        var oldPos, kept, nKV, headDim, ringCap, kvMode, isSliding, pad: UInt32
        var keptNodes: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                        UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct KVSlice: Sendable { let isSliding: Bool; let startPos: Int; let count: Int; let kData: Data; let vData: Data }

    let hal: QWHal
    let config: QWModelDescriptor.Inkling2DSparkConfig
    let ink2: QWModelDescriptor.Inkling2Config
    let capacity: Int
    let fullVocab: Int
    let embedRmsEps: Float
    let lmHeadStorage: QWHeadStorage
    let lmHeadBank: QWGemmFP.BF16Bank?
    let embedTokens: Ink2Slice
    let embedNorm: Ink2Slice
    let foldWIsAQ8: Bool
    let treeMode: DSparkTreeMode

    let weightsBuf: any MTLBuffer
    let foldWBuf: any MTLBuffer
    let foldBBuf: any MTLBuffer
    let layers: [DraftLayerW]
    let maskEmbed: Slice
    let finalNorm: Slice
    let markovW1: Slice
    let rnnProjW: Slice
    let rnnProjB: Slice
    let markovDFlareNorm: Slice
    let alphas: [(Float, Float, Float, Float)]

    let dummyB: any MTLBuffer
    let dummyF: any MTLBuffer
    let tapBuf: any MTLBuffer
    let lastTapBuf: any MTLBuffer
    let ctxMixBuf: any MTLBuffer
    let ctxKVProjBuf: any MTLBuffer
    let propXBuf: any MTLBuffer
    let propResBuf: any MTLBuffer
    let propNormBuf: any MTLBuffer
    let propQKVBuf: any MTLBuffer
    let propAttnBuf: any MTLBuffer
    let propAttnPartOBuf: any MTLBuffer
    let propAttnPartMLBuf: any MTLBuffer
    let propGUBuf: any MTLBuffer
    let propActBuf: any MTLBuffer
    let propGUSlice: QWFPBufferSlice
    let propActSlice: QWFPBufferSlice
    let rotF16Buf: any MTLBuffer
    let f32OutBuf: any MTLBuffer
    let rotF16TV: QWTensorValue
    let f32OutTV: QWTensorValue
    let f32OutElems: Int
    let baseLogitsBF16Buf: any MTLBuffer
    let baseLogitsF16InBuf: any MTLBuffer
    let baseLogitsF32Buf: any MTLBuffer
    let baseLogitsF16InTV: QWTensorValue
    let baseLogitsF32TV: QWTensorValue
    let fRnnBuf: any MTLBuffer
    let markovCtxF32Buf: any MTLBuffer
    let markovStateBF16Buf: any MTLBuffer
    let markovLatentBF16Buf: any MTLBuffer
    let markovStateBatchBufA: any MTLBuffer
    let markovStateBatchBufB: any MTLBuffer
    let markovLatentBatchBuf: any MTLBuffer
    let draftTokBuf: any MTLBuffer
    let treeTokBuf: any MTLBuffer
    let argmaxPartValBuf: any MTLBuffer
    let argmaxPartIdxBuf: any MTLBuffer
    let topkPartValBuf: any MTLBuffer
    let topkPartIdxBuf: any MTLBuffer

    let pEmbed: any MTLComputePipelineState
    let pCopyBF16: any MTLComputePipelineState
    let pBF16ToF16FWHT32: any MTLComputePipelineState
    let pF32ToBF16: any MTLComputePipelineState
    let pSConvVerifyCommit: any MTLComputePipelineState
    let pSConvVerifyCommitTree: any MTLComputePipelineState
    let pTreeKVCompact: any MTLComputePipelineState
    let pMixNorm: any MTLComputePipelineState
    let pMixNormMap: any MTLComputePipelineState
    let pCtxKV: any MTLComputePipelineState
    let pCtxKVFused: any MTLComputePipelineState
    let pAddNorm: any MTLComputePipelineState
    let pPropQKR: any MTLComputePipelineState
    let pPropQKRStrided: any MTLComputePipelineState
    let pAttnGQA: any MTLComputePipelineState
    let pAttnGQABatch: any MTLComputePipelineState
    let pAttnGQASplit: any MTLComputePipelineState
    let pAttnGQACombine: any MTLComputePipelineState
    let pMarkovCtx: any MTLComputePipelineState
    let pMarkovStep: any MTLComputePipelineState
    let pMarkovStepBatch: any MTLComputePipelineState
    let pLogitsStage1: any MTLComputePipelineState
    let pLogitsAQ8Stage1: any MTLComputePipelineState
    let pArgmaxFinal: any MTLComputePipelineState
    let pTopKStage1: any MTLComputePipelineState
    let pTopKAQ8Stage1: any MTLComputePipelineState
    let pTopKFinal: any MTLComputePipelineState

    var ctxLen: Int = 0
    let totalBytes: Int
    let residentBuffers: [any MTLBuffer]

    static let maxSplitKV = 16

    init(
        hal: QWHal,
        mtpURL: URL,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config,
        config: QWModelDescriptor.Inkling2DSparkConfig,
        weights: QWInkling2Weights,
        capacity: Int,
        maxPrefillChunk: Int = QWInkling2Session.maxPrefillChunk
    ) throws {
        switch weights.lmHeadStorage {
        case .bf16(let tv):
            guard tv.region.buffer.storageMode == .shared else {
                throw QWInkling2Error.badState("DSpark requires shared storage for lm_head fold")
            }
        case .q4g(let hw):
            guard hw.view.blocks.region.buffer.storageMode == .shared else {
                throw QWInkling2Error.badState("DSpark requires shared storage for q4g lm_head fold")
            }
        case .f16:
            throw QWInkling2Error.badState("unsupported .f16 lmHeadStorage for DSpark")
        }
        guard config.targetLayerIds.count == 4, config.numLayers >= 1, config.blockSize >= 2 else {
            throw QWInkling2Error.badState("unsupported DSpark mtp_config geometry")
        }
        self.hal = hal
        self.config = config
        self.ink2 = ink2
        self.capacity = capacity
        self.fullVocab = descriptor.vocab
        self.embedRmsEps = descriptor.rmsEps
        self.lmHeadStorage = weights.lmHeadStorage
        self.lmHeadBank = weights.lmHeadBank
        self.embedTokens = weights.embedTokens
        self.embedNorm = weights.embedNorm
        self.treeMode = DSparkTreeMode.resolveFromEnvironment()

        let device = hal.device
        let src: DSparkSource = mtpURL.pathExtension.lowercased() == "qwp"
            ? .qwp(try QWQuant.Pack(url: mtpURL, descriptor: descriptor, skipIdentityCheck: true))
            : .safetensors(try QWSafeTensorsContainer(url: mtpURL))
        func align256(_ n: Int) -> Int { max(256, ((n + 255) / 256) * 256) }
        func alloc(_ bytes: Int, label: String) throws -> any MTLBuffer {
            let len = align256(bytes)
            guard let b = device.makeBuffer(length: len, options: .storageModeShared) else {
                throw QWInkling2Error.allocationFailed("dspark \(label) \(len) bytes")
            }
            return b
        }

        let H = config.hiddenSize
        let R = config.markovRank
        let nH = config.qHeads
        let nKV = config.kvHeads
        let hd = config.headDim
        let qDim = nH * hd
        let kvDim = nKV * hd
        let mlpI = config.intermediateSize
        let rnnProjKey = "markov_head.rnn_proj.weight"
        let rnnProjBF16Bytes = 3 * R * (2 * R + 2 * H) * 2

        let orderedKeys: [String] = {
            var k = ["mask_embed", "norm.weight", "markov_head.markov_w1.weight",
                     rnnProjKey, "markov_head.rnn_proj.bias", "markov_head.dflare_norm.weight"]
            for l in 0..<config.numLayers {
                let p = "layers.\(l)."
                k.append(contentsOf: [
                    p + "dflare_norm.weight", p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                    p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight", p + "self_attn.k_norm_ctx.weight",
                    p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
                    p + "self_attn.o_proj.weight", p + "self_attn.k_proj_ctx.weight", p + "self_attn.v_proj_ctx.weight",
                    p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"
                ])
            }
            return k
        }()

        var rawSizes: [String: Int] = [:]
        var kinds: [String: QWWeightView.ContainerKind?] = [:]
        for name in orderedKeys {
            rawSizes[name] = (name == rnnProjKey) ? rnnProjBF16Bytes : (try src.byteLength(of: name))
            kinds[name] = try src.containerKind(of: name)
        }
        func sameKind(_ a: String, _ b: String) -> Bool {
            kinds[a].flatMap { $0 } == kinds[b].flatMap { $0 }
        }
        func isRaw(_ name: String) -> Bool { (try? src.isRawBF16(of: name)) ?? true }
        func pairTight(_ prev: String, _ cur: String) -> Bool {
            if isRaw(prev) && isRaw(cur) { return true }
            return !isRaw(prev) && !isRaw(cur) && sameKind(prev, cur)
        }

        var offsets: [String: Int] = [:]
        var cursor = 0
        for (idx, name) in orderedKeys.enumerated() {
            let byteLen = rawSizes[name]!
            let prev = idx > 0 ? orderedKeys[idx - 1] : ""
            let isTightQKV = (name.hasSuffix("self_attn.k_proj.weight") && prev.hasSuffix("self_attn.q_proj.weight"))
                || (name.hasSuffix("self_attn.v_proj.weight") && prev.hasSuffix("self_attn.k_proj.weight"))
            let isTightKVCtx = name.hasSuffix("self_attn.v_proj_ctx.weight") && prev.hasSuffix("self_attn.k_proj_ctx.weight")
            let isTightGU = name.hasSuffix("mlp.up_proj.weight") && prev.hasSuffix("mlp.gate_proj.weight")
            if (isTightQKV || isTightKVCtx || isTightGU) && pairTight(prev, name) {
                offsets[name] = cursor
                cursor += byteLen
            } else {
                cursor = (cursor + 255) & ~255
                offsets[name] = cursor
                cursor += byteLen
            }
        }
        let wBuf = try alloc(cursor, label: "weights")
        let wBufPtr = wBuf.contents()
        try withExtendedLifetime(wBuf) {
            for name in orderedKeys {
                guard let off = offsets[name] else { continue }
                let dstPtr = wBufPtr.advanced(by: off)
                if name == rnnProjKey && !isRaw(rnnProjKey) {
                    let f32 = try Self.dequantizeMatrixToF32(src: src, name: rnnProjKey, nRows: 3 * R, kCols: 2 * R + 2 * H)
                    let dst = dstPtr.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<f32.count { dst[i] = Self.f32ToBF16(f32[i]) }
                } else {
                    try src.withBytes(of: name) { bytes in
                        guard let base = bytes.baseAddress else { return }
                        dstPtr.copyMemory(from: base, byteCount: bytes.count)
                    }
                }
            }
        }
        self.weightsBuf = wBuf
        func slice(_ name: String) -> Slice { Slice(buffer: wBuf, offset: offsets[name]!) }
        func singleW(_ name: String, n: Int, k: Int) throws -> Ink2LinearW {
            try Self.linearWeight(buffer: wBuf, offset: offsets[name]!, byteLength: rawSizes[name]!,
                                  n: n, k: k, kind: kinds[name] ?? nil, name: name)
        }
        self.maskEmbed = slice("mask_embed")
        self.finalNorm = slice("norm.weight")
        self.markovW1 = slice("markov_head.markov_w1.weight")
        self.rnnProjW = slice(rnnProjKey)
        self.rnnProjB = slice("markov_head.rnn_proj.bias")
        self.markovDFlareNorm = slice("markov_head.dflare_norm.weight")
        self.alphas = try Self.computeAlphas(src: src, rows: config.numLayers + 1)

        let V = ink2.unpaddedVocabSize
        let useAQ8 = (R >= 32) && (R % 32 == 0)
        self.foldWIsAQ8 = useAQ8
        let foldWBytes = useAQ8 ? (V * (R / 32) * 36) : (V * R * 2)
        let fWBuf = try alloc(foldWBytes, label: "foldW")
        let fBBuf = try alloc(V * 2, label: "foldB")
        try Self.foldMarkovHead(
            src: src, lmHeadStorage: weights.lmHeadStorage, V: V, H: H, R: R,
            foldWIsAQ8: useAQ8, dstW: fWBuf, dstB: fBBuf
        )
        self.foldWBuf = fWBuf
        self.foldBBuf = fBBuf

        let B = config.blockSize
        let draftCount = B - 1
        let maxT = min(maxPrefillChunk, max(512, capacity))
        let localSet = Set(config.localLayerIds)

        var builtLayers: [DraftLayerW] = []
        builtLayers.reserveCapacity(config.numLayers)
        for l in 0..<config.numLayers {
            let p = "layers.\(l)."
            let isSliding = localSet.contains(l)
            let ringCap = isSliding
                ? min(capacity, config.slidingWindowSize + maxT)
                : Self.globalKVCapacity(for: capacity)
            let usesRingBuffer = isSliding || capacity > ringCap
            let effectiveWindow = isSliding ? config.slidingWindowSize : ringCap
            let rowBytes = kvDim * 2
            let kc = try alloc(ringCap * rowBytes, label: "kvK[\(l)]")
            let vc = try alloc(ringCap * rowBytes, label: "kvV[\(l)]")
            let qN = p + "self_attn.q_proj.weight", kN = p + "self_attn.k_proj.weight", vN = p + "self_attn.v_proj.weight"
            let kcN = p + "self_attn.k_proj_ctx.weight", vcN = p + "self_attn.v_proj_ctx.weight"
            let gN = p + "mlp.gate_proj.weight", uN = p + "mlp.up_proj.weight"

            let qkvW: FusedQKVW
            if offsets[kN]! == offsets[qN]! + rawSizes[qN]! && offsets[vN]! == offsets[kN]! + rawSizes[kN]! {
                let fused = try Self.linearWeight(
                    buffer: wBuf, offset: offsets[qN]!, byteLength: rawSizes[qN]! + rawSizes[kN]! + rawSizes[vN]!,
                    n: qDim + 2 * kvDim, k: H, kind: kinds[qN] ?? nil, name: p + "qkv"
                )
                qkvW = .fused(fused)
            } else {
                qkvW = .split(q: try singleW(qN, n: qDim, k: H),
                              k: try singleW(kN, n: kvDim, k: H),
                              v: try singleW(vN, n: kvDim, k: H))
            }
            let kvCtxW: FusedKVCtxW
            if offsets[vcN]! == offsets[kcN]! + rawSizes[kcN]! {
                let fused = try Self.linearWeight(
                    buffer: wBuf, offset: offsets[kcN]!, byteLength: rawSizes[kcN]! + rawSizes[vcN]!,
                    n: 2 * kvDim, k: H, kind: kinds[kcN] ?? nil, name: p + "kvCtx"
                )
                kvCtxW = .fused(fused)
            } else {
                kvCtxW = .split(k: try singleW(kcN, n: kvDim, k: H), v: try singleW(vcN, n: kvDim, k: H))
            }
            guard offsets[uN]! == offsets[gN]! + rawSizes[gN]! else {
                throw QWInkling2Error.badState("mismatched DSpark gate/up containers at layer \(l)")
            }
            let guW = try Self.linearWeight(
                buffer: wBuf, offset: offsets[gN]!, byteLength: rawSizes[gN]! + rawSizes[uN]!,
                n: 2 * mlpI, k: H, kind: kinds[gN] ?? nil, name: p + "gateUp"
            )
            builtLayers.append(DraftLayerW(
                dflareNorm: slice(p + "dflare_norm.weight"),
                inputNorm: slice(p + "input_layernorm.weight"),
                postAttnNorm: slice(p + "post_attention_layernorm.weight"),
                qNorm: slice(p + "self_attn.q_norm.weight"),
                kNorm: slice(p + "self_attn.k_norm.weight"),
                kNormCtx: slice(p + "self_attn.k_norm_ctx.weight"),
                qkvW: qkvW,
                oW: try singleW(p + "self_attn.o_proj.weight", n: H, k: qDim),
                kvCtxW: kvCtxW,
                gateUpW: guW,
                downW: try singleW(p + "mlp.down_proj.weight", n: H, k: mlpI),
                isSliding: isSliding, ringCap: ringCap,
                usesRingBuffer: usesRingBuffer, effectiveWindow: effectiveWindow,
                kCache: kc, vCache: vc
            ))
        }
        self.layers = builtLayers

        self.dummyB = try alloc(256, label: "dummyB")
        self.dummyF = try alloc(256, label: "dummyF")
        self.tapBuf = try alloc(4 * maxT * H * 2, label: "tapBuf")
        self.lastTapBuf = try alloc(4 * H * 2, label: "lastTapBuf")
        self.ctxMixBuf = try alloc(maxT * H * 2, label: "ctxMixBuf")
        self.ctxKVProjBuf = try alloc(maxT * 2 * kvDim * 2, label: "ctxKVProj")
        self.propXBuf = try alloc(B * H * 2, label: "propX")
        self.propResBuf = try alloc(B * H * 2, label: "propRes")
        self.propNormBuf = try alloc(B * H * 2, label: "propNorm")
        self.propQKVBuf = try alloc(B * (qDim + 2 * kvDim) * 2, label: "propQKV")
        self.propAttnBuf = try alloc(B * qDim * 2, label: "propAttn")
        self.propAttnPartOBuf = try alloc(B * nH * Self.maxSplitKV * hd * 4, label: "propAttnPartO")
        self.propAttnPartMLBuf = try alloc(B * nH * Self.maxSplitKV * 8, label: "propAttnPartML")
        let guBuf = try alloc(B * 2 * mlpI * 2, label: "propGU")
        let actBuf = try alloc(B * mlpI * 2, label: "propAct")
        self.propGUBuf = guBuf
        self.propActBuf = actBuf
        self.propGUSlice = QWFPBufferSlice(buffer: guBuf, offset: 0, bytes: guBuf.length)
        self.propActSlice = QWFPBufferSlice(buffer: actBuf, offset: 0, bytes: actBuf.length)

        let rotCount = max(maxT * H, B * max(H, max(qDim, mlpI)))
        let r16Buf = try alloc(rotCount * 2, label: "rotF16")
        let f32Count = max(maxT * 2 * kvDim, B * max(qDim + 2 * kvDim, 2 * mlpI))
        let f32Out = try alloc(f32Count * 4, label: "f32Out")
        self.rotF16Buf = r16Buf
        self.f32OutBuf = f32Out
        self.f32OutElems = f32Count
        self.rotF16TV = QWTensorValue(region: QWMRegion(buffer: r16Buf, base: 0, length: r16Buf.length),
                                      offset: 0, shape: QWTensorShape([rotCount], .f16))
        self.f32OutTV = QWTensorValue(region: QWMRegion(buffer: f32Out, base: 0, length: f32Out.length),
                                      offset: 0, shape: QWTensorShape([f32Count], .f32))

        self.baseLogitsBF16Buf = try alloc(draftCount * V * 2, label: "baseLogits")
        let isQuantHead = if case .q4g = weights.lmHeadStorage { true } else { false }
        let f16InCount = isQuantHead ? (B * H) : 128
        let f32LogitsCount = isQuantHead ? (draftCount * V) : 64
        let f16InBuf = try alloc(f16InCount * 2, label: "baseLogitsF16In")
        let f32Buf = try alloc(f32LogitsCount * 4, label: "baseLogitsF32")
        self.baseLogitsF16InBuf = f16InBuf
        self.baseLogitsF32Buf = f32Buf
        self.baseLogitsF16InTV = QWTensorValue(region: QWMRegion(buffer: f16InBuf, base: 0, length: f16InBuf.length),
                                               offset: 0, shape: QWTensorShape([f16InCount], .f16))
        self.baseLogitsF32TV = QWTensorValue(region: QWMRegion(buffer: f32Buf, base: 0, length: f32Buf.length),
                                             offset: 0, shape: QWTensorShape([f32LogitsCount], .f32))
        self.fRnnBuf = try alloc(H * 2, label: "fRnn")
        self.markovCtxF32Buf = try alloc(draftCount * 3 * R * 4, label: "markovCtx")
        self.markovStateBF16Buf = try alloc(R * 2, label: "markovState")
        self.markovLatentBF16Buf = try alloc(R * 2, label: "markovLatent")
        self.markovStateBatchBufA = try alloc(4 * R * 2, label: "markovStA")
        self.markovStateBatchBufB = try alloc(4 * R * 2, label: "markovStB")
        self.markovLatentBatchBuf = try alloc(4 * R * 2, label: "markovLatB")
        self.draftTokBuf = try alloc(B * 4, label: "draftTok")
        self.treeTokBuf = try alloc(16 * 4, label: "treeTok")
        self.argmaxPartValBuf = try alloc(1024 * 4, label: "argmaxV")
        self.argmaxPartIdxBuf = try alloc(1024 * 4, label: "argmaxI")
        self.topkPartValBuf = try alloc(1024 * 12 * 4, label: "topkV")
        self.topkPartIdxBuf = try alloc(1024 * 12 * 4, label: "topkI")

        self.pEmbed = try hal.pipeline("qwg_ink2_embed_norm")
        self.pCopyBF16 = try hal.pipeline("qwg_ink2_copy_bf16")
        self.pBF16ToF16FWHT32 = try hal.pipeline("qwg_ink2_bf16_to_f16_fwht32")
        self.pF32ToBF16 = try hal.pipeline("qwg_ink2_f32_to_bf16")
        self.pSConvVerifyCommit = try hal.pipeline("qwg_ink2_sconv_verify_commit")
        self.pSConvVerifyCommitTree = try hal.pipeline("qwg_ink2_sconv_verify_commit_tree")
        self.pTreeKVCompact = try hal.pipeline("qwg_ink2_tree_kv_compact")
        self.pMixNorm = try hal.pipeline("qwg_ink2_dspark_mix_norm")
        self.pMixNormMap = try hal.pipeline("qwg_ink2_dspark_mix_norm_map")
        self.pCtxKV = try hal.pipeline("qwg_ink2_dspark_ctx_kv_write")
        self.pCtxKVFused = try hal.pipeline("qwg_ink2_dspark_ctx_kv_write_fused")
        self.pAddNorm = try hal.pipeline("qwg_ink2_dspark_add_norm")
        self.pPropQKR = try hal.pipeline("qwg_ink2_dspark_prop_qk_rope")
        self.pPropQKRStrided = try hal.pipeline("qwg_ink2_dspark_prop_qk_rope_strided")
        self.pAttnGQA = try hal.pipeline("qwg_ink2_dspark_attn_gqa")
        self.pAttnGQABatch = try hal.pipeline("qwg_ink2_dspark_attn_gqa_batch")
        self.pAttnGQASplit = try hal.pipeline("qwg_ink2_dspark_attn_gqa_split")
        self.pAttnGQACombine = try hal.pipeline("qwg_ink2_dspark_attn_gqa_combine")
        self.pMarkovCtx = try hal.pipeline("qwg_ink2_dspark_markov_ctx")
        self.pMarkovStep = try hal.pipeline("qwg_ink2_dspark_markov_step_state")
        self.pMarkovStepBatch = try hal.pipeline("qwg_ink2_dspark_markov_step_batch")
        self.pLogitsStage1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_stage1")
        self.pLogitsAQ8Stage1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_aq8_stage1")
        self.pArgmaxFinal = try hal.pipeline("qwg_ink2_dspark_markov_argmax_final")
        self.pTopKStage1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_bf16_topk_stage1")
        self.pTopKAQ8Stage1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_aq8_topk_stage1")
        self.pTopKFinal = try hal.pipeline("qwg_ink2_dspark_markov_topk_final")

        var bufs: [any MTLBuffer] = [
            wBuf, fWBuf, fBBuf, dummyB, dummyF, tapBuf, lastTapBuf, ctxMixBuf, ctxKVProjBuf,
            propXBuf, propResBuf, propNormBuf, propQKVBuf, propAttnBuf, propAttnPartOBuf, propAttnPartMLBuf, guBuf, actBuf, r16Buf, f32Out,
            baseLogitsBF16Buf, f16InBuf, f32Buf, fRnnBuf, markovCtxF32Buf, markovStateBF16Buf,
            markovLatentBF16Buf, markovStateBatchBufA, markovStateBatchBufB, markovLatentBatchBuf,
            draftTokBuf, treeTokBuf, argmaxPartValBuf, argmaxPartIdxBuf, topkPartValBuf, topkPartIdxBuf
        ]
        for l in builtLayers { bufs.append(l.kCache); bufs.append(l.vCache) }
        self.residentBuffers = bufs
        self.totalBytes = bufs.reduce(0) { $0 + $1.length }
        src.adviseDontNeed()
    }

    static func estimateWeightBytes(mtpURL: URL, config: QWModelDescriptor.Inkling2DSparkConfig, unpaddedVocab: Int) -> Int {
        let attr = try? FileManager.default.attributesOfItem(atPath: mtpURL.path)
        let fileSize = (attr?[.size] as? Int) ?? (610 << 20)
        let r = config.markovRank
        let foldWBytes = (r >= 32 && r % 32 == 0) ? (unpaddedVocab * (r / 32) * 36) : (unpaddedVocab * r * 2)
        return fileSize + foldWBytes + unpaddedVocab * 2 + (64 << 20)
    }

    /// Maximum global KV cache capacity (tokens) retained by DSpark drafters on long target contexts.
    static let maxGlobalKVCapacity = 16_384

    /// Resolved DSpark global KV capacity for a target engine context `capacity`.
    static func globalKVCapacity(for capacity: Int) -> Int {
        min(max(1, capacity), maxGlobalKVCapacity)
    }

    /// Estimates DSpark's total fixed + capped-global KV cache bytes (bounded at <= 16K global tokens).
    static func estimateKVBytes(
        config: QWModelDescriptor.Inkling2DSparkConfig,
        capacity: Int,
        maxPrefillChunk: Int
    ) -> Int {
        let rowTotal = 2 * config.kvHeads * config.headDim * 2
        let localLayers = config.localLayerIds.count
        let localCap = min(max(1, capacity), config.slidingWindowSize + maxPrefillChunk)
        let globalLayers = max(0, config.numLayers - localLayers)
        let globalCap = globalKVCapacity(for: capacity)
        return (localLayers * localCap + globalLayers * globalCap) * rowTotal
    }

    func reset() { ctxLen = 0 }
}

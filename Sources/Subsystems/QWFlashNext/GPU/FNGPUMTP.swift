import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

/// Backward-compatible alias for the unified `QWContextCopyDrafter` in `QWCore`.
public typealias FNContextCopyDrafter = QWContextCopyDrafter


/// 1-layer Multi-Token Prediction (`mtp.*`) module on the GPU for Qwen3.8-Flash-Next speculative decoding.
/// Fuses the trunk's Layer-47 4-branch residual stream with the next token's embedding, runs `mtp.layers.0`
/// (Full-Attention QSA + 512-expert MoE), and projects through `mtp.hyper_connection_mixer` + shared `lm_head`.
public final class FNGPUMTP: @unchecked Sendable {
    /// Rows of the LM head the MTP drafts over (`QWG_DRAFT_VOCAB`, default 98304, 0 = full vocab).
    static func draftVocab(full: Int) -> Int {
        let v = Int(ProcessInfo.processInfo.environment["QWG_DRAFT_VOCAB"] ?? "") ?? 98304
        return v > 0 ? min(full, (v + 255) & ~255) : full
    }
    public let preNormEmb: QWTensorValue
    public let preNormHid: QWTensorValue
    public let fcEmbedding: QWWeightView
    public let fcHidden: QWWeightView
    public let layer: FNGPULayerWeights
    public let attnState1: FNGPUAttnDecodeState
    public let attnState2: FNGPUAttnDecodeState
    public let mtpHead: FNGPUHead
    public let onesInj: QWTensorValue
    public let tokensBuf: QWTensorValue
    /// Chained MTP drafts `[d1, d2, d3, ...]` written on the device.
    public let chainTokens: QWTensorValue
    public let scratch1: FNGPUScratch
    public let scratch2: FNGPUScratch
    public let hidden: Int, hc: Int, width: Int, eps: Float
    public private(set) var lastGPUMillis: Double = 0
    private let arena: QWArena
    private let ctx: FNGPUContext
    private let embedTable: FNGPUEmbedTable

    public static func isAvailable(in pack: QWQuant.Pack) -> Bool {
        (try? pack.entry("mtp.pre_fc_norm_embedding.weight")) != nil
    }

    public static func isAvailable(in loader: FNPackLoader) -> Bool {
        if let pack = loader.pack { return isAvailable(in: pack) }
        return loader.st?.tensor("mtp.pre_fc_norm_embedding.weight") != nil
    }

    public init(arena: QWArena, ctx: FNGPUContext, loader: FNPackLoader,
                embedTable: FNGPUEmbedTable, headWeight: QWWeightView,
                capacity: Int, maxPrefillRows: Int = 512,
                kvMode: FNKVStorageMode = .f16,
                sharedRope: (cos: QWTensorValue, sin: QWTensorValue, bcos: QWTensorValue, bsin: QWTensorValue)? = nil,
                sharedQSAScores1: QWTensorValue? = nil,
                sharedQSAScores2: QWTensorValue? = nil,
                staging: (any MTLBuffer)? = nil,
                yarnFactor: Float? = nil) throws {
        self.arena = arena
        self.ctx = ctx
        self.embedTable = embedTable
        let d = loader.descriptor
        let H = d.hidden, C = d.hcCount ?? 4, W = C * H, lowrank = d.hcLowrank ?? 320
        self.hidden = H; self.hc = C; self.width = W; self.eps = d.rmsEps

        preNormEmb = try FNGPULoader.f32Buffer(arena, try loader.tensor("mtp.pre_fc_norm_embedding.weight"))
        preNormHid = try FNGPULoader.f32Buffer(arena, try loader.tensor("mtp.pre_fc_norm_hidden.weight"))
        fcEmbedding = try FNGPULoader.hq4Bank(arena, loader, "mtp.fc_embedding.weight", n: H, k: H, staging: staging)
        fcHidden = try FNGPULoader.hq4Bank(arena, loader, "mtp.fc_hidden.weight", n: H, k: H, staging: staging)

        let gw = try FNGPULoader.layer(d.numLayers, loader: loader, arena: arena, prefix: "mtp.layers.0.", staging: staging,
                                       yarnFactor: yarnFactor)
        self.layer = gw
        guard case .attention(let attnM) = gw.mixer else {
            throw FNGPUError.notHQ4("mtp.layers.0.self_attn")
        }
        attnState1 = try FNGPUAttnDecodeState(arena: arena, mixer: attnM, hidden: H, capacity: capacity,
                                              kvMode: kvMode,
                                              sharedRope: sharedRope, sharedQSAScores: sharedQSAScores1)
        attnState2 = try FNGPUAttnDecodeState(sharingCachesOf: attnState1, arena: arena, mixer: attnM, rows: 8,
                                              sharedQSAScores: sharedQSAScores2)
        scratch1 = try FNGPUScratch(arena: arena, T: 1, hidden: H, hc: C, lowrank: lowrank,
                                    moe: gw.moe, ple: nil, gdn: nil, attn: attnM, maxCapacity: capacity)
        scratch2 = try FNGPUScratch(arena: arena, T: 8, hidden: H, hc: C, lowrank: lowrank,
                                    moe: gw.moe, ple: nil, gdn: nil, attn: attnM, maxCapacity: capacity)

        let mp = "mtp.hyper_connection_mixer."
        let mixerDown = try FNGPULoader.hq4Bank(arena, loader, mp + "input_mix_weight_down.weight", n: lowrank, k: W, staging: staging)
        let mixerUp = try FNGPULoader.hq4Bank(arena, loader, mp + "input_mix_weight_up.weight", n: W, k: lowrank, staging: staging)
        let mixerNorm = try FNGPULoader.f32Buffer(arena, try loader.tensor(mp + "hc_norm.weight"))
        // Drafts only need a likely token, so the MTP head may argmax over a frequent-id prefix of the
        // vocab (BPE ids are roughly frequency-ordered); verification still runs the full head.
        let draftVocab = Self.draftVocab(full: headWeight.n)
        let draftHead = draftVocab < headWeight.n
            ? (headWeight.subview(blocks: headWeight.blocks, n: draftVocab, k: headWeight.k) ?? headWeight)
            : headWeight
        mtpHead = try FNGPUHead(arena: arena, hal: ctx.hal, head: draftHead, mixerDown: mixerDown, mixerUp: mixerUp,
                                normWeight: mixerNorm, eps: d.rmsEps, hidden: H, hc: C, lowrank: lowrank, maxRows: 8)

        let maxR = max(8, maxPrefillRows)
        onesInj = try FNGPULoader.f32Buffer(arena, [Float](repeating: 1.0, count: maxR * C))
        tokensBuf = try arena.activationsRegion().allocate(QWTensorShape([maxR], .i32))
        chainTokens = try arena.activationsRegion().allocate(QWTensorShape([8], .i32))
    }

    public func resetState(usedPositions: Int? = nil) {
        let cap = attnState1.capacity
        let n = min(cap, max(0, usedPositions ?? cap))
        guard n > 0 else { return }
        let idxRowBytes = (attnState1.indexK.bytes / cap) * n
        let poolBlocks = (n + 3) / 4
        let poolBytes = min(attnState1.pooled.bytes, (attnState1.pooled.bytes / max(1, cap / 4)) * poolBlocks)
        memset(arena.cpuPointer(of: attnState1.indexK), 0, idxRowBytes)
        memset(arena.cpuPointer(of: attnState1.pooled), 0, poolBytes)
    }

    public func resetStateTail(from startPos: Int, usedPositions: Int) {
        let cap = attnState1.capacity
        let s = min(cap, max(0, startPos))
        let e = min(cap, max(s, usedPositions))
        if e > s {
            let rowStride = attnState1.indexK.bytes / cap
            memset(arena.cpuPointer(of: attnState1.indexK).advanced(by: s * rowStride), 0, (e - s) * rowStride)
        }
        let bStart = s / 4
        let bEnd = (e + 3) / 4
        if bEnd > bStart {
            let blockStride = attnState1.pooled.bytes / max(1, cap / 4)
            let startByte = min(attnState1.pooled.bytes, bStart * blockStride)
            let endByte = min(attnState1.pooled.bytes, bEnd * blockStride)
            if endByte > startByte {
                memset(arena.cpuPointer(of: attnState1.pooled).advanced(by: startByte), 0, endByte - startByte)
            }
        }
    }

    /// Encodes a T-row MTP pass into `enc` using `streamPrev [T, W]` and on-device `tokens [T]` at `pos`.
    /// When `usePrefillPath` is true (or `T > 4`), runs `layerPrefill` with `scratch`; else runs decode/verify `layer`.
    public func encode(
        _ enc: any MTLComputeCommandEncoder,
        streamPrev: QWTensorValue,
        tokens: QWTensorValue,
        pos: Int,
        rows T: Int,
        scratch customScratch: FNGPUScratch? = nil,
        preInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil,
        projectRows: Int = 1,
        usePrefillPath: Bool = false,
        draftTokens: QWTensorValue? = nil
    ) throws {
        let hal = ctx.hal
        if customScratch == nil && T > 1 {
            scratch2.T = T
        }
        let s = customScratch ?? (T == 1 ? scratch1 : scratch2)
        precondition(s.T == T, "MTP scratch T (\(s.T)) must match rows (\(T))")
        if let inj = preInject {
            try QWGemm.encodeFnInject(enc, hal: hal, base: inj.base, src: inj.src, inj: inj.scale,
                                      out: streamPrev, rows: T, hidden: hidden, hc: hc)
            enc.memoryBarrier(scope: .buffers)
        }
        let emb = s.mixedA
        let xnEmb = s.mixedB
        let rotEmb = s.oG
        let embProj = s.pleV
        let xnHid = s.pleKeyN
        let rotHid = s.pleQueryN
        let hidProj = s.pleGV
        let streamIn = s.pleOut
        let streamOut = s.pleGVN

        try embedTable.gather(enc: enc, hal: hal, token: tokens,
                              out: emb, rows: T, hc: 1)
        try QWGemm.encodeFnHCNormRot(enc, hal: hal, x: streamPrev, weight: preNormHid,
                                     xn: xnHid, rot: rotHid, rows: T, width: width, hidden: width, eps: eps)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeFnHCNormRot(enc, hal: hal, x: emb, weight: preNormEmb,
                                     xn: xnEmb, rot: rotEmb, rows: T, width: hidden, hidden: hidden, eps: eps)
        try FNGPULayerPasses.denseProj(enc, hal: hal, s: s, weight: fcHidden, x: rotHid, y: hidProj, rows: hc * T)
        enc.memoryBarrier(scope: .buffers)
        try FNGPULayerPasses.denseProj(enc, hal: hal, s: s, weight: fcEmbedding, x: rotEmb, y: embProj, rows: T)
        enc.memoryBarrier(scope: .buffers)

        let inInj = (base: hidProj, src: embProj, scale: onesInj)
        let usedTailScratch = usePrefillPath && T > 1 && projectRows <= 1 && layer.ple == nil
        if usedTailScratch {
            try FNGPULayerPasses.mtpPrefillCache(enc, ctx: ctx, layer: layer, s: s, stream: streamIn,
                                                  state: attnState1, pos: pos, inInject: inInj,
                                                  lastRowScratch: projectRows == 1 ? scratch1 : nil)
        } else if usePrefillPath || T > 8 {
            try FNGPULayerPasses.layerPrefill(enc, ctx: ctx, layer: layer, s: s, stream: streamIn, out: streamOut,
                                              attnState: attnState1, pos: pos, inInject: inInj, deferOutInject: true)
        } else {
            let st = (T == 1) ? attnState1 : attnState2
            try FNGPULayerPasses.layer(enc, ctx: ctx, layer: layer, s: s, stream: streamIn, out: streamOut,
                                       attnState: st, pos: pos, inInject: inInj, deferOutInject: true)
        }
        guard projectRows > 0 else { return }
        if usedTailScratch {
            let tailInj = (base: scratch1.h2, src: scratch1.moeOut, scale: scratch1.injB)
            try mtpHead.encode(enc, hal: hal, stream: scratch1.pleGVN, inInject: tailInj, rows: 1)
        } else if let draft = draftTokens, T >= 2 && T <= 8 {
            let tailInj = (base: s.h2, src: s.moeOut, scale: s.injB)
            try mtpHead.encode(enc, hal: hal, stream: streamOut, inInject: tailInj, rows: 1,
                               selectRowFrom: (verified: tokens, draft: draft, totalRows: T))
        } else if projectRows == T {
            let tailInj = (base: s.h2, src: s.moeOut, scale: s.injB)
            try mtpHead.encode(enc, hal: hal, stream: streamOut, inInject: tailInj, rows: T)
        } else {
            let r = T - 1
            let lastStream = streamOut.slice(byteOffset: r * width * 4, count: width, dtype: .f32)
            let tailInj = (base: s.h2.slice(byteOffset: r * width * 4, count: width, dtype: .f32),
                           src: s.moeOut.slice(byteOffset: r * hidden * 4, count: hidden, dtype: .f32),
                           scale: s.injB.slice(byteOffset: r * hc * 4, count: hc, dtype: .f32))
            try mtpHead.encode(enc, hal: hal, stream: lastStream, inInject: tailInj, rows: 1)
        }
    }

    /// The materialized MTP output stream of the row the last verify pass projected.
    public func lastProjectedStream(verifyRows k: Int) -> QWTensorValue {
        let s = k == 1 ? scratch1.pleGVN : scratch2.pleGVN
        return s.slice(byteOffset: 0, count: width, dtype: .f32)
    }

    /// Chains `passes` single-row MTP passes after a verify round, each feeding the
    /// previous pass's materialized stream and draft token, writing drafts to `chainTokens[1...]`.
    /// `chainTokens[0]` receives the verify pass's own draft first.
    public func encodeChain(_ enc: any MTLComputeCommandEncoder, firstStream: QWTensorValue,
                            startPos: Int, passes: Int) throws {
        let hal = ctx.hal
        let copy = try hal.pipeline("qwg_verify_copy")
        func copyToken(to slot: Int) {
            enc.setComputePipelineState(copy)
            mtpHead.token.slice(byteOffset: 0, count: 1, dtype: .i32).bind(enc, index: 0)
            chainTokens.slice(byteOffset: slot * 4, count: 1, dtype: .i32).bind(enc, index: 1)
            var one: UInt32 = 1
            enc.setBytes(&one, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
        copyToken(to: 0)
        var prev = firstStream
        for p in 0..<passes {
            try encode(enc, streamPrev: prev, tokens: chainTokens.slice(byteOffset: p * 4, count: 1, dtype: .i32),
                       pos: startPos + p, rows: 1, projectRows: 1)
            enc.memoryBarrier(scope: .buffers)
            copyToken(to: p + 1)
            prev = scratch1.pleGVN
        }
    }

    public func readChain(count: Int) -> [Int] {
        arena.read(chainTokens, as: Int32.self, count: count).map(Int.init)
    }

    public func writeTokens(_ nextTokens: [Int32]) {
        arena.write(tokensBuf, from: nextTokens)
    }

    /// Runs MTP over a prefill chunk of `nextTokens.count` rows starting at `startPos`, reusing `prefillScratch`
    /// and materializing Layer 47's deferred output stream into `streamBuf` when `preInject` is supplied.
    public func forwardPrefill(
        streamBuf: QWTensorValue,
        preInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)?,
        nextTokens: [Int32],
        startPos: Int,
        prefillScratch: FNGPUScratch,
        projectLastRow: Bool = true
    ) throws -> Int {
        let T = nextTokens.count
        precondition(T >= 1 && prefillScratch.T == T)
        arena.write(tokensBuf, from: nextTokens)
        guard let (cb, enc) = ctx.hal.begin(.concurrent) else {
            throw NSError(domain: "FNGPUMTP", code: 1, userInfo: [NSLocalizedDescriptionKey: "no command buffer"])
        }
        try encode(enc, streamPrev: streamBuf, tokens: tokensBuf, pos: startPos, rows: T,
                   scratch: prefillScratch, preInject: preInject,
                   projectRows: projectLastRow ? 1 : 0, usePrefillPath: T > 1)
        ctx.hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        lastGPUMillis = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        return projectLastRow ? mtpHead.argmaxToken() : 0
    }
}

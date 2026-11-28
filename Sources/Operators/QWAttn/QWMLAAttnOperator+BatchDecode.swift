// QWMLAAttnOperator+BatchDecode.swift — Multi-session batched MLA decode (`B = 1..8`)
// and per-session `SessionState` allocation / stash helpers for `DriverBatchDecode`.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWKV

extension QWMLAAttnOperator {
    public static let maxDecodeBatchRows = 8

    /// Standalone per-session MLA/DSA state (`pooledKeys`, `kPeCache`, `kPending`, `gatePending`)
    /// enabling `B = 2..8` concurrent sessions to advance without clobbering each other.
    public final class SessionState: @unchecked Sendable {
        public let region: QWMRegion
        public let layers: [LayerState]
        public let maxContext: Int
        public let maxBlocks: Int

        internal init(region: QWMRegion, layers: [LayerState], maxContext: Int, maxBlocks: Int) {
            self.region = region
            self.layers = layers
            self.maxContext = maxContext
            self.maxBlocks = maxBlocks
        }
    }

    internal func buildDSALatentPrepParams(
        mlaW: LayerWeights,
        state: LayerState,
        kvMode: QWKVMode,
        pos0: Int,
        rows: Int,
        hasMRoPE: Bool
    ) -> DSALatentPrepParams {
        let effDense = isLayerDenseOnly(mlaW, state: state)
        let effIdxDim = effDense ? 0 : idxDim
        let callLatStride = latentStride(for: mlaW, state: state)
        let kNormRMS = mlaW.indexerKNormIsRMS ?? indexerKNormIsRMS
        let ropeFlags: UInt32 = (ropeInterleave ? 1 : 0) | (hasMRoPE ? 2 : 0) | (kNormRMS ? 4 : 0)

        var gatedFlags: UInt32 = 0
        let qAW = mlaW.qAGatedNorm?.weight ?? mlaW.qANormW
        if qAW.shape.dtype == .bf16 { gatedFlags |= 2 }
        if let qG = mlaW.qAGatedNorm {
            gatedFlags |= 1
            if qG.gateUp.shape.dtype == .bf16 { gatedFlags |= 4 }
            if qG.gateDown.shape.dtype == .bf16 { gatedFlags |= 8 }
            if qG.zeroCentered { gatedFlags |= 16 }
        }
        let kvAW = mlaW.kvAGatedNorm?.weight ?? mlaW.kvANormW
        if kvAW.shape.dtype == .bf16 { gatedFlags |= 64 }
        if let kvG = mlaW.kvAGatedNorm {
            gatedFlags |= 32
            if kvG.gateUp.shape.dtype == .bf16 { gatedFlags |= 128 }
            if kvG.gateDown.shape.dtype == .bf16 { gatedFlags |= 256 }
            if kvG.zeroCentered { gatedFlags |= 512 }
        }
        if mlaW.kNormW.shape.dtype == .bf16 { gatedFlags |= 1024 }

        return DSALatentPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qLoraRank: UInt32(qLoraRank),
            kvLoraRank: UInt32(kvLoraRank),
            ropeDim: UInt32(ropeDim),
            idxDim: UInt32(effIdxDim),
            latentStride: UInt32(callLatStride),
            kvMode: Self.kvModeCode(kvMode),
            rmsEps: geometry.rmsEps,
            kNormEps: 1e-6,
            mscale: ropeMscale,
            useHadamard: useFastHadamard ? 1 : 0,
            ropeFlags: ropeFlags,
            gatedFlags: gatedFlags,
            qAGateRank: UInt32(mlaW.qAGatedNorm?.rank ?? 0),
            kvAGateRank: UInt32(mlaW.kvAGatedNorm?.rank ?? 0),
            qAGatingWeight: mlaW.qAGatedNorm?.gatingWeight ?? 1.0,
            kvAGatingWeight: mlaW.kvAGatedNorm?.gatingWeight ?? 1.0,
            alphaQ: mlaW.alphaQ ?? alphaQ,
            alphaKV: mlaW.alphaKV ?? alphaKV
        )
    }

    /// Allocates an isolated `SessionState` matching this operator's layer topology.
    public func makeSessionState(maxContext overrideContext: Int? = nil) throws -> SessionState {
        let ctx = max(512, overrideContext ?? self.maxContext)
        let blocks: Int
        if !hasIndexer {
            blocks = max(128, ctx)
        } else if isDecoupledDSA {
            blocks = max(128, ctx >> poolShift)
        } else {
            blocks = max(128, (ctx + 3) / 4)
        }
        let peCapRows = max(blocks, ctx)
        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }
        var offs: [(kPend: Int, gatePend: Int, pooled: Int, kPe: Int, hasIdx: Bool)] = []
        offs.reserveCapacity(layers.count)
        for ls in layers {
            let hasIdx = ls.hasIndexer
            let kOff = (isDecoupledDSA || !hasIdx) ? 0 : reserve(ringCap * idxDim * 2)
            let gOff = (isDecoupledDSA || !hasIdx) ? 0 : reserve(ringCap * idxDim * 2)
            let pOff = hasIdx ? reserve(((blocks + 7) & ~7) * idxDim * 2) : reserve(0)
            let peOff = (isDecoupledDSA && ropeDim > 0) ? reserve(((peCapRows + 7) & ~7) * ropeDim * 2) : 0
            offs.append((kOff, gOff, pOff, peOff, hasIdx))
        }
        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "mla-batch-session", asked: totalBytes, free: 0)
        }
        buf.label = "mla-batch-session"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        let storedIdxCols = max(1, idxDim)
        var built: [LayerState] = []
        built.reserveCapacity(layers.count)
        for (i, o) in offs.enumerated() {
            let kPe: QWTensorValue? = (isDecoupledDSA && ropeDim > 0)
                ? QWTensorValue(region: reg, offset: o.kPe, shape: QWTensorShape([peCapRows, ropeDim], .f16))
                : nil
            built.append(LayerState(
                fullAttentionIndex: i,
                kPending: QWTensorValue(region: reg, offset: o.kPend, shape: QWTensorShape([isDecoupledDSA || !o.hasIdx ? 1 : ringCap, storedIdxCols], .f16)),
                gatePending: QWTensorValue(region: reg, offset: o.gatePend, shape: QWTensorShape([isDecoupledDSA || !o.hasIdx ? 1 : ringCap, storedIdxCols], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([o.hasIdx ? blocks : 1, storedIdxCols], .f16)),
                kPeCache: kPe,
                coarseCSR: nil,
                hasIndexer: o.hasIdx
            ))
        }
        return SessionState(region: reg, layers: built, maxContext: ctx, maxBlocks: blocks)
    }

    /// Restores a captured live session stash (`captureLiveSessionStash`) into `sessionState`.
    public func restoreLiveSessionStash(
        _ data: Data,
        into sessionState: SessionState,
        trunkPosition: Int,
        mtpLayerIndex: Int?,
        mtpPosition: Int
    ) -> Bool {
        guard trunkPosition >= 0, mtpPosition >= 0,
              sessionState.layers.count == layers.count,
              data.count == liveSessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition) else {
            return false
        }
        let base = sessionState.region.pointer
        let capLimit = max(sessionState.maxBlocks, sessionState.maxContext)
        for (i, ls) in sessionState.layers.enumerated() {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            let limit = isDecoupledDSA ? (ls.hasIndexer ? sessionState.maxBlocks : capLimit) : (sessionState.maxBlocks * 4)
            if pos > limit { return false }
        }
        withExtendedLifetime(sessionState.region.buffer) {
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let srcBase = raw.baseAddress else { return }
                var src = 0
                if isDecoupledDSA {
                    for (i, ls) in sessionState.layers.enumerated() {
                        let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                        let idxBytes = ls.hasIndexer ? (pos * idxDim * 2) : 0
                        let peBytes = pos * ropeDim * 2
                        if idxBytes > 0 {
                            base.advanced(by: ls.pooledKeys.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: idxBytes)
                            src += idxBytes
                        }
                        if peBytes > 0, let kPe = ls.kPeCache {
                            base.advanced(by: kPe.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: peBytes)
                            src += peBytes
                        }
                    }
                } else {
                    let rowBytes = idxDim * 2
                    let ringBytes = ringCap * rowBytes
                    for (i, ls) in sessionState.layers.enumerated() {
                        let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                        let pooledBytes = ls.hasIndexer ? ((pos / 4) * rowBytes) : 0
                        if pooledBytes > 0 {
                            base.advanced(by: ls.pooledKeys.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: pooledBytes)
                            src += pooledBytes
                        }
                        if ls.hasIndexer {
                            base.advanced(by: ls.kPending.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: ringBytes)
                            src += ringBytes
                            base.advanced(by: ls.gatePending.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: ringBytes)
                            src += ringBytes
                        }
                    }
                }
            }
        }
        return true
    }

    /// Captures a live session stash from `sessionState` matching `captureLiveSessionStash` layout.
    public func captureLiveSessionStash(
        from sessionState: SessionState,
        trunkPosition: Int,
        mtpLayerIndex: Int?,
        mtpPosition: Int
    ) -> Data {
        precondition(trunkPosition >= 0 && mtpPosition >= 0 && sessionState.layers.count == layers.count)
        var out = Data(capacity: liveSessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition))
        let base = sessionState.region.pointer
        withExtendedLifetime(sessionState.region.buffer) {
            if isDecoupledDSA {
                for (i, ls) in sessionState.layers.enumerated() {
                    let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                    let idxBytes = ls.hasIndexer ? (pos * idxDim * 2) : 0
                    let peBytes = pos * ropeDim * 2
                    if idxBytes > 0 {
                        out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self), count: idxBytes)
                    }
                    if peBytes > 0, let kPe = ls.kPeCache {
                        out.append(base.advanced(by: kPe.offset).assumingMemoryBound(to: UInt8.self), count: peBytes)
                    }
                }
            } else {
                let rowBytes = idxDim * 2
                let ringBytes = ringCap * rowBytes
                for (i, ls) in sessionState.layers.enumerated() {
                    let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                    let pooledBytes = ls.hasIndexer ? ((pos / 4) * rowBytes) : 0
                    if pooledBytes > 0 {
                        out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self), count: pooledBytes)
                    }
                    if ls.hasIndexer {
                        out.append(base.advanced(by: ls.kPending.offset).assumingMemoryBound(to: UInt8.self), count: ringBytes)
                        out.append(base.advanced(by: ls.gatePending.offset).assumingMemoryBound(to: UInt8.self), count: ringBytes)
                    }
                }
            }
        }
        return out
    }

    internal func encodeMLAWeightViewH16(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        weight: QWWeightView,
        y16: QWTensorValue,
        rows: Int
    ) throws {
        guard rows > 0 else { return }
        if weight.bits == 16 && weight.blocks.shape.dtype == .f16 {
            var hgP = HeadGateProjParams(
                T: UInt32(rows),
                hidden: UInt32(weight.k),
                nGate: UInt32(weight.n)
            )
            enc.setComputePipelineState(psoHeadGateProj)
            x.bind(enc, index: 0)
            weight.blocks.bind(enc, index: 1)
            y16.bind(enc, index: 2)
            enc.setBytes(&hgP, length: MemoryLayout<HeadGateProjParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: weight.n, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        switch weight.containerKind {
        case .shvq(let bits, let group), .rotatedHQ(let bits, let group), .nativeAffine(let bits, let group):
            try QWGemmAQ.encodeDecodeT4(
                enc, hal: hal, blocks: weight.blocks, x: x, y16: y16,
                n: weight.n, k: weight.k, bits: bits, group: group,
                rows: rows, isSHVQ: weight.containerKind.isSHVQ
            )
        case .incumbentHQ4G32:
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: weight.n, k: weight.k), dtype: .q4g, epi: .none, outDtype: .f16)
            for b in 0..<rows {
                let xRow = (rows == 1) ? x : x.slice(byteOffset: b * weight.k * 2, count: weight.k, dtype: .f16)
                let yRow = (rows == 1) ? y16 : y16.slice(byteOffset: b * weight.n * 2, count: weight.n, dtype: .f16)
                try QWGemm.encodeH16(enc, hal: hal, plan: plan, x: xRow, wQ4: weight, y16: yRow)
            }
        }
    }

    /// Encodes batched MLA decode (`B = 1..8` sessions) assuming `[B, latentStride]` `qkvProj` output
    /// is already written in `mlaLatentScratch`. Shares `qBProj`, `W_UK`, `W_UV`, and `vhaGateRot`
    /// weight reads across all `B` sessions while isolating per-session KV views, positions, and state.
    public func encodeDecodeBatch(
        _ enc: any MTLComputeCommandEncoder,
        ln1: QWTensorValue,
        mlaW: LayerWeights,
        slots: [DecodeBatchSlot],
        attOut: QWTensorValue,
        qFullPrecomputed: Bool = false,
        externalGatePrecomputed: Bool = false,
        encodeBatchProj: ((_ enc: any MTLComputeCommandEncoder, _ x: QWTensorValue, _ weight: QWWeightView, _ y16: QWTensorValue, _ rows: Int) throws -> Void)? = nil
    ) throws {
        let B = slots.count
        guard (1...Self.maxDecodeBatchRows).contains(B), B <= maxPrefillChunk else {
            throw QWAttnError.badShape("MLA encodeDecodeBatch requires 1...\(Self.maxDecodeBatchRows) slots (got \(B))")
        }
        enc.useResource(region.buffer, usage: [.read, .write])
        for slot in slots {
            enc.useResource(slot.state.pooledKeys.region.buffer, usage: [.read, .write])
            if let kPe = slot.state.kPeCache {
                enc.useResource(kPe.region.buffer, usage: [.read, .write])
            }
        }

        if !isDecoupledDSA {
            try encodeNoPEDecodeBatch(
                enc, ln1: ln1, mlaW: mlaW, slots: slots, attOut: attOut,
                qFullPrecomputed: qFullPrecomputed, encodeBatchProj: encodeBatchProj
            )
            return
        }

        let refState = slots[0].state
        let effDense = isLayerDenseOnly(mlaW, state: refState)
        let effIdxHeads = effDense ? 0 : idxHeads
        let effIdxDim = effDense ? 0 : idxDim
        let effTopK = effDense ? 0 : topKPools
        let callLatStride = latentStride(for: mlaW, state: refState)
        let callQChannels = qChannels(for: mlaW)
        let callQFullStride = qFullStride(for: mlaW, state: refState)
        let effOGateInQKV = oGateInQKV || mlaW.oGateInQKV
        let H = geometry.hidden
        let tgHBytes = ((max(512, max(qLoraRank, kvLoraRank)) * 2 + 15) & ~15)
        let qAW = mlaW.qAGatedNorm?.weight ?? mlaW.qANormW
        let kvAW = mlaW.kvAGatedNorm?.weight ?? mlaW.kvANormW

        for b in 0..<B {
            let slot = slots[b]
            let ls = slot.state
            let kPe = ls.kPeCache ?? mlaLatentScratch
            let latRow = mlaLatentScratch.slice(byteOffset: b * callLatStride * 2, count: callLatStride, dtype: .f16)
            let qResRow = qLoraRank > 0
                ? qResidScratch.slice(byteOffset: b * qLoraRank * 2, count: qLoraRank, dtype: .f16)
                : qResidScratch
            let hasMRoPE = stageMRoPEPositions(slot.mropePositions, rows: 1)
            var dsaP = buildDSALatentPrepParams(
                mlaW: mlaW, state: ls, kvMode: slot.kv.k.mode, pos0: slot.position, rows: 1, hasMRoPE: hasMRoPE
            )
            enc.setComputePipelineState(psoDSALatentPrep)
            latRow.bind(enc, index: 0)
            qAW.bind(enc, index: 1)
            kvAW.bind(enc, index: 2)
            mlaW.kNormW.bind(enc, index: 3)
            mlaW.kNormB.bind(enc, index: 4)
            invFreqBuf.bind(enc, index: 5)
            qResRow.bind(enc, index: 6)
            slot.kv.k.pool.bindTo(enc, index: 7)
            slot.kv.v.pool.bindTo(enc, index: 8)
            slot.kv.k.table.bind(enc, index: 9)
            slot.kv.v.table.bind(enc, index: 10)
            kPe.bind(enc, index: 11)
            ls.pooledKeys.bind(enc, index: 12)
            enc.setBytes(&dsaP, length: MemoryLayout<DSALatentPrepParams>.stride, index: 13)
            mropePosScratch.bind(enc, index: 14)
            ropeAxisMapBuf.bind(enc, index: 15)
            (mlaW.qAGatedNorm?.gateUp ?? qAW).bind(enc, index: 16)
            (mlaW.qAGatedNorm?.gateDown ?? qAW).bind(enc, index: 17)
            (mlaW.kvAGatedNorm?.gateUp ?? kvAW).bind(enc, index: 18)
            (mlaW.kvAGatedNorm?.gateDown ?? kvAW).bind(enc, index: 19)
            enc.setThreadgroupMemoryLength(tgHBytes, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            if !effDense, effIdxHeads > 0, effIdxDim > 0 {
                let ln1Row = ln1.slice(byteOffset: b * H * 2, count: H, dtype: .f16)
                let idxWRow = idxWeightsScratch.slice(byteOffset: b * effIdxHeads * 4, count: effIdxHeads, dtype: .f32)
                var gwP = LatentPrepParams(
                    T: 1, pos0: UInt32(slot.position), hidden: UInt32(H),
                    qLoraRank: UInt32(qLoraRank), kvLoraRank: UInt32(kvLoraRank),
                    idxDim: UInt32(effIdxDim), idxHeads: UInt32(effIdxHeads),
                    ringMask: UInt32(ringCap - 1), rmsEps: geometry.rmsEps,
                    kNormEps: 1e-6, kvMode: Self.kvModeCode(slot.kv.k.mode) | 0x100
                )
                enc.setComputePipelineState(psoGateWeightsProj)
                ln1Row.bind(enc, index: 0)
                mlaW.wGateWeights.bind(enc, index: 1)
                ls.gatePending.bind(enc, index: 2)
                idxWRow.bind(enc, index: 3)
                enc.setBytes(&gwP, length: MemoryLayout<LatentPrepParams>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (effIdxHeads + 7) / 8, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.memoryBarrier(scope: .buffers)

        if !qFullPrecomputed {
            let qBIn = isDirectQ
                ? ln1.slice(byteOffset: 0, count: B * H, dtype: .f16)
                : qResidScratch.slice(byteOffset: 0, count: B * qLoraRank, dtype: .f16)
            let qFullRows = mlaQFullScratch.slice(byteOffset: 0, count: B * callQFullStride, dtype: .f16)
            if let proj = encodeBatchProj {
                try proj(enc, qBIn, mlaW.qBProj, qFullRows, B)
            } else {
                try encodeMLAWeightViewH16(enc, x: qBIn, weight: mlaW.qBProj, y16: qFullRows, rows: B)
            }
        }

        let extGateMode = !effOGateInQKV && (
            outputGateMode == .headWiseFromHidden
            || outputGateMode == .elementWiseFromHidden
            || mlaW.oGateProj != nil
            || mlaW.oGateF16 != nil
        )
        let extGateCols = outputGateMode == .headWiseFromHidden ? qHeads : (qHeads * vHeadDim)
        if !externalGatePrecomputed, extGateMode {
            let ln1Rows = ln1.slice(byteOffset: 0, count: B * H, dtype: .f16)
            let gateOut = oGateScratch.slice(byteOffset: 0, count: B * extGateCols, dtype: .f16)
            if let gateProj = mlaW.oGateProj {
                if let proj = encodeBatchProj {
                    try proj(enc, ln1Rows, gateProj, gateOut, B)
                } else {
                    try encodeMLAWeightViewH16(enc, x: ln1Rows, weight: gateProj, y16: gateOut, rows: B)
                }
            } else if let gateF16 = mlaW.oGateF16 {
                var hgP = HeadGateProjParams(T: UInt32(B), hidden: UInt32(H), nGate: UInt32(extGateCols))
                enc.setComputePipelineState(psoHeadGateProj)
                ln1Rows.bind(enc, index: 0)
                gateF16.bind(enc, index: 1)
                gateOut.bind(enc, index: 2)
                enc.setBytes(&hgP, length: MemoryLayout<HeadGateProjParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: extGateCols, height: B, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        enc.memoryBarrier(scope: .buffers)

        let baseInc: UInt32 = mlaW.kvBProj.bits == 16
            ? 3
            : (mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0))
        let hasInlineOrFlagGate = mlaW.hasOGate
            || outputGateMode == .elementWiseFromQLora
            || (geometry.mlaConfig == nil && geometry.attnOutputGate)
        let hasAnyOGate = effOGateInQKV || extGateMode || hasInlineOrFlagGate
        let needsPostEpilogue = (mlaW.vhaMix != nil) || hasAnyOGate
        var absP = AbsorbParams(
            T: UInt32(B),
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank),
            qFullStride: UInt32(callQFullStride),
            bits: UInt32(mlaW.kvBProj.bits),
            group: UInt32(mlaW.kvBProj.groupSize),
            isIncumbent: baseInc | (needsPostEpilogue ? 0x10000 : 0)
        )
        let mTiles = (B + 7) / 8
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: B, mTiles: mTiles)

        let sinkTensor = mlaW.attnSink ?? zeroSinkBuf
        let softScale = effectiveSoftmaxScale(for: mlaW)
        let supportsNAXRopeDim = (ropeDim == 0 || ropeDim == 32 || ropeDim == 64 || ropeDim == 128)
        let split = (sparseAttnMMAEnabled && qHeads % 16 == 0 && kvLoraRank == 512 && supportsNAXRopeDim)
            ? psoDSASparseAttnSplit : nil

        for b in 0..<B {
            let slot = slots[b]
            let qFullRow = mlaQFullScratch.slice(byteOffset: b * callQFullStride * 2, count: callQFullStride, dtype: .f16)
            let hasMRoPE = stageMRoPEPositions(slot.mropePositions, rows: 1)
            let ropeFlags: UInt32 = (ropeInterleave ? 1 : 0) | (hasMRoPE ? 2 : 0)
            var qrP = QRopeIdxPrepParams(
                T: 1, pos0: UInt32(slot.position), qHeads: UInt32(qHeads), idxHeads: UInt32(effIdxHeads),
                ropeDim: UInt32(ropeDim), idxDim: UInt32(effIdxDim), qRopeOffset: UInt32(qRopeOffset),
                qChannels: UInt32(callQChannels), qFullStride: UInt32(callQFullStride),
                mscale: ropeMscale, useHadamard: useFastHadamard ? 1 : 0, ropeFlags: ropeFlags
            )
            enc.setComputePipelineState(psoQRopeIdxPrep)
            qFullRow.bind(enc, index: 0)
            invFreqBuf.bind(enc, index: 1)
            enc.setBytes(&qrP, length: MemoryLayout<QRopeIdxPrepParams>.stride, index: 2)
            mropePosScratch.bind(enc, index: 3)
            ropeAxisMapBuf.bind(enc, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: qHeads + effIdxHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)

        for b in 0..<B {
            let slot = slots[b]
            let ls = slot.state
            let kPe = ls.kPeCache ?? mlaLatentScratch
            let qFullRow = mlaQFullScratch.slice(byteOffset: b * callQFullStride * 2, count: callQFullStride, dtype: .f16)
            let idxWRow = idxWeightsScratch.slice(byteOffset: b * max(1, effIdxHeads) * 4, count: max(1, effIdxHeads), dtype: .f32)
            let listRow = blockListScratch.slice(byteOffset: b * listStride * 4, count: listStride, dtype: .i32)
            let countRow = blockCountScratch.slice(byteOffset: b * 4, count: 1, dtype: .i32)
            if !effDense, slot.position + 1 > effTopK + windowTokens {
                let effSplit = (effIdxHeads == 32 || effIdxHeads == 64) ? psoIndexerSplit : nil
                let packedIdxHeads: UInt32 = (poolShift == 2 && windowTokens == 0)
                    ? UInt32(effIdxHeads)
                    : (UInt32(effIdxHeads & 0xFF) | 0x1000 | (UInt32(poolShift & 0xF) << 8) | (UInt32(windowTokens) << 16))
                var idxP = IndexerParams(
                    T: 1, pos0: UInt32(slot.position), qChannels: UInt32(callQChannels),
                    idxHeads: packedIdxHeads, idxDim: UInt32(effIdxDim),
                    topKPools: UInt32(effTopK), listStride: UInt32(listStride), scoreStride: UInt32(maxBlocks)
                )
                qFullRow.bind(enc, index: 0)
                idxWRow.bind(enc, index: 1)
                ls.pooledKeys.bind(enc, index: 2)
                scoreScratch.bind(enc, index: 3)
                listRow.bind(enc, index: 4)
                countRow.bind(enc, index: 5)
                enc.setBytes(&idxP, length: MemoryLayout<IndexerParams>.stride, index: 6)
                if let effSplit {
                    let midPools = ((slot.position + 1) >> poolShift) - windowTokens
                    enc.setComputePipelineState(effSplit.score)
                    enc.dispatchThreadgroups(MTLSize(width: max(1, (midPools + 63) / 64), height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(effSplit.select)
                } else {
                    enc.setComputePipelineState(psoIndexerScoreSelect)
                }
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            let qAbsRow = qAbsScratch.slice(byteOffset: b * qHeads * kvLoraRank * 2, count: qHeads * kvLoraRank, dtype: .f16)
            let uLatRow = uLatScratch.slice(byteOffset: b * qHeads * kvLoraRank * 2, count: qHeads * kvLoraRank, dtype: .f16)
            let kvCode = Self.kvModeCode(slot.kv.k.mode)
            var attnP = DSASparseAttnParams(
                T: 1, pos0: UInt32(slot.position), qHeads: UInt32(qHeads), headDim: UInt32(nopeDim + ropeDim),
                kvLoraRank: UInt32(kvLoraRank), topK: UInt32(effTopK), windowTokens: UInt32(windowTokens),
                listStride: UInt32(listStride), kvMode: kvCode, ropeDim: UInt32(ropeDim),
                qRopeOffset: UInt32(qRopeOffset), qFullStride: UInt32(callQFullStride),
                hasSink: mlaW.attnSink != nil ? 1 : 0, softmaxScale: softScale
            )
            let useSplit = (split != nil && kvCode <= 2)
            enc.setComputePipelineState(useSplit ? split!.attend : psoDSASparseAttn)
            qAbsRow.bind(enc, index: 0)
            slot.kv.k.pool.bindTo(enc, index: 1)
            slot.kv.v.pool.bindTo(enc, index: 2)
            slot.kv.k.table.bind(enc, index: 3)
            slot.kv.v.table.bind(enc, index: 4)
            listRow.bind(enc, index: 5)
            (useSplit ? partOScratch : uLatRow).bind(enc, index: 6)
            enc.setBytes(&attnP, length: MemoryLayout<DSASparseAttnParams>.stride, index: 7)
            if useSplit, let split {
                partMLScratch.bind(enc, index: 8)
                qFullRow.bind(enc, index: 9)
                kPe.bind(enc, index: 10)
                sinkTensor.bind(enc, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: qHeads / 16, depth: attnSplits),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                var nSplitU = UInt32(attnSplits)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(split.combine)
                partOScratch.bind(enc, index: 0)
                partMLScratch.bind(enc, index: 1)
                uLatRow.bind(enc, index: 2)
                enc.setBytes(&attnP, length: MemoryLayout<DSASparseAttnParams>.stride, index: 3)
                enc.setBytes(&nSplitU, length: 4, index: 4)
                sinkTensor.bind(enc, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                if b + 1 < B { enc.memoryBarrier(scope: .buffers) }
            } else {
                qFullRow.bind(enc, index: 8)
                kPe.bind(enc, index: 9)
                sinkTensor.bind(enc, index: 10)
                enc.dispatchThreadgroups(MTLSize(width: qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.memoryBarrier(scope: .buffers)

        let projDst = needsPostEpilogue ? yVhaScratch : attOut
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: projDst, absP: &absP,
                     absorbMMA: absorbMMA, rows: B, mTiles: mTiles)

        if needsPostEpilogue {
            enc.memoryBarrier(scope: .buffers)
            let gateCode: UInt32 = switch outputGateMode {
            case .headWiseFromHidden: 2
            case .elementWiseFromHidden, .elementWiseFromQLora: 1
            case .none: hasAnyOGate ? 1 : 0
            }
            let gateSrc: QWTensorValue
            let resolvedGateStride: Int
            let resolvedGateOffset: Int
            if effOGateInQKV {
                gateSrc = mlaLatentScratch
                resolvedGateStride = callLatStride
                resolvedGateOffset = qLoraRank + kvLoraRank + ropeDim + effIdxDim
            } else if extGateMode {
                gateSrc = oGateScratch
                resolvedGateStride = extGateCols
                resolvedGateOffset = 0
            } else {
                gateSrc = mlaQFullScratch
                resolvedGateStride = callQFullStride
                resolvedGateOffset = qHeads * (nopeDim + ropeDim)
            }
            var vgP = VHAGateParams(
                T: UInt32(B), qHeads: UInt32(qHeads), vHeadDim: UInt32(vHeadDim),
                gateStride: UInt32(resolvedGateStride), oGateOffset: UInt32(resolvedGateOffset),
                hasVHA: mlaW.vhaMix != nil ? 1 : 0, hasOGate: gateCode
            )
            enc.setComputePipelineState(psoVHAGateRot)
            yVhaScratch.bind(enc, index: 0)
            gateSrc.bind(enc, index: 1)
            (mlaW.vhaMix ?? yVhaScratch).bind(enc, index: 2)
            attOut.bind(enc, index: 3)
            enc.setBytes(&vgP, length: MemoryLayout<VHAGateParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: vHeadDim / 32, height: B, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    private func encodeNoPEDecodeBatch(
        _ enc: any MTLComputeCommandEncoder,
        ln1: QWTensorValue,
        mlaW: LayerWeights,
        slots: [DecodeBatchSlot],
        attOut: QWTensorValue,
        qFullPrecomputed: Bool,
        encodeBatchProj: ((_ enc: any MTLComputeCommandEncoder, _ x: QWTensorValue, _ weight: QWWeightView, _ y16: QWTensorValue, _ rows: Int) throws -> Void)?
    ) throws {
        let B = slots.count
        let H = geometry.hidden
        let callLatStride = latentStride
        let callQFullStride = qFullStride
        for b in 0..<B {
            let slot = slots[b]
            let ls = slot.state
            let latRow = mlaLatentScratch.slice(byteOffset: b * callLatStride * 2, count: callLatStride, dtype: .f16)
            let ln1Row = ln1.slice(byteOffset: b * H * 2, count: H, dtype: .f16)
            let qResRow = qLoraRank > 0
                ? qResidScratch.slice(byteOffset: b * qLoraRank * 2, count: qLoraRank, dtype: .f16)
                : qResidScratch
            let idxWRow = idxWeightsScratch.slice(byteOffset: b * max(1, idxHeads) * 4, count: max(1, idxHeads), dtype: .f32)
            var prepP = LatentPrepParams(
                T: 1, pos0: UInt32(slot.position), hidden: UInt32(H),
                qLoraRank: UInt32(qLoraRank), kvLoraRank: UInt32(kvLoraRank),
                idxDim: UInt32(idxDim), idxHeads: UInt32(idxHeads),
                ringMask: UInt32(ringCap - 1), rmsEps: geometry.rmsEps,
                kNormEps: 1e-6, kvMode: Self.kvModeCode(slot.kv.k.mode)
            )
            enc.setComputePipelineState(psoLatentPrep)
            latRow.bind(enc, index: 0)
            ln1Row.bind(enc, index: 1)
            mlaW.qANormW.bind(enc, index: 2)
            mlaW.kvANormW.bind(enc, index: 3)
            mlaW.kNormW.bind(enc, index: 4)
            mlaW.kNormB.bind(enc, index: 5)
            mlaW.wGateWeights.bind(enc, index: 6)
            qResRow.bind(enc, index: 7)
            slot.kv.k.pool.bindTo(enc, index: 8)
            slot.kv.v.pool.bindTo(enc, index: 9)
            slot.kv.k.table.bind(enc, index: 10)
            slot.kv.v.table.bind(enc, index: 11)
            ls.kPending.bind(enc, index: 12)
            ls.gatePending.bind(enc, index: 13)
            idxWRow.bind(enc, index: 14)
            enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 15)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoGateWeightsProj)
            ln1Row.bind(enc, index: 0)
            mlaW.wGateWeights.bind(enc, index: 1)
            ls.gatePending.bind(enc, index: 2)
            idxWRow.bind(enc, index: 3)
            enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (idxDim + idxHeads + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            let numBlocks = ((slot.position + 1) / 4) - (slot.position / 4)
            if numBlocks > 0, let ape = mlaW.ape {
                enc.memoryBarrier(scope: .buffers)
                var sB = UInt32(slot.position / 4), nB = UInt32(numBlocks), dU = UInt32(idxDim), rM = UInt32(ringCap - 1)
                enc.setComputePipelineState(psoPoolKeys)
                ls.kPending.bind(enc, index: 0)
                ls.gatePending.bind(enc, index: 1)
                ape.bind(enc, index: 2)
                ls.pooledKeys.bind(enc, index: 3)
                enc.setBytes(&sB, length: 4, index: 4)
                enc.setBytes(&nB, length: 4, index: 5)
                enc.setBytes(&dU, length: 4, index: 6)
                enc.setBytes(&rM, length: 4, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: min(128, idxDim), height: 1, depth: 1))
            }
        }
        enc.memoryBarrier(scope: .buffers)

        if !qFullPrecomputed {
            let qBIn = isDirectQ
                ? ln1.slice(byteOffset: 0, count: B * H, dtype: .f16)
                : qResidScratch.slice(byteOffset: 0, count: B * qLoraRank, dtype: .f16)
            let qFullRows = mlaQFullScratch.slice(byteOffset: 0, count: B * callQFullStride, dtype: .f16)
            if let proj = encodeBatchProj {
                try proj(enc, qBIn, mlaW.qBProj, qFullRows, B)
            } else {
                try encodeMLAWeightViewH16(enc, x: qBIn, weight: mlaW.qBProj, y16: qFullRows, rows: B)
            }
            enc.memoryBarrier(scope: .buffers)
        }

        let isInc: UInt32 = mlaW.kvBProj.bits == 16 ? 3
            : (mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0))
        var absP = AbsorbParams(
            T: UInt32(B), qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim), vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank), qFullStride: UInt32(callQFullStride),
            bits: UInt32(mlaW.kvBProj.bits), group: UInt32(mlaW.kvBProj.groupSize), isIncumbent: isInc
        )
        let mTiles = (B + 7) / 8
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: B, mTiles: mTiles)
        enc.memoryBarrier(scope: .buffers)

        for b in 0..<B {
            let slot = slots[b]
            let ls = slot.state
            let qFullRow = mlaQFullScratch.slice(byteOffset: b * callQFullStride * 2, count: callQFullStride, dtype: .f16)
            let idxWRow = idxWeightsScratch.slice(byteOffset: b * max(1, idxHeads) * 4, count: max(1, idxHeads), dtype: .f32)
            let listRow = blockListScratch.slice(byteOffset: b * listStride * 4, count: listStride, dtype: .i32)
            let countRow = blockCountScratch.slice(byteOffset: b * 4, count: 1, dtype: .i32)
            if ls.hasIndexer, idxHeads > 0 {
                var idxP = IndexerParams(
                    T: 1, pos0: UInt32(slot.position), qChannels: UInt32(qChannels),
                    idxHeads: UInt32(idxHeads), idxDim: UInt32(idxDim),
                    topKPools: UInt32(topKPools), listStride: UInt32(listStride), scoreStride: UInt32(maxBlocks)
                )
                qFullRow.bind(enc, index: 0)
                idxWRow.bind(enc, index: 1)
                ls.pooledKeys.bind(enc, index: 2)
                scoreScratch.bind(enc, index: 3)
                listRow.bind(enc, index: 4)
                countRow.bind(enc, index: 5)
                enc.setBytes(&idxP, length: MemoryLayout<IndexerParams>.stride, index: 6)
                enc.setComputePipelineState(psoIndexerScoreSelect)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
            let qAbsRow = qAbsScratch.slice(byteOffset: b * qHeads * kvLoraRank * 2, count: qHeads * kvLoraRank, dtype: .f16)
            let uLatRow = uLatScratch.slice(byteOffset: b * qHeads * kvLoraRank * 2, count: qHeads * kvLoraRank, dtype: .f16)
            var attnP = SparseAttnParams(
                T: 1, pos0: UInt32(slot.position), qHeads: UInt32(qHeads), headDim: UInt32(nopeDim),
                kvLoraRank: UInt32(kvLoraRank), topKPools: UInt32(topKPools), listStride: UInt32(listStride),
                kvMode: Self.kvModeCode(slot.kv.k.mode)
            )
            enc.setComputePipelineState(psoSparseAttn)
            qAbsRow.bind(enc, index: 0)
            slot.kv.k.pool.bindTo(enc, index: 1)
            slot.kv.v.pool.bindTo(enc, index: 2)
            slot.kv.k.table.bind(enc, index: 3)
            slot.kv.v.table.bind(enc, index: 4)
            listRow.bind(enc, index: 5)
            uLatRow.bind(enc, index: 6)
            enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: attOut, absP: &absP,
                     absorbMMA: absorbMMA, rows: B, mTiles: mTiles)
    }
}

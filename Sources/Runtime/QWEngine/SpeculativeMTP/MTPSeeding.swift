// MTPSeeding.swift — QWEngineDriver MTPSeeding family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    func prepareMTPSeeding(_ session: QWEngineSession, count: Int) throws -> Int {
        guard let mtpCache else { return 0 }
        guard !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              let handle = session.mtpKV, session.trunkHidden != nil,
              session.position == 0 || session.trunkHiddenValid else {
            throw QWEngineDriverError.state("MTP seeding requires a live session and previous trunk hidden")
        }
        try mtpCache.validate(handle)
        let rows = mtpCache.written(handle)
        guard rows == session.roundRecord.mtpPhysicalRows else {
            throw QWEngineDriverError.state("MTP seeding cursor mismatch")
        }
        let end = rows + QWMTPPromptSeeder.pairCount(position: session.position, count: count)
        if end > handle.tokenCapacity {
            try mtpCache.grow(handle, to: end)
            session.mtpKVViews = [try mtpCache.layerViews(handle, layer: 0)]
        }
        try mtpCache.prepareWrites(handle, positions: rows..<end)
        return rows
    }

    /// The trunk encoder has ended. Seeding uses a serial encoder even when
    /// ordinary decode uses explicit-barrier concurrent dispatch.
    func encodeMTPSeeding(_ cb: any MTLCommandBuffer, session: QWEngineSession,
                                  embeddings: QWTensorValue, hidden: QWTensorValue,
                                  count: Int, cacheRows: Int,
                                  dflashCapture: QWDFlashConfig.PrefillCaptureSlice? = nil,
                                  dsparkCapture: QWDSparkDrafter.PrefillCaptureSlice? = nil) throws {
        if isWeightFreeSpeculation { return }
        if dsparkDrafter != nil {
            if let dsparkDrafter, let dsSlot = session.dsparkSlot {
                if let cap = dsparkCapture {
                    try dsparkDrafter.encodeContextIngestCB(
                        cb, driver: self, slot: dsSlot, pos0: cap.ctxStartPos, rows: cap.captureRows)
                } else if count == 1 {
                    try dsparkDrafter.encodeContextIngestCB(
                        cb, driver: self, slot: dsSlot, pos0: session.position, rows: 1)
                }
            }
            return
        }
        if dflashDrafter != nil {
            if let dflashDrafter, let dSlot = session.dflashSlot {
                if let cap = dflashCapture {
                    try dflashDrafter.encodeContextUpdate(
                        cb, slot: dSlot, startPosition: cap.ctxStartPos, totalCount: cap.captureRows)
                } else if count == 1 {
                    try dflashDrafter.encodeContextUpdate(
                        cb, slot: dSlot, startPosition: session.position, totalCount: 1)
                }
            }
            return
        }
        if let echoDrafter, let previous = session.trunkHidden, let views = session.mtpKVViews.first {
            try encodeEchoSeeding(cb, echoDrafter: echoDrafter, embeddings: embeddings,
                                  hidden: hidden, previous: previous,
                                  position: session.position, count: count,
                                  cacheRows: cacheRows, views: views)
            return
        }
        guard let mtpDraftEncoder, let mtpWeights, let previous = session.trunkHidden,
              let views = session.mtpKVViews.first else {
            throw QWEngineDriverError.state("MTP prompt seeding infrastructure is absent")
        }
        let pos = session.position
        let pairs = QWMTPPromptSeeder.pairCount(position: pos, count: count)
        if pairs >= 2 && count <= psc.maxChunk && geometry.hidden % 256 == 0 {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("MTP batched seed encoder")
            }
            hal.touchKeepAlive(enc)
            do {
                let seedMRoPE = mlaOperator.flatMap {
                    session.mropeSlice(pos: pos, count: pairs, mropeLayout: $0.mropeLayout)
                }
                let tStart = pos == 0 ? 1 : 0
                let seedTurnTable = session.mropeTurnTableSlice(
                    pos: pos + tStart, count: pairs, rotaryDims: geometry.rotaryDims
                )
                try encodeBatchedMTPPairSeed(
                    enc, mtpWeights: mtpWeights, embeddings: embeddings, hidden: hidden,
                    previous: previous, pos: pos, tStart: tStart,
                    pairs: pairs, cacheRows: cacheRows, views: views,
                    mropePositions: seedMRoPE, mropeDelta: session.roundRecord.mropeDelta,
                    mropeTurnTable: seedTurnTable)
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWEngineDriverError.state("MTP hidden retention encoder")
            }
            let H = geometry.hidden
            blit.copy(from: hidden.region.buffer,
                      sourceOffset: hidden.region.base + hidden.offset + (count - 1) * H * 2,
                      to: previous.region.buffer,
                      destinationOffset: previous.region.base + previous.offset,
                      size: previous.bytes)
            blit.endEncoding()
            return
        }
        try QWMTPPromptSeeder.encodeChunk(cb, draft: mtpDraftEncoder, embeddings: embeddings,
                                          trunkHidden: hidden, previousHidden: previous,
                                          position: session.position, count: count, cacheRows: cacheRows, kv: views,
                                          mropeDelta: session.roundRecord.mropeDelta)
    }

    func encodeBatchedMTPPairSeed(
        _ enc: any MTLComputeCommandEncoder,
        mtpWeights: QWMTPDraftWeights,
        embeddings: QWTensorValue,
        hidden: QWTensorValue,
        previous: QWTensorValue,
        pos: Int,
        tStart: Int,
        pairs: Int,
        cacheRows: Int,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        mropePositions: [Int32]? = nil,
        mropeDelta: Int = 0,
        mropeTurnTable: QWTensorValue? = nil
    ) throws {
        let H = geometry.hidden
        let concatBuf = prefixRows(psc.guRaw16, rows: pairs, cols: 2 * H, dt: .f16)
        let fcOutBuf = prefixRows(psc.o16, rows: pairs, cols: H, dt: .f16)
        let ln1Buf = prefixRows(psc.dn16, rows: pairs, cols: H, dt: .f16)
        if let gn = mtpWeights.gatedNorms {
            for p in 0..<pairs {
                let embRow = stridedRow(embeddings, row: tStart + p, cols: H, dt: .f16)
                let hidRow = (tStart == 0 && p == 0)
                    ? prefixRows(previous, rows: 1, cols: H, dt: .f16)
                    : stridedRow(hidden, row: tStart + p - 1, cols: H, dt: .f16)
                let eOut = concatBuf.slice(byteOffset: p * 2 * H * 2, count: H, dtype: .f16)
                let hOut = concatBuf.slice(byteOffset: (p * 2 * H + H) * 2, count: H, dtype: .f16)
                try QWNorm.encodeGatedRMSNorm(
                    enc, hal: hal, x: embRow, weights: gn.enorm,
                    out: eOut, hidden: H, applyFwht32: mtpWeights.fc.isRotated)
                try QWNorm.encodeGatedRMSNorm(
                    enc, hal: hal, x: hidRow, weights: gn.hnorm,
                    out: hOut, hidden: H, applyFwht32: mtpWeights.fc.isRotated)
            }
            try gemmBatchedH16(enc, m: pairs, n: H, k: 2 * H, x: concatBuf, wQ4: mtpWeights.fc, y16: fcOutBuf)
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: hal, x: fcOutBuf, weights: gn.input,
                out: ln1Buf, hidden: H, applyFwht32: mtpWeights.qkvProj.isRotated)
        } else {
            try QWNorm.encodeMTPPreFcNorm(enc, hal: hal,
                                          embeddings: embeddings, hidden: hidden,
                                          previous: previous,
                                          gammaEmb: mtpWeights.preFcNormEmbedding,
                                          gammaHid: mtpWeights.preFcNormHidden,
                                          out: concatBuf,
                                          tStart: tStart, pairs: pairs,
                                          geometry: geometry)
            try gemmBatchedH16(enc, m: pairs, n: H, k: 2 * H, x: concatBuf, wQ4: mtpWeights.fc, y16: fcOutBuf)
            try encodeRMSNormFast(enc, x: fcOutBuf, gamma: mtpWeights.inputNorm, out: ln1Buf, rows: pairs)
        }
        if let mla = mlaOperator, let mlaW = mtpWeights.mla {
            let latentN = mtpWeights.qkvProj.n
            let latentBuf = prefixRows(mla.mlaLatentScratch, rows: pairs, cols: latentN, dt: .f16)
            try gemmBatchedH16(enc, m: pairs, n: latentN, k: H, x: ln1Buf, wQ4: mtpWeights.qkvProj, y16: latentBuf)
            mla.encodeLatentPrepAndPool(
                enc, ln1: ln1Buf, mlaW: mlaW,
                kv: views, fullAttnLayer: attnLayers.count,
                pos0: cacheRows, rows: pairs, mropePositions: mropePositions)
        } else {
            guard let wq = mtpWeights.wq, let wkTiled = mtpWeights.wkTiled else {
                throw QWEngineDriverError.state("MTP seeding Q/K norms are missing")
            }
            let qkvBuf = prefixRows(psc.attnY16, rows: pairs, cols: geometry.attnTotalN, dt: .f16)
            let qOutBuf = prefixRows(psc.attQR16, rows: pairs, cols: geometry.qProjOut, dt: .f16)
            let kOutBuf = prefixRows(psc.attKR16, rows: pairs, cols: geometry.kvChannels, dt: .f16)
            let vOutBuf = prefixRows(psc.attV16, rows: pairs, cols: geometry.kvChannels, dt: .f16)
            try gemmBatchedH16(enc, m: pairs, n: geometry.attnTotalN, k: H, x: ln1Buf, wQ4: mtpWeights.qkvProj, y16: qkvBuf)
            try ropeApply.encodeExtractQK(enc, hal: hal, attnY: qkvBuf,
                                          qOut: qOutBuf, kOut: kOutBuf, vOut: vOutBuf,
                                          wq: wq, wk: wkTiled,
                                          qkvBias: mtpWeights.qkvBias,
                                          positionStart: max(0, pos + tStart + mropeDelta), count: pairs,
                                          customTable: mropeTurnTable)
            encodePagedKVWrite(enc, kv: views, k: kOutBuf, v: vOutBuf, startPos: cacheRows, t: pairs)
        }
    }

    private func encodeEchoSeeding(_ cb: any MTLCommandBuffer,
                                   echoDrafter: QWEchoDraftEncoder,
                                   embeddings: QWTensorValue,
                                   hidden: QWTensorValue,
                                   previous: QWTensorValue,
                                   position: Int,
                                   count: Int,
                                   cacheRows: Int,
                                   views: (k: QWKVLayerView, v: QWKVLayerView)) throws {
        let h = geometry.hidden
        let streamElems = echoDrafter.streams * h
        let streamBytes = streamElems * 4
        let pairs = QWMTPPromptSeeder.pairCount(position: position, count: count)
        if pairs > 0 {
            let maxSeed = echoDrafter.maxSeedRows
            if position > 0 && count > 1 {
                let chunk0 = min(pairs, maxSeed)
                guard let prepBlit = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("ECHO seed stage blit encoder")
                }
                prepBlit.copy(from: previous.region.buffer,
                              sourceOffset: previous.region.base + previous.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset,
                              size: streamBytes)
                prepBlit.copy(from: hidden.region.buffer,
                              sourceOffset: hidden.region.base + hidden.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset + streamBytes,
                              size: (chunk0 - 1) * streamBytes)
                prepBlit.endEncoding()
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("ECHO seed compute encoder")
            }
            do {
                hal.touchKeepAlive(enc)
                enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
                var r0 = 0
                while r0 < pairs {
                    let chunk = min(maxSeed, pairs - r0)
                    let prevSlice: QWTensorValue
                    let embSlice: QWTensorValue
                    if position == 0 {
                        prevSlice = hidden.slice(byteOffset: r0 * streamBytes, count: chunk * streamElems, dtype: .f32)
                        embSlice = embeddings.slice(byteOffset: (r0 + 1) * h * 2, count: chunk * h, dtype: .f16)
                    } else if count == 1 {
                        prevSlice = previous
                        embSlice = embeddings.slice(byteOffset: 0, count: h, dtype: .f16)
                    } else if r0 == 0 {
                        prevSlice = echoDrafter.prevStreamBuf.slice(byteOffset: 0, count: chunk * streamElems, dtype: .f32)
                        embSlice = embeddings.slice(byteOffset: 0, count: chunk * h, dtype: .f16)
                    } else {
                        prevSlice = hidden.slice(byteOffset: (r0 - 1) * streamBytes, count: chunk * streamElems, dtype: .f32)
                        embSlice = embeddings.slice(byteOffset: r0 * h * 2, count: chunk * h, dtype: .f16)
                    }
                    try echoDrafter.encodeSeedChunk(enc, driver: self, prevStream: prevSlice,
                                                    embeddings: embSlice, kv: views,
                                                    cacheRows: cacheRows + r0, rows: chunk)
                    enc.memoryBarrier(scope: .buffers)
                    r0 += chunk
                }
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
        }
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw QWEngineDriverError.state("ECHO stream retention blit encoder")
        }
        blit.copy(from: hidden.region.buffer,
                  sourceOffset: hidden.region.base + hidden.offset + (count - 1) * streamBytes,
                  to: previous.region.buffer,
                  destinationOffset: previous.region.base + previous.offset,
                  size: streamBytes)
        blit.endEncoding()
    }

    func commitMTPSeeding(_ session: QWEngineSession, position: Int, count: Int, cacheRows: Int) {
        session.roundRecord.consumedCount = position + count
        session.roundRecord.absolutePosition = position + count
        if verifyWindow != nil {
            session.trunkHiddenValid = true
        }
        if dflashDrafter != nil, session.dflashSlot != nil {
            session.dflashCacheOffset = position + count
        }
        if dsparkDrafter != nil, session.dsparkSlot != nil {
            session.dsparkCacheOffset = position + count
        }
        guard let mtpCache, let handle = session.mtpKV else { return }
        let rows = cacheRows + QWMTPPromptSeeder.pairCount(position: position, count: count)
        mtpCache.markWritten(handle, upTo: rows)
        session.roundRecord.mtpPhysicalRows = rows
        session.trunkHiddenValid = true
    }

    /// One greedy decode step: token in → argmax token out. Full 64-layer
    /// cadence, ONE command buffer (spec-21: CPU must never be the tail;
    /// end-before-wait law: hal.end commits, THEN waitUntilCompleted).
}

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
import QWState

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
                                  count: Int, cacheRows: Int) throws {
        if let dflashDrafter, let dSlot = session.dflashSlot {
            try dflashDrafter.encodeContextUpdate(cb, slot: dSlot, startPosition: session.position, totalCount: count)
            return
        }
        guard let mtpDraftEncoder, let mtpWeights, let previous = session.trunkHidden,
              let views = session.mtpKVViews.first else {
            throw QWEngineDriverError.state("MTP prompt seeding infrastructure is absent")
        }
        let pos = session.position
        let pairs = QWMTPPromptSeeder.pairCount(position: pos, count: count)
        if pairs >= 16 && count <= psc.maxChunk {
            let H = geometry.hidden
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("MTP batched seed encoder")
            }
            do {
                let tStart = (pos == 0) ? 1 : 0
                let concatBuf = prefixRows(psc.guRaw16, rows: pairs, cols: 2 * H, dt: .f16)
                try QWNorm.encodeMTPPreFcNorm(enc, hal: hal,
                                              embeddings: embeddings, hidden: hidden,
                                              previous: previous,
                                              gammaEmb: mtpWeights.preFcNormEmbedding,
                                              gammaHid: mtpWeights.preFcNormHidden,
                                              out: concatBuf,
                                              tStart: tStart, pairs: pairs,
                                              geometry: geometry)
                let fcOutBuf = prefixRows(psc.o16, rows: pairs, cols: H, dt: .f16)
                let ln1Buf = prefixRows(psc.dn16, rows: pairs, cols: H, dt: .f16)
                gemmBatchedH16(enc, m: pairs, n: H, k: 2 * H, x: concatBuf, wQ4: mtpWeights.fc, y16: fcOutBuf)
                try encodeRMSNormFast(enc, x: fcOutBuf, gamma: mtpWeights.inputNorm, out: ln1Buf, rows: pairs)
                if let mla = mlaOperator, let mlaW = mtpWeights.mla {
                    let latentN = geometry.qLoraRank + geometry.kvLoraRank + geometry.indexerHeadDim
                    let latentBuf = prefixRows(mla.mlaLatentScratch, rows: pairs, cols: latentN, dt: .f16)
                    gemmBatchedH16(enc, m: pairs, n: latentN, k: H, x: ln1Buf, wQ4: mtpWeights.qkvProj, y16: latentBuf)
                    mla.encodeLatentPrepAndPool(
                        enc, ln1: ln1Buf, mlaW: mlaW,
                        kv: views, fullAttnLayer: attnLayers.count,
                        pos0: cacheRows, rows: pairs)
                } else {
                    guard let wq = mtpWeights.wq, let wkTiled = mtpWeights.wkTiled else {
                        throw QWEngineDriverError.state("MTP seeding Q/K norms are missing")
                    }
                    let qkvBuf = prefixRows(psc.attnY16, rows: pairs, cols: geometry.attnTotalN, dt: .f16)
                    let qOutBuf = prefixRows(psc.attQR16, rows: pairs, cols: geometry.qProjOut, dt: .f16)
                    let kOutBuf = prefixRows(psc.attKR16, rows: pairs, cols: geometry.kvChannels, dt: .f16)
                    let vOutBuf = prefixRows(psc.attV16, rows: pairs, cols: geometry.kvChannels, dt: .f16)
                    gemmBatchedH16(enc, m: pairs, n: geometry.attnTotalN, k: H, x: ln1Buf, wQ4: mtpWeights.qkvProj, y16: qkvBuf)
                    try ropeApply.encodeExtractQK(enc, hal: hal, attnY: qkvBuf,
                                                  qOut: qOutBuf, kOut: kOutBuf, vOut: vOutBuf,
                                                  wq: wq, wk: wkTiled,
                                                  positionStart: pos + tStart, count: pairs)
                    encodePagedKVWrite(enc, kv: views, k: kOutBuf, v: vOutBuf, startPos: cacheRows, t: pairs)
                }
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWEngineDriverError.state("MTP hidden retention encoder")
            }
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
                                          position: session.position, count: count, cacheRows: cacheRows, kv: views)
    }

    func commitMTPSeeding(_ session: QWEngineSession, position: Int, count: Int, cacheRows: Int) {
        guard let mtpCache, let handle = session.mtpKV else { return }
        let rows = cacheRows + QWMTPPromptSeeder.pairCount(position: position, count: count)
        mtpCache.markWritten(handle, upTo: rows)
        session.roundRecord.mtpPhysicalRows = rows
        session.roundRecord.consumedCount = position + count
        session.roundRecord.absolutePosition = position + count
        session.trunkHiddenValid = true
        if dflashDrafter != nil {
            session.dflashCacheOffset = position + count
        }
    }

    /// One greedy decode step: token in → argmax token out. Full 64-layer
    /// cadence, ONE command buffer (spec-21: CPU must never be the tail;
    /// end-before-wait law: hal.end commits, THEN waitUntilCompleted).
}

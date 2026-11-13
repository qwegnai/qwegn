// QWEchoDraftEncoder.swift — Zero-copy shared-last-layer ECHO speculative drafter.
// Aliases trunk Layer N-1 MLA-DSA + Latent-MoE + mHC weights while applying unique
// ECHO entry projections (`enorm`, `hnorm`, `e_proj`, `h_proj`), `hc_head`, and `MultiMax`.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWKV
import QWMLP
import QWNorm

final class QWEchoDraftEncoder: @unchecked Sendable {
    let geometry: QWModelGeometry
    let weights: QWEngineDriver.EchoDraftWeights
    let layer: QWEngineDriver.LayerW
    let mlaOperator: QWMLAAttnOperator
    let mlaLayerIndex: Int
    let maxSeedRows: Int
    let streams: Int
    let region: QWMRegion

    let embedBuf: QWTensorValue
    let eNormed16: QWTensorValue
    let eOut16: QWTensorValue
    let prevStreamBuf: QWTensorValue
    let prev16: QWTensorValue
    let hNormed16: QWTensorValue
    let hOut16: QWTensorValue
    let streamScratch: QWTensorValue
    let draftPing: QWTensorValue
    let draftPong: QWTensorValue
    let controlsScratch: QWTensorValue
    let ln1: QWTensorValue
    let attOut: QWTensorValue
    let oOut16: QWTensorValue
    let ln2: QWTensorValue
    let mlpInter: QWTensorValue
    let dn32: QWTensorValue
    let dn16: QWTensorValue
    let postNorm: QWTensorValue
    let gsHead: QWTensorValue
    let draftLogits: QWTensorValue
    let greedySelector: QWMTPGreedySelector
    let partials: QWMRegion
    let moe: QWEngineDriver.MoEScratch?

    private let psoEchoStreamInit: any MTLComputePipelineState

    init(geometry: QWModelGeometry,
         hal: QWHal,
         weights: QWEngineDriver.EchoDraftWeights,
         trunkLayer: QWEngineDriver.LayerW,
         mlaOperator: QWMLAAttnOperator,
         mlaLayerIndex: Int,
         maxSeedRows: Int) throws {
        guard let mhcCfg = geometry.mhcConfig,
              trunkLayer.mhc != nil,
              let attn = trunkLayer.attn,
              attn.mla != nil else {
            throw QWEngineDriverError.state("ECHO drafter requires mHC + MLA-DSA on the shared trunk layer")
        }
        self.geometry = geometry
        self.weights = weights
        self.mlaOperator = mlaOperator
        self.mlaLayerIndex = mlaLayerIndex
        self.maxSeedRows = max(16, min(512, maxSeedRows))
        self.streams = mhcCfg.streams

        let aliasedMoE: QWEngineDriver.MoEW?
        if let tMoe = trunkLayer.moe {
            aliasedMoE = QWEngineDriver.MoEW(
                router: tMoe.router,
                gateUp: tMoe.gateUp,
                down: tMoe.down,
                numExperts: tMoe.numExperts,
                expertsPerTok: tMoe.expertsPerTok,
                sharedGateUp: tMoe.sharedGateUp,
                sharedDown: tMoe.sharedDown,
                sharedGateScalar: tMoe.sharedGateScalar,
                routerBias: weights.routerBias ?? tMoe.routerBias,
                routedScaleParam: tMoe.routedScaleParam,
                splitFeatureRouting: tMoe.splitFeatureRouting,
                latent: tMoe.latent,
                coldGateUp: tMoe.coldGateUp,
                coldDown: tMoe.coldDown,
                coldStreamer: tMoe.coldStreamer
            )
        } else {
            aliasedMoE = nil
        }
        self.layer = QWEngineDriver.LayerW(
            ln1: trunkLayer.ln1,
            ln2: trunkLayer.ln2,
            gu: trunkLayer.gu,
            down: trunkLayer.down,
            kind: trunkLayer.kind,
            gdn: nil,
            attn: attn,
            moe: aliasedMoE,
            mhc: trunkLayer.mhc,
            hca: nil
        )
        self.psoEchoStreamInit = try hal.pipeline("qwg_echo_stream_init_f16")

        let h = geometry.hidden
        let c = self.streams
        let rows = self.maxSeedRows
        let vocab = geometry.vocab
        let ctrlWidth = c + c * c
        let vCols = max(attn.o.k, geometry.qHeads * geometry.vHeadDim)
        let denseInter = max(128, trunkLayer.down.k)
        let partialBytes = max(4096, 2 * ((vocab + 31) / 32) * 4 + 512)
        let moeBytes = aliasedMoE != nil
            ? QWEngineDriver.moeScratchAllocationBytes(rows: 1, geometry: geometry, useGroupedNAX: false)
            : 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        let estimatedBytes = align256(rows * h * 2) * 4
            + align256(rows * c * h * 4) * 2
            + align256(rows * c * h * 2) * 3
            + align256(c * h * 4) * 2
            + align256(rows * ctrlWidth * 4)
            + align256(vCols * 2)
            + align256(h * 2) * 5
            + align256(denseInter * 2)
            + align256(h * 4) * 2
            + align256(vocab * 4)
            + partialBytes * 3
            + moeBytes
            + (2 << 20)

        guard let buf = hal.device.makeBuffer(length: estimatedBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "echo-draft", asked: estimatedBytes, free: 0)
        }
        buf.label = "echo-draft"
        memset(buf.contents(), 0, estimatedBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: estimatedBytes)
        self.region = reg
        let sub = QWSubAllocator(region: reg)
        func tv(_ n: Int, _ dt: QWDType = .f16) throws -> QWTensorValue {
            try sub.allocate(QWTensorShape([max(1, n)], dt))
        }

        self.embedBuf = try tv(rows * h, .f16)
        self.eNormed16 = try tv(rows * h, .f16)
        self.eOut16 = try tv(rows * h, .f16)
        self.prevStreamBuf = try tv(rows * c * h, .f32)
        self.prev16 = try tv(rows * c * h, .f16)
        self.hNormed16 = try tv(rows * c * h, .f16)
        self.hOut16 = try tv(rows * c * h, .f16)
        self.streamScratch = try tv(rows * c * h, .f32)
        self.draftPing = try tv(c * h, .f32)
        self.draftPong = try tv(c * h, .f32)
        self.controlsScratch = try tv(rows * ctrlWidth, .f32)
        self.ln1 = try tv(rows * h, .f16)
        self.attOut = try tv(vCols, .f16)
        self.oOut16 = try tv(h, .f16)
        self.ln2 = try tv(h, .f16)
        self.mlpInter = try tv(denseInter, .f16)
        self.dn32 = try tv(h, .f32)
        self.dn16 = try tv(h, .f16)
        self.postNorm = try tv(h, .f16)
        self.gsHead = try tv(max(1, h / 128), .f32)
        self.draftLogits = try tv(vocab, .f32)
        self.greedySelector = try QWMTPGreedySelector(geometry: geometry, allocator: sub, vocabulary: vocab)
        let pOff = try sub.allocate(bytes: partialBytes)
        self.partials = QWMRegion(buffer: buf, base: pOff, length: partialBytes)
        if aliasedMoE != nil {
            self.moe = try QWEngineDriver.makeMoEScratch(tv, rows: 1, geometry: geometry, useGroupedNAX: false)
        } else {
            self.moe = nil
        }
    }

    func stageEmbeddings(driver: QWEngineDriver, tokens: ArraySlice<Int>) throws -> QWTensorValue {
        let count = tokens.count
        guard count > 0, count <= maxSeedRows else {
            throw QWEngineDriverError.state("ECHO seed token count \(count) exceeds capacity \(maxSeedRows)")
        }
        let h = geometry.hidden
        for (idx, tok) in tokens.enumerated() {
            let dst = embedBuf.slice(byteOffset: idx * h * 2, count: h, dtype: .f16)
            try driver.copyEmbedRow(tok, to: dst)
        }
        return embedBuf.slice(byteOffset: 0, count: count * h, dtype: .f16)
    }

    private func encodeEntryAndLatentPrep(_ enc: any MTLComputeCommandEncoder,
                                          driver: QWEngineDriver,
                                          prevStream: QWTensorValue,
                                          embeddings: QWTensorValue,
                                          outStream: QWTensorValue,
                                          kv: (k: QWKVLayerView, v: QWKVLayerView),
                                          cacheRows: Int,
                                          rows: Int) throws {
        guard let mhc = layer.mhc, let attn = layer.attn, let mlaW = attn.mla else {
            throw QWEngineDriverError.state("ECHO layer weights are incomplete")
        }
        let h = geometry.hidden
        let c = streams
        let streamRows = rows * c
        let streamElems = streamRows * h
        let ctrlWidth = c + c * c

        let prev16Slice = prev16.slice(byteOffset: 0, count: streamElems, dtype: .f16)
        let hNormSlice = hNormed16.slice(byteOffset: 0, count: streamElems, dtype: .f16)
        let hOutSlice = hOut16.slice(byteOffset: 0, count: streamElems, dtype: .f16)
        let eNormSlice = eNormed16.slice(byteOffset: 0, count: rows * h, dtype: .f16)
        let eOutSlice = eOut16.slice(byteOffset: 0, count: rows * h, dtype: .f16)
        let ctrlSlice = controlsScratch.slice(byteOffset: 0, count: rows * ctrlWidth, dtype: .f32)
        let ln1Slice = ln1.slice(byteOffset: 0, count: rows * h, dtype: .f16)

        try QWGemm.encodeCast(enc, hal: driver.hal, x32: prevStream, y16: prev16Slice, count: streamElems)
        enc.memoryBarrier(scope: .buffers)

        try driver.encodeRMSNormFast(enc, x: prev16Slice, gamma: weights.hNorm, out: hNormSlice, rows: streamRows)
        try driver.encodeRMSNormFast(enc, x: embeddings, gamma: weights.eNorm, out: eNormSlice, rows: rows)
        enc.memoryBarrier(scope: .buffers)

        driver.gemmBatchedH16(enc, m: streamRows, n: h, k: h,
                              x: hNormSlice, wQ4: weights.hProj, y16: hOutSlice)
        if rows == 1 {
            try driver.gemvH16T4(enc, n: h, k: h, x: eNormSlice, gs: driver.sc.gs,
                                 wQ4: weights.eProj, y16: eOutSlice)
        } else {
            driver.gemmBatchedH16(enc, m: rows, n: h, k: h,
                                  x: eNormSlice, wQ4: weights.eProj, y16: eOutSlice)
        }
        enc.memoryBarrier(scope: .buffers)

        var hiddenU = UInt32(h)
        enc.setComputePipelineState(psoEchoStreamInit)
        hOutSlice.bind(enc, index: 0)
        eOutSlice.bind(enc, index: 1)
        outStream.bind(enc, index: 2)
        enc.setBytes(&hiddenU, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (h + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        try driver.encodeMHCRouterNorm(enc, stream: outStream, w: mhc.attnW, b: mhc.attnB,
                                       gamma: layer.ln1, controls: ctrlSlice, out: ln1Slice, rows: rows)
        enc.memoryBarrier(scope: .buffers)

        if rows == 1 {
            let latent1 = QWTensorValue(region: mlaOperator.mlaLatentScratch.region,
                                        offset: mlaOperator.mlaLatentScratch.offset,
                                        shape: QWTensorShape([attn.qkvProj.n], .f16))
            try driver.gemvH16T4(enc, n: attn.qkvProj.n, k: h, x: ln1Slice, gs: driver.sc.gs,
                                 wQ4: attn.qkvProj, y16: latent1)
        } else {
            driver.gemmBatchedH16(enc, m: rows, n: attn.qkvProj.n, k: h,
                                  x: ln1Slice, wQ4: attn.qkvProj, y16: mlaOperator.mlaLatentScratch,
                                  yStride: mlaOperator.latentStride)
        }
        enc.memoryBarrier(scope: .buffers)

        mlaOperator.encodeLatentPrepAndPool(enc, ln1: ln1Slice, mlaW: mlaW,
                                            kv: kv, fullAttnLayer: mlaLayerIndex,
                                            pos0: cacheRows, rows: rows)
    }

    func encodeSeedChunk(_ enc: any MTLComputeCommandEncoder,
                         driver: QWEngineDriver,
                         prevStream: QWTensorValue,
                         embeddings: QWTensorValue,
                         kv: (k: QWKVLayerView, v: QWKVLayerView),
                         cacheRows: Int,
                         rows: Int) throws {
        guard rows > 0, rows <= maxSeedRows else {
            throw QWEngineDriverError.state("ECHO seed chunk rows \(rows) out of range 1...\(maxSeedRows)")
        }
        let outStream = streamScratch.slice(byteOffset: 0, count: rows * streams * geometry.hidden, dtype: .f32)
        try encodeEntryAndLatentPrep(enc, driver: driver, prevStream: prevStream, embeddings: embeddings,
                                     outStream: outStream, kv: kv, cacheRows: cacheRows, rows: rows)
    }

    func encodeStep(_ enc: inout any MTLComputeCommandEncoder,
                    driver: QWEngineDriver,
                    prevStream: QWTensorValue,
                    embedding: QWTensorValue,
                    kv: (k: QWKVLayerView, v: QWKVLayerView),
                    cacheRows: Int,
                    outStream: QWTensorValue,
                    writeLogits: Bool,
                    selectGreedy: Bool) throws {
        guard let mhcCfg = geometry.mhcConfig,
              let mhc = layer.mhc,
              let attn = layer.attn,
              let mlaW = attn.mla else {
            throw QWEngineDriverError.state("ECHO layer weights are incomplete")
        }
        let h = geometry.hidden
        let c = streams
        let ctrl1 = controlsScratch.slice(byteOffset: 0, count: c + c * c, dtype: .f32)

        try encodeEntryAndLatentPrep(enc, driver: driver, prevStream: prevStream, embeddings: embedding,
                                     outStream: outStream, kv: kv, cacheRows: cacheRows, rows: 1)
        enc.memoryBarrier(scope: .buffers)

        let qResid1 = QWTensorValue(region: mlaOperator.qResidScratch.region,
                                    offset: mlaOperator.qResidScratch.offset,
                                    shape: QWTensorShape([mlaW.qBProj.k], .f16))
        let qFull1 = QWTensorValue(region: mlaOperator.mlaQFullScratch.region,
                                   offset: mlaOperator.mlaQFullScratch.offset,
                                   shape: QWTensorShape([mlaW.qBProj.n], .f16))
        try driver.gemvH16T4(enc, n: mlaW.qBProj.n, k: mlaW.qBProj.k,
                             x: qResid1, gs: driver.sc.gs, wQ4: mlaW.qBProj, y16: qFull1)
        enc.memoryBarrier(scope: .buffers)

        mlaOperator.encodeScoreSelectAbsorbAttendAndProjUV(
            enc, mlaW: mlaW, kv: kv, attOut: attOut,
            fullAttnLayer: mlaLayerIndex, pos0: cacheRows, rows: 1)
        enc.memoryBarrier(scope: .buffers)

        try driver.gemvH16T4(enc, n: h, k: attn.o.k, x: attOut, gs: driver.sc.gs48,
                             wQ4: attn.o, y16: oOut16)
        enc.memoryBarrier(scope: .buffers)

        try driver.encodeMHCRouterNorm(enc, stream: outStream, w: mhc.ffnW, b: mhc.ffnB,
                                       gamma: layer.ln2, controls: ctrl1, out: ln2,
                                       rows: 1, mergeBranchOut: oOut16)
        enc.memoryBarrier(scope: .buffers)

        if let moeW = layer.moe, let moeSc = self.moe {
            try QWMoEChain.encode(enc: &enc, hal: driver.hal, moe: moeW, s: moeSc,
                                  x16: ln2, dn32: dn32, dn16: dn16,
                                  rows: 1, geometry: geometry, driver: driver)
        } else {
            driver.mlpSiluPairH16(enc, x: ln2, gs: driver.sc.gs, wQ4: layer.gu.view,
                                  h16: mlpInter, fuseFwht32: true)
            enc.memoryBarrier(scope: .buffers)
            try driver.gemvH16T4(enc, n: h, k: layer.down.k, x: mlpInter, gs: driver.sc.gsI,
                                 wQ4: layer.down, y16: dn16)
        }
        enc.memoryBarrier(scope: .buffers)

        try driver.encodeMHCMerge(enc, stream: outStream, branchOut: dn16, controls: ctrl1,
                                  xOut: oOut16, nextGamma: weights.sharedHeadNorm, nextNormOut: postNorm,
                                  rows: 1, isLeave: false, sinkhornIters: mhcCfg.iterations,
                                  hasNextNorm: false)
        enc.memoryBarrier(scope: .buffers)

        try driver.encodeMHCExitHead(enc, stream: outStream, headW: weights.hcHeadW, headB: weights.hcHeadB,
                                     normGamma: weights.sharedHeadNorm, outNorm: postNorm,
                                     rows: 1, applyNorm: true)
        enc.memoryBarrier(scope: .buffers)

        let vocab = draftLogits.shape.elementCount
        let needFullLogits = writeLogits || driver.multimaxParams != nil || !selectGreedy
        if !needFullLogits, driver.mtpDraftHead == nil, case .q4g(let headW) = driver.headStorage,
           !headW.view.isIncumbentContainer {
            let nTG = (vocab + 31) / 32
            let keyBytes = nTG * 4
            let keysHi = QWTensorValue(region: partials, offset: 0, shape: QWTensorShape([nTG], .i32))
            let keysLo = QWTensorValue(region: partials, offset: keyBytes, shape: QWTensorShape([nTG], .i32))
            try QWGemmAQ.encodeHeadArgmax(enc, hal: driver.hal, view: headW.view, x: postNorm,
                                          y: draftLogits, keysHi: keysHi, keysLo: keysLo, writeLogits: false)
            var nTG32 = UInt32(nTG)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try driver.hal.pipeline("qwg_head_argmax_fin"))
            keysHi.bind(enc, index: 0)
            keysLo.bind(enc, index: 1)
            greedySelector.token.bind(enc, index: 2)
            enc.setBytes(&nTG32, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            return
        }

        if let draftHead = driver.mtpDraftHead {
            try draftHead.encode(enc, hal: driver.hal, x: postNorm, gs: gsHead, y: draftLogits)
        } else {
            switch driver.headStorage {
            case .f16(let w):
                try QWGemm.encodeFwht32(enc, hal: driver.hal, x: postNorm)
                enc.memoryBarrier(scope: .buffers)
                try QWHead.encodeF16(enc, hal: driver.hal, w: w, x: postNorm,
                                     y: draftLogits, geometry: geometry, N: vocab)
            case .q4g(let w):
                try QWHead.encodeQ4g(enc, hal: driver.hal, weight: w,
                                     x: postNorm, gs: gsHead, y: draftLogits)
            }
        }
        if let mm = driver.multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try driver.head.encodeMultiMax(enc, logits: draftLogits, params: mm, count: vocab)
        }
        if selectGreedy {
            enc.memoryBarrier(scope: .buffers)
            try greedySelector.encode(enc, hal: driver.hal, logits: draftLogits,
                                      ordering: .numericalDraft, destination: greedySelector.token)
        }
    }

    func propose(driver: QWEngineDriver,
                 initialStream: QWTensorValue,
                 kv: (k: QWKVLayerView, v: QWKVLayerView),
                 pendingToken: Int,
                 cacheRows: Int,
                 drafts: Int,
                 sampledToken: ((QWTensorValue) throws -> Int)? = nil,
                 shouldStopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = nil,
                 writeLogitsOnFinalDraft: Bool = false,
                 completed: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> [Int] {
        guard drafts > 0 else { return [] }
        let h = geometry.hidden
        var currentToken = pendingToken
        var proposed: [Int] = []
        proposed.reserveCapacity(drafts)

        for step in 0..<drafts {
            let embSlice = embedBuf.slice(byteOffset: 0, count: h, dtype: .f16)
            try driver.copyEmbedRow(currentToken, to: embSlice)
            let inStream = step == 0 ? initialStream : ((step % 2 == 1) ? draftPing : draftPong)
            let outStream = (step % 2 == 0) ? draftPing : draftPong
            let isLast = (step + 1 == drafts)
            let needLogits = (sampledToken != nil) || (shouldStopAfterDraft != nil)
                || (isLast && writeLogitsOnFinalDraft)

            guard let cb = driver.hal.queue.makeCommandBuffer(),
                  var enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("ECHO draft command buffer creation failed")
            }
            driver.bindResidentResources(enc)
            enc.useResource(region.buffer, usage: [.read, .write])
            driver.activeCB = cb
            driver.activeEnc = enc
            var open = true
            do {
                try encodeStep(&enc, driver: driver, prevStream: inStream, embedding: embSlice,
                               kv: kv, cacheRows: cacheRows + step, outStream: outStream,
                               writeLogits: needLogits, selectGreedy: sampledToken == nil)
                if let active = driver.activeEnc { enc = active }
                let finalCB = driver.activeCB ?? cb
                enc.endEncoding()
                open = false
                driver.activeEnc = nil
                driver.activeCB = nil
                finalCB.commit()
                finalCB.waitUntilCompleted()
                guard finalCB.status == .completed else {
                    driver.commandBufferErrors += 1
                    throw QWEngineDriverError.state("ECHO draft GPU failure: \(String(describing: finalCB.error))")
                }
                try completed?(finalCB)
            } catch {
                if open { enc.endEncoding() }
                driver.activeEnc = nil
                driver.activeCB = nil
                throw error
            }

            let chosen: Int
            if let sampledToken {
                chosen = try sampledToken(draftLogits)
            } else {
                chosen = try greedySelector.completedToken()
            }
            proposed.append(chosen)
            currentToken = chosen
            if let shouldStopAfterDraft,
               try shouldStopAfterDraft(draftLogits, proposed.count) {
                break
            }
        }
        return proposed
    }
}

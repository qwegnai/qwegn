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
import QWMTP
import QWNorm

final class QWEchoDraftEncoder: @unchecked Sendable {
    static let maxProposalRows = 15

    let geometry: QWModelGeometry
    let weights: QWEngineDriver.EchoDraftWeights
    let layer: QWEngineDriver.LayerW
    let mlaOperator: QWMLAAttnOperator?
    let mlaLayerIndex: Int
    let attnDecode: QWAttnDecode?
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
    let qkvScratch: QWTensorValue?
    let qRScratch: QWTensorValue?
    let kRScratch: QWTensorValue?
    let vScratch: QWTensorValue?
    let gqaPartials: QWMRegion?
    let attOut: QWTensorValue
    let oOut16: QWTensorValue
    let ln2: QWTensorValue
    let mlpInter: QWTensorValue
    let dn32: QWTensorValue
    let dn16: QWTensorValue
    let postNorm: QWTensorValue
    let gsHead: QWTensorValue
    let draftLogits: QWTensorValue
    let draftLogitHistory: QWTensorValue
    let selector: QWTokenSelector
    let partials: QWMRegion
    let moe: QWEngineDriver.MoEScratch?

    private let psoEchoStreamInit: any MTLComputePipelineState

    static func allocationBytes(
        geometry: QWModelGeometry,
        maxSeedRows: Int,
        streams: Int,
        vCols: Int,
        denseInter: Int,
        hasMLA: Bool,
        hasMoE: Bool
    ) -> Int {
        let rows = max(16, min(512, maxSeedRows))
        let h = geometry.hidden, vocab = geometry.vocab, c = streams
        let ctrlWidth = c + c * c
        let partialBytes = max(4096, 2 * ((vocab + 31) / 32) * 4 + 512)
        let moeBytes = hasMoE
            ? QWEngineDriver.moeScratchAllocationBytes(rows: 1, geometry: geometry, useGroupedNAX: false) : 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        let gqaPartialsBytes = !hasMLA
            ? QWAttnDecode.decodePartialsBytes(geometry: geometry, maxSplits: QWAttnDecode.maxLongContextSplits) : 0
        let gqaScratchBytes = !hasMLA
            ? (align256(rows * geometry.attnTotalN * 2) + align256(rows * geometry.qProjOut * 2)
                + align256(rows * geometry.kvChannels * 2) * 2 + align256(gqaPartialsBytes)) : 0
        let selectorBytes = QWTokenSelector.allocationBytes(
            vocabLimit: vocab, vocabulary: vocab, maxRows: maxProposalRows)
        return align256(rows * h * 2) * 4 + align256(rows * c * h * 4) * 2
            + align256(rows * c * h * 2) * 3 + align256(c * h * 4) * 2
            + align256(rows * ctrlWidth * 4) + align256(vCols * 2)
            + align256(h * 2) * 5 + align256(denseInter * 2) + align256(h * 4) * 2
            + align256(vocab * 4) + align256(maxProposalRows * vocab * 4)
            + selectorBytes + partialBytes * 3 + moeBytes + gqaScratchBytes + (2 << 20)
    }

    init(geometry: QWModelGeometry, hal: QWHal, weights: QWEngineDriver.EchoDraftWeights,
         trunkLayer: QWEngineDriver.LayerW, mlaOperator: QWMLAAttnOperator?,
         mlaLayerIndex: Int, maxSeedRows: Int, kvMode: QWKVMode = .f16) throws {
        guard let mhcCfg = geometry.mhcConfig, trunkLayer.mhc != nil,
              let attn = trunkLayer.attn, (mlaOperator == nil || attn.mla != nil) else {
            throw QWEngineDriverError.state("ECHO drafter requires mHC + compatible attention on the draft layer")
        }
        self.geometry = geometry; self.weights = weights; self.mlaOperator = mlaOperator
        self.mlaLayerIndex = mlaLayerIndex
        self.attnDecode = mlaOperator == nil ? try QWAttnDecode(hal: hal, kvMode: kvMode, geometry: geometry) : nil
        self.maxSeedRows = max(16, min(512, maxSeedRows)); self.streams = mhcCfg.streams

        let aliasedMoE: QWEngineDriver.MoEW? = trunkLayer.moe.map { t in
            QWEngineDriver.MoEW(
                router: t.router, gateUp: t.gateUp, down: t.down, numExperts: t.numExperts,
                expertsPerTok: t.expertsPerTok, sharedGateUp: t.sharedGateUp, sharedDown: t.sharedDown,
                sharedGateScalar: t.sharedGateScalar, routerBias: weights.routerBias ?? t.routerBias,
                routedScaleParam: t.routedScaleParam, splitFeatureRouting: t.splitFeatureRouting,
                latent: t.latent, coldGateUp: t.coldGateUp, coldDown: t.coldDown, coldStreamer: t.coldStreamer)
        }
        self.layer = QWEngineDriver.LayerW(
            ln1: trunkLayer.ln1, ln2: trunkLayer.ln2, gu: trunkLayer.gu, down: trunkLayer.down,
            kind: trunkLayer.kind, gdn: nil, attn: attn, moe: aliasedMoE, mhc: trunkLayer.mhc, hca: nil)
        self.psoEchoStreamInit = try hal.pipeline("qwg_echo_stream_init_f16")

        let rows = self.maxSeedRows, h = geometry.hidden, c = self.streams, vocab = geometry.vocab
        let ctrlWidth = c + c * c, vCols = max(attn.o.k, geometry.qHeads * geometry.vHeadDim)
        let denseInter = max(128, trunkLayer.down.k)
        let partialBytes = max(4096, 2 * ((vocab + 31) / 32) * 4 + 512)
        let gqaPartialsBytes = mlaOperator == nil
            ? QWAttnDecode.decodePartialsBytes(geometry: geometry, maxSplits: QWAttnDecode.maxLongContextSplits) : 0
        let estimatedBytes = Self.allocationBytes(
            geometry: geometry, maxSeedRows: maxSeedRows, streams: c, vCols: vCols,
            denseInter: denseInter, hasMLA: mlaOperator != nil, hasMoE: aliasedMoE != nil)

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

        embedBuf = try tv(rows * h); eNormed16 = try tv(rows * h); eOut16 = try tv(rows * h)
        prevStreamBuf = try tv(rows * c * h, .f32); prev16 = try tv(rows * c * h)
        hNormed16 = try tv(rows * c * h); hOut16 = try tv(rows * c * h)
        streamScratch = try tv(rows * c * h, .f32)
        draftPing = try tv(c * h, .f32); draftPong = try tv(c * h, .f32)
        controlsScratch = try tv(rows * ctrlWidth, .f32); ln1 = try tv(rows * h)
        if mlaOperator == nil {
            qkvScratch = try tv(rows * geometry.attnTotalN)
            qRScratch = try tv(rows * geometry.qProjOut)
            kRScratch = try tv(rows * geometry.kvChannels); vScratch = try tv(rows * geometry.kvChannels)
            let gpOff = try sub.allocate(bytes: gqaPartialsBytes)
            gqaPartials = QWMRegion(buffer: buf, base: gpOff, length: gqaPartialsBytes)
        } else {
            qkvScratch = nil; qRScratch = nil; kRScratch = nil; vScratch = nil; gqaPartials = nil
        }
        attOut = try tv(vCols); oOut16 = try tv(h); ln2 = try tv(h); mlpInter = try tv(denseInter)
        dn32 = try tv(h, .f32); dn16 = try tv(h); postNorm = try tv(h)
        gsHead = try tv(max(1, h / 128), .f32); draftLogits = try tv(vocab, .f32)
        draftLogitHistory = try tv(Self.maxProposalRows * vocab, .f32)
        selector = try QWTokenSelector(
            vocabLimit: vocab, allocator: sub, vocabulary: vocab, maxRows: Self.maxProposalRows)
        let pOff = try sub.allocate(bytes: partialBytes)
        partials = QWMRegion(buffer: buf, base: pOff, length: partialBytes)
        moe = aliasedMoE != nil ? try QWEngineDriver.makeMoEScratch(tv, rows: 1, geometry: geometry, useGroupedNAX: false) : nil
    }

    func draftLogitRow(_ row: Int) throws -> QWTensorValue {
        let vocab = geometry.vocab
        guard (0..<Self.maxProposalRows).contains(row) else {
            throw QWEngineDriverError.state("ECHO draft logit history row \(row) out of range")
        }
        return QWTensorValue(
            region: draftLogitHistory.region,
            offset: draftLogitHistory.offset + row * vocab * 4,
            shape: QWTensorShape([vocab], .f32))
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
                                          rows: Int,
                                          writeSeedKVOnly: Bool) throws {
        guard let mhc = layer.mhc, let attn = layer.attn else {
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

        try driver.gemmBatchedH16(enc, m: streamRows, n: h, k: h,
                                  x: hNormSlice, wQ4: weights.hProj, y16: hOutSlice)
        if rows == 1 {
            try driver.gemvH16T4(enc, n: h, k: h, x: eNormSlice, gs: driver.sc.gs,
                                 wQ4: weights.eProj, y16: eOutSlice)
        } else {
            try driver.gemmBatchedH16(enc, m: rows, n: h, k: h,
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

        if let mla = mlaOperator, let mlaW = attn.mla {
            if rows == 1 {
                let latent1 = QWTensorValue(region: mla.mlaLatentScratch.region,
                                            offset: mla.mlaLatentScratch.offset,
                                            shape: QWTensorShape([attn.qkvProj.n], .f16))
                try driver.gemvH16T4(enc, n: attn.qkvProj.n, k: h, x: ln1Slice, gs: driver.sc.gs,
                                     wQ4: attn.qkvProj, y16: latent1)
            } else {
                try driver.gemmBatchedH16(enc, m: rows, n: attn.qkvProj.n, k: h,
                                          x: ln1Slice, wQ4: attn.qkvProj, y16: mla.mlaLatentScratch,
                                          yStride: mla.latentStride)
            }
            enc.memoryBarrier(scope: .buffers)
            mla.encodeLatentPrepAndPool(enc, ln1: ln1Slice, mlaW: mlaW,
                                        kv: kv, fullAttnLayer: mlaLayerIndex,
                                        pos0: cacheRows, rows: rows)
        } else {
            guard let qkvAll = qkvScratch, let qRAll = qRScratch,
                  let kRAll = kRScratch, let vAll = vScratch else {
                throw QWEngineDriverError.state("ECHO GQA scratch missing")
            }
            let qkvSlice = qkvAll.slice(byteOffset: 0, count: rows * geometry.attnTotalN, dtype: .f16)
            let qRSlice = qRAll.slice(byteOffset: 0, count: rows * geometry.qProjOut, dtype: .f16)
            let kRSlice = kRAll.slice(byteOffset: 0, count: rows * geometry.kvChannels, dtype: .f16)
            let vSlice = vAll.slice(byteOffset: 0, count: rows * geometry.kvChannels, dtype: .f16)
            if rows == 1 {
                try driver.gemvH16T4(enc, n: geometry.attnTotalN, k: h,
                                     x: ln1Slice, gs: driver.sc.gs, wQ4: attn.qkvProj, y16: qkvSlice)
            } else {
                try driver.gemmBatchedH16(enc, m: rows, n: geometry.attnTotalN, k: h,
                                          x: ln1Slice, wQ4: attn.qkvProj, y16: qkvSlice)
            }
            enc.memoryBarrier(scope: .buffers)
            try driver.ropeApply.encodeExtractQK(
                enc, hal: driver.hal, attnY: qkvSlice,
                qOut: qRSlice, kOut: kRSlice, vOut: vSlice,
                wq: attn.wq, wk: attn.wkTiled, qkvBias: attn.qkvBias,
                positionStart: cacheRows, count: rows)
            if writeSeedKVOnly {
                enc.memoryBarrier(scope: .buffers)
                driver.encodePagedKVWrite(enc, kv: kv, k: kRSlice, v: vSlice, startPos: cacheRows, t: rows)
            }
        }
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
                                     outStream: outStream, kv: kv, cacheRows: cacheRows, rows: rows,
                                     writeSeedKVOnly: true)
    }

    func encodeStep(_ enc: inout any MTLComputeCommandEncoder,
                    driver: QWEngineDriver,
                    prevStream: QWTensorValue,
                    embedding: QWTensorValue,
                    kv: (k: QWKVLayerView, v: QWKVLayerView),
                    cacheRows: Int,
                    outStream: QWTensorValue,
                    writeLogits: Bool,
                    selectGreedy: Bool,
                    logitsDestination: QWTensorValue? = nil) throws {
        guard let mhcCfg = geometry.mhcConfig,
              let mhc = layer.mhc,
              let attn = layer.attn else {
            throw QWEngineDriverError.state("ECHO layer weights are incomplete")
        }
        let h = geometry.hidden
        let c = streams
        let ctrl1 = controlsScratch.slice(byteOffset: 0, count: c + c * c, dtype: .f32)

        try encodeEntryAndLatentPrep(enc, driver: driver, prevStream: prevStream, embeddings: embedding,
                                     outStream: outStream, kv: kv, cacheRows: cacheRows, rows: 1,
                                     writeSeedKVOnly: false)
        enc.memoryBarrier(scope: .buffers)

        if let mla = mlaOperator, let mlaW = attn.mla {
            let ln1Single = ln1.slice(byteOffset: 0, count: h, dtype: .f16)
            let qIn1 = mla.isDirectQ
                ? ln1Single
                : QWTensorValue(region: mla.qResidScratch.region,
                                offset: mla.qResidScratch.offset,
                                shape: QWTensorShape([mlaW.qBProj.k], .f16))
            let qFull1 = QWTensorValue(region: mla.mlaQFullScratch.region,
                                       offset: mla.mlaQFullScratch.offset,
                                       shape: QWTensorShape([mlaW.qBProj.n], .f16))
            try driver.gemvH16T4(enc, n: mlaW.qBProj.n, k: mlaW.qBProj.k,
                                 x: qIn1, gs: driver.sc.gs, wQ4: mlaW.qBProj, y16: qFull1)
            enc.memoryBarrier(scope: .buffers)

            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: attOut,
                fullAttnLayer: mlaLayerIndex, pos0: cacheRows, rows: 1,
                ln1: ln1Single)
        } else {
            guard let aDec = attnDecode,
                  let qRAll = qRScratch, let kRAll = kRScratch,
                  let vAll = vScratch, let gPart = gqaPartials else {
                throw QWEngineDriverError.state("ECHO GQA decode state missing")
            }
            let qR1 = qRAll.slice(byteOffset: 0, count: geometry.qProjOut, dtype: .f16)
            let kR1 = kRAll.slice(byteOffset: 0, count: geometry.kvChannels, dtype: .f16)
            let v1 = vAll.slice(byteOffset: 0, count: geometry.kvChannels, dtype: .f16)
            try aDec.encode(
                enc, qgate: qR1, kv: kv, knew: kR1, vnew: v1,
                partials: gPart, out: attOut, C: cacheRows,
                splits: QWAttnDecode.splits(forKeys: cacheRows + 1, kvHeads: geometry.kvHeads, gpuCores: driver.hal.caps.gpuCores))
        }
        enc.memoryBarrier(scope: .buffers)

        try driver.gemvH16T4(enc, n: h, k: attn.o.k, x: attOut, gs: driver.sc.gs48,
                             wQ4: attn.o, y16: oOut16)
        enc.memoryBarrier(scope: .buffers)
        if let oBias = attn.oBias {
            try QWNorm.encodeAddBiasF16(enc, hal: driver.hal, x: oOut16, bias: oBias, rows: 1, cols: h)
            enc.memoryBarrier(scope: .buffers)
        }

        try driver.encodeMHCRouterNorm(enc, stream: outStream, w: mhc.ffnW, b: mhc.ffnB,
                                       gamma: layer.ln2, controls: ctrl1, out: ln2,
                                       rows: 1, mergeBranchOut: oOut16)
        enc.memoryBarrier(scope: .buffers)

        if let moeW = layer.moe, let moeSc = self.moe {
            let prevEncID = ObjectIdentifier(enc)
            try QWMoEChain.encode(enc: &enc, hal: driver.hal, moe: moeW, s: moeSc,
                                  x16: ln2, dn32: dn32, dn16: dn16,
                                  rows: 1, geometry: geometry, driver: driver,
                                  cb: driver.activeCB, skipOut32: true)
            if ObjectIdentifier(enc) != prevEncID {
                driver.hal.touchKeepAlive(enc)
                enc.useResource(region.buffer, usage: [.read, .write])
            }
        } else {
            try driver.mlpSiluPairH16(enc, x: ln2, gs: driver.sc.gs, wQ4: layer.gu.view,
                                      h16: mlpInter, fuseFwht32: layer.gu.view.isRotated)
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

        let logitsOut = logitsDestination ?? draftLogits
        let vocab = logitsOut.shape.elementCount
        let needFullLogits = writeLogits || driver.multimaxParams != nil || !selectGreedy
        if !needFullLogits, driver.mtpDraftHead == nil, case .q4g(let headW) = driver.headStorage,
           !headW.view.isIncumbentContainer {
            let nTG = (vocab + 31) / 32
            let keyBytes = nTG * 4
            let keysHi = QWTensorValue(region: partials, offset: 0, shape: QWTensorShape([nTG], .i32))
            let keysLo = QWTensorValue(region: partials, offset: keyBytes, shape: QWTensorShape([nTG], .i32))
            try QWGemmAQ.encodeHeadArgmax(enc, hal: driver.hal, view: headW.view, x: postNorm,
                                          y: logitsOut, keysHi: keysHi, keysLo: keysLo, writeLogits: false)
            var nTG32 = UInt32(nTG)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try driver.hal.pipeline("qwg_head_argmax_fin"))
            keysHi.bind(enc, index: 0)
            keysLo.bind(enc, index: 1)
            selector.token.bind(enc, index: 2)
            enc.setBytes(&nTG32, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            return
        }

        if let draftHead = driver.mtpDraftHead {
            try draftHead.encode(enc, hal: driver.hal, x: postNorm, gs: gsHead, y: logitsOut)
        } else {
            switch driver.headStorage {
            case .f16(let w):
                if driver.hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: driver.hal, x: postNorm)
                    enc.memoryBarrier(scope: .buffers)
                }
                try QWHead.encodeF16(enc, hal: driver.hal, w: w, x: postNorm,
                                     y: logitsOut, geometry: geometry, N: vocab)
            case .bf16(let w):
                if driver.hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: driver.hal, x: postNorm)
                    enc.memoryBarrier(scope: .buffers)
                }
                try QWHead.encodeBF16(enc, hal: driver.hal, w: w, x: postNorm,
                                      y: logitsOut, geometry: geometry, N: vocab)
            case .q4g(let w):
                try QWHead.encodeQ4g(enc, hal: driver.hal, weight: w,
                                     x: postNorm, gs: gsHead, y: logitsOut)
            }
        }
        if let mm = driver.multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try driver.head.encodeMultiMax(enc, logits: logitsOut, params: mm, count: vocab)
        }
        if selectGreedy {
            enc.memoryBarrier(scope: .buffers)
            try selector.encodeGreedy(enc, hal: driver.hal, logits: logitsOut,
                                      ordering: .numericalDraft, destination: selector.token)
        }
    }

    private static func drainPendingStepCBs(_ cbs: [any MTLCommandBuffer], driver: QWEngineDriver) {
        for cb in cbs {
            cb.waitUntilCompleted()
            if cb.status != .completed || cb.error != nil { driver.commandBufferErrors += 1 }
        }
    }

    @discardableResult
    private static func drainColdEvents(driver: QWEngineDriver) -> (any Error)? {
        var coldFailure: (any Error)?
        if let pipe = driver.coldEvents {
            pipe.waitUntilIdle()
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            driver.decodeTel.coldWaitUs += r.waitUs
            driver.decodeTel.coldIOUs += r.ioUs
            driver.decodeTel.coldStaged += r.staged
            coldFailure = r.failure
        }
        driver.flushPendingColdUnloads()
        return coldFailure
    }

    func enqueueSingleCB(driver: QWEngineDriver,
                       table: QWMTPEmbeddingTable,
                       initialStream: QWTensorValue,
                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                       pendingToken: Int,
                       cacheRows: Int,
                       drafts: Int,
                       writeLogitsOnFinalDraft: Bool = false,
                       journalTokenOffset: Int = 0,
                       stochastic: QWStochasticSelectionParams? = nil) throws -> [any MTLCommandBuffer] {
        let h = geometry.hidden
        guard drafts >= 1, stochastic == nil || drafts <= Self.maxProposalRows,
              journalTokenOffset >= 0, (journalTokenOffset + drafts) * 4 <= table.tokens.length else {
            throw QWEngineDriverError.state("ECHO single-CB journal range is invalid")
        }
        selector.token.region.pointer.advanced(by: selector.token.offset)
            .assumingMemoryBound(to: Int32.self).pointee = Int32(pendingToken)
        guard let cb = driver.hal.makeUnretainedCommandBuffer(),
              var enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("ECHO draft command buffer creation failed")
        }
        func bindEchoResources(_ targetEnc: any MTLComputeCommandEncoder) {
            driver.hal.touchKeepAlive(targetEnc)
            targetEnc.useResource(region.buffer, usage: [.read, .write])
            targetEnc.useResource(table.tokens, usage: [.read, .write])
        }
        bindEchoResources(enc)
        var boundEncID = ObjectIdentifier(enc)
        driver.stepCBs = []
        driver.activeCB = cb
        driver.activeEnc = enc
        defer { driver.stepCBs = nil; driver.activeEnc = nil; driver.activeCB = nil }
        let journalTensor = QWTensorValue(
            region: QWMRegion(buffer: table.tokens, base: 0, length: table.tokens.length),
            offset: 0, shape: QWTensorShape([table.tokens.length / 4], .i32))
        do {
            for step in 0..<drafts {
                let embSlice = embedBuf.slice(byteOffset: 0, count: h, dtype: .f16)
                try table.encode(enc, token: selector.token, destination: embSlice)
                let inStream = step == 0 ? initialStream : ((step % 2 == 1) ? draftPing : draftPong)
                let outStream = (step % 2 == 0) ? draftPing : draftPong
                let isLast = (step + 1 == drafts)
                let logitsOut = stochastic != nil ? try draftLogitRow(step) : draftLogits
                try encodeStep(&enc, driver: driver, prevStream: inStream, embedding: embSlice,
                               kv: kv, cacheRows: cacheRows + step, outStream: outStream,
                               writeLogits: (isLast && writeLogitsOnFinalDraft) || stochastic != nil,
                               selectGreedy: stochastic == nil, logitsDestination: logitsOut)
                if let active = driver.activeEnc { enc = active }
                if ObjectIdentifier(enc) != boundEncID {
                    bindEchoResources(enc)
                    boundEncID = ObjectIdentifier(enc)
                }
                enc.memoryBarrier(scope: .buffers)
                if var stepParams = stochastic {
                    stepParams.philoxOffsetBase &+= UInt64(step)
                    try selector.encodeStochasticDraft(
                        enc, hal: driver.hal, logits: logitsOut, params: stepParams, rows: 1,
                        proposalSlotBase: step, destination: selector.token,
                        journal: journalTensor, journalOffset: journalTokenOffset + step)
                } else {
                    let saved = QWTensorValue(
                        region: QWMRegion(buffer: table.tokens, base: 0, length: table.tokens.length),
                        offset: (journalTokenOffset + step) * 4, shape: QWTensorShape([1], .i32))
                    try QWVerifyWindow.encodeCopy(enc, hal: driver.hal, source: selector.token, destination: saved)
                }
                if !isLast { enc.memoryBarrier(scope: .buffers) }
            }
            let finalEnc = driver.activeEnc ?? enc
            let finalCB = driver.activeCB ?? cb
            let extraCBs = driver.stepCBs ?? []
            driver.activeEnc = nil
            driver.activeCB = nil
            finalEnc.endEncoding()
            finalCB.commit()
            return extraCBs + [finalCB]
        } catch {
            if let openEnc = driver.activeEnc { driver.activeEnc = nil; openEnc.endEncoding() }
            Self.drainPendingStepCBs(driver.stepCBs ?? [], driver: driver)
            _ = Self.drainColdEvents(driver: driver)
            throw error
        }
    }

    func readSparseProposals(drafts: Int) throws -> [QWSpeculativeRoundResolver.DraftProposal] {
        var proposals: [QWSpeculativeRoundResolver.DraftProposal] = []
        proposals.reserveCapacity(drafts)
        let vocab = geometry.vocab
        for row in 0..<drafts {
            let buf = try selector.readSparseProposal(row: row)
            if buf.isSparse {
                proposals.append(try QWSpeculativeRoundResolver.DraftProposal.fromSparseBuffer(buf))
            } else {
                let rowLogits = try draftLogitRow(row)
                let ptr = rowLogits.region.pointer.advanced(by: rowLogits.offset).assumingMemoryBound(to: Float.self)
                proposals.append(try QWSpeculativeRoundResolver.DraftProposal.fromSparseBuffer(
                    buf, logits: UnsafeBufferPointer(start: ptr, count: vocab)))
            }
        }
        return proposals
    }

    private static func waitAndDrainCBs(
        _ cbs: [any MTLCommandBuffer],
        driver: QWEngineDriver,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws {
        var allDrained = false
        defer {
            if !allDrained {
                Self.drainPendingStepCBs(cbs, driver: driver)
                _ = Self.drainColdEvents(driver: driver)
            }
        }
        for sub in cbs { sub.waitUntilCompleted() }
        let coldFailure = Self.drainColdEvents(driver: driver)
        allDrained = true
        for sub in cbs where sub.status != .completed || sub.error != nil { driver.commandBufferErrors += 1 }
        if let coldFailure { throw QWEngineDriverError.state("ECHO cold event staging: \(coldFailure)") }
        for sub in cbs {
            if let err = sub.error { throw QWEngineDriverError.state("ECHO draft GPU failure: \(err.localizedDescription)") }
            if sub.status != .completed {
                throw QWEngineDriverError.state("ECHO draft GPU completion failed (status=\(sub.status.rawValue))")
            }
            try completed?(sub)
        }
    }

    private func proposeSingleCB(
        driver: QWEngineDriver, table: QWMTPEmbeddingTable, initialStream: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView), pendingToken: Int, cacheRows: Int,
        drafts: Int, writeLogitsOnFinalDraft: Bool,
        stochastic: QWStochasticSelectionParams? = nil,
        completed: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> [Int] {
        let cbs = try enqueueSingleCB(
            driver: driver, table: table, initialStream: initialStream, kv: kv,
            pendingToken: pendingToken, cacheRows: cacheRows, drafts: drafts,
            writeLogitsOnFinalDraft: writeLogitsOnFinalDraft, stochastic: stochastic)
        try Self.waitAndDrainCBs(cbs, driver: driver, completed: completed)
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        return try (0..<drafts).map { step in
            let id = Int(ids[step])
            guard (0..<geometry.vocab).contains(id) else {
                throw QWEngineDriverError.state("ECHO single-CB draft invalid token \(id) at step \(step)")
            }
            return id
        }
    }

    func proposeStochastic(
        driver: QWEngineDriver, initialStream: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
        pendingToken: Int, cacheRows: Int, drafts: Int, params: QWStochasticSelectionParams,
        writeLogitsOnFinalDraft: Bool = false, completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> (drafts: [Int], proposals: [QWSpeculativeRoundResolver.DraftProposal]) {
        guard drafts > 0, drafts <= Self.maxProposalRows, let table = driver.mtpDeviceEmbedding,
              table.vocabulary == geometry.vocab, table.hidden == geometry.hidden else {
            throw QWEngineDriverError.state("ECHO stochastic device chain requires compatible device embedding")
        }
        let tokens = try proposeSingleCB(
            driver: driver, table: table, initialStream: initialStream, kv: kv,
            pendingToken: pendingToken, cacheRows: cacheRows, drafts: drafts,
            writeLogitsOnFinalDraft: writeLogitsOnFinalDraft, stochastic: params, completed: completed)
        return (tokens, try readSparseProposals(drafts: drafts))
    }

    func propose(
        driver: QWEngineDriver, initialStream: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
        pendingToken: Int, cacheRows: Int, drafts: Int,
        sampledToken: ((QWTensorValue) throws -> Int)? = nil,
        shouldStopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = nil,
        writeLogitsOnFinalDraft: Bool = false, completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> [Int] {
        guard drafts > 0 else { return [] }
        let h = geometry.hidden
        if drafts >= 1, sampledToken == nil, shouldStopAfterDraft == nil, drafts <= 64,
           let table = driver.mtpDeviceEmbedding, table.vocabulary == geometry.vocab, table.hidden == h {
            return try proposeSingleCB(
                driver: driver, table: table, initialStream: initialStream, kv: kv,
                pendingToken: pendingToken, cacheRows: cacheRows, drafts: drafts,
                writeLogitsOnFinalDraft: writeLogitsOnFinalDraft, completed: completed)
        }
        var currentToken = pendingToken
        var proposed: [Int] = []
        proposed.reserveCapacity(drafts)
        for step in 0..<drafts {
            let embSlice = embedBuf.slice(byteOffset: 0, count: h, dtype: .f16)
            try driver.copyEmbedRow(currentToken, to: embSlice)
            let inStream = step == 0 ? initialStream : ((step % 2 == 1) ? draftPing : draftPong)
            let outStream = (step % 2 == 0) ? draftPing : draftPong
            let needLogits = (sampledToken != nil) || (shouldStopAfterDraft != nil)
                || (step + 1 == drafts && writeLogitsOnFinalDraft)
            guard let cb = driver.hal.makeUnretainedCommandBuffer(),
                  var enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("ECHO draft command buffer creation failed")
            }
            driver.hal.touchKeepAlive(enc)
            enc.useResource(region.buffer, usage: [.read, .write])
            driver.stepCBs = []; driver.activeCB = cb; driver.activeEnc = enc
            var stepDrained = false
            defer {
                if let openEnc = driver.activeEnc { driver.activeEnc = nil; openEnc.endEncoding() }
                if !stepDrained {
                    Self.drainPendingStepCBs(driver.stepCBs ?? [], driver: driver)
                    _ = Self.drainColdEvents(driver: driver)
                }
                driver.stepCBs = nil; driver.activeEnc = nil; driver.activeCB = nil
            }
            try encodeStep(&enc, driver: driver, prevStream: inStream, embedding: embSlice,
                           kv: kv, cacheRows: cacheRows + step, outStream: outStream,
                           writeLogits: needLogits, selectGreedy: sampledToken == nil)
            let finalEnc = driver.activeEnc ?? enc, finalCB = driver.activeCB ?? cb
            let stepCbs = (driver.stepCBs ?? []) + [finalCB]
            driver.activeEnc = nil; driver.activeCB = nil
            finalEnc.endEncoding(); finalCB.commit()
            stepDrained = true
            try Self.waitAndDrainCBs(stepCbs, driver: driver, completed: completed)
            let chosen = if let sampledToken { try sampledToken(draftLogits) } else { try selector.completedToken() }
            proposed.append(chosen); currentToken = chosen
            if let shouldStopAfterDraft, try shouldStopAfterDraft(draftLogits, proposed.count) { break }
        }
        return proposed
    }
}

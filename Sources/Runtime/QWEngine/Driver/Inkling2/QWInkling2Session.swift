// QWInkling2Session.swift — Preallocated KV, conv rings, and zero-alloc scratch workspace for Inkling-2.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWKV

struct Ink2VisitKV: @unchecked Sendable {
    let isSliding: Bool
    let ringCap: Int
    let kCache: any MTLBuffer
    let vCache: any MTLBuffer
    let kMeta: any MTLBuffer
    let vMeta: any MTLBuffer
    let rowBytes: Int
    let metaRowBytes: Int
    let kvCacheView: QWSinkRelKVCache
}

struct Ink2SlotState: @unchecked Sendable {
    let slotIndex: Int
    let capacity: Int
    let visitKVs: [Ink2VisitKV]
    let convStatesSlab: any MTLBuffer
    let residentStateBytes: Int
    let stateBuffers: [any MTLBuffer]

    func resetConvStates() {
        memset(convStatesSlab.contents(), 0, convStatesSlab.length)
    }
}

final class QWInkling2Session: @unchecked Sendable {
    static let maxPrefillChunk = 4096

    static func resolveMaxChunk(prefillChunkSize: Int, capacity: Int) -> Int {
        min(max(32, min(8192, prefillChunkSize)), max(32, capacity))
    }

    let capacity: Int
    let maxChunk: Int
    let kvModeRaw: UInt32
    let kvMode: QWKVMode
    let descriptor: QWModelDescriptor
    let geometry: QWModelGeometry
    let ink2: QWModelDescriptor.Inkling2Config
    let executionPlan: [Int]
    let slidingAttnConfig: QWSinkRelConfig
    let globalAttnConfig: QWSinkRelConfig
    let primarySlotState: Ink2SlotState
    let visitKVs: [Ink2VisitKV]
    let convStatesSlab: any MTLBuffer
    let convStateOffsets: [Int]
    let convStateKvStrideElems: Int
    let convStateHStrideElems: Int
    let convVerifyKvStrideElems: Int
    let convVerifyHStrideElems: Int
    let dummyB: any MTLBuffer
    let dummyF: any MTLBuffer

    // Preallocated scratch buffers (sized once for maxT = maxChunk)
    let tokIdsBuf: any MTLBuffer
    let xBuf: any MTLBuffer
    let preludeBuf: any MTLBuffer
    let blocksRingBuf: any MTLBuffer
    let attnResOutBuf: any MTLBuffer
    let bankBuf: any MTLBuffer
    let msgOutBuf: any MTLBuffer
    let ctrlPairBuf: any MTLBuffer
    let ctrlHidBuf: any MTLBuffer
    let ctrlOutBuf: any MTLBuffer
    let normBuf: any MTLBuffer
    let resF32Buf: any MTLBuffer
    let qProjBuf: any MTLBuffer
    let qPrepBuf: any MTLBuffer
    let kProjBuf: any MTLBuffer
    let kConvBuf: any MTLBuffer
    let vProjBuf: any MTLBuffer
    let vConvBuf: any MTLBuffer
    let rProjBuf: any MTLBuffer
    let rPrepBuf: any MTLBuffer
    let sinkBuf: any MTLBuffer
    let gateBuf: any MTLBuffer
    let attnOutBuf: any MTLBuffer
    let oOutBuf: any MTLBuffer
    let fusedAttnProjBuf: any MTLBuffer
    let dynWF32Buf: any MTLBuffer
    let partOBuf: any MTLBuffer
    let partMLBuf: any MTLBuffer
    let moeTopIdsBuf: any MTLBuffer
    let moeTopWBuf: any MTLBuffer
    let routerLogitsF32Buf: any MTLBuffer
    let moeCountsBuf: any MTLBuffer
    let moeOffsetsBuf: any MTLBuffer
    let moeSortedBuf: any MTLBuffer
    let moeSortedTokBuf: any MTLBuffer
    let moeInvPermBuf: any MTLBuffer
    let moeActiveExpertsBuf: any MTLBuffer
    let moeActiveArgsBuf: any MTLBuffer
    let moeTileEBuf: any MTLBuffer
    let moeTileRBuf: any MTLBuffer
    let moeTileCountBuf: any MTLBuffer
    let moePrefillGUBuf: any MTLBuffer
    let moeActBuf: any MTLBuffer
    let moeSlotOutBuf: any MTLBuffer
    let moeOutBuf: any MTLBuffer
    let logitsF32Buf: any MTLBuffer
    let headKeysHiBuf: any MTLBuffer
    let headKeysLoBuf: any MTLBuffer
    let headTokensBuf: any MTLBuffer
    let qwpF16InBuf: any MTLBuffer
    let qwpF16OutBuf: any MTLBuffer
    let qwpF32OutBuf: any MTLBuffer
    let qwpMoERotF32Buf: any MTLBuffer
    let qwpMoEGUF32Buf: any MTLBuffer
    let qwpMoEActF32Buf: any MTLBuffer
    let qwpMoEOutF32Buf: any MTLBuffer
    let qwpF32OutElems: Int

    // Zero-alloc tensor/slice views for generalized QWGemmFP / QWGemm / QWAttnSinkRel / QWHead operators
    let normSlice: QWFPBufferSlice
    let moeOutSlice: QWFPBufferSlice
    let routerLogitsSlice: QWFPBufferSlice
    let ctrlPairSlice: QWFPBufferSlice
    let qwpF16InSlice: QWFPBufferSlice
    let qwpF16InTV: QWTensorValue
    let qwpF16OutTV: QWTensorValue
    let qwpF32OutTV: QWTensorValue
    let qwpMoERotF32TV: QWTensorValue
    let qwpMoEGUF32TV: QWTensorValue
    let qwpMoEGUF32Slice: QWFPBufferSlice
    let qwpMoEActF32TV: QWTensorValue
    let qwpMoEActF32Slice: QWFPBufferSlice
    let qwpMoEOutF32TV: QWTensorValue
    let headKeysHiTV: QWTensorValue
    let headKeysLoTV: QWTensorValue
    let headTokensTV: QWTensorValue
    let logitsF32TV: QWTensorValue
    let qPrepTV: QWTensorValue
    let rPrepTV: QWTensorValue
    let attnOutTV: QWTensorValue
    let partialsView: QWSinkRelPartials
    let moeTopIdsTV: QWTensorValue
    let moeTopWTV: QWTensorValue
    let moeOffsetsTV: QWTensorValue
    let moeSortedTV: QWTensorValue
    let moeSortedTokTV: QWTensorValue
    let moeInvPermTV: QWTensorValue
    let moeActiveExpertsTV: QWTensorValue
    let moeActiveArgsTV: QWTensorValue
    let moeTileETV: QWTensorValue
    let moeTileRTV: QWTensorValue
    let moeTileCountTV: QWTensorValue
    let moePrefillGUTV: QWTensorValue
    let moeActTV: QWTensorValue
    let moeSlotOutTV: QWTensorValue
    let moePrefillGUF16TV: QWTensorValue
    let moeActF16TV: QWTensorValue
    let moeSlotOutF16TV: QWTensorValue

    let convPreVerifyBuf: (any MTLBuffer)?
    let convVerifyInputsBuf: (any MTLBuffer)?
    let convVerifyOffsets: [Int]
    let convVerifyChannels: [Int]
    let convSlots: Int
    let tokenRecycler = QWTokenRecycler()

    let residentStateBytes: Int
    let scratchBytes: Int

    init(
        device: any MTLDevice,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config,
        capacity: Int,
        kvModeRaw: UInt32,
        enableSpeculative: Bool = false,
        enableQWPScratch: Bool = false,
        prefillChunkSize: Int = QWInkling2Session.maxPrefillChunk
    ) throws {
        self.descriptor = descriptor
        self.geometry = QWModelGeometry(descriptor: descriptor)
        self.ink2 = ink2
        self.capacity = capacity
        self.maxChunk = Self.resolveMaxChunk(prefillChunkSize: prefillChunkSize, capacity: capacity)
        self.kvModeRaw = kvModeRaw
        let resolvedKVMode: QWKVMode = switch kvModeRaw {
        case 0: .f16
        case 1: .q8
        default: .q4
        }
        self.kvMode = resolvedKVMode
        self.executionPlan = descriptor.executionPlan
        let maxT = maxChunk
        let H = descriptor.hidden
        let nH = descriptor.qHeads
        let nKV = descriptor.kvHeads
        let headDim = descriptor.headDim
        let qDim = nH * headDim
        let kvDim = nKV * headDim
        let dRel = ink2.dRel
        let moeI = descriptor.moeIntermediate ?? 1152
        let nExp = descriptor.numExperts ?? 128
        let topK = descriptor.expertsPerTok ?? 8
        let convSlots = max(1, ink2.sconvKernelSize - 1)
        let numVisits = executionPlan.count

        self.slidingAttnConfig = QWSinkRelConfig(
            nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: ink2.relExtent,
            slidingWindow: ink2.slidingWindowSize, hasSink: true, hasRelBias: true,
            hasOutGate: true, kvMode: resolvedKVMode
        )
        self.globalAttnConfig = QWSinkRelConfig(
            nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: ink2.relExtent,
            slidingWindow: 0, hasSink: true, hasRelBias: true,
            hasOutGate: true, kvMode: resolvedKVMode
        )

        func alloc(_ bytes: Int, label: String) throws -> any MTLBuffer {
            try Self.allocAlignedBuffer(device: device, bytes: bytes, label: label)
        }

        self.dummyB = try alloc(256, label: "dummyB")
        self.dummyF = try alloc(256, label: "dummyF")

        let slot0 = try Self.allocateSlotStorage(
            device: device, descriptor: descriptor, ink2: ink2, executionPlan: executionPlan,
            slotIndex: 0, capacity: capacity, maxChunk: maxT,
            kvModeRaw: kvModeRaw, kvMode: resolvedKVMode
        )
        self.convStateKvStrideElems = slot0.kvStrideElems
        self.convStateHStrideElems = slot0.hStrideElems
        self.convStateOffsets = slot0.convOffsets
        self.visitKVs = slot0.state.visitKVs
        self.convStatesSlab = slot0.state.convStatesSlab
        self.residentStateBytes = slot0.state.residentStateBytes
        self.primarySlotState = slot0.state

        self.tokIdsBuf = try alloc(maxT * 4, label: "tokIds")
        self.xBuf = try alloc(maxT * H * 2, label: "x")
        self.preludeBuf = try alloc(maxT * H * 2, label: "prelude")
        self.blocksRingBuf = try alloc(8 * maxT * H * 2, label: "blocksRing")
        self.attnResOutBuf = try alloc(3 * maxT * H * 4, label: "attnRes")
        self.bankBuf = try alloc(maxT * 6 * H * 4, label: "bank")
        self.msgOutBuf = try alloc(4 * maxT * H * 4, label: "msgOut")
        self.ctrlPairBuf = try alloc(3 * maxT * 2 * H * 2, label: "ctrlPair")
        self.ctrlHidBuf = try alloc(3 * maxT * ink2.loopGateRank * 4, label: "ctrlHid")
        self.ctrlOutBuf = try alloc(6 * maxT * H * 4, label: "ctrlOut")
        self.normBuf = try alloc(3 * maxT * H * 2, label: "norm")
        self.resF32Buf = try alloc(maxT * H * 4, label: "resF32")
        self.qProjBuf = try alloc(maxT * qDim * 2, label: "qProj")
        self.qPrepBuf = try alloc(maxT * qDim * 2, label: "qPrep")
        self.kProjBuf = try alloc(maxT * kvDim * 2, label: "kProj")
        self.kConvBuf = try alloc(maxT * kvDim * 2, label: "kConv")
        self.vProjBuf = try alloc(maxT * kvDim * 2, label: "vProj")
        self.vConvBuf = try alloc(maxT * kvDim * 2, label: "vConv")
        self.rProjBuf = try alloc(maxT * nH * dRel * 2, label: "rProj")
        self.rPrepBuf = try alloc(maxT * nH * dRel * 2, label: "rPrep")
        self.sinkBuf = try alloc(maxT * nH * 2, label: "sink")
        self.gateBuf = try alloc(maxT * qDim * 2, label: "gate")
        self.attnOutBuf = try alloc(maxT * qDim * 2, label: "attnOut")
        self.oOutBuf = try alloc(maxT * H * 2, label: "oOut")
        let fusedProjElems = 2 * qDim + nH * dRel + nH + 2 * kvDim
        self.fusedAttnProjBuf = try alloc(maxT * fusedProjElems * 2, label: "fusedAttnProj")
        self.dynWF32Buf = try alloc(maxT * nKV * 4 * 4, label: "dynW")
        let partsBytes = QWAttnSinkRel.partialsBytes(
            tokens: QWAttnSinkRel.maxDecodeBatchTokens, qHeads: nH,
            headDim: headDim, splits: QWAttnSinkRel.maxSplits
        )
        self.partOBuf = try alloc(partsBytes.partOBytes, label: "partO")
        self.partMLBuf = try alloc(partsBytes.partMLBytes, label: "partML")
        let maxPairs = maxT * topK
        let maxMoETiles = QWGemm.moeMaxTiles(experts: nExp, pairs: maxPairs)
        let maxTilePlan = QWGemm.MoETilePlan(experts: nExp, pairs: maxPairs, gateUpRows: 2 * moeI, downRows: H)
        let activeArgsBytes = QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes
        self.moeTopIdsBuf = try alloc(maxPairs * 4, label: "topIds")
        self.moeTopWBuf = try alloc(maxPairs * 4, label: "topW")
        self.routerLogitsF32Buf = try alloc(maxT * nExp * 4, label: "routerLogits")
        self.moeCountsBuf = try alloc(nExp * 4, label: "moeCounts")
        self.moeOffsetsBuf = try alloc((nExp + 1) * 4, label: "moeOffsets")
        self.moeSortedBuf = try alloc(maxPairs * 4, label: "moeSorted")
        self.moeSortedTokBuf = try alloc(maxPairs * 4, label: "moeSortedTok")
        self.moeInvPermBuf = try alloc(maxPairs * 4, label: "moeInvPerm")
        self.moeActiveExpertsBuf = try alloc(nExp * 4, label: "moeActiveExperts")
        self.moeActiveArgsBuf = try alloc(activeArgsBytes, label: "moeActiveArgs")
        self.moeTileEBuf = try alloc(maxMoETiles * 4, label: "moeTileE")
        self.moeTileRBuf = try alloc(maxMoETiles * 4, label: "moeTileR")
        self.moeTileCountBuf = try alloc(maxTilePlan.scratchInts * 4, label: "moeTileCount")
        self.moePrefillGUBuf = try alloc(maxPairs * 2 * moeI * 2, label: "moePrefillGU")
        self.moeActBuf = try alloc(maxPairs * moeI * 2, label: "moeAct")
        self.moeSlotOutBuf = try alloc(maxPairs * H * 2, label: "moeSlotOut")
        self.moeOutBuf = try alloc(maxT * H * 2, label: "moeOut")
        self.logitsF32Buf = try alloc(16 * ink2.unpaddedVocabSize * 4, label: "logitsF32")
        let headSlots = max(1, QWHead.batchedArgmaxSlots(ink2.unpaddedVocabSize))
        self.headKeysHiBuf = try alloc(16 * headSlots * 4, label: "headKeysHi")
        self.headKeysLoBuf = try alloc(16 * headSlots * 4, label: "headKeysLo")
        self.headTokensBuf = try alloc(16 * 4, label: "headTokens")
        let qwpInElems = enableQWPScratch ? max(maxT * max(2 * H, qDim), 3 * 16 * H) : 128
        let qwpOut16Elems = enableQWPScratch ? maxT * max(fusedProjElems, max(qDim, max(H, nKV * 4))) : 128
        let f32OutElems = enableQWPScratch ? max(16 * max(fusedProjElems, max(qDim, 2 * H)), min(maxT, 256) * max(fusedProjElems, max(qDim, 2 * H))) : 64
        let maxDecT = min(maxT, 16)
        let qwpMoERotElems = enableQWPScratch ? maxDecT * H : 4
        let qwpMoEGUElems = enableQWPScratch ? maxDecT * topK * 2 * moeI : 4
        let qwpMoEActElems = enableQWPScratch ? maxDecT * topK * moeI : 4
        let qwpMoEOutElems = enableQWPScratch ? maxDecT * H : 4
        self.qwpF32OutElems = f32OutElems
        self.qwpF16InBuf = try alloc(qwpInElems * 2, label: "qwpF16In")
        self.qwpF16OutBuf = try alloc(qwpOut16Elems * 2, label: "qwpF16Out")
        self.qwpF32OutBuf = try alloc(f32OutElems * 4, label: "qwpF32Out")
        self.qwpMoERotF32Buf = try alloc(qwpMoERotElems * 4, label: "qwpMoERotF32")
        self.qwpMoEGUF32Buf = try alloc(qwpMoEGUElems * 4, label: "qwpMoEGUF32")
        self.qwpMoEActF32Buf = try alloc(qwpMoEActElems * 4, label: "qwpMoEActF32")
        self.qwpMoEOutF32Buf = try alloc(qwpMoEOutElems * 4, label: "qwpMoEOutF32")

        func tensor(_ b: any MTLBuffer, count: Int, _ dt: QWDType) -> QWTensorValue {
            QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length),
                          offset: 0, shape: QWTensorShape([count], dt))
        }
        self.normSlice = QWFPBufferSlice(buffer: normBuf, offset: 0, bytes: normBuf.length)
        self.moeOutSlice = QWFPBufferSlice(buffer: moeOutBuf, offset: 0, bytes: moeOutBuf.length)
        self.routerLogitsSlice = QWFPBufferSlice(buffer: routerLogitsF32Buf, offset: 0, bytes: routerLogitsF32Buf.length)
        self.ctrlPairSlice = QWFPBufferSlice(buffer: ctrlPairBuf, offset: 0, bytes: ctrlPairBuf.length)
        self.qwpF16InSlice = QWFPBufferSlice(buffer: qwpF16InBuf, offset: 0, bytes: qwpF16InBuf.length)
        self.qwpF16InTV = tensor(qwpF16InBuf, count: qwpInElems, .f16)
        self.qwpF16OutTV = tensor(qwpF16OutBuf, count: qwpOut16Elems, .f16)
        self.qwpF32OutTV = tensor(qwpF32OutBuf, count: f32OutElems, .f32)
        self.qwpMoERotF32TV = tensor(qwpMoERotF32Buf, count: qwpMoERotElems, .f32)
        self.qwpMoEGUF32TV = tensor(qwpMoEGUF32Buf, count: qwpMoEGUElems, .f32)
        self.qwpMoEGUF32Slice = QWFPBufferSlice(buffer: qwpMoEGUF32Buf, offset: 0, bytes: qwpMoEGUF32Buf.length)
        self.qwpMoEActF32TV = tensor(qwpMoEActF32Buf, count: qwpMoEActElems, .f32)
        self.qwpMoEActF32Slice = QWFPBufferSlice(buffer: qwpMoEActF32Buf, offset: 0, bytes: qwpMoEActF32Buf.length)
        self.qwpMoEOutF32TV = tensor(qwpMoEOutF32Buf, count: qwpMoEOutElems, .f32)
        self.headKeysHiTV = tensor(headKeysHiBuf, count: 16 * headSlots, .i32)
        self.headKeysLoTV = tensor(headKeysLoBuf, count: 16 * headSlots, .i32)
        self.headTokensTV = tensor(headTokensBuf, count: 16, .i32)
        self.logitsF32TV = tensor(logitsF32Buf, count: 16 * ink2.unpaddedVocabSize, .f32)
        self.qPrepTV = tensor(qPrepBuf, count: maxT * qDim, .bf16)
        self.rPrepTV = tensor(rPrepBuf, count: maxT * nH * dRel, .bf16)
        self.attnOutTV = tensor(attnOutBuf, count: maxT * qDim, .bf16)
        self.partialsView = QWSinkRelPartials(
            partO: QWMRegion(buffer: partOBuf, base: 0, length: partOBuf.length),
            partML: QWMRegion(buffer: partMLBuf, base: 0, length: partMLBuf.length)
        )
        self.moeTopIdsTV = tensor(moeTopIdsBuf, count: maxPairs, .i32)
        self.moeTopWTV = tensor(moeTopWBuf, count: maxPairs, .f32)
        self.moeOffsetsTV = tensor(moeOffsetsBuf, count: nExp + 1, .i32)
        self.moeSortedTV = tensor(moeSortedBuf, count: maxPairs, .i32)
        self.moeSortedTokTV = tensor(moeSortedTokBuf, count: maxPairs, .i32)
        self.moeInvPermTV = tensor(moeInvPermBuf, count: maxPairs, .i32)
        self.moeActiveExpertsTV = tensor(moeActiveExpertsBuf, count: nExp, .i32)
        self.moeActiveArgsTV = tensor(moeActiveArgsBuf, count: activeArgsBytes / 4, .i32)
        self.moeTileETV = tensor(moeTileEBuf, count: maxMoETiles, .i32)
        self.moeTileRTV = tensor(moeTileRBuf, count: maxMoETiles, .i32)
        self.moeTileCountTV = tensor(moeTileCountBuf, count: maxTilePlan.scratchInts, .i32)
        self.moePrefillGUTV = tensor(moePrefillGUBuf, count: maxPairs * 2 * moeI, .bf16)
        self.moeActTV = tensor(moeActBuf, count: maxPairs * moeI, .bf16)
        self.moeSlotOutTV = tensor(moeSlotOutBuf, count: maxPairs * H, .bf16)
        self.moePrefillGUF16TV = tensor(moePrefillGUBuf, count: maxPairs * 2 * moeI, .f16)
        self.moeActF16TV = tensor(moeActBuf, count: maxPairs * moeI, .f16)
        self.moeSlotOutF16TV = tensor(moeSlotOutBuf, count: maxPairs * H, .f16)

        self.convSlots = convSlots
        var cvOffsets: [Int] = []
        var cvChans: [Int] = []
        var cvCursor = 0
        let maxVerifyRows = 16
        let verifyKvBytes = ((maxVerifyRows * kvDim * 2 + 255) / 256) * 256
        let verifyHBytes = ((maxVerifyRows * H * 2 + 255) / 256) * 256
        self.convVerifyKvStrideElems = verifyKvBytes / 2
        self.convVerifyHStrideElems = verifyHBytes / 2
        for _ in 0..<numVisits {
            for (ch, stepBytes) in [(kvDim, verifyKvBytes), (kvDim, verifyKvBytes), (H, verifyHBytes), (H, verifyHBytes)] {
                cvOffsets.append(cvCursor)
                cvChans.append(ch)
                cvCursor += stepBytes
            }
        }
        self.convVerifyOffsets = cvOffsets
        self.convVerifyChannels = cvChans
        if enableSpeculative {
            self.convPreVerifyBuf = try alloc(slot0.state.convStatesSlab.length, label: "convPreVerify")
            self.convVerifyInputsBuf = try alloc(cvCursor, label: "convVerifyInputs")
        } else {
            self.convPreVerifyBuf = nil
            self.convVerifyInputsBuf = nil
        }

        var allScratch: [any MTLBuffer] = [
            tokIdsBuf, xBuf, preludeBuf, blocksRingBuf, attnResOutBuf, bankBuf, msgOutBuf,
            ctrlPairBuf, ctrlHidBuf, ctrlOutBuf, normBuf, resF32Buf, qProjBuf, qPrepBuf,
            kProjBuf, kConvBuf, vProjBuf, vConvBuf, rProjBuf, rPrepBuf, sinkBuf, gateBuf,
            attnOutBuf, oOutBuf, fusedAttnProjBuf, dynWF32Buf, partOBuf, partMLBuf,
            moeTopIdsBuf, moeTopWBuf, routerLogitsF32Buf, moeCountsBuf, moeOffsetsBuf,
            moeSortedBuf, moeSortedTokBuf, moeInvPermBuf, moeActiveExpertsBuf, moeActiveArgsBuf,
            moeTileEBuf, moeTileRBuf, moeTileCountBuf, moePrefillGUBuf, moeActBuf, moeSlotOutBuf,
            moeOutBuf, logitsF32Buf, headKeysHiBuf, headKeysLoBuf, headTokensBuf,
            qwpF16InBuf, qwpF16OutBuf, qwpF32OutBuf,
            qwpMoERotF32Buf, qwpMoEGUF32Buf, qwpMoEActF32Buf, qwpMoEOutF32Buf,
            dummyB, dummyF
        ]
        if let pre = convPreVerifyBuf { allScratch.append(pre) }
        if let vIn = convVerifyInputsBuf { allScratch.append(vIn) }
        self.scratchBytes = allScratch.reduce(0) { $0 + $1.length }
        resetConvStates()
    }

    private static func allocAlignedBuffer(
        device: any MTLDevice,
        bytes: Int,
        label: String
    ) throws -> any MTLBuffer {
        let len = max(16, ((bytes + 255) / 256) * 256)
        guard let b = device.makeBuffer(length: len, options: .storageModeShared) else {
            throw QWInkling2Error.allocationFailed("\(label) \(len) bytes")
        }
        return b
    }

    private static func allocateSlotStorage(
        device: any MTLDevice,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config,
        executionPlan: [Int],
        slotIndex: Int,
        capacity: Int,
        maxChunk: Int,
        kvModeRaw: UInt32,
        kvMode: QWKVMode
    ) throws -> (state: Ink2SlotState, convOffsets: [Int], kvStrideElems: Int, hStrideElems: Int) {
        let H = descriptor.hidden
        let kvDim = descriptor.kvHeads * descriptor.headDim
        let nKV = descriptor.kvHeads
        let convSlots = max(1, ink2.sconvKernelSize - 1)
        let numVisits = executionPlan.count
        let payRowBytes: Int = switch kvModeRaw {
        case 0: kvDim * 2
        case 1: kvDim
        default: kvDim / 2
        }
        let metaRowBytes: Int = switch kvModeRaw {
        case 0: 4
        case 1: nKV * 2
        default: nKV * 4
        }

        let stateKvBytes = ((convSlots * kvDim * 4 + 255) / 256) * 256
        let stateHBytes = ((convSlots * H * 4 + 255) / 256) * 256
        let visitStateBytes = 2 * stateKvBytes + 2 * stateHBytes
        let convSlab = try allocAlignedBuffer(
            device: device, bytes: numVisits * visitStateBytes, label: "slot\(slotIndex).convStatesSlab"
        )
        var kvs: [Ink2VisitKV] = []
        var cOffsets: [Int] = []
        var stateBuffers: [any MTLBuffer] = [convSlab]
        kvs.reserveCapacity(numVisits)
        cOffsets.reserveCapacity(numVisits * 4)
        stateBuffers.reserveCapacity(numVisits * 4 + 1)
        var kvMem = convSlab.length

        for v in 0..<numVisits {
            let l = executionPlan[v]
            let isSliding = ink2.isSlidingLayer(l)
            let ringCap = isSliding ? min(capacity, ink2.slidingWindowSize + maxChunk) : capacity
            let kc = try allocAlignedBuffer(device: device, bytes: ringCap * payRowBytes, label: "s\(slotIndex).kvK[\(v)]")
            let vc = try allocAlignedBuffer(device: device, bytes: ringCap * payRowBytes, label: "s\(slotIndex).kvV[\(v)]")
            let km = try allocAlignedBuffer(device: device, bytes: max(16, ringCap * metaRowBytes), label: "s\(slotIndex).kvKM[\(v)]")
            let vm = try allocAlignedBuffer(device: device, bytes: max(16, ringCap * metaRowBytes), label: "s\(slotIndex).kvVM[\(v)]")
            kvMem += kc.length + vc.length + km.length + vm.length
            stateBuffers.append(kc); stateBuffers.append(vc); stateBuffers.append(km); stateBuffers.append(vm)
            let kMetaReg = kvMode == .f16 ? nil : QWMRegion(buffer: km, base: 0, length: km.length)
            let vMetaReg = kvMode == .f16 ? nil : QWMRegion(buffer: vm, base: 0, length: vm.length)
            let cacheView = QWSinkRelKVCache(
                kCache: QWMRegion(buffer: kc, base: 0, length: kc.length),
                vCache: QWMRegion(buffer: vc, base: 0, length: vc.length),
                kMeta: kMetaReg, vMeta: vMetaReg, ringCap: ringCap, mode: kvMode
            )
            kvs.append(Ink2VisitKV(
                isSliding: isSliding, ringCap: ringCap,
                kCache: kc, vCache: vc, kMeta: km, vMeta: vm,
                rowBytes: payRowBytes, metaRowBytes: metaRowBytes,
                kvCacheView: cacheView
            ))
            let stBase = v * visitStateBytes
            cOffsets.append(stBase)
            cOffsets.append(stBase + stateKvBytes)
            cOffsets.append(stBase + 2 * stateKvBytes)
            cOffsets.append(stBase + 2 * stateKvBytes + stateHBytes)
        }
        let state = Ink2SlotState(
            slotIndex: slotIndex,
            capacity: capacity,
            visitKVs: kvs,
            convStatesSlab: convSlab,
            residentStateBytes: kvMem,
            stateBuffers: stateBuffers
        )
        state.resetConvStates()
        return (state, cOffsets, stateKvBytes / 4, stateHBytes / 4)
    }

    static func makeCompanionSlotState(
        device: any MTLDevice,
        descriptor: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config,
        slotIndex: Int,
        capacity: Int,
        maxChunk: Int,
        kvModeRaw: UInt32,
        kvMode: QWKVMode
    ) throws -> Ink2SlotState {
        try allocateSlotStorage(
            device: device, descriptor: descriptor, ink2: ink2, executionPlan: descriptor.executionPlan,
            slotIndex: slotIndex, capacity: capacity, maxChunk: maxChunk,
            kvModeRaw: kvModeRaw, kvMode: kvMode
        ).state
    }

    func resetConvStates() {
        primarySlotState.resetConvStates()
    }

    private struct SConvVerifyCommitParams {
        var numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride: UInt32
    }

    func commitVerifyConv(hal: QWHal, kept: Int, profiler: QWInkling2Profiler? = nil) throws {
        guard kept > 0, convVerifyInputsBuf != nil else { return }
        let cbOpt = (profiler == nil)
            ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
            : hal.queue.makeCommandBuffer()
        guard let cb = cbOpt else { throw QWInkling2Error.badState("failed to create sconv verify commit CB") }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        profiler?.beginStage(.sconv, on: cb, encoder: &encOpt, hal: hal)
        if encOpt == nil { encOpt = cb.makeComputeCommandEncoder() }
        guard let enc = encOpt else { throw QWInkling2Error.badState("failed to create sconv commit encoder") }
        try encodeCommitVerifyConv(enc, hal: hal, kept: kept)
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw QWInkling2Error.badState("sconv verify commit failed: \(err.localizedDescription)") }
    }

    func encodeCommitVerifyConv(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        kept: Int
    ) throws {
        guard kept > 0, let vInBuf = convVerifyInputsBuf else { return }
        let H = descriptor.hidden, kvDim = descriptor.kvHeads * descriptor.headDim, numVisits = executionPlan.count
        var p = SConvVerifyCommitParams(
            numVisits: UInt32(numVisits), kvDim: UInt32(kvDim), H: UInt32(H), kept: UInt32(kept),
            stateKvStride: UInt32(convStateKvStrideElems), stateHStride: UInt32(convStateHStrideElems),
            verifyKvStride: UInt32(convVerifyKvStrideElems), verifyHStride: UInt32(convVerifyHStrideElems)
        )
        let totalCh = 2 * kvDim + 2 * H
        enc.setComputePipelineState(try hal.pipeline("qwg_ink2_sconv_verify_commit"))
        enc.setBuffer(convStatesSlab, offset: 0, index: 0)
        enc.setBuffer(vInBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SConvVerifyCommitParams>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: totalCh, height: numVisits, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, totalCh), height: 1, depth: 1))
    }

    func commitVerifyConvStates(kept: Int) {
        guard kept > 0 else { return }
        commitVerifyConvStates(keptNodes: Array(0..<kept))
    }

    func commitVerifyConvStates(keptNodes: [Int]) {
        let kept = keptNodes.count
        guard kept > 0,
              let preBuf = convPreVerifyBuf,
              let vInBuf = convVerifyInputsBuf else { return }
        let preBase = preBuf.contents()
        let slabBase = convStatesSlab.contents()
        let inBase = vInBuf.contents()
        for (cIdx, stOff) in convStateOffsets.enumerated() {
            let ch = convVerifyChannels[cIdx]
            let pre = preBase.advanced(by: stOff).assumingMemoryBound(to: Float.self)
            let xIn = inBase.advanced(by: convVerifyOffsets[cIdx]).assumingMemoryBound(to: UInt16.self)
            let dst = slabBase.advanced(by: stOff).assumingMemoryBound(to: Float.self)
            @inline(__always) func xAt(_ r: Int, _ c: Int) -> Float {
                Float(bitPattern: UInt32(xIn[r * ch + c]) << 16)
            }
            if kept >= 3 {
                let r0 = keptNodes[kept - 3], r1 = keptNodes[kept - 2], r2 = keptNodes[kept - 1]
                for c in 0..<ch {
                    dst[c] = xAt(r0, c)
                    dst[ch + c] = xAt(r1, c)
                    dst[2 * ch + c] = xAt(r2, c)
                }
            } else if kept == 2 {
                let r0 = keptNodes[0], r1 = keptNodes[1]
                for c in 0..<ch {
                    dst[c] = pre[2 * ch + c]
                    dst[ch + c] = xAt(r0, c)
                    dst[2 * ch + c] = xAt(r1, c)
                }
            } else {
                let r0 = keptNodes[0]
                for c in 0..<ch {
                    dst[c] = pre[ch + c]
                    dst[ch + c] = pre[2 * ch + c]
                    dst[2 * ch + c] = xAt(r0, c)
                }
            }
        }
    }

    var allStateBuffers: [any MTLBuffer] {
        var out: [any MTLBuffer] = []
        out.reserveCapacity(primarySlotState.stateBuffers.count + 64)
        out.append(contentsOf: primarySlotState.stateBuffers)
        out.append(contentsOf: [
            tokIdsBuf, xBuf, preludeBuf, blocksRingBuf, attnResOutBuf, bankBuf, msgOutBuf,
            ctrlPairBuf, ctrlHidBuf, ctrlOutBuf, normBuf, resF32Buf, qProjBuf, qPrepBuf,
            kProjBuf, kConvBuf, vProjBuf, vConvBuf, rProjBuf, rPrepBuf, sinkBuf, gateBuf,
            attnOutBuf, oOutBuf, fusedAttnProjBuf, dynWF32Buf, partOBuf, partMLBuf,
            moeTopIdsBuf, moeTopWBuf, routerLogitsF32Buf, moeCountsBuf, moeOffsetsBuf,
            moeSortedBuf, moeSortedTokBuf, moeInvPermBuf, moeActiveExpertsBuf, moeActiveArgsBuf,
            moeTileEBuf, moeTileRBuf, moeTileCountBuf, moePrefillGUBuf, moeActBuf, moeSlotOutBuf,
            moeOutBuf, logitsF32Buf, headKeysHiBuf, headKeysLoBuf, headTokensBuf,
            qwpF16InBuf, qwpF16OutBuf, qwpF32OutBuf,
            qwpMoERotF32Buf, qwpMoEGUF32Buf, qwpMoEActF32Buf, qwpMoEOutF32Buf,
            dummyB, dummyF
        ])
        if let pre = convPreVerifyBuf { out.append(pre) }
        if let vIn = convVerifyInputsBuf { out.append(vIn) }
        return out
    }
}

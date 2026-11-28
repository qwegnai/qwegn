// QWDriverStandardState.swift — Storage box for standard-driver runtime state.
// Separated from QWEngineDriver so HyperStream (hcCount > 1) allocates zero standard KV/scratch/GDN objects.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWGDN
import QWHAL
import QWHead
import QWKV
import QWMLP
import QWMTP
import QWNorm
import QWQuant
import QWRope

final class QWDriverStandardState: @unchecked Sendable {
    let pack: QWQuant.Pack?
    let embedReader: QWSafeTensorsReader?
    let embedScalesReader: QWSafeTensorsReader?
    let embedBiasesReader: QWSafeTensorsReader?
    let arena: QWArena
    let kvCache: QWKVCache
    let ropeApply: QWRopeApply
    let attnLayers: [Int]
    let hcaLayers: [Int]
    let gdnLayers: [Int]
    let attnDecode: QWAttnDecode?
    let pagedAttnPrefill: QWAttnPagedPrefill
    let pagedAttnPrefillGeneral: QWAttnPagedPrefillGeneral
    let gdnConv: QWGDNConv
    let gdnDecode: QWGDNDecode
    let gdnPrefill: QWGDNPrefill
    let head: QWHead
    let pipelines: QWEngineDriver.CorePipelines
    let mhcSplit: QWMHCSplitRouter?
    let attnPrefill: QWAttnPrefill
    let gdnIndexMap: [Int]
    let attnIndexMap: [Int]
    let hcaIndexMap: [Int]
    let layersW: [QWEngineDriver.LayerW]
    let mtpWeights: QWEngineDriver.MTPWeights?
    let echoWeights: QWEngineDriver.EchoDraftWeights?
    let mhcExitHead: QWEngineDriver.MHCExitHeadWeights?
    let multimaxParams: QWTensorValue?
    let embedPlane: String
    let embedRawOff: Int
    let mtpDeviceEmbedding: QWMTPEmbeddingTable?
    let mtpBatchedMLP: QWMTPBatchedMLP?
    let mtpBatchedTrunk: QWMTPBatchedTrunk?
    let mtpSharedAttentionPartials: QWMRegion?
    let embedStaging: any MTLBuffer
    let headStorage: QWHeadStorage
    let finalNorm: QWTensorValue
    let finalGatedNorm: QWGatedNormWeights?
    let mtpDraftHead: QWMTPTDraftHead?
    let mtpScratch: QWEngineDriver.MTPScratch?
    let mtpDraftEncoder: QWMTPDraftEncoder?
    let echoDrafter: QWEchoDraftEncoder?
    let dflashDrafter: QWDFlashDrafter?
    let dsparkDrafter: QWDSparkDrafter?
    let gdnReplayCheckpoint: QWGDNReplayCheckpoint?
    let mtpPromptEmbeddings: QWTensorValue?
    let verifyWindow: QWVerifyWindow?
    var prefixStore: QWEnginePrefixStore<QWEngineCachedPrefix>?
    var cacheDiskManager: QWCacheDiskManager?
    let coldWriteController = QWAsyncColdWriteController()
    let mtpCache: QWKVCache?
    let sc: QWEngineDriver.Scratch
    let psc: QWEngineDriver.PrefillScratch
    let lsaOperator: QWLSAOperator?
    let mlaOperator: QWMLAAttnOperator?
    let mbsaOperator: QWMBSAAttnOperator?
    let hcaOperator: QWHCAAttnOperator?
    let hcaLsaBuffer: (any MTLBuffer)?
    let pscBuffer: any MTLBuffer
    let pscByteOffset: Int
    let pscByteLength: Int

    init(
        pack: QWQuant.Pack?,
        embedReader: QWSafeTensorsReader?,
        embedScalesReader: QWSafeTensorsReader?,
        embedBiasesReader: QWSafeTensorsReader?,
        arena: QWArena,
        kvCache: QWKVCache,
        ropeApply: QWRopeApply,
        attnLayers: [Int],
        hcaLayers: [Int],
        gdnLayers: [Int],
        attnDecode: QWAttnDecode?,
        pagedAttnPrefill: QWAttnPagedPrefill,
        pagedAttnPrefillGeneral: QWAttnPagedPrefillGeneral,
        gdnConv: QWGDNConv,
        gdnDecode: QWGDNDecode,
        gdnPrefill: QWGDNPrefill,
        head: QWHead,
        pipelines: QWEngineDriver.CorePipelines,
        mhcSplit: QWMHCSplitRouter?,
        attnPrefill: QWAttnPrefill,
        gdnIndexMap: [Int],
        attnIndexMap: [Int],
        hcaIndexMap: [Int],
        layersW: [QWEngineDriver.LayerW],
        mtpWeights: QWEngineDriver.MTPWeights?,
        echoWeights: QWEngineDriver.EchoDraftWeights?,
        mhcExitHead: QWEngineDriver.MHCExitHeadWeights?,
        multimaxParams: QWTensorValue?,
        embedPlane: String,
        embedRawOff: Int,
        mtpDeviceEmbedding: QWMTPEmbeddingTable?,
        mtpBatchedMLP: QWMTPBatchedMLP?,
        mtpBatchedTrunk: QWMTPBatchedTrunk?,
        mtpSharedAttentionPartials: QWMRegion?,
        embedStaging: any MTLBuffer,
        headStorage: QWHeadStorage,
        finalNorm: QWTensorValue,
        finalGatedNorm: QWGatedNormWeights? = nil,
        mtpDraftHead: QWMTPTDraftHead?,
        mtpScratch: QWEngineDriver.MTPScratch?,
        mtpDraftEncoder: QWMTPDraftEncoder?,
        echoDrafter: QWEchoDraftEncoder?,
        dflashDrafter: QWDFlashDrafter?,
        dsparkDrafter: QWDSparkDrafter?,
        gdnReplayCheckpoint: QWGDNReplayCheckpoint?,
        mtpPromptEmbeddings: QWTensorValue?,
        verifyWindow: QWVerifyWindow?,
        mtpCache: QWKVCache?,
        sc: QWEngineDriver.Scratch,
        psc: QWEngineDriver.PrefillScratch,
        lsaOperator: QWLSAOperator?,
        mlaOperator: QWMLAAttnOperator?,
        mbsaOperator: QWMBSAAttnOperator?,
        hcaOperator: QWHCAAttnOperator?,
        hcaLsaBuffer: (any MTLBuffer)?,
        pscBuffer: any MTLBuffer,
        pscByteOffset: Int,
        pscByteLength: Int
    ) {
        self.pack = pack
        self.embedReader = embedReader
        self.embedScalesReader = embedScalesReader
        self.embedBiasesReader = embedBiasesReader
        self.arena = arena
        self.kvCache = kvCache
        self.ropeApply = ropeApply
        self.attnLayers = attnLayers
        self.hcaLayers = hcaLayers
        self.gdnLayers = gdnLayers
        self.attnDecode = attnDecode
        self.pagedAttnPrefill = pagedAttnPrefill
        self.pagedAttnPrefillGeneral = pagedAttnPrefillGeneral
        self.gdnConv = gdnConv
        self.gdnDecode = gdnDecode
        self.gdnPrefill = gdnPrefill
        self.head = head
        self.pipelines = pipelines
        self.mhcSplit = mhcSplit
        self.attnPrefill = attnPrefill
        self.gdnIndexMap = gdnIndexMap
        self.attnIndexMap = attnIndexMap
        self.hcaIndexMap = hcaIndexMap
        self.layersW = layersW
        self.mtpWeights = mtpWeights
        self.echoWeights = echoWeights
        self.mhcExitHead = mhcExitHead
        self.multimaxParams = multimaxParams
        self.embedPlane = embedPlane
        self.embedRawOff = embedRawOff
        self.mtpDeviceEmbedding = mtpDeviceEmbedding
        self.mtpBatchedMLP = mtpBatchedMLP
        self.mtpBatchedTrunk = mtpBatchedTrunk
        self.mtpSharedAttentionPartials = mtpSharedAttentionPartials
        self.embedStaging = embedStaging
        self.headStorage = headStorage
        self.finalNorm = finalNorm
        self.finalGatedNorm = finalGatedNorm
        self.mtpDraftHead = mtpDraftHead
        self.mtpScratch = mtpScratch
        self.mtpDraftEncoder = mtpDraftEncoder
        self.echoDrafter = echoDrafter
        self.dflashDrafter = dflashDrafter
        self.dsparkDrafter = dsparkDrafter
        self.gdnReplayCheckpoint = gdnReplayCheckpoint
        self.mtpPromptEmbeddings = mtpPromptEmbeddings
        self.verifyWindow = verifyWindow
        self.mtpCache = mtpCache
        self.sc = sc
        self.psc = psc
        self.lsaOperator = lsaOperator
        self.mlaOperator = mlaOperator
        self.mbsaOperator = mbsaOperator
        self.hcaOperator = hcaOperator
        self.hcaLsaBuffer = hcaLsaBuffer
        self.pscBuffer = pscBuffer
        self.pscByteOffset = pscByteOffset
        self.pscByteLength = pscByteLength
    }
}

extension QWEngineDriver {
    func requireStandard() -> QWDriverStandardState {
        guard let s = standardState else {
            preconditionFailure("standard runtime state unavailable in HyperStream mode")
        }
        return s
    }

    var pack: QWQuant.Pack? { standardState?.pack }
    var embedReader: QWSafeTensorsReader? { standardState?.embedReader }
    var embedScalesReader: QWSafeTensorsReader? { standardState?.embedScalesReader }
    var embedBiasesReader: QWSafeTensorsReader? { standardState?.embedBiasesReader }
    var arena: QWArena { requireStandard().arena }
    var kvCache: QWKVCache { requireStandard().kvCache }
    var ropeApply: QWRopeApply { requireStandard().ropeApply }
    var attnLayers: [Int] { requireStandard().attnLayers }
    var hcaLayers: [Int] { requireStandard().hcaLayers }
    var gdnLayers: [Int] { requireStandard().gdnLayers }
    var attnDecode: QWAttnDecode? { standardState?.attnDecode }
    var pagedAttnPrefill: QWAttnPagedPrefill { requireStandard().pagedAttnPrefill }
    var pagedAttnPrefillGeneral: QWAttnPagedPrefillGeneral { requireStandard().pagedAttnPrefillGeneral }
    var gdnConv: QWGDNConv { requireStandard().gdnConv }
    var gdnDecode: QWGDNDecode { requireStandard().gdnDecode }
    var gdnPrefill: QWGDNPrefill { requireStandard().gdnPrefill }
    var head: QWHead { requireStandard().head }
    var psoBf16F16: any MTLComputePipelineState { requireStandard().pipelines.psoBf16F16 }
    var psoGEMVDxH16T4: any MTLComputePipelineState { requireStandard().pipelines.psoGEMVDxH16T4 }
    var psoGEMVDxH16DualT4: any MTLComputePipelineState { requireStandard().pipelines.psoGEMVDxH16DualT4 }
    var psoGEMVDxH16GdnInProjT4: any MTLComputePipelineState { requireStandard().pipelines.psoGEMVDxH16GdnInProjT4 }
    var psoGEMVSiluPairH16: any MTLComputePipelineState { requireStandard().pipelines.psoGEMVSiluPairH16 }
    var psoFwht32: any MTLComputePipelineState { requireStandard().pipelines.psoFwht32 }
    var psoRMSNorm: any MTLComputePipelineState { requireStandard().pipelines.psoRMSNorm }
    var psoRMSNormResidual: any MTLComputePipelineState { requireStandard().pipelines.psoRMSNormResidual }
    var psoRMSNormResidualWide: any MTLComputePipelineState { requireStandard().pipelines.psoRMSNormResidualWide }
    var psoRMSNormResidualMoE: any MTLComputePipelineState { requireStandard().pipelines.psoRMSNormResidualMoE }
    var psoRMSNormResidualMoEWide: any MTLComputePipelineState { requireStandard().pipelines.psoRMSNormResidualMoEWide }
    var psoHeadF16: any MTLComputePipelineState { requireStandard().pipelines.psoHeadF16 }
    var psoGemmBatchedH16: (any MTLComputePipelineState)? { standardState?.pipelines.psoGemmBatchedH16 }
    var psoGemmBatchedH16SwiGLU: (any MTLComputePipelineState)? { standardState?.pipelines.psoGemmBatchedH16SwiGLU }
    var psoGemmBatchedH16GdnInProj: (any MTLComputePipelineState)? { standardState?.pipelines.psoGemmBatchedH16GdnInProj }
    var psoSiluMulF16: any MTLComputePipelineState { requireStandard().pipelines.psoSiluMulF16 }
    var psoKVPagedWrite: any MTLComputePipelineState { requireStandard().pipelines.psoKVPagedWrite }
    var psoKVPagedWriteGeneral: any MTLComputePipelineState { requireStandard().pipelines.psoKVPagedWriteGeneral }
    var psoZeroPadAttn: any MTLComputePipelineState { requireStandard().pipelines.psoZeroPadAttn }
    var psoMHCEnter: (any MTLComputePipelineState)? { standardState?.pipelines.psoMHCEnter }
    var psoMHCRouterNorm: (any MTLComputePipelineState)? { standardState?.pipelines.psoMHCRouterNorm }
    var psoMHCRouterNormRows: (any MTLComputePipelineState)? { standardState?.pipelines.psoMHCRouterNormRows }
    var psoMHCMerge: (any MTLComputePipelineState)? { standardState?.pipelines.psoMHCMerge }
    var psoMHCExitHead: (any MTLComputePipelineState)? { standardState?.pipelines.psoMHCExitHead }
    var mhcSplit: QWMHCSplitRouter? { standardState?.mhcSplit }
    var attnPrefill: QWAttnPrefill { requireStandard().attnPrefill }
    var gdnIndexMap: [Int] { requireStandard().gdnIndexMap }
    var attnIndexMap: [Int] { requireStandard().attnIndexMap }
    var hcaIndexMap: [Int] { requireStandard().hcaIndexMap }
    var layersW: [LayerW] { requireStandard().layersW }
    var mtpWeights: MTPWeights? { standardState?.mtpWeights }
    var echoWeights: EchoDraftWeights? { standardState?.echoWeights }
    var mhcExitHead: MHCExitHeadWeights? { standardState?.mhcExitHead }
    var multimaxParams: QWTensorValue? { standardState?.multimaxParams }
    var embedPlane: String { requireStandard().embedPlane }
    var embedRawOff: Int { requireStandard().embedRawOff }
    var mtpDeviceEmbedding: QWMTPEmbeddingTable? { standardState?.mtpDeviceEmbedding }
    var mtpBatchedMLP: QWMTPBatchedMLP? { standardState?.mtpBatchedMLP }
    var mtpBatchedTrunk: QWMTPBatchedTrunk? { standardState?.mtpBatchedTrunk }
    var mtpSharedAttentionPartials: QWMRegion? { standardState?.mtpSharedAttentionPartials }
    var embedStaging: any MTLBuffer { requireStandard().embedStaging }
    var headStorage: QWHeadStorage { requireStandard().headStorage }
    var finalNorm: QWTensorValue { requireStandard().finalNorm }
    var finalGatedNorm: QWGatedNormWeights? { standardState?.finalGatedNorm }
    var mtpDraftHead: QWMTPTDraftHead? { standardState?.mtpDraftHead }
    var mtpScratch: MTPScratch? { standardState?.mtpScratch }
    var mtpDraftEncoder: QWMTPDraftEncoder? { standardState?.mtpDraftEncoder }
    var echoDrafter: QWEchoDraftEncoder? { standardState?.echoDrafter }
    var dflashDrafter: QWDFlashDrafter? { standardState?.dflashDrafter }
    var dsparkDrafter: QWDSparkDrafter? { standardState?.dsparkDrafter }
    var gdnReplayCheckpoint: QWGDNReplayCheckpoint? { standardState?.gdnReplayCheckpoint }
    var mtpPromptEmbeddings: QWTensorValue? { standardState?.mtpPromptEmbeddings }
    var verifyWindow: QWVerifyWindow? { standardState?.verifyWindow }
    var prefixStore: QWEnginePrefixStore<QWEngineCachedPrefix>? {
        get { standardState?.prefixStore }
        set { standardState?.prefixStore = newValue }
    }
    var cacheDiskManager: QWCacheDiskManager? {
        get { standardState?.cacheDiskManager }
        set { standardState?.cacheDiskManager = newValue }
    }
    var coldWriteController: QWAsyncColdWriteController { requireStandard().coldWriteController }
    var mtpCache: QWKVCache? { standardState?.mtpCache }
    var sc: Scratch { requireStandard().sc }
    var psc: PrefillScratch { requireStandard().psc }
    public var lsaOperator: QWLSAOperator? { standardState?.lsaOperator }
    public var mlaOperator: QWMLAAttnOperator? { standardState?.mlaOperator }
    public var mbsaOperator: QWMBSAAttnOperator? { standardState?.mbsaOperator }
    public var hcaOperator: QWHCAAttnOperator? { standardState?.hcaOperator }
    var hcaLsaBuffer: (any MTLBuffer)? { standardState?.hcaLsaBuffer }
    var pscBuffer: any MTLBuffer { requireStandard().pscBuffer }
    var pscByteOffset: Int { requireStandard().pscByteOffset }
    var pscByteLength: Int { requireStandard().pscByteLength }
}

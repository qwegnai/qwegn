// QWEngineDriver+Init.swift — Startup resource builders for QWEngineDriver initialization.

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
import QWLoad

extension QWEngineDriver {
    struct StartupContainers {
        let cfg: StartupConfig
        let tuning: QWEngineTuning
        let osReserveGiB: Double?
        let geometry: QWModelGeometry
        let hal: QWHal
        let pack: QWQuant.Pack?
        let stContainer: QWSafeTensorsContainer?
        let weightFormat: QWWeightFormat
        let weightFormatDeclineReason: String?
        let mtpDrafts: Int
        let mtpWeightDrafts: Int
        let prefillChunkSize: Int
    }

    struct CorePipelines {
        let psoBf16F16: any MTLComputePipelineState
        let psoGEMVDxH16T4: any MTLComputePipelineState
        let psoGEMVDxH16DualT4: any MTLComputePipelineState
        let psoGEMVDxH16GdnInProjT4: any MTLComputePipelineState
        let psoGEMVSiluPairH16: any MTLComputePipelineState
        let psoFwht32: any MTLComputePipelineState
        let psoRMSNorm: any MTLComputePipelineState
        let psoRMSNormResidual: any MTLComputePipelineState
        let psoRMSNormResidualWide: any MTLComputePipelineState
        let psoRMSNormResidualMoE: any MTLComputePipelineState
        let psoRMSNormResidualMoEWide: any MTLComputePipelineState
        let psoHeadF16: any MTLComputePipelineState
        let psoGemmBatchedH16: (any MTLComputePipelineState)?
        let psoGemmBatchedH16SwiGLU: (any MTLComputePipelineState)?
        let psoGemmBatchedH16GdnInProj: (any MTLComputePipelineState)?
        let psoSiluMulF16: any MTLComputePipelineState
        let psoKVPagedWrite: any MTLComputePipelineState
        let psoKVPagedWriteGeneral: any MTLComputePipelineState
        let psoZeroPadAttn: any MTLComputePipelineState
        let psoMHCEnter: (any MTLComputePipelineState)?
        let psoMHCRouterNorm: (any MTLComputePipelineState)?
        let psoMHCRouterNormRows: (any MTLComputePipelineState)?
        let psoMHCMerge: (any MTLComputePipelineState)?
        let psoMHCExitHead: (any MTLComputePipelineState)?
    }

    struct ArenaBudgetPlan {
        let effectiveMaxTokens: Int
        let effectiveRopeRows: Int
        let weightPlan: QWWeightAllocationPlan
        let trunkKVBytes: Int
        let mtpKVBytes: Int
        let actBytes: Int
        let gdnBytes: Int
        let fixedBytes: Int
        let mtpPartialsOff: Int?
        let pCarveBytes: Int
        let useGroupedNAX: Bool
        let capacityPlan: QWCapacityPlan
        let capacityLogMessage: String
        let arena: QWArena
        let availBytes: Int
    }

    struct EmbeddingAndHeadSetup {
        let embedPlane: String
        let embedRawOff: Int
        let embedReader: QWSafeTensorsReader?
        let embedScalesReader: QWSafeTensorsReader?
        let embedBiasesReader: QWSafeTensorsReader?
        let mtpDeviceEmbedding: QWMTPEmbeddingTable?
        let embedStaging: any MTLBuffer
        let mtpDraftHead: QWMTPTDraftHead?
        let headStorage: QWHeadStorage
    }

    struct MTPVerificationResources {
        let gdnReplayCheckpoint: QWGDNReplayCheckpoint?
        let verifyWindow: QWVerifyWindow?
        let mtpBatchedMLP: QWMTPBatchedMLP?
        let mtpBatchedTrunk: QWMTPBatchedTrunk?
        let mtpSharedAttentionPartials: QWMRegion?
        let mtpScratch: MTPScratch?
        let mtpPromptEmbeddings: QWTensorValue?
        let mtpDraftEncoder: QWMTPDraftEncoder?
        let echoDrafter: QWEchoDraftEncoder?
    }

    func prefixPolicyKey(mode: GDNPrefillMode) -> String {
        Self.prefixPolicyKey(backend: mtpBackend, drafts: mtpDrafts, mode: mode, cacheSalt: activeCacheSalt)
    }

    func flushPendingColdUnloads() {
        commandFrame.flushPendingColdUnloads()
    }

    static func prepareContainersAndGeometry(
        packPath: String, maxTokens: Int, ropeRows: Int, kvMode: QWKVMode,
        prefillChunkSize: Int, mtpDrafts requestedMTPDrafts: Int,
        residentExpertsPerLayer: Int?, osReserveGiB explicitOSReserveGiB: Double?,
        cfg: StartupConfig, baseTuning: QWEngineTuning,
        requestedSessions: Int? = nil
    ) throws -> StartupContainers {
        let tuning = baseTuning.overlaid(environment: cfg.environment)
        let osReserveGiB = explicitOSReserveGiB ?? tuning.osReserveGiB
        let explicitResE = residentExpertsPerLayer ?? tuning.residentExpertsPerLayer
        var rawGeometry = try QWModelGeometry.load(packPath: packPath).withResidentExperts(explicitResE)
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
        let stContainer = isSafetensors ? try QWSafeTensorsContainer(url: root.weights) : nil
        let pack = isSafetensors ? nil : try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        func ckptHasTensor(_ name: String) -> Bool {
            func rawExists(_ candidate: String) -> Bool {
                if let pk = pack { return (try? pk.entry(candidate)) != nil }
                if let st = stContainer { return st.tensor(candidate) != nil }
                return false
            }
            let resolved = QWCheckpointTensorLoader.resolveName(
                name,
                defaultPrefix: Self.P,
                tieWordEmbeddings: rawGeometry.descriptor.tieWordEmbeddings,
                rawExists: rawExists
            )
            return rawExists(resolved)
        }
        if let firstAttnLayer = rawGeometry.descriptor.fullAttentionLayers.first,
           !rawGeometry.usesLatentOrCompressedAttention {
            let attnPfx = "\(Self.P)layers.\(firstAttnLayer).self_attn."
            let ckptHasQKV = ckptHasTensor(attnPfx + "q_proj.bias")
                || ckptHasTensor(attnPfx + "k_proj.bias")
                || ckptHasTensor(attnPfx + "v_proj.bias")
            let ckptHasO = ckptHasTensor(attnPfx + "o_proj.bias")
            let resolvedQKV = rawGeometry.descriptor.hasQKVBias || rawGeometry.descriptor.hasAttnOutputBias || ckptHasQKV
            if resolvedQKV != rawGeometry.descriptor.hasQKVBias || ckptHasO != rawGeometry.descriptor.hasAttnOutputBias {
                rawGeometry = QWModelGeometry(
                    descriptor: rawGeometry.descriptor.withAttentionBias(qkv: resolvedQKV, output: ckptHasO))
            }
        }
        let nextNMTPName = "model.language_model.layers.\(rawGeometry.numLayers).enorm.weight"
        let nextNMTPAltName = "model.layers.\(rawGeometry.numLayers).enorm.weight"
        let maxwellMTPName = "mtp_layers.0.enorm.weight"
        let maxwellMTPAltName = "model.mtp_layers.0.enorm.weight"
        let hasMTPWeights: Bool
        if let st = stContainer {
            hasMTPWeights = st.tensor("mtp.pre_fc_norm_embedding.weight") != nil
                || st.tensor(nextNMTPName) != nil
                || st.tensor(nextNMTPAltName) != nil
                || st.tensor(maxwellMTPName) != nil
                || st.tensor(maxwellMTPAltName) != nil
        } else if let pk = pack {
            hasMTPWeights = (try? pk.entry("mtp.pre_fc_norm_embedding.weight")) != nil
                || (try? pk.entry(nextNMTPName)) != nil
                || (try? pk.entry(nextNMTPAltName)) != nil
                || (try? pk.entry(maxwellMTPName)) != nil
                || (try? pk.entry(maxwellMTPAltName)) != nil
        } else {
            hasMTPWeights = false
        }
        let resolvedDraftStrategy = (try? QWDraftStrategy.resolve(cfg.mtpDraftStrategy)) ?? .hybrid
        let allowWeightFreeMTP = !hasMTPWeights && cfg.mtpBackend != .dflash2 && cfg.mtpBackend != .dspark && resolvedDraftStrategy == .ngram
        let mtpDrafts = (cfg.mtpBackend == .dflash2 || cfg.mtpBackend == .dspark || hasMTPWeights)
            ? requestedMTPDrafts
            : (allowWeightFreeMTP ? min(requestedMTPDrafts, 15) : 0)
        let mtpWeightDrafts = (cfg.mtpBackend == .dflash2 || cfg.mtpBackend == .dspark) ? 0 : (hasMTPWeights ? mtpDrafts : 0)
        try Self.validateGeometry(rawGeometry, kvMode: kvMode)
        let choice = try Self.weightFormatChoice(for: root, quantization: rawGeometry.descriptor.quantization,
                                                 geometry: rawGeometry, mtpDrafts: mtpWeightDrafts, pack: pack)
        let headSpec = Self.resolveHeadSpec(st: stContainer, pack: pack, trunkFormat: choice.format,
                                            geometry: rawGeometry)
        let moeRouterIsBF16 = Self.resolveMoERouterIsBF16(
            pack: pack, geometry: rawGeometry, mtpWeightDrafts: mtpWeightDrafts)
        let resolvedCfg = try cfg.resolvingHead(
            isQ4g: headSpec.isQ4g, format: headSpec.format, moeRouterIsBF16: moeRouterIsBF16)
        let prelimHal = try QWHal(
            geometry: rawGeometry,
            weightBasis: choice.format.basis,
            gdnNormSigmoid: rawGeometry.gdnGatingConfig.map { $0.outputNormAct == .sigmoid },
            gdnGateScale: rawGeometry.gdnGatingConfig?.outputGateScale ?? 1.0
        )
        let allowExpertReduction = (explicitResE == nil && rawGeometry.numExperts > 0)
        let (geometry, plannedChunk) = try Self.planResidencyAndChunk(
            raw: rawGeometry, cfg: resolvedCfg, hal: prelimHal, format: choice.format, tuning: tuning,
            osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
            ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts, kvMode: kvMode,
            allowExpertReduction: allowExpertReduction, requestedSessions: requestedSessions)
        return StartupContainers(
            cfg: resolvedCfg,
            tuning: tuning, osReserveGiB: osReserveGiB, geometry: geometry, hal: prelimHal,
            pack: pack, stContainer: stContainer, weightFormat: choice.format,
            weightFormatDeclineReason: choice.declineReason,
            mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
            prefillChunkSize: plannedChunk)
    }

    static func resolveCorePipelines(
        hal: QWHal, weightFormat: QWWeightFormat, mhcConfig: QWModelDescriptor.MHCConfig?
    ) throws -> CorePipelines {
        let psoBf16F16 = try hal.pipeline("qwg_bf16_to_f16")
        let psoGEMVDxH16T4 = try hal.pipeline(Self.decodeKernelName(format: weightFormat))
        let psoGEMVDxH16DualT4 = try hal.pipeline("qwg_gemv_q4g_dx_h16_dual_t4")
        let psoGEMVDxH16GdnInProjT4 = try hal.pipeline(Self.gdnInProjDecodeKernelName(format: weightFormat))
        let psoGEMVSiluPairH16 = try hal.pipeline(Self.siluPairKernelName(format: weightFormat))
        let psoFwht32 = try hal.pipeline("qwg_fwht32_f16")
        let psoRMSNorm = try hal.pipeline("qwg_rmsnorm")
        let psoRMSNormResidual = try hal.pipeline("qwg_rmsnorm_residual")
        let psoRMSNormResidualWide = try hal.pipeline("qwg_rmsnorm_residual_wide")
        let psoRMSNormResidualMoE = try hal.pipeline("qwg_rmsnorm_residual_moe")
        let psoRMSNormResidualMoEWide = try hal.pipeline("qwg_rmsnorm_residual_moe_wide")
        let psoHeadF16 = try hal.pipeline("qwg_gemv_f16_v4")
        let useNAX = hal.caps.isMetal4TensorCapable
        let canSG = hal.caps.supportsSimdgroupMMA
        let psoGemmBatchedH16 = canSG
            ? try hal.pipeline(Self.gemmKernelName(format: weightFormat, useNAX: useNAX)) : nil
        let psoGemmBatchedH16SwiGLU = canSG
            ? try hal.pipeline(Self.swigluGemmKernelName(format: weightFormat, useNAX: useNAX)) : nil
        let psoGemmBatchedH16GdnInProj = canSG
            ? try hal.pipeline(Self.gdnInProjPrefillKernelName(format: weightFormat, useNAX: useNAX)) : nil
        let psoSiluMulF16 = try hal.pipeline("qwg_silu_mul_f16")
        let psoKVPagedWrite = try hal.pipeline("qwg_kv_paged_write")
        let psoKVPagedWriteGeneral = try hal.pipeline("qwg_kv_paged_write_general")
        let psoZeroPadAttn = try hal.pipeline("qwg_zero_pad_attn_f16")
        let psoMHCEnter: (any MTLComputePipelineState)?
        let psoMHCRouterNorm: (any MTLComputePipelineState)?
        let psoMHCRouterNormRows: (any MTLComputePipelineState)?
        let psoMHCMerge: (any MTLComputePipelineState)?
        let psoMHCExitHead: (any MTLComputePipelineState)?
        if let mhcCfg = mhcConfig {
            psoMHCEnter = try hal.pipeline("qwg_mhc_enter_f16")
            psoMHCRouterNorm = try hal.pipeline("qwg_mhc_router_norm_f16")
            psoMHCRouterNormRows = ProcessInfo.processInfo.environment["QW_MHC_ROWS_KERNEL"] == "0"
                ? nil : try hal.pipeline("qwg_mhc_router_norm_rows_f16")
            psoMHCMerge = try hal.pipeline("qwg_mhc_merge_f16")
            psoMHCExitHead = mhcCfg.hasExitHead ? try hal.pipeline("qwg_mhc_exit_head_f16") : nil
        } else {
            psoMHCEnter = nil; psoMHCRouterNorm = nil; psoMHCRouterNormRows = nil
            psoMHCMerge = nil; psoMHCExitHead = nil
        }
        return CorePipelines(
            psoBf16F16: psoBf16F16, psoGEMVDxH16T4: psoGEMVDxH16T4,
            psoGEMVDxH16DualT4: psoGEMVDxH16DualT4,
            psoGEMVDxH16GdnInProjT4: psoGEMVDxH16GdnInProjT4,
            psoGEMVSiluPairH16: psoGEMVSiluPairH16, psoFwht32: psoFwht32,
            psoRMSNorm: psoRMSNorm, psoRMSNormResidual: psoRMSNormResidual,
            psoRMSNormResidualWide: psoRMSNormResidualWide,
            psoRMSNormResidualMoE: psoRMSNormResidualMoE,
            psoRMSNormResidualMoEWide: psoRMSNormResidualMoEWide,
            psoHeadF16: psoHeadF16,
            psoGemmBatchedH16: psoGemmBatchedH16,
            psoGemmBatchedH16SwiGLU: psoGemmBatchedH16SwiGLU,
            psoGemmBatchedH16GdnInProj: psoGemmBatchedH16GdnInProj,
            psoSiluMulF16: psoSiluMulF16, psoKVPagedWrite: psoKVPagedWrite,
            psoKVPagedWriteGeneral: psoKVPagedWriteGeneral, psoZeroPadAttn: psoZeroPadAttn,
            psoMHCEnter: psoMHCEnter, psoMHCRouterNorm: psoMHCRouterNorm,
            psoMHCRouterNormRows: psoMHCRouterNormRows, psoMHCMerge: psoMHCMerge,
            psoMHCExitHead: psoMHCExitHead)
    }

    static func planArenaAndBudget(
        geometry: QWModelGeometry, hal: QWHal, weightFormat: QWWeightFormat,
        weightFormatDeclineReason: String?, tuning: QWEngineTuning, osReserveGiB: Double?,
        prefillChunkSize: Int, maxTokens: Int, ropeRows: Int, kvMode: QWKVMode,
        attnLayerCount: Int, gdnLayerCount: Int, cfg: StartupConfig,
        mtpDrafts: Int, mtpWeightDrafts: Int,
        requestedSessions: Int? = nil
    ) throws -> ArenaBudgetPlan {
        try QWScratchArenaBuilder.planArenaAndBudget(
            geometry: geometry, hal: hal, weightFormat: weightFormat,
            weightFormatDeclineReason: weightFormatDeclineReason, tuning: tuning,
            osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize,
            maxTokens: maxTokens, ropeRows: ropeRows, kvMode: kvMode,
            attnLayerCount: attnLayerCount, gdnLayerCount: gdnLayerCount,
            cfg: cfg, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
            requestedSessions: requestedSessions)
    }

    static func loadEmbeddingsAndHead(
        loader L: inout WeightLoader, pack: QWQuant.Pack?, stContainer: QWSafeTensorsContainer?,
        packPath: String, hal: QWHal, geometry: QWModelGeometry, cfg: StartupConfig,
        mtpDrafts: Int, hasMTPDrafter: Bool, budget: ArenaBudgetPlan
    ) throws -> EmbeddingAndHeadSetup {
        let embedName = L.resolveName(Self.P + "embed_tokens.weight")
        let embedBytes = geometry.vocab * geometry.hidden * 2
        let devEmbedEnv = cfg.environment["QW_MTP_DEVICE_EMBEDDING"] ?? cfg.environment["QW_MTP_DEVICE_EMBED"]
        let fixedBytes = budget.fixedBytes
        let allowDeviceEmbed = devEmbedEnv != "0" && hasMTPDrafter && mtpDrafts > 0 && (
            devEmbedEnv == "1"
            || (geometry.effectiveResidentExperts == geometry.numExperts
                && fixedBytes + budget.trunkKVBytes + budget.mtpKVBytes + embedBytes + (1 << 30) <= budget.availBytes)
        )
        let embedRawOff: Int
        let embedReader: QWSafeTensorsReader?
        let embedScalesReader: QWSafeTensorsReader?
        let embedBiasesReader: QWSafeTensorsReader?
        let mtpDeviceEmbedding: QWMTPEmbeddingTable?
        if let pk = pack {
            guard let rawOff = try pk.entry(embedName).rawOff else {
                throw QWEngineDriverError.weightMissing(embedName, "not a bf16 raw plane")
            }
            embedRawOff = rawOff; embedReader = nil; embedScalesReader = nil; embedBiasesReader = nil
            mtpDeviceEmbedding = allowDeviceEmbed
                ? try QWMTPEmbeddingTable(hal: hal, path: pk.url.path, offset: rawOff,
                                         vocabulary: geometry.vocab, hidden: geometry.hidden) : nil
        } else if let st = stContainer {
            embedRawOff = 0
            let eReader = try st.reader(of: embedName)
            embedReader = eReader
            if eReader.tensorInfo.dtype == .u32, embedName.hasSuffix(".weight") {
                let stem = String(embedName.dropLast(".weight".count))
                embedScalesReader = try st.reader(of: stem + ".scales")
                embedBiasesReader = try st.reader(of: stem + ".biases")
            } else {
                embedScalesReader = nil; embedBiasesReader = nil
            }
            if allowDeviceEmbed, eReader.tensorInfo.dtype == .bf16,
               let sURL = st.shardURL(of: embedName), let rng = st.byteRange(of: embedName),
               rng.lowerBound % 2 == 0 {
                mtpDeviceEmbedding = try QWMTPEmbeddingTable(hal: hal, path: sURL.path, offset: rng.lowerBound,
                                                             vocabulary: geometry.vocab, hidden: geometry.hidden)
            } else {
                mtpDeviceEmbedding = nil
            }
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }
        guard let embedStaging = hal.device.makeBuffer(length: geometry.hidden * 2, options: .storageModeShared) else {
            throw QWEngineDriverError.state("failed to allocate embed staging buffer")
        }
        let mtpDraftHead: QWMTPTDraftHead?
        if cfg.draftHeadEnabled, let sidecarPath = cfg.environment["QW_MTP_DRAFT_HEAD_SIDECAR"] {
            let sidecar = try QWQuant.Pack(url: URL(fileURLWithPath: sidecarPath),
                                           descriptor: geometry.descriptor, skipIdentityCheck: true)
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: sidecar)
        } else if cfg.draftHeadEnabled, cfg.headIsQ4g, let pk = pack {
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: pk)
        } else {
            if cfg.draftHeadEnabled && pack != nil {
                throw QWEngineDriverError.state("QW_MTP_DRAFT_HEAD=1 requires QW_MTP_DRAFT_HEAD_SIDECAR")
            }
            mtpDraftHead = nil
        }
        let headStorage = try Self.loadHead(loader: &L, headIsQ4g: cfg.headIsQ4g,
                                            headFormat: cfg.headFormat, geometry: geometry)
        if case .q4g(let hw) = headStorage {
            switch hw.view.containerKind {
            case .incumbentHQ4G32:
                break
            case .shvq(let bits, let group), .rotatedHQ(let bits, let group), .nativeAffine(let bits, let group):
                let isSHVQ = hw.view.containerKind.isSHVQ
                for name in [QWGemmAQ.kernelName(bits: bits, group: group, isSHVQ: isSHVQ),
                             QWGemmAQ.decodeKernelName(bits: bits, group: group, isSHVQ: isSHVQ),
                             QWGemmAQ.decodeM4KernelName(bits: bits, group: group, isSHVQ: isSHVQ),
                             QWGemmAQ.headArgmaxKernelName(bits: bits, group: group, isSHVQ: isSHVQ),
                             QWGemmAQ.headArgmaxM4KernelName(bits: bits, group: group, isSHVQ: isSHVQ),
                             QWGemmAQ.headArgmaxM16KernelName(bits: bits, group: group, isSHVQ: isSHVQ)].compactMap({ $0 }) {
                    _ = try hal.pipeline(name)
                }
            }
        }
        return EmbeddingAndHeadSetup(
            embedPlane: embedName, embedRawOff: embedRawOff, embedReader: embedReader,
            embedScalesReader: embedScalesReader, embedBiasesReader: embedBiasesReader,
            mtpDeviceEmbedding: mtpDeviceEmbedding, embedStaging: embedStaging,
            mtpDraftHead: mtpDraftHead, headStorage: headStorage)
    }

    static func buildGDNSlots(
        gsub: QWSubAllocator, geometry: QWModelGeometry, gdnLayerCount: Int,
        mtpDrafts: Int, dflashDrafter: QWDFlashDrafter?, dsparkDrafter: QWDSparkDrafter? = nil,
        slotCount: Int? = nil
    ) throws -> [GdnSlot] {
        var slots: [GdnSlot] = []
        let H = geometry.hidden
        let resolvedSlotCount = slotCount ?? Self.gdnSlotCount()
        for slotIdx in 0..<resolvedSlotCount {
            let recSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.recElems(geometry: geometry))], .f32))
            let ringSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.ringElems(geometry: geometry))], .f16))
            let recViews = Self.slotViews(recSlab, elem: QWSlotGeometry.recElems(geometry: geometry), dt: .f32, layers: gdnLayerCount, geometry: geometry)
            let ringViews = Self.slotViews(ringSlab, elem: QWSlotGeometry.ringElems(geometry: geometry), dt: .f16, layers: gdnLayerCount, geometry: geometry)
            func disjoint(_ views: [QWTensorValue]) -> Bool {
                guard views.count > 1 else { return true }
                var spans = views.map { ($0.region.base + $0.offset, $0.bytes) }
                spans.sort { $0.0 < $1.0 }
                for i in 1..<spans.count where spans[i].0 < spans[i - 1].0 + spans[i - 1].1 { return false }
                return true
            }
            precondition(recViews.count == gdnLayerCount && disjoint(recViews),
                         "QWEngineDriver: GDN rec state views ALIAS (per-layer state law)")
            precondition(ringViews.count == gdnLayerCount && disjoint(ringViews),
                         "QWEngineDriver: GDN ring views ALIAS (per-layer state law)")
            let trunkHidShape = geometry.usesMultiStreamMTPHidden
                ? QWTensorShape([geometry.mhcConfig!.streams * H], .f32) : QWTensorShape([H], .f16)
            let dflashSlot = (dflashDrafter.map { slotIdx < $0.sessionSlots.count ? $0.sessionSlots[slotIdx] : nil }) ?? nil
            let dsparkSlot = (dsparkDrafter.map { slotIdx < $0.sessionSlots.count ? $0.sessionSlots[slotIdx] : nil }) ?? nil
            slots.append(GdnSlot(recSlab: recSlab, ringSlab: ringSlab,
                                 rec: recViews, ring: ringViews,
                                 trunkHidden: mtpDrafts > 0 ? try gsub.allocate(trunkHidShape) : nil,
                                 dflashSlot: dflashSlot,
                                 dsparkSlot: dsparkSlot))
        }
        return slots
    }

    private static func prewarmBatchedDecodePipelines(
        hal: QWHal, geometry: QWModelGeometry, layersW: [LayerW], headAQ4: Bool, hasMLA: Bool
    ) throws {
        // Always prewarm B=2..8 batched head pipelines before hal.freezePipelines().
        for m in 2...8 {
            _ = try hal.pipeline("qwg_head_f16_m\(m)")
            _ = try hal.pipeline("qwg_head_q4g_m\(m)", isAQ4: headAQ4)
            _ = try hal.pipeline("qwg_head_q4g_argmax_m\(m)", isAQ4: headAQ4)
            if m >= 4 {
                _ = try hal.pipeline("qwg_head_q4g_rp_m\(m)", isAQ4: headAQ4)
            }
        }
        guard geometry.hcaConfig == nil && geometry.blockSparseGQAConfig == nil else {
            return
        }
        for m in 2...8 {
            _ = try hal.pipeline("qwm_mlp_gate_up_m\(m)")
            _ = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)")
            _ = try hal.pipeline("qwm_mlp_down_m\(m)")
            if m >= 4 {
                _ = try hal.pipeline("qwm_mlp_gate_up_rp_m\(m)")
                _ = try hal.pipeline("qwm_mlp_down_rp_m\(m)")
            }
        }
        var mlpAQ4Prewarmed = false
        var seenGemms = Set<String>()
        var seenAqDecode = Set<String>()
        var seenAqSilu = Set<String>()
        func prewarmWeight(_ w: QWWeightView) throws {
            let useNAX = hal.caps.isMetal4TensorCapable
            guard hal.caps.supportsSimdgroupMMA else { return }
            switch w.containerKind {
            case .incumbentHQ4G32:
                return
            case .shvq(let bits, let group), .rotatedHQ(let bits, let group), .nativeAffine(let bits, let group):
                guard let name = QWGemmAQ.gemmKernelName(bits: bits, group: group,
                                                         isSHVQ: w.containerKind.isSHVQ,
                                                         useNAX: useNAX),
                      seenGemms.insert(name).inserted else { return }
                _ = try hal.pipeline(name)
            }
        }
        func prewarmBatchedProj(_ w: QWWeightView) throws {
            if QWMTPBatchedMLP.usesQ4GMKernel(w) {
                if w.containerKind != .incumbentHQ4G32 && !mlpAQ4Prewarmed {
                    for m in 2...8 {
                        _ = try hal.pipeline("qwm_mlp_gate_up_m\(m)", isAQ4: true)
                        _ = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)", isAQ4: true)
                        _ = try hal.pipeline("qwm_mlp_down_m\(m)", isAQ4: true)
                        if m >= 4 {
                            _ = try hal.pipeline("qwm_mlp_gate_up_rp_m\(m)", isAQ4: true)
                            _ = try hal.pipeline("qwm_mlp_down_rp_m\(m)", isAQ4: true)
                        }
                    }
                    mlpAQ4Prewarmed = true
                }
            } else if let name = QWGemmAQ.decodeKernelName(bits: w.containerKind.bits,
                                                           group: w.containerKind.groupSize,
                                                           isSHVQ: w.containerKind.isSHVQ),
                      seenAqDecode.insert(name).inserted {
                _ = try hal.pipeline(name)
            }
        }
        var gdnHQ4Prewarmed = false
        var gdnAQ4Prewarmed = false
        var seenGdnInProj = Set<String>()
        for lw in layersW {
            try prewarmWeight(lw.gu.view)
            try prewarmWeight(lw.down)
            if lw.moe == nil {
                let gu = lw.gu.view
                if QWMTPBatchedMLP.usesQ4GMKernel(gu) {
                    try prewarmBatchedProj(gu)
                } else if let name = QWGemmAQ.siluPairKernelName(bits: gu.bits, group: gu.groupSize, isSHVQ: gu.isSHVQ),
                          seenAqSilu.insert(name).inserted {
                    _ = try hal.pipeline(name)
                }
                try prewarmBatchedProj(lw.down)
            }
            if let a = lw.attn {
                try prewarmWeight(a.qkvProj)
                try prewarmWeight(a.o)
                try prewarmBatchedProj(a.qkvProj)
                try prewarmBatchedProj(a.o)
                if let mla = a.mla {
                    try prewarmWeight(mla.qBProj)
                    try prewarmWeight(mla.kvBProj)
                    try prewarmBatchedProj(mla.qBProj)
                    try prewarmBatchedProj(mla.kvBProj)
                }
            }
            if let g = lw.gdn {
                try prewarmWeight(g.inProj)
                if geometry.gdnVectorDecay { try prewarmBatchedProj(g.inProj) }
                if let fgb = g.fgbProj {
                    try prewarmWeight(fgb)
                    try prewarmBatchedProj(fgb)
                }
                try prewarmWeight(g.out)
                try prewarmBatchedProj(g.out)
                if !geometry.gdnVectorDecay {
                    let v = g.inProj
                    switch v.containerKind {
                    case .incumbentHQ4G32, .rotatedHQ(4, 32), .nativeAffine(4, 32):
                        let isAQ4 = v.containerKind != .incumbentHQ4G32
                        if (isAQ4 && !gdnAQ4Prewarmed) || (!isAQ4 && !gdnHQ4Prewarmed) {
                            for m in 2...8 {
                                _ = try hal.pipeline("qwm_gdn_projection_m\(m)", isAQ4: isAQ4)
                                if m >= 4 { _ = try hal.pipeline("qwm_gdn_projection_rp_m\(m)", isAQ4: isAQ4) }
                            }
                            if isAQ4 { gdnAQ4Prewarmed = true } else { gdnHQ4Prewarmed = true }
                        }
                    case .shvq(let bits, let group), .rotatedHQ(let bits, let group), .nativeAffine(let bits, let group):
                        if let name = QWGemmAQ.gdnInProjDecodeKernelName(bits: bits, group: group,
                                                                         isSHVQ: v.containerKind.isSHVQ),
                           seenGdnInProj.insert(name).inserted {
                            _ = try hal.pipeline(name)
                        }
                    }
                }
            }
        }
    }

    static func buildMTPVerificationResources(
        geometry: QWModelGeometry, hal: QWHal, asub: QWSubAllocator, gsub: QWSubAllocator,
        attnDecode: QWAttnDecode?, attnLayers: [Int], gdnLayerCount: Int,
        mlaOperator: QWMLAAttnOperator?, layersW: [LayerW],
        mtpWeights: MTPWeights?, echoWeights: EchoDraftWeights?, dflashDrafter: QWDFlashDrafter?,
        mtpDrafts: Int, effectiveMaxTokens: Int, maxT: Int, mtpPartialsOff: Int, pCarveBytes: Int,
        headIsQ4g: Bool, verifyHeadQ4Enabled: Bool, useGroupedNAX: Bool,
        scratch: Scratch, ropeApply: QWRopeApply, effectiveRopeRows: Int, kvMode: QWKVMode
    ) throws -> MTPVerificationResources {
        let headAQ4 = (hal.weightBasis == .standard)
            || (layersW.first.map { $0.gu.view.containerKind != .incumbentHQ4G32 } ?? false)
        try Self.prewarmBatchedDecodePipelines(hal: hal, geometry: geometry, layersW: layersW,
                                               headAQ4: headAQ4, hasMLA: mlaOperator != nil)
        guard mtpDrafts > 0 else {
            return MTPVerificationResources(
                gdnReplayCheckpoint: nil, verifyWindow: nil, mtpBatchedMLP: nil,
                mtpBatchedTrunk: nil, mtpSharedAttentionPartials: nil, mtpScratch: nil,
                mtpPromptEmbeddings: nil, mtpDraftEncoder: nil, echoDrafter: nil)
        }
        let gdnReplayCheckpoint = try QWGDNReplayCheckpoint(
            geometry: geometry, allocator: gsub, layers: gdnLayerCount, rows: mtpDrafts + 1)
        _ = try hal.pipeline("qwg_gdn_replay_state")
        _ = try hal.pipeline("qwg_verify_copy")
        let verifyWindow = try QWVerifyWindow(
            geometry: geometry, allocator: asub, rows: mtpDrafts + 1, layerMajor: true,
            gpuTargets: true, q4Head: verifyHeadQ4Enabled || headIsQ4g, useGroupedNAX: useGroupedNAX)
        let mtpBatchedMLP = try QWMTPBatchedMLP(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
        let mtpBatchedTrunk = try QWMTPBatchedTrunk(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
        let mtpSharedAttentionPartials: QWMRegion?
        if attnDecode != nil {
            let maxDecodeSplits = effectiveMaxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
            let bytes = QWAttnDecode.decodePartialsBytesBatched(T: min(8, mtpDrafts + 1), geometry: geometry, maxSplits: maxDecodeSplits)
            let offset = try asub.allocate(bytes: bytes)
            mtpSharedAttentionPartials = QWMRegion(buffer: asub.region.buffer, base: asub.region.base + offset, length: bytes)
        } else {
            mtpSharedAttentionPartials = nil
        }
        _ = try hal.pipeline("qwg_mtp_argmax_target_checked")
        let needsNeuralScratch = mtpWeights != nil || echoWeights != nil
        let partials = QWMRegion(buffer: gsub.region.buffer, base: gsub.region.base + mtpPartialsOff, length: pCarveBytes)
        let mtpDenseInter = (mtpWeights != nil && mtpWeights?.moe == nil) ? mtpWeights?.down.k : nil
        let mtpScratch = needsNeuralScratch
            ? try MTPScratch(geometry: geometry, allocator: asub, partials: partials,
                             intermediate: mtpDenseInter)
            : nil
        let mtpPromptEmbeddings = needsNeuralScratch
            ? try asub.allocate(QWTensorShape([maxT * geometry.hidden], .f16))
            : nil
        let mtpDraftEncoder: QWMTPDraftEncoder? = try mtpWeights.flatMap { w in
            try mtpScratch.map { scratch in
                try QWMTPDraftEncoder(geometry: geometry, hal: hal, weights: w, scratch: scratch,
                                      rope: ropeApply, ropeRows: effectiveRopeRows, kvMode: kvMode,
                                      mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count)
            }
        }
        let echoDrafter: QWEchoDraftEncoder?
        if let ew = echoWeights, let draftLayer = ew.dedicatedLayer ?? layersW.last,
           (mlaOperator != nil || ew.dedicatedLayer != nil) {
            echoDrafter = try QWEchoDraftEncoder(
                geometry: geometry, hal: hal, weights: ew, trunkLayer: draftLayer,
                mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count,
                maxSeedRows: maxT, kvMode: kvMode)
        } else {
            echoDrafter = nil
        }
        return MTPVerificationResources(
            gdnReplayCheckpoint: gdnReplayCheckpoint, verifyWindow: verifyWindow,
            mtpBatchedMLP: mtpBatchedMLP, mtpBatchedTrunk: mtpBatchedTrunk,
            mtpSharedAttentionPartials: mtpSharedAttentionPartials, mtpScratch: mtpScratch,
            mtpPromptEmbeddings: mtpPromptEmbeddings, mtpDraftEncoder: mtpDraftEncoder,
            echoDrafter: echoDrafter)
    }
}

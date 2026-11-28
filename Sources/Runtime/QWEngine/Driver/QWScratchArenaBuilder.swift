// QWScratchArenaBuilder.swift — Shared scratch sizing, arena budgeting, and tensor carving.
import Foundation
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

/// Single source of truth for MoE/prefill/decode scratch sizing, UMA residency
/// search, arena budgeting, and scratch carving across `QWEngineDriver`.
enum QWScratchArenaBuilder {
    struct MoEScratchPlan {
        let rows: Int
        let geometry: QWModelGeometry
        let isPrefill: Bool
        let aliasGuDnAcc: Bool
        let r1Rows: Int
        let dnAccElems: Int
        let tcInts: Int
        let latentDim: Int
        let sharedIntermediate: Int
        let routerCols: Int
        let needsActiveBuckets: Bool
        let needsPairGrouping: Bool

        init(rows: Int, geometry: QWModelGeometry, useGroupedNAX: Bool) {
            self.rows = rows
            self.geometry = geometry
            self.isPrefill = rows > 16 && geometry.supportsGroupedNAXMoEPrefill && useGroupedNAX
            self.aliasGuDnAcc = isPrefill && geometry.effectiveResidentExperts >= geometry.numExperts
            self.r1Rows = isPrefill ? min(rows, 8) : rows
            let outDim = geometry.moeRoutedDim
            let k = geometry.expertsPerTok
            self.dnAccElems = isPrefill
                ? max(r1Rows * k * outDim, (rows * k * outDim + 1) / 2)
                : (rows * k * outDim)
            self.tcInts = isPrefill
                ? QWGemm.MoETilePlan(
                    experts: geometry.numExperts,
                    pairs: rows * k,
                    gateUpRows: 2 * geometry.moeIntermediate,
                    downRows: outDim).scratchInts
                : 0
            self.latentDim = geometry.moeLatentSize ?? 0
            self.sharedIntermediate = geometry.sharedExpertIntermediate
            self.routerCols = geometry.moeSplitFeatureRouting ? (2 * geometry.numExperts) : geometry.numExperts
            self.needsActiveBuckets = rows >= 2 && geometry.numExperts > 0
            self.needsPairGrouping = self.isPrefill || self.needsActiveBuckets
        }

        private static func alignedBytes(_ count: Int, _ dt: QWDType) -> Int {
            let elemBytes = Int((Double(max(1, count)) * dt.bytesPerParam).rounded())
            return ((elemBytes + 255) / 256) * 256
        }

        var allocationBytes: Int {
            guard geometry.numExperts > 0 else { return 0 }
            let H = geometry.hidden
            let I = geometry.moeIntermediate
            let E = geometry.numExperts
            let K = geometry.expertsPerTok
            let ISh = sharedIntermediate
            var total = 2 * Self.alignedBytes(rows * H, .f32)
                + Self.alignedBytes(rows * routerCols, .f32)
                + Self.alignedBytes(rows * K, .i32)
                + Self.alignedBytes(rows * K, .f32)
                + Self.alignedBytes(r1Rows * K * 2 * I, .f32)
                + Self.alignedBytes(r1Rows * K * I, .f32)
                + (ISh > 0 ? Self.alignedBytes(rows * 2 * ISh, .f32) + Self.alignedBytes(rows * ISh, .f32) : 0)
                + Self.alignedBytes(rows * H, .f32)
                + Self.alignedBytes(rows, .f32)
                + (aliasGuDnAcc ? 0 : Self.alignedBytes(dnAccElems, .f32))
            if latentDim > 0 {
                total += 2 * Self.alignedBytes(rows * latentDim, .f32)
                if isPrefill { total += Self.alignedBytes(rows * latentDim, .f16) }
            }
            if isPrefill {
                let guBytes = Self.alignedBytes(rows * K * 2 * I, .f16)
                let dnBytes = Self.alignedBytes(dnAccElems, .f32)
                total += Self.alignedBytes(E + 1, .i32)
                    + 5 * Self.alignedBytes(rows * K, .i32)
                    + Self.alignedBytes(tcInts, .i32)
                    + (aliasGuDnAcc ? max(guBytes, dnBytes) : guBytes)
                    + Self.alignedBytes(rows * K * I, .f16)
                    + Self.alignedBytes(rows * (H + ISh), .f16)
            } else if needsPairGrouping {
                total += Self.alignedBytes(E + 1, .i32) + 2 * Self.alignedBytes(rows * K, .i32)
            }
            if needsActiveBuckets {
                let indInts = QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4
                total += Self.alignedBytes(E, .i32) + Self.alignedBytes(indInts, .i32)
            }
            return total
        }

        func carve(
            _ alloc: (Int, QWDType) throws -> QWTensorValue,
            persistentAlloc: ((Int, QWDType) throws -> QWTensorValue)? = nil
        ) throws -> QWEngineDriver.MoEScratch {
            func pAlloc(_ n: Int, _ dt: QWDType) throws -> QWTensorValue {
                if let persistentAlloc { return try persistentAlloc(n, dt) }
                return try alloc(n, dt)
            }
            let H = geometry.hidden, I = geometry.moeIntermediate
            let E = geometry.numExperts, K = geometry.expertsPerTok, ISh = sharedIntermediate
            let latentSc: QWMoEOperator.LatentScratch? = latentDim > 0 ? QWMoEOperator.LatentScratch(
                bufA32: try alloc(rows * latentDim, .f32),
                bufB32: try alloc(rows * latentDim, .f32),
                bufB16: isPrefill ? try alloc(rows * latentDim, .f16) : nil
            ) : nil
            let xRot32 = try alloc(rows * H, .f32)
            let xPlain32 = try alloc(rows * H, .f32)
            let logits = try alloc(rows * routerCols, .f32)
            let idx = try alloc(rows * K, .i32)
            let wts = try alloc(rows * K, .f32)
            let gu32 = try alloc(r1Rows * K * 2 * I, .f32)
            let act32 = try alloc(r1Rows * K * I, .f32)
            let guSh32 = ISh > 0 ? try alloc(rows * 2 * ISh, .f32) : nil
            let actSh32 = ISh > 0 ? try alloc(rows * ISh, .f32) : nil
            let shared = try pAlloc(rows * H, .f32)
            let gate = try pAlloc(rows, .f32)
            let expOff = needsPairGrouping ? try alloc(E + 1, .i32) : nil
            let sortedPair = needsPairGrouping ? try alloc(rows * K, .i32) : nil
            let sortedTok = needsPairGrouping ? try alloc(rows * K, .i32) : nil
            let invPerm = isPrefill ? try alloc(rows * K, .i32) : nil
            let tileE = isPrefill ? try alloc(rows * K, .i32) : nil
            let tileR = isPrefill ? try alloc(rows * K, .i32) : nil
            let tileCount = isPrefill ? try alloc(tcInts, .i32) : nil
            let indInts = QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4
            let activeExperts = needsActiveBuckets ? try alloc(E, .i32) : nil
            let activeIndirect = needsActiveBuckets ? try alloc(indInts, .i32) : nil
            let gu16Value: QWTensorValue?
            let dnAccValue: QWTensorValue
            if aliasGuDnAcc {
                let guElems = max(1, rows * K * 2 * I)
                let dnElems = max(1, dnAccElems)
                let sharedSlab = try alloc(max(guElems, (dnElems * 4 + 1) / 2), .f16)
                gu16Value = QWTensorValue(region: sharedSlab.region, offset: sharedSlab.offset,
                                          shape: QWTensorShape([guElems], .f16))
                dnAccValue = QWTensorValue(region: sharedSlab.region, offset: sharedSlab.offset,
                                           shape: QWTensorShape([dnElems], .f32))
            } else {
                gu16Value = isPrefill ? try alloc(rows * K * 2 * I, .f16) : nil
                dnAccValue = try alloc(dnAccElems, .f32)
            }
            let s = QWEngineDriver.MoEScratch(
                xRot32: xRot32, xPlain32: xPlain32, logits: logits, idx: idx, wts: wts,
                gu32: gu32, act32: act32, guSh32: guSh32, actSh32: actSh32,
                shared: shared, gate: gate, expOff: expOff,
                sortedPair: sortedPair, sortedTok: sortedTok, invPerm: invPerm,
                tileE: tileE, tileR: tileR, tileCount: tileCount,
                gu16: gu16Value,
                dnHi: isPrefill ? try alloc(rows * K * I, .f16) : nil,
                dnAcc: dnAccValue,
                naxX16: isPrefill ? try alloc(rows * (H + ISh), .f16) : nil,
                latent: latentSc,
                activeExperts: activeExperts,
                activeIndirect: activeIndirect
            )
            memset(s.shared.region.pointer.advanced(by: s.shared.offset), 0, s.shared.bytes)
            if ISh > 0 && !geometry.sharedExpertHasGate {
                let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<rows { gPtr[i] = 1.0 }
            } else {
                memset(s.gate.region.pointer.advanced(by: s.gate.offset), 0, s.gate.bytes)
            }
            return s
        }
    }

    static func prefillTransientBreakdown(
        geometry: QWModelGeometry,
        maxPrefillChunk: Int = 4096,
        useGroupedNAX: Bool = true
    ) -> (persistent: Int, attnTransient: Int, mlpTransient: Int) {
        let H = geometry.hidden, I = geometry.intermediate
        let qC = geometry.qChannels, qOut = geometry.qProjOut, kvE = geometry.kvChannels
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let vh = geometry.gdnVHeads, E = geometry.numExperts
        let usesMLA = geometry.usesLatentOrCompressedAttention
        let mhcC = geometry.mhcConfig?.streams ?? 0
        let glogC = geometry.gdnGlogChannels
        let persistentF16 = 5 * H
        let persistentF32 = H
            + (mhcC > 0 ? (mhcC * H + mhcC + mhcC * mhcC) : 0)
            + (E > 0 ? (H + 1) : 0)
        let persistentBytes = persistentF16 * 2 + persistentF32 * 4

        let attnScratchF16 = usesMLA ? 0 : (geometry.attnProjTotalN + 2 * kvE + qOut)
        let attOutCols = max(qC, geometry.qHeads * geometry.vHeadDim)
        let fgbC = geometry.gdnVectorDecay ? geometry.kdaFgbChannels : 0
        let gdnYF32 = geometry.gdnVectorDecay ? 0 : (2 * vh)
        let needsWy = !geometry.gdnVectorDecay && geometry.gdnLayerCount > 0 && maxPrefillChunk > 4096
        let attnF16 = attnScratchF16 + 2 * qkvC + 2 * zC + attOutCols + fgbC
        let attnF32 = gdnYF32 + zC + vh + glogC + (needsWy ? 4 * zC : 0)
        let attnTransientBytes = attnF16 * 2 + attnF32 * 4

        let mlpDenseBytes = (max(2 * I, 2 * H) + I) * 2
        var moeTransientBytes = 0
        if E > 0 {
            let isGroupedNAX = geometry.supportsGroupedNAXMoEPrefill && useGroupedNAX
            let aliasGuDnAcc = isGroupedNAX && geometry.effectiveResidentExperts >= E
            let guDnAliasSaved = aliasGuDnAcc
                ? min(4 * geometry.expertsPerTok * geometry.moeIntermediate,
                      2 * geometry.expertsPerTok * geometry.moeRoutedDim)
                : 0
            let totalMoe = QWEngineDriver.moeBytesPerToken(geometry: geometry)
                + QWEngineDriver.moeGroupedPrefillExtraBytesPerToken(
                    geometry: geometry, useGroupedNAX: useGroupedNAX)
            moeTransientBytes = max(0, totalMoe - (H + 1) * 4 - guDnAliasSaved)
        }
        return (persistentBytes, attnTransientBytes, mlpDenseBytes + moeTransientBytes)
    }

    static func prefillBytesPerToken(
        geometry: QWModelGeometry,
        maxPrefillChunk: Int = 4096,
        useGroupedNAX: Bool = true,
        aliasBranches: Bool = true
    ) -> Int {
        let b = prefillTransientBreakdown(
            geometry: geometry, maxPrefillChunk: maxPrefillChunk, useGroupedNAX: useGroupedNAX)
        return aliasBranches
            ? (b.persistent + max(b.attnTransient, b.mlpTransient))
            : (b.persistent + b.attnTransient + b.mlpTransient)
    }

    static func activationBytes(
        geometry: QWModelGeometry,
        prefillChunkRows: Int,
        mtpDrafts: Int,
        maxTokens: Int,
        useGroupedNAX: Bool,
        maxBufferLength: Int?
    ) -> (actBytes: Int, aliasBranches: Bool, bytesPerToken: Int) {
        let decodeActBytes = QWEngineDriver.decodeAndMTPScratchBytes(
            geometry: geometry, prefillChunkRows: prefillChunkRows,
            mtpDrafts: mtpDrafts, maxTokens: maxTokens, useGroupedNAX: useGroupedNAX)
        let aliasedBpt = prefillBytesPerToken(
            geometry: geometry, maxPrefillChunk: prefillChunkRows,
            useGroupedNAX: useGroupedNAX, aliasBranches: true)
        let aliasBranches = maxBufferLength.map { cap in
            let maxBufLen = max(256, (cap / 256) * 256)
            return (prefillChunkRows * aliasedBpt + decodeActBytes) <= maxBufLen
        } ?? true
        let bytesPerToken = prefillBytesPerToken(
            geometry: geometry, maxPrefillChunk: prefillChunkRows,
            useGroupedNAX: useGroupedNAX, aliasBranches: aliasBranches)
        return (prefillChunkRows * bytesPerToken + decodeActBytes, aliasBranches, bytesPerToken)
    }

    static func mlaPerTokenBytes(geometry: QWModelGeometry, activeAttnLayers: Int) -> Int {
        if geometry.usesNoPEMLA {
            return activeAttnLayers * ((geometry.indexerHeadDim * 2) / 4)
        }
        if geometry.usesDecoupledRopeMLA {
            let ropeDim = geometry.mlaConfig?.qkRopeHeadDim ?? geometry.hybridMLAConfig?.qkRopeHeadDim ?? 64
            let idxDim = geometry.mlaHasIndexer ? max(32, geometry.indexerHeadDim) : 0
            return activeAttnLayers * ((idxDim + ropeDim) * 2)
        }
        return 0
    }

    struct FixedComponentFootprint {
        let gdnBytes: Int
        let extraFixedBytes: Int
        let dflashWeights: Int
        let activeAttnLayers: Int
        let attnLayerCount: Int
        let mlaPerTokBytes: Int
        let hcaPerTokBytes: Int
        let mbsaPerTokBytes: Int
        var perTokenBytes: Int { mlaPerTokBytes + hcaPerTokBytes + mbsaPerTokBytes }

        static func compute(
            geometry: QWModelGeometry,
            prefillChunkSize: Int,
            maxTokens: Int,
            mtpDrafts: Int,
            mtpWeightDrafts: Int,
            dflashFootprint: (weights: Int, scratch: Int),
            tuning: QWEngineTuning,
            physicalMemoryBytes: Int?,
            requestedSessions: Int? = nil
        ) -> FixedComponentFootprint {
            let numV = geometry.numVirtualLayers
            let attnCount = (0..<numV).filter { geometry.isVirtualLayerFullAttention($0) }.count
            let gdnCount = geometry.gdnLayerCount
            let slotCount = QWEngineDriver.gdnSlotCount(
                physicalMemoryBytes: physicalMemoryBytes, requestedSessions: requestedSessions)
            let slotBytes = gdnCount * (QWSlotGeometry.recElems(geometry: geometry) * 4
                + QWSlotGeometry.ringElems(geometry: geometry) * 2)
            let trunkHidBytes = geometry.usesMultiStreamMTPHidden
                ? geometry.mhcConfig!.streams * geometry.hidden * 4
                : geometry.hidden * 2
            let needsMTPCache = mtpWeightDrafts > 0 && mtpDrafts > 0
            let gdnBytes = slotCount * slotBytes
                + QWEngineDriver.partialsScratchBytes(
                    geometry: geometry, mtpDrafts: mtpWeightDrafts > 0 ? mtpDrafts : 0, maxTokens: maxTokens)
                + (mtpDrafts > 0
                    ? QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: gdnCount, rows: mtpDrafts + 1)
                    : 0)
                + (mtpDrafts > 0 ? slotCount * trunkHidBytes : 0)
                + (1 << 20)
            let activeAttn = attnCount + (needsMTPCache ? 1 : 0)
            let extraFixed = QWEngineDriver.mlaFixedScratchBytes(
                    geometry: geometry, prefillChunkRows: prefillChunkSize, activeAttnLayers: activeAttn)
                + QWEngineDriver.hcaFixedScratchBytes(geometry: geometry, prefillChunkRows: prefillChunkSize)
                + QWEngineDriver.mbsaFixedScratchBytes(
                    geometry: geometry, prefillChunkRows: prefillChunkSize, attnLayerCount: attnCount)
                + (mtpWeightDrafts > 0 && (geometry.mtpSharedLastLayer || geometry.hasDedicatedMHCMTP) ? (112 << 20) : 0)
                + QWEngineDriver.mhcSplitBufferBytes(geometry: geometry)
                + QWEngineDriver.externalMoEDecodeBufferBytes(
                    geometry: geometry, tuning: tuning, prefillChunkSize: prefillChunkSize)
                + QWEngineDriver.sharedBatchMLPFixedBytes(geometry: geometry, mtpDrafts: mtpDrafts)
                + dflashFootprint.scratch
            return FixedComponentFootprint(
                gdnBytes: gdnBytes,
                extraFixedBytes: extraFixed,
                dflashWeights: dflashFootprint.weights,
                activeAttnLayers: activeAttn,
                attnLayerCount: attnCount,
                mlaPerTokBytes: mlaPerTokenBytes(geometry: geometry, activeAttnLayers: activeAttn),
                hcaPerTokBytes: QWEngineDriver.hcaPerTokenBytes(geometry: geometry),
                mbsaPerTokBytes: QWEngineDriver.mbsaPerTokenBytes(geometry: geometry, attnLayerCount: attnCount)
            )
        }
    }

    static func stepDownResidentExperts(
        raw: QWModelGeometry,
        ropeRows: Int,
        mtpWeightDrafts: Int,
        headQ4g: Bool,
        maxMTP: Int,
        format: QWWeightFormat,
        headFormat: QWWeightFormat? = nil,
        moeRouterIsBF16: Bool = false,
        dflashWeights: Int,
        safeWeightCap: Int
    ) -> Int {
        let minE = min(raw.numExperts, max(8, raw.expertsPerTok))
        var cand = ((raw.numExperts - 2) / 2) * 2
        while cand >= minE {
            let candGeo = raw.withResidentExperts(cand)
            if let p = try? QWWeightAllocationPlan(
                geometry: candGeo, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
                headQ4g: headQ4g, maxMTPDrafts: maxMTP, format: format,
                headFormat: headFormat, moeRouterIsBF16: moeRouterIsBF16),
               p.bytes + dflashWeights <= safeWeightCap {
                return cand
            }
            cand -= 2
        }
        return minE
    }

    static func planResidencyAndChunk(
        raw: QWModelGeometry, cfg: QWEngineDriver.StartupConfig, hal: QWHal, format: QWWeightFormat,
        tuning: QWEngineTuning, osReserveGiB: Double?, prefillChunkSize: Int, maxTokens: Int,
        ropeRows: Int, mtpDrafts: Int, mtpWeightDrafts: Int, kvMode: QWKVMode,
        allowExpertReduction: Bool = true,
        requestedSessions: Int? = nil
    ) throws -> (geometry: QWModelGeometry, prefillChunkSize: Int) {
        let maxMTP0 = 15
        let arenaCap0 = max(0, hal.caps.maxWorkingSetBytes - (64 << 20))
        let isMoE0 = raw.numExperts > 0
        let avail0 = QWEngineDriver.residencyBudgetBytes(
            policy: tuning.memoryPolicy,
            osReserveGiB: isMoE0 ? osReserveGiB : (osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30)),
            arenaCap: arenaCap0
        )
        let chunk0 = max(32, min(QWPrefillLimits.maxRows, (prefillChunkSize / 32) * 32))
        let useNax0 = hal.caps.supportsSimdgroupMMA
        let hasMTPDrafter0 = (mtpWeightDrafts > 0 || cfg.mtpBackend == .dflash2 || cfg.mtpBackend == .dspark) && mtpDrafts > 0
        let needsMTPCache = mtpWeightDrafts > 0 && mtpDrafts > 0
        let modelCap0 = raw.maxPositionEmbeddings > 0 ? min(maxTokens, raw.maxPositionEmbeddings) : maxTokens
        let minFitTokens0 = min(modelCap0, max(16_384, min(maxTokens, 32_768)))
        let dflash0 = QWEngineDriver.dflashFootprintBytes(
            draftPackPath: cfg.dflashDraftPackPath,
            descriptor: raw.descriptor,
            mtpDrafts: mtpDrafts,
            isDFlash: cfg.mtpBackend == .dflash2 || cfg.mtpBackend == .dspark,
            isDSpineBackend: cfg.dflashDraftPackPath.lowercased().contains("dspine"),
            isDSparkBackend: cfg.mtpBackend == .dspark,
            selectorTopK: cfg.dflashSelectorTopK,
            physicalMemoryBytes: QWAdmission.physicalMemoryBytes(),
            requestedSessions: requestedSessions
        )
        let fullWPlan = try QWWeightAllocationPlan(
            geometry: raw, ropeRows: max(ropeRows, maxTokens),
            mtpDrafts: mtpWeightDrafts, headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP0,
            format: format, headFormat: cfg.headFormat, moeRouterIsBF16: cfg.moeRouterIsBF16)
        let fullBytes0 = fullWPlan.bytes + dflash0.weights
        let embedBytes0 = raw.vocab * raw.hidden * 2
        let devEmbedEnv = cfg.environment["QW_MTP_DEVICE_EMBEDDING"] ?? cfg.environment["QW_MTP_DEVICE_EMBED"]
        let forceDeviceEmbed = devEmbedEnv == "1"
        let disableDeviceEmbed = devEmbedEnv == "0"
        let maxBuf0 = Int(hal.device.maxBufferLength)

        func safeWeightCap(forChunk chunkRows: Int) -> Int {
            let act = activationBytes(
                geometry: raw, prefillChunkRows: chunkRows, mtpDrafts: mtpDrafts,
                maxTokens: maxTokens, useGroupedNAX: useNax0, maxBufferLength: maxBuf0).actBytes
            let fixed = FixedComponentFootprint.compute(
                geometry: raw, prefillChunkSize: chunkRows, maxTokens: maxTokens,
                mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
                dflashFootprint: dflash0, tuning: tuning,
                physicalMemoryBytes: QWAdmission.physicalMemoryBytes(),
                requestedSessions: requestedSessions)
            let kv0 = QWEngineDriver.kvBudgetBytes(
                maxTokens: minFitTokens0, mtpDrafts: needsMTPCache ? mtpDrafts : 0,
                kvMode: kvMode, geometry: raw, prefillChunk: chunkRows)
                + (fixed.hcaPerTokBytes + fixed.mbsaPerTokBytes) * minFitTokens0
            let baseReserve = act + fixed.gdnBytes + fixed.extraFixedBytes + kv0 + (256 << 20)
            let safeCapWithoutEmbed = max(0, avail0 - baseReserve)
            let canFitDeviceEmbed = fullBytes0 > 0 && (fullBytes0 + embedBytes0 + (1 << 30) <= safeCapWithoutEmbed)
            let mtpEmbed = (!disableDeviceEmbed && hasMTPDrafter0
                && (forceDeviceEmbed || canFitDeviceEmbed)) ? embedBytes0 : 0
            return max(0, avail0 - (baseReserve + mtpEmbed))
        }

        if fullBytes0 <= safeWeightCap(forChunk: chunk0) {
            return (raw, chunk0)
        }
        for candChunk in [3584, 3072, 2560, 2048, 1536, 1024, 512] where candChunk < chunk0 {
            if fullBytes0 <= safeWeightCap(forChunk: candChunk) {
                return (raw, candChunk)
            }
        }
        let fallbackChunk = min(chunk0, 1024)
        guard allowExpertReduction, isMoE0 else {
            return (raw, fallbackChunk)
        }
        let chosenE = stepDownResidentExperts(
            raw: raw, ropeRows: max(ropeRows, maxTokens), mtpWeightDrafts: mtpWeightDrafts,
            headQ4g: cfg.headIsQ4g, maxMTP: maxMTP0, format: format, headFormat: cfg.headFormat,
            moeRouterIsBF16: cfg.moeRouterIsBF16,
            dflashWeights: dflash0.weights, safeWeightCap: safeWeightCap(forChunk: fallbackChunk))
        return (raw.withResidentExperts(chosenE), fallbackChunk)
    }

    static func planArenaAndBudget(
        geometry: QWModelGeometry, hal: QWHal, weightFormat: QWWeightFormat,
        weightFormatDeclineReason: String?, tuning: QWEngineTuning, osReserveGiB: Double?,
        prefillChunkSize: Int, maxTokens: Int, ropeRows: Int, kvMode: QWKVMode,
        attnLayerCount: Int, gdnLayerCount: Int, cfg: QWEngineDriver.StartupConfig,
        mtpDrafts: Int, mtpWeightDrafts: Int,
        requestedSessions: Int? = nil
    ) throws -> QWEngineDriver.ArenaBudgetPlan {
        let useGroupedNAX = hal.caps.supportsSimdgroupMMA
        let actBytes = activationBytes(
            geometry: geometry, prefillChunkRows: prefillChunkSize, mtpDrafts: mtpDrafts,
            maxTokens: maxTokens, useGroupedNAX: useGroupedNAX,
            maxBufferLength: Int(hal.device.maxBufferLength)).actBytes
        let dflashFootprint = QWEngineDriver.dflashFootprintBytes(
            draftPackPath: cfg.dflashDraftPackPath,
            descriptor: geometry.descriptor,
            mtpDrafts: mtpDrafts,
            isDFlash: cfg.mtpBackend == .dflash2 || cfg.mtpBackend == .dspark,
            isDSpineBackend: cfg.dflashDraftPackPath.lowercased().contains("dspine"),
            isDSparkBackend: cfg.mtpBackend == .dspark,
            selectorTopK: cfg.dflashSelectorTopK,
            physicalMemoryBytes: QWAdmission.physicalMemoryBytes(),
            requestedSessions: requestedSessions
        )
        let fixed = FixedComponentFootprint.compute(
            geometry: geometry, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
            mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
            dflashFootprint: dflashFootprint, tuning: tuning,
            physicalMemoryBytes: QWAdmission.physicalMemoryBytes(),
            requestedSessions: requestedSessions)
        let needsMTPCache = mtpWeightDrafts > 0 && mtpDrafts > 0
        let extraFixedBytes = fixed.extraFixedBytes + dflashFootprint.weights
        let kvSplit = swaLinearAndFixedKV(
            geometry: geometry, activeAttnLayers: fixed.activeAttnLayers, needsMTPCache: needsMTPCache,
            maxTokens: maxTokens, prefillChunk: prefillChunkSize, kvMode: kvMode)
        let perTokBytes = 2 * kvSplit.linearLayers
            * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens)
            + fixed.perTokenBytes
        let maxMTP = 15
        let prelimPlan = try QWWeightAllocationPlan(
            geometry: geometry, ropeRows: max(ropeRows, maxTokens),
            mtpDrafts: mtpWeightDrafts, headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP,
            format: weightFormat, headFormat: cfg.headFormat, moeRouterIsBF16: cfg.moeRouterIsBF16)
        let arenaCap = max(0, hal.caps.maxWorkingSetBytes - (64 << 20))
        let isMoE = geometry.numExperts > 0
        let isColdStreamingMoE = isMoE && geometry.effectiveResidentExperts < geometry.numExperts
        let effectiveReserveGiB = isMoE
            ? QWEngineDriver.effectiveOSReserveGiB(policy: tuning.memoryPolicy, explicitReserveGiB: osReserveGiB)
            : osReserveGiB
        let availBytes = QWEngineDriver.residencyBudgetBytes(
            policy: tuning.memoryPolicy,
            osReserveGiB: isMoE ? effectiveReserveGiB : (osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30)),
            arenaCap: arenaCap
        )
        let requestedMaxTokens = (isColdStreamingMoE && maxTokens == geometry.maxPositionEmbeddings)
            ? max(16_384, min(maxTokens, 32_768)) : maxTokens
        let prelimCap = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings, requested: requestedMaxTokens,
            perTokenBytes: perTokBytes,
            fixedBytes: prelimPlan.bytes + actBytes + fixed.gdnBytes + extraFixedBytes + kvSplit.swaFixedKVBytes,
            availableBytes: availBytes, yarnAllowed: true)
        let effectiveMaxTokens = (!prelimCap.fits && requestedMaxTokens == geometry.maxPositionEmbeddings
            && prelimCap.maxFittingContext >= QWCapacityPlan.kvStepTokens)
            ? (prelimCap.maxFittingContext / QWCapacityPlan.kvStepTokens) * QWCapacityPlan.kvStepTokens
            : requestedMaxTokens
        let effectiveRopeRows = max(ropeRows, effectiveMaxTokens)
        let weightPlan = try QWWeightAllocationPlan(
            geometry: geometry, ropeRows: effectiveRopeRows, mtpDrafts: mtpWeightDrafts,
            headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP,
            format: weightFormat, headFormat: cfg.headFormat, moeRouterIsBF16: cfg.moeRouterIsBF16)
        let weightsBytes = weightPlan.bytes
        let trunkKVBytes = QWEngineDriver.kvBudgetBytes(
            maxTokens: effectiveMaxTokens, mtpDrafts: 0, kvMode: kvMode,
            geometry: geometry, prefillChunk: prefillChunkSize)
        let mtpKVBytes = needsMTPCache
            ? (QWEngineDriver.kvBudgetBytes(
                maxTokens: effectiveMaxTokens, mtpDrafts: mtpDrafts, kvMode: kvMode,
                geometry: geometry, prefillChunk: prefillChunkSize)
                - trunkKVBytes)
            : 0
        let kvBytes = trunkKVBytes + mtpKVBytes
        let fixedBytes = weightsBytes + actBytes + fixed.gdnBytes + extraFixedBytes
        let effectiveSwaKV = swaLinearAndFixedKV(
            geometry: geometry, activeAttnLayers: fixed.activeAttnLayers, needsMTPCache: needsMTPCache,
            maxTokens: effectiveMaxTokens, prefillChunk: prefillChunkSize, kvMode: kvMode).swaFixedKVBytes
        let capPlan = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings, requested: effectiveMaxTokens,
            perTokenBytes: perTokBytes, fixedBytes: fixedBytes + effectiveSwaKV,
            availableBytes: availBytes, yarnAllowed: true)
        let formatSuffix = " weights_format=\(weightFormat.name) weights_bpw="
            + String(format: "%.2f", weightFormat.effectiveBitsPerWeight(geometry: geometry))
            + (weightFormatDeclineReason.map { " weights_declined=\($0)" } ?? "")
        let capLog = try QWEngineBootstrap.admit(
            weightsBytes: weightsBytes,
            fixedScratchBytes: actBytes + fixed.gdnBytes + extraFixedBytes,
            kvMode: String(describing: kvMode), perTokenBytes: perTokBytes,
            maxTokens: effectiveMaxTokens, totalPlannedBytes: fixedBytes + kvBytes,
            availableBytes: availBytes, osReserveGiB: effectiveReserveGiB, logSuffix: formatSuffix)
        let arena = try QWArena(hal: hal, weights: weightsBytes, activations: actBytes,
                                kv: kvBytes, gdnStates: fixed.gdnBytes)
        return QWEngineDriver.ArenaBudgetPlan(
            effectiveMaxTokens: effectiveMaxTokens, effectiveRopeRows: effectiveRopeRows,
            weightPlan: weightPlan, trunkKVBytes: trunkKVBytes, mtpKVBytes: mtpKVBytes,
            actBytes: actBytes, gdnBytes: fixed.gdnBytes, fixedBytes: fixedBytes, mtpPartialsOff: nil,
            pCarveBytes: QWEngineDriver.partialsCarveBytes(geometry: geometry, maxTokens: effectiveMaxTokens),
            useGroupedNAX: useGroupedNAX, capacityPlan: capPlan, capacityLogMessage: capLog,
            arena: arena, availBytes: availBytes)
    }

    static func estimateDriverContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String?,
        contextTokens: Int,
        kvMode: QWKVMode,
        mtpDrafts: Int,
        mtpBackend: String,
        prefillChunkSize: Int,
        tuning: QWEngineTuning,
        physicalMemoryBytes: Int?,
        resolveFormat: (String?, QWModelDescriptor, QWModelGeometry, Int) -> (QWWeightFormat, Bool, QWWeightFormat?, Bool)
    ) -> QWContextMemoryEstimate {
        let clampedContext = max(64, min(1_048_576, contextTokens))
        let baseGeometry = QWModelGeometry(descriptor: descriptor)
        let rawGeometry = baseGeometry.numExperts > 0
            ? baseGeometry.withResidentExperts(tuning.residentExpertsPerLayer) : baseGeometry
        let effDrafts = max(0, mtpDrafts)
        let lowerBackend = mtpBackend.lowercased()
        let isDFlash = ["dflash2", "dflash", "dspine", "dspark"].contains(lowerBackend)
        let maxMTP = 15
        let mtpWeightDrafts = (descriptor.mtpLayers > 0 && !isDFlash) ? min(effDrafts, maxMTP) : 0
        let needsMTPCache = mtpWeightDrafts > 0 && effDrafts > 0
        let hasMTPDrafter = (mtpWeightDrafts > 0 || isDFlash) && effDrafts > 0
        let (format, headQ4g, headFormat, moeRouterIsBF16) = resolveFormat(packPath, descriptor, rawGeometry, mtpWeightDrafts)
        let chunkRows = max(32, min(QWPrefillLimits.maxRows, (prefillChunkSize / 32) * 32))
        let maxBufferCap = QWDeviceCaps.current?.maxBufferLength
        let actBytes = activationBytes(
            geometry: rawGeometry, prefillChunkRows: chunkRows, mtpDrafts: effDrafts,
            maxTokens: clampedContext, useGroupedNAX: true, maxBufferLength: maxBufferCap).actBytes
        let dflashBytes = QWEngineDriver.dflashFootprintBytes(
            draftPackPath: nil, targetPackPath: packPath, descriptor: descriptor, mtpDrafts: effDrafts,
            isDFlash: isDFlash, isDSpineBackend: lowerBackend == "dspine",
            isDSparkBackend: lowerBackend == "dspark",
            physicalMemoryBytes: physicalMemoryBytes)
        let rawFixed = FixedComponentFootprint.compute(
            geometry: rawGeometry, prefillChunkSize: chunkRows, maxTokens: clampedContext,
            mtpDrafts: effDrafts, mtpWeightDrafts: mtpWeightDrafts,
            dflashFootprint: dflashBytes, tuning: tuning, physicalMemoryBytes: physicalMemoryBytes)
        let ropeRows = max(rawGeometry.maxPositionEmbeddings, clampedContext)
        let isMoE = rawGeometry.numExperts > 0
        let reserveGiB = isMoE
            ? QWEngineDriver.effectiveOSReserveGiB(policy: tuning.memoryPolicy, explicitReserveGiB: tuning.osReserveGiB)
            : (tuning.osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30))
        let availBytes = max(0, (physicalMemoryBytes ?? 0) - max(0, Int((reserveGiB * Double(1 << 30)).rounded())))
        var planningFailed = false
        let geometry: QWModelGeometry = {
            guard isMoE, tuning.residentExpertsPerLayer == nil, availBytes > 0 else { return rawGeometry }
            let modelCap = rawGeometry.maxPositionEmbeddings > 0
                ? min(clampedContext, rawGeometry.maxPositionEmbeddings) : clampedContext
            let minFitTokens = min(modelCap, max(16_384, min(clampedContext, 32_768)))
            let minFitKV = QWEngineDriver.kvBudgetBytes(
                maxTokens: minFitTokens, mtpDrafts: needsMTPCache ? effDrafts : 0,
                kvMode: kvMode, geometry: rawGeometry, prefillChunk: chunkRows)
                + (rawFixed.hcaPerTokBytes + rawFixed.mbsaPerTokBytes) * minFitTokens
            let safeCap = max(0, availBytes - (actBytes + rawFixed.gdnBytes + rawFixed.extraFixedBytes
                + minFitKV + (256 << 20)))
            let fullBytes: Int
            do {
                fullBytes = try QWWeightAllocationPlan(
                    geometry: rawGeometry, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
                    headQ4g: headQ4g, maxMTPDrafts: maxMTP, format: format, headFormat: headFormat,
                    moeRouterIsBF16: moeRouterIsBF16).bytes + dflashBytes.weights
            } catch {
                planningFailed = true
                return rawGeometry
            }
            guard fullBytes > safeCap else { return rawGeometry }
            let chosenE = stepDownResidentExperts(
                raw: rawGeometry, ropeRows: ropeRows, mtpWeightDrafts: mtpWeightDrafts,
                headQ4g: headQ4g, maxMTP: maxMTP, format: format, headFormat: headFormat,
                moeRouterIsBF16: moeRouterIsBF16,
                dflashWeights: dflashBytes.weights, safeWeightCap: safeCap)
            return rawGeometry.withResidentExperts(chosenE)
        }()
        let weightsBytes: Int
        do {
            weightsBytes = try QWWeightAllocationPlan(
                geometry: geometry, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
                headQ4g: headQ4g, maxMTPDrafts: maxMTP, format: format, headFormat: headFormat,
                moeRouterIsBF16: moeRouterIsBF16).bytes + dflashBytes.weights
        } catch {
            planningFailed = true
            weightsBytes = 0
        }
        let fixed = geometry.effectiveResidentExperts == rawGeometry.effectiveResidentExperts
            ? rawFixed
            : FixedComponentFootprint.compute(
                geometry: geometry, prefillChunkSize: chunkRows, maxTokens: clampedContext,
                mtpDrafts: effDrafts, mtpWeightDrafts: mtpWeightDrafts,
                dflashFootprint: dflashBytes, tuning: tuning, physicalMemoryBytes: physicalMemoryBytes)
        let kvSplit = swaLinearAndFixedKV(
            geometry: geometry, activeAttnLayers: fixed.activeAttnLayers, needsMTPCache: needsMTPCache,
            maxTokens: clampedContext, prefillChunk: chunkRows, kvMode: kvMode)
        let perTokBytes = 2 * kvSplit.linearLayers
            * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens)
            + fixed.perTokenBytes
        let kvBytes = QWEngineDriver.kvBudgetBytes(
            maxTokens: clampedContext, mtpDrafts: needsMTPCache ? effDrafts : 0,
            kvMode: kvMode, geometry: geometry, prefillChunk: chunkRows)
            + fixed.perTokenBytes * clampedContext
        let baseScratchBytes = actBytes + fixed.gdnBytes + fixed.extraFixedBytes
        let embedBytes = geometry.vocab * geometry.hidden * 2
        let hasDeviceEmbed = hasMTPDrafter && geometry.effectiveResidentExperts == geometry.numExperts
            && !planningFailed
            && (availBytes == 0 || weightsBytes + baseScratchBytes + kvBytes + embedBytes + (1 << 30) <= availBytes)
        let scratchBytes = baseScratchBytes + (hasDeviceEmbed ? embedBytes : 0)
        if planningFailed {
            return QWContextMemoryEstimate(
                contextTokens: clampedContext, weightsBytes: 0, kvBytes: kvBytes,
                scratchAndStateBytes: scratchBytes, requiredBytes: 0,
                availableBytes: availBytes, maxFittingContext: 0, fits: false)
        }
        let fixedBytes = weightsBytes + scratchBytes
        let requiredBytes = fixedBytes + kvBytes
        let effectiveFixedForMax = fixedBytes + kvSplit.swaFixedKVBytes
        let maxFitting = availBytes > effectiveFixedForMax
            ? min(1_048_576, (availBytes - effectiveFixedForMax) / max(1, perTokBytes)) : 0
        return QWContextMemoryEstimate(
            contextTokens: clampedContext, weightsBytes: weightsBytes, kvBytes: kvBytes,
            scratchAndStateBytes: scratchBytes, requiredBytes: requiredBytes,
            availableBytes: availBytes, maxFittingContext: maxFitting,
            fits: availBytes == 0 || requiredBytes <= availBytes)
    }

    private static func swaLinearAndFixedKV(
        geometry: QWModelGeometry, activeAttnLayers: Int, needsMTPCache: Bool,
        maxTokens: Int, prefillChunk: Int, kvMode: QWKVMode
    ) -> (linearLayers: Int, swaFixedKVBytes: Int) {
        guard geometry.hasAnySlidingWindow, !geometry.usesLatentOrCompressedAttention else {
            return (activeAttnLayers, 0)
        }
        let pageBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geometry)
        var fullCount = 0
        var swaBytes = 0
        for v in 0..<geometry.numVirtualLayers where geometry.isVirtualLayerFullAttention(v) {
            let w = geometry.virtualLayerSlidingWindow(v)
            if w > 0 {
                let cap = min(maxTokens, w + prefillChunk + 512)
                swaBytes += QWEngineDriver.kvPagesPerPool(maxTokens: cap) * 2 * pageBytes
                    + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
            } else {
                fullCount += 1
            }
        }
        return (fullCount + (needsMTPCache ? 1 : 0), swaBytes)
    }
}

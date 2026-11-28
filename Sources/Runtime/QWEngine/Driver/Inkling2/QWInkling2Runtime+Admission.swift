// QWInkling2Runtime+Admission.swift — Unified UMA memory sizing and 1M admission for Inkling-2.

import Foundation
import QWAttn
import QWCore
import QWKV
import QWQuant

extension QWInkling2Runtime {
    struct AdmissionSizing {
        let payRowBytes: Int
        let metaRowBytes: Int
        let kvRowTotal: Int
        let slidingVisits: Int
        let globalVisits: Int
        let slidingCap: Int
        let convBytes: Int
        let weightsEst: Int
        let fixedScratchEst: Int
        let perTokBytes: Int
        let availableBytes: Int
        let capacityPlan: QWCapacityPlan
        let alignedCap: Int
        let finalCap: Int
        let totalPlannedBytes: Int
    }

    struct WaveSlotPlan {
        let targetSlots: Int
        let waveSlotCap: Int
        let extraSlotBytes: Int
    }

    static func planCompanionWaveSlots(
        sizing: AdmissionSizing,
        maxConcurrentSessions: Int
    ) -> WaveSlotPlan {
        guard maxConcurrentSessions > 1, sizing.finalCap < 524_288 else {
            return WaveSlotPlan(targetSlots: 1, waveSlotCap: sizing.finalCap, extraSlotBytes: 0)
        }
        let maxCandidates = min(4, maxConcurrentSessions)
        let waveSlotCap = min(sizing.finalCap, 32_768)
        let extraSlotBytes = sizing.slidingVisits * sizing.slidingCap * sizing.kvRowTotal
            + sizing.globalVisits * waveSlotCap * sizing.kvRowTotal
            + sizing.convBytes
        var targetSlots = 1
        if maxCandidates >= 2 {
            for candidate in 2...maxCandidates {
                if sizing.totalPlannedBytes + (candidate - 1) * extraSlotBytes + (4 << 30) <= sizing.availableBytes {
                    targetSlots = candidate
                } else {
                    break
                }
            }
        }
        return WaveSlotPlan(targetSlots: targetSlots, waveSlotCap: waveSlotCap, extraSlotBytes: extraSlotBytes)
    }

    static func computeAdmissionSizing(
        descriptor desc: QWModelDescriptor,
        ink2: QWModelDescriptor.Inkling2Config,
        baseModelBytes: Int,
        dsparkWeightBytes: Int,
        dsparkCfg: QWModelDescriptor.Inkling2DSparkConfig?,
        activeMtpDrafts: Int,
        rawKVMode: String,
        requestedTokens: Int?,
        prefillChunkSize: Int,
        lsaConfig: QWLSAConfig,
        lsaIndexDim: Int = QWLSAConfig.defaultIndexDim,
        lsaMultiHeadDirect: Bool? = nil,
        extraFixedScratchBytes: Int = 0,
        availableBytes: Int
    ) throws -> AdmissionSizing {
        let elemsPerRow = desc.kvHeads * desc.headDim
        let payRowBytes: Int
        let metaRowBytes: Int
        switch rawKVMode {
        case "f16":
            payRowBytes = elemsPerRow * 2
            metaRowBytes = 0
        case "q8":
            payRowBytes = elemsPerRow
            metaRowBytes = desc.kvHeads * 2
        case "q4":
            payRowBytes = elemsPerRow / 2
            metaRowBytes = desc.kvHeads * 4
        default:
            throw QWEngineDriverError.unsupportedKVMode(rawKVMode)
        }
        let weightsEst = baseModelBytes + dsparkWeightBytes
        let numVisits = desc.executionPlan.count
        let slidingVisits = desc.executionPlan.filter { ink2.isSlidingLayer($0) }.count
        let globalVisits = numVisits - slidingVisits
        let kvRowTotal = 2 * (payRowBytes + metaRowBytes)
        let perTokBytes = max(1, globalVisits * kvRowTotal)

        let maxChunkPlan = max(32, min(8192, prefillChunkSize))
        let sconvSlots = max(1, ink2.sconvKernelSize - 1)
        let slidingCap = ink2.slidingWindowSize + maxChunkPlan
        let convBytes = numVisits * sconvSlots * (2 * desc.hidden + 2 * elemsPerRow) * 4
        let targetCtxForFixed = min(desc.maxPositionEmbeddings, max(1, requestedTokens ?? desc.maxPositionEmbeddings))
        let dsparkFixedEst: Int = (activeMtpDrafts > 0 && dsparkCfg != nil)
            ? (QWInkling2DSpark.estimateKVBytes(
                config: dsparkCfg!, capacity: targetCtxForFixed, maxPrefillChunk: maxChunkPlan) + (32 << 20))
            : 0
        let globalVisitIndices = desc.executionPlan.indices.filter { !ink2.isSlidingLayer(desc.executionPlan[$0]) }
        let effectiveLSA = lsaConfig.resolvingAdaptiveBudget(forMaxContext: targetCtxForFixed)
        let lsaBytes = QWLSAOperator.estimateRegionBytes(
            geometry: QWModelGeometry(descriptor: desc),
            lsaConfig: effectiveLSA,
            attnLayers: globalVisitIndices,
            maxContext: targetCtxForFixed,
            maxPrefillChunk: QWInkling2Session.resolveMaxChunk(prefillChunkSize: maxChunkPlan, capacity: targetCtxForFixed),
            indexDim: lsaIndexDim,
            isMultiHeadDirect: lsaMultiHeadDirect
        )
        let fixedScratchEst = slidingVisits * slidingCap * kvRowTotal + convBytes
            + dsparkFixedEst + extraFixedScratchBytes + lsaBytes + (640 << 20)

        let capPlan = try QWCapacityPlan.plan(
            nativeWindow: desc.maxPositionEmbeddings,
            requested: requestedTokens,
            perTokenBytes: perTokBytes,
            fixedBytes: weightsEst + fixedScratchEst,
            availableBytes: availableBytes,
            yarnAllowed: false
        )
        let alignedCap = QWCapacityPlan.steppedCapacity(capPlan.context)
        let resolvedCap = QWCapacityBudget.resolveKVCapacity(
            alignedCap: alignedCap,
            fixedBytes: weightsEst + fixedScratchEst,
            perTokenBytes: perTokBytes,
            availableBytes: availableBytes
        )
        let finalCap = min(desc.maxPositionEmbeddings, max(capPlan.context, resolvedCap))
        let totalPlanned = weightsEst + fixedScratchEst + finalCap * perTokBytes
        return AdmissionSizing(
            payRowBytes: payRowBytes,
            metaRowBytes: metaRowBytes,
            kvRowTotal: kvRowTotal,
            slidingVisits: slidingVisits,
            globalVisits: globalVisits,
            slidingCap: slidingCap,
            convBytes: convBytes,
            weightsEst: weightsEst,
            fixedScratchEst: fixedScratchEst,
            perTokBytes: perTokBytes,
            availableBytes: availableBytes,
            capacityPlan: capPlan,
            alignedCap: alignedCap,
            finalCap: finalCap,
            totalPlannedBytes: totalPlanned
        )
    }

    /// Estimates Inkling-2 unified-memory footprint for UI, admission, and `1M` `q4` gates.
    public static func estimateContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String? = nil,
        contextTokens: Int,
        kvMode: QWKVMode = .q4,
        mtpDrafts: Int = 0,
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        lsaConfig: QWLSAConfig = QWLSAConfig(),
        tuning: QWEngineTuning = .default,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate {
        let clampedCtx = max(1, min(contextTokens, descriptor.maxPositionEmbeddings))
        let reserve = tuning.osReserveGiB.map { max(0, Int(($0 * Double(1 << 30)).rounded())) }
            ?? QWCapacityBudget.osReserveBytes
        let availBytes = max(0, (physicalMemoryBytes ?? 0) - reserve)
        guard let ink2 = descriptor.inkling2Config else {
            return QWContextMemoryEstimate(
                contextTokens: clampedCtx, weightsBytes: 0, kvBytes: 0, scratchAndStateBytes: 0,
                requiredBytes: 0, availableBytes: availBytes, maxFittingContext: 0, fits: false)
        }
        let rawKVMode = String(describing: kvMode)
        var dsparkURL: URL? = nil
        var baseModelBytes: Int? = nil
        if let packPath, !packPath.isEmpty,
           let loc = try? QWModelLocation(url: URL(fileURLWithPath: packPath.expandingTildePath)) {
            let wDir = loc.weights.hasDirectoryPath ? loc.weights : loc.weights.deletingLastPathComponent()
            dsparkURL = [wDir.appendingPathComponent("dspark.qwp"),
                         loc.assets.appendingPathComponent("dspark.qwp")]
                .first { FileManager.default.fileExists(atPath: $0.path) }
            if loc.kind == .qwp1,
               let attrs = try? FileManager.default.attributesOfItem(atPath: loc.weights.path),
               let fileSize = attrs[.size] as? Int {
                baseModelBytes = fileSize
            }
        }
        let fallbackModelBytes = (try? QWAdmissionPlan.modelResidentBytes(descriptor: descriptor)) ?? 0
        let resolvedBase = baseModelBytes ?? fallbackModelBytes
        let dsparkCfg = (mtpDrafts > 0) ? ink2.dspark : nil
        let activeDrafts = dsparkCfg.map { min(max(0, mtpDrafts), max(0, $0.blockSize - 1)) } ?? 0
        let dsparkWeightEst: Int = (activeDrafts > 0 && dsparkCfg != nil && dsparkURL != nil)
            ? QWInkling2DSpark.estimateWeightBytes(
                mtpURL: dsparkURL!, config: dsparkCfg!, unpaddedVocab: ink2.unpaddedVocabSize)
            : 0
        let lsaHint = resolveLSAAdmissionHint(packPath: packPath, descriptor: descriptor, lsaConfig: lsaConfig)
        guard let sizing = try? computeAdmissionSizing(
            descriptor: descriptor,
            ink2: ink2,
            baseModelBytes: resolvedBase,
            dsparkWeightBytes: dsparkWeightEst,
            dsparkCfg: dsparkCfg,
            activeMtpDrafts: activeDrafts,
            rawKVMode: rawKVMode,
            requestedTokens: clampedCtx,
            prefillChunkSize: prefillChunkSize,
            lsaConfig: lsaConfig,
            lsaIndexDim: lsaHint.indexDim,
            lsaMultiHeadDirect: lsaHint.isMultiHeadDirect,
            availableBytes: availBytes
        ) else {
            return QWContextMemoryEstimate(
                contextTokens: clampedCtx, weightsBytes: resolvedBase + dsparkWeightEst,
                kvBytes: 0, scratchAndStateBytes: 0,
                requiredBytes: resolvedBase + dsparkWeightEst,
                availableBytes: availBytes, maxFittingContext: 0, fits: false)
        }
        let kvBytes = max(0, sizing.finalCap * sizing.perTokBytes)
        return QWContextMemoryEstimate(
            contextTokens: clampedCtx,
            weightsBytes: sizing.weightsEst,
            kvBytes: kvBytes,
            scratchAndStateBytes: sizing.fixedScratchEst,
            requiredBytes: sizing.totalPlannedBytes,
            availableBytes: sizing.availableBytes,
            maxFittingContext: sizing.capacityPlan.maxFittingContext,
            fits: sizing.capacityPlan.fits
        )
    }

    static func resolveLSAAdmissionHint(
        packPath: String?,
        descriptor: QWModelDescriptor,
        lsaConfig: QWLSAConfig
    ) -> (indexDim: Int, isMultiHeadDirect: Bool) {
        let emeans = (lsaConfig.alignedClusterBlock == QWLSAConfig.emeansClusterBlock)
        let supportsFullMH = !emeans && descriptor.kvHeads * descriptor.lsaHeadSliceDim > 0
            && QWLSAConfig.supportsFullSliceQ4Direct(
                qHeads: descriptor.qHeads, kvHeads: descriptor.kvHeads, sliceDim: descriptor.lsaHeadSliceDim)
        guard lsaConfig.isEnabled, lsaConfig.mode != .svd, let packPath, !packPath.isEmpty,
              let url = QWLSAConfig.resolveSidecarURL(packPath: packPath, explicitSidecarPath: lsaConfig.sidecarPath),
              let info = try? QWLSASidecar.peekInfo(url: url, descriptor: descriptor),
              info.geometryMatches, info.indexDim > 0 else {
            return (QWLSAConfig.defaultIndexDim, supportsFullMH)
        }
        let sidecarMH = supportsFullMH && QWLSAConfig.isTrainingFreeSidecarMethod(info.method)
        return (info.indexDim, sidecarMH)
    }

    static func resolveLSAIndexDim(
        packPath: String?,
        descriptor: QWModelDescriptor,
        lsaConfig: QWLSAConfig
    ) -> Int {
        resolveLSAAdmissionHint(packPath: packPath, descriptor: descriptor, lsaConfig: lsaConfig).indexDim
    }
}

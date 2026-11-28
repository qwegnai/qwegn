// DriverConfig.swift — QWEngineDriver DriverConfig family (split from QWEngineDriver.swift; same-module extension).
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

/// The container a model is held in, and why the size-preserving family was
/// declined when it was. Each reason is a different fact about the model, so
/// a user can tell them apart (specs/04-qwquant.md).
struct QWWeightFormatChoice {
    let format: QWWeightFormat
    /// Why the family was not chosen, or nil when it was.
    let declineReason: String?
}

extension QWEngineDriver {
    static func resolveMTPFlag(name: String, value: String?, drafts: Int, defaultValue: Bool = false, maxDrafts: Int = 15) throws -> Bool {
        guard (0...maxDrafts).contains(drafts) else { throw QWEngineDriverError.badMTPDrafts(drafts) }
        switch value {
        case nil: return drafts > 0 && defaultValue
        case "0": return false
        case "1": return drafts > 0
        case .some(let value):
            throw QWEngineDriverError.state("\(name)='\(value)' invalid; expected 0|1")
        }
    }

    /// C08-adjacent (4b): QW_ROPE_YARN_FACTOR — YaRN context extension,
    /// HF yarn-maths (mscale = 0.1·ln(f)+1 applied onto cos/sin; the table
    /// rows are engine positions, HF positions map 1:1). Unset = native
    static func resolveYaRNFactor(_ value: String?) throws -> Float? {
        guard let value else { return nil }
        guard let factor = Float(value) else {
            throw QWEngineDriverError.badYaRN("'\(value)' is not a number")
        }
        guard factor.isFinite, factor > 0, factor <= 64 else {
            throw QWEngineDriverError.badYaRN("factor \(value) outside (0, 64]")
        }
        return factor > 1 ? factor : nil
    }

    static func resolveMTPAdaptiveEntropyThreshold(_ value: String?) throws -> Double {
        guard let value else { return 0.5 }
        guard let threshold = Double(value), threshold.isFinite, (0...1).contains(threshold) else {
            throw QWEngineDriverError.state(
                "QW_MTP_ADAPTIVE_ENTROPY_THRESHOLD='\(value)' invalid; expected a finite number in 0...1")
        }
        return threshold
    }

    /// KV arena budget law (P4-AD-58): each pool holds one page MORE than the
    /// largest single session's page count. A shared-prefix adopter admits
    /// exactly ceil(maxTokens/256) pages; the last partially-written prefix
    static func kvPagesPerPool(maxTokens: Int) -> Int {
        (maxTokens + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens + 1
    }

    /// The decode scratch's partials budget — the 27B's default used to
    /// size it. The budget and the carve are ONE expression, so they
    /// cannot drift apart.
    static func partialsScratchBytes(geometry: QWModelGeometry, mtpDrafts: Int, maxTokens: Int = 262_144) -> Int {
        partialsCarveBytes(geometry: geometry, maxTokens: maxTokens) * (mtpDrafts > 0 ? 2 : 1)
    }

    /// Partials buffer floor for latent/compressed attention models that run
    /// fused MTP draft-head argmax instead of GQA split-K attention decode.
    static func draftHeadArgmaxPartialsBytes(vocab: Int) -> Int {
        let clampedVocab = max(1, vocab)
        let slots = max((clampedVocab + 31) / 32, QWHead.gemvArgmaxSlots(clampedVocab))
        return max(4096, ((2 * slots * 4) + 4095) & ~4095)
    }

    /// The ONE carve's size — the same number the budget counts.
    static func partialsCarveBytes(geometry: QWModelGeometry, maxTokens: Int = 262_144) -> Int {
        if geometry.usesLatentOrCompressedAttention {
            return draftHeadArgmaxPartialsBytes(vocab: geometry.vocab)
        }
        let splits = maxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
        return QWAttnDecode.decodePartialsBytes(geometry: geometry, maxSplits: splits)
    }

    /// Adaptive prefill chunk ceiling at ultra-long context (`> 131K` tokens)
    /// for quadratic full attention; subquadratic MLA/LSA keeps `maxChunk`.
    public static func adaptivePrefillChunkCap(
        position: Int,
        maxChunk: Int,
        isSubquadraticAttention: Bool = false
    ) -> Int {
        guard !isSubquadraticAttention else { return maxChunk }
        if position >= 524_288 {
            return max(32, min(maxChunk, 128))
        } else if position >= 131_072 {
            return max(32, min(maxChunk, 256))
        }
        return maxChunk
    }

    /// The o_proj's group-sum width — the larger of the attention's and
    /// the GDN's. The 27B's `gdnVHeads` equalled it by coincidence.
    static func oProjGroupSumFloats(geometry: QWModelGeometry) -> Int {
        let hcaOutCols = geometry.hcaConfig.map { max(1, $0.oGroups) * $0.oLoraRank } ?? 0
        return max((geometry.hidden + 127) / 128,
                   max((geometry.gdnZChannels + 127) / 128,
                       max((geometry.qHeads * geometry.vHeadDim + 127) / 128,
                           (hcaOutCols + 127) / 128)))
    }

    /// The sparse-MoE chain's per-token scratch — the tensors the routed
    /// layer needs beyond the dense ones.
    static func moeBytesPerToken(geometry: QWModelGeometry) -> Int {
        guard geometry.numExperts > 0 else { return 0 }
        let H = geometry.hidden, I = geometry.moeIntermediate
        let outDim = geometry.moeRoutedDim
        let E = geometry.numExperts, K = geometry.expertsPerTok
        let routerCols = geometry.moeSplitFeatureRouting ? (2 * E) : E
        let latentDim = geometry.moeLatentSize ?? 0
        let ISh = geometry.sharedExpertIntermediate
        let shF32 = ISh > 0 ? 3 * ISh : 0
        let baseF32 = 3 * H + routerCols + K + 3 * K * I + K * outDim + shF32 + 2 * latentDim + 1
        let baseI32 = K
        return baseF32 * 4 + baseI32 * 4
    }

    /// Per-token delta of grouped-NAX MoE prefill relative to `moeBytesPerToken`:
    /// `.f16` `gu16`/`dnHi`/`dnAcc` replace `.f32` `gu32`/`act32`/`dnAcc` (`r1Rows <= 8`).
    static func moeGroupedPrefillExtraBytesPerToken(
        geometry: QWModelGeometry,
        useGroupedNAX: Bool = true
    ) -> Int {
        guard geometry.supportsGroupedNAXMoEPrefill && useGroupedNAX else { return 0 }
        let H = geometry.hidden, I = geometry.moeIntermediate
        let outDim = geometry.moeRoutedDim
        let latentDim = geometry.moeLatentSize ?? 0
        let K = geometry.expertsPerTok, ISh = geometry.sharedExpertIntermediate
        let addedBytes = 5 * K * 4 + (H + ISh + latentDim) * 2
        let savedBytes = (3 * K * I + K * outDim) * 2
        return addedBytes - savedBytes
    }

    /// Exact 256-byte-aligned suballocator footprint of `makeMoEScratch(rows:geometry:useGroupedNAX:)`.
    static func moeScratchAllocationBytes(
        rows: Int,
        geometry: QWModelGeometry,
        useGroupedNAX: Bool = true
    ) -> Int {
        QWScratchArenaBuilder.MoEScratchPlan(
            rows: rows, geometry: geometry, useGroupedNAX: useGroupedNAX).allocationBytes
    }

    /// The sparse-MoE chain's scratch for `rows` rows. Sized for decode
    /// or chunked prefill; shared-expert tensors are zeroed once.
    static func makeMoEScratch(_ alloc: (Int, QWDType) throws -> QWTensorValue,
                               persistentAlloc: ((Int, QWDType) throws -> QWTensorValue)? = nil,
                               rows: Int, geometry: QWModelGeometry,
                               useGroupedNAX: Bool = true) throws -> MoEScratch {
        try QWScratchArenaBuilder.MoEScratchPlan(
            rows: rows, geometry: geometry, useGroupedNAX: useGroupedNAX
        ).carve(alloc, persistentAlloc: persistentAlloc)
    }

    /// The prefill scratch's per-token footprint, accounting for disjoint Attention/GDN vs MLP/MoE lifetimes.
    static func prefillBytesPerToken(
        geometry: QWModelGeometry,
        maxPrefillChunk: Int = 4096,
        useGroupedNAX: Bool = true
    ) -> Int {
        QWScratchArenaBuilder.prefillBytesPerToken(
            geometry: geometry, maxPrefillChunk: maxPrefillChunk, useGroupedNAX: useGroupedNAX)
    }

    /// Non-prefill-chunk activation scratch (`sc` + fixed `psc` tails + optional MTP verify/seeding).
    static func decodeAndMTPScratchBytes(
        geometry: QWModelGeometry,
        prefillChunkRows: Int,
        mtpDrafts: Int,
        maxTokens: Int,
        useGroupedNAX: Bool = true
    ) -> Int {
        let scBytes = moeScratchAllocationBytes(rows: 1, geometry: geometry, useGroupedNAX: useGroupedNAX)
            + geometry.vocab * 4 + QWHead.gemvArgmaxSlots(geometry.vocab) * 8 + (16 << 20)
        let pscFixedTail = (8 << 20)
        guard mtpDrafts > 0 else {
            return max(64 << 20, scBytes + pscFixedTail)
        }
        let verifyRows = max(2, min(16, mtpDrafts + 1))
        let verifyBytes = QWVerifyWindow.allocationBytes(
            geometry: geometry, rows: verifyRows, layerMajor: true, gpuTargets: true,
            q4Head: true, useGroupedNAX: useGroupedNAX)
        let promptEmbedBytes = prefillChunkRows * geometry.hidden * 2
        let maxDecodeSplits = maxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
        let sharedAttnBytes = geometry.usesLatentOrCompressedAttention
            ? draftHeadArgmaxPartialsBytes(vocab: geometry.vocab)
            : QWAttnDecode.decodePartialsBytesBatched(
                T: min(8, verifyRows), geometry: geometry, maxSplits: maxDecodeSplits)
        let mtpLongExtra = maxTokens > 262_144
            ? verifyRows * partialsCarveBytes(geometry: geometry, maxTokens: maxTokens) : 0
        let mtpMiscBytes = (16 << 20)
        return max(64 << 20, scBytes + pscFixedTail + verifyBytes + promptEmbedBytes
            + sharedAttnBytes + mtpLongExtra + mtpMiscBytes)
    }

    /// The KV arena budget law, sized from the model's **own** geometry: the
    /// pool holds the model's full-attention layers, not the 27B's 16.
    static func kvBudgetBytes(maxTokens: Int, mtpDrafts: Int, kvMode: QWKVMode = .f16,
                              geometry: QWModelGeometry,
                              prefillChunk: Int = 4096) -> Int {
        let attnLayers = (0..<geometry.numVirtualLayers).filter { geometry.isVirtualLayerFullAttention($0) }
        let pageBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geometry)
        let trunkTables = QWKVCache.tableSlotBytes * 2 * attnLayers.count * QWKVCache.maxSeqs
        let trunkPoolBytes: Int
        if geometry.hasAnySlidingWindow {
            let pagesSum = attnLayers.reduce(0) { acc, v in
                let w = geometry.virtualLayerSlidingWindow(v)
                let cap = w > 0 ? min(maxTokens, w + prefillChunk + 512) : maxTokens
                return acc + kvPagesPerPool(maxTokens: cap)
            }
            trunkPoolBytes = pagesSum * 2 * pageBytes
        } else {
            trunkPoolBytes = kvPagesPerPool(maxTokens: maxTokens) * 2 * attnLayers.count * pageBytes
        }
        let trunk = trunkPoolBytes + trunkTables
        let mtp = mtpDrafts > 0
            ? (kvPagesPerPool(maxTokens: maxTokens) * 2 * pageBytes + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs)
            : 0
        return trunk + mtp
    }

    /// Modes whose DECODE KERNELS actually read the pool (4c lane shipped):
    /// qwattn_decode / _batched / _shared bind the pools with a runtime
    /// `q8mode` constant (0 f16, 1 q8, 2 q4) — quantise-on-write epilogue +
    static let kernelSupportedKVModes: Set<QWKVMode> = [.f16, .q8, .q4]

    /// The geometry's **serving law**, checked before any GPU work: the
    /// compressed KV modes are the 27B's layout (C103, C108), and a head's
    /// width must be a multiple of 32, up to 256 halves (C112).
    static func validateGeometry(_ geometry: QWModelGeometry, kvMode: QWKVMode) throws {
        if geometry.usesLatentOrCompressedAttention {
            try validateKVMode(mode: kvMode)
            if geometry.mlaLayerCount > 0 {
                guard geometry.kvLoraRank > 0, geometry.kvLoraRank % 32 == 0 else {
                    throw QWEngineDriverError.unsupportedModel(
                        "kv_lora_rank \(geometry.kvLoraRank) must be a positive multiple of 32")
                }
            }
            if let hca = geometry.hcaConfig {
                guard hca.headDim > 0, hca.headDim % 32 == 0 else {
                    throw QWEngineDriverError.unsupportedModel(
                        "HCA head_dim \(hca.headDim) must be a positive multiple of 32")
                }
            }
            return
        }
        try validateServing(headDim: geometry.headDim, kvHeads: geometry.kvHeads,
                            qHeads: geometry.qHeads, kvMode: String(describing: kvMode))
    }

    /// The container a checkpoint's weights are held in, and so the basis
    /// its activations are in (specs/02-qwhal.md). The reason the family was
    /// declined, when it was, rides along with the choice (specs/04).
    static func weightFormatChoice(for root: QWModelRoot, quantization: QWQuantizationSpec?,
                                    geometry: QWModelGeometry,
                                    mtpDrafts: Int,
                                    pack: QWQuant.Pack? = nil) throws -> QWWeightFormatChoice {
        switch root.kind {
        case .safetensorsSingle, .safetensorsSharded:
            if case .recurrentLoop(_, let ink) = geometry.descriptor.executionTopology, let ink {
                let counts = ink.expertModesCount(numLayers: geometry.numLayers)
                return QWWeightFormatChoice(
                    format: .nativeMixedFP(denseBits: 16, nvfp4Layers: counts.nvfp4, mxfp8Layers: counts.mxfp8),
                    declineReason: nil)
            }
            guard let q = quantization else {
                return QWWeightFormatChoice(
                    format: .hq4G32,
                    declineReason: "the checkpoint says nothing about its quantisation")
            }
            if let method = q.method, !method.isEmpty {
                guard method.lowercased() == "affine" else {
                    throw QWEngineDriverError.badConfig(
                        "quantization.method '\(method)' is not implemented — this engine reads "
                        + "the affine scheme only")
                }
            }
            if let mode = q.mode, !mode.isEmpty {
                guard mode.lowercased() == "affine" else {
                    throw QWEngineDriverError.badConfig(
                        "quantization.mode '\(mode)' is not implemented — this engine reads the "
                        + "affine scheme only")
                }
            }
            guard let bits = q.bits, let groupSize = q.groupSize else {
                return QWWeightFormatChoice(
                    format: .hq4G32,
                    declineReason: "the checkpoint names no (bits, group) pair")
            }
            // The family must hold the pair at all...
            guard QWAQBlock.isSupported(bits: bits, group: groupSize) else {
                return QWWeightFormatChoice(
                    format: .hq4G32,
                    declineReason: "the family holds no container for a \(bits)-bit "
                    + "group \(groupSize) checkpoint")
            }
            // ...and every weight kernel this model dispatches must have a
            // twin in the family. A path without one would read the
            // family's blocks with the incumbent's kernel.
            if let reason = familyDeclineReason(geometry: geometry, mtpDrafts: mtpDrafts) {
                return QWWeightFormatChoice(format: .hq4G32, declineReason: reason)
            }
            // The environment may force the incumbent's container, so the
            // two routes can be measured against each other in one build.
            if forcedWeightFormat() == .hq4G32 {
                return QWWeightFormatChoice(
                    format: .hq4G32,
                    declineReason: "QW_WEIGHT_FORMAT forces the incumbent's container")
            }
            return QWWeightFormatChoice(
                format: .nativeAffine(bits: bits, groupSize: groupSize), declineReason: nil)
        case .coreAIModel:
            throw QWEngineDriverError.badConfig(
                "CoreAI .aimodel bundles run via QWCoreAIDecisionAdapter, not Metal DriverConfig")
        case .qwp1, nil:
            if let pack {
                typealias FormatSpec = (bits: Int, group: Int, incumbent: Bool, shvq: Bool)
                typealias FormatEntry = (name: String, spec: FormatSpec)
                var denseBest: FormatEntry?
                var denseAuxFallback: FormatEntry?
                var expBest: FormatEntry?
                var expDownBest: FormatEntry?
                // Track the lexicographically smallest name per bucket so SipHash order
                // cannot let auxiliary tensors (e.g. hq8 lm_head) override trunk format.
                for (name, entry) in pack.tensors where entry.hq4Off != nil {
                    let spec: FormatSpec = (bits: entry.hqBits, group: entry.hqGroup,
                                            incumbent: entry.isIncumbentHQ4, shvq: entry.isSHVQ)
                    if entry.shape.count == 3 || (name.contains(".experts.") && !name.contains("shared_expert")) {
                        if name.contains("down_proj") || name.contains(".w2.") {
                            if expDownBest.map({ name < $0.name }) ?? true { expDownBest = (name, spec) }
                        } else if expBest.map({ name < $0.name }) ?? true {
                            expBest = (name, spec)
                        }
                    } else if entry.shape.count == 2 {
                        if Self.isAuxiliary2DTensor(name) {
                            if denseAuxFallback.map({ name < $0.name }) ?? true { denseAuxFallback = (name, spec) }
                        } else if denseBest.map({ name < $0.name }) ?? true {
                            denseBest = (name, spec)
                        }
                    }
                }
                let d = (denseBest ?? denseAuxFallback)?.spec
                    ?? (bits: 4, group: 32, incumbent: true, shvq: false)
                let e = expBest?.spec ?? expDownBest?.spec ?? d
                let ed = expDownBest?.spec ?? e
                if !(d.incumbent && e.incumbent && ed.incumbent) {
                    return QWWeightFormatChoice(
                        format: .hqBlock(bits: d.bits, groupSize: d.group,
                                         expertBits: e.bits, expertGroupSize: e.group,
                                         denseIncumbent: d.incumbent, expertIncumbent: e.incumbent,
                                         expertDownBits: ed.bits, expertDownGroupSize: ed.group,
                                         expertDownIncumbent: ed.incumbent,
                                         denseSHVQ: d.shvq, expertSHVQ: e.shvq,
                                         expertDownSHVQ: ed.shvq),
                        declineReason: nil)
                }
            }
            return QWWeightFormatChoice(
                format: .hq4G32,
                declineReason: "the engine's own pack holds the incumbent's container")
        }
    }

    private static func isAuxiliary2DTensor(_ name: String) -> Bool {
        name == "lm_head.weight"
            || name.hasSuffix(".lm_head.weight")
            || name.contains("embed_tokens")
            || name.contains("vision_tower.")
            || name.contains(".visual.")
            || name.hasPrefix("visual.")
    }

    /// The container alone, for the callers that only need to dispatch.
    static func weightFormat(for root: QWModelRoot, quantization: QWQuantizationSpec?,
                             geometry: QWModelGeometry,
                             mtpDrafts: Int,
                             pack: QWQuant.Pack? = nil) throws -> QWWeightFormat {
        try weightFormatChoice(for: root, quantization: quantization,
                               geometry: geometry, mtpDrafts: mtpDrafts, pack: pack).format
    }

    /// The container the environment forces, when it forces one. Only the
    /// INCUMBENT can be forced: a checkpoint the family cannot hold has no
    /// native container to fall back to.
    static func forcedWeightFormat() -> QWWeightFormat? {
        switch ProcessInfo.processInfo.environment["QW_WEIGHT_FORMAT"]?.lowercased() {
        case "incumbent", "hq4g32": return .hq4G32
        default: return nil
        }
    }

    /// Why the family cannot serve this model, when it cannot.
    /// Both standard and MTP paths have native affine twins; supported
    /// MTP configurations return `nil`.
    static func familyDeclineReason(geometry: QWModelGeometry, mtpDrafts: Int) -> String? {
        _ = (geometry, mtpDrafts)
        return nil
    }

    /// The same law over the numbers a **plan** carries — the module's
    /// pre-flight and the driver's check read one predicate, so the two
    /// cannot disagree about what the standard path can serve (C110, C112).
    static func validateServing(headDim: Int, kvHeads: Int, qHeads: Int,
                                 kvMode: String, allowF32: Bool = false) throws {
        let lowerMode = kvMode.lowercased()
        let allowed = allowF32 ? ["f32", "f16", "q8", "q4"] : ["f16", "q8", "q4"]
        guard allowed.contains(lowerMode) else {
            throw QWEngineDriverError.unsupportedKVMode(kvMode)
        }
        guard headDim % 32 == 0 else {
            throw QWEngineDriverError.unsupportedModel(
                "head_dim \(headDim) is not a multiple of 32")
        }
        guard headDim <= 256 else {
            throw QWEngineDriverError.unsupportedModel(
                "head_dim \(headDim) is wider than the 256 halves the lane mapping covers")
        }
    }

    static func validateKVMode(mode: QWKVMode) throws {
        guard kernelSupportedKVModes.contains(mode) else {
            throw QWEngineDriverError.unsupportedKVMode(String(describing: mode))
        }
    }

    /// EXP-12: the batched q4 VERIFY head rides the draft head's sidecar
    /// planes — without them there is no q4 copy of lm_head to verify with.
    static func validateMTPVerifyHeadQ4(enabled: Bool, draftHead: Bool) throws {
        guard !enabled || draftHead else {
            throw QWEngineDriverError.state("QW_MTP_VERIFY_HEAD_Q4=1 requires QW_MTP_DRAFT_HEAD=1 (the q4 planes ride its sidecar)")
        }
    }

    /// Non-KV fixed buffer footprint of `QWMLAAttnOperator` (`mla-cache`):
    /// pending rings, chunk latent/query/output scratch, split partials, and indexer workspace.
    static func mlaFixedScratchBytes(
        geometry: QWModelGeometry,
        prefillChunkRows: Int,
        activeAttnLayers: Int
    ) -> Int {
        guard geometry.usesAnyMLA, activeAttnLayers > 0 else { return 0 }
        let maxT = max(32, prefillChunkRows)
        var ringCap = 4096
        while ringCap < maxT * 2 { ringCap <<= 1 }
        let hasIndexer = geometry.mlaHasIndexer
        let idxDim = hasIndexer ? max(32, geometry.indexerHeadDim) : 0
        let idxHeads = hasIndexer ? max(1, geometry.indexerHeads) : 0
        let ropeDim = geometry.usesNoPEMLA ? 0 : geometry.qkRopeHeadDim
        let nopeDim = geometry.qkNopeHeadDim > 0 ? geometry.qkNopeHeadDim : geometry.headDim
        let gateMode = geometry.mlaConfig?.outputGateMode
            ?? (geometry.attnOutputGate ? .elementWiseFromQLora : .none)
        let hmla = geometry.hybridMLAConfig
        let fuseOGateInQKV = (hmla == nil && gateMode == .elementWiseFromHidden)
        let inlineGate = (gateMode == .elementWiseFromQLora) ? geometry.qHeads * geometry.vHeadDim : 0
        let extGateCols = gateMode == .headWiseFromHidden
            ? geometry.qHeads
            : (gateMode == .elementWiseFromHidden ? geometry.qHeads * geometry.vHeadDim : 0)
        let yVhaCols = geometry.usesDecoupledRopeMLA ? (geometry.qHeads * geometry.vHeadDim) : 0
        let extraQ = geometry.usesNoPEMLA ? 0 : (geometry.qHeads * ropeDim + inlineGate)
        let latentStride = (geometry.mlaHasDirectQ ? 0 : geometry.qLoraRank)
            + geometry.kvLoraRank + ropeDim + idxDim + (fuseOGateInQKV ? geometry.qHeads * geometry.vHeadDim : 0)
        let qFullStride = geometry.qHeads * nopeDim + extraQ + idxHeads * idxDim
        let perLayerRings = (geometry.usesDecoupledRopeMLA || !hasIndexer)
            ? 0
            : (activeAttnLayers * 2 * ringCap * idxDim * 2)
        let perTokChunk = (latentStride + geometry.qLoraRank + qFullStride
            + 2 * geometry.qHeads * geometry.kvLoraRank + extGateCols + yVhaCols) * 2
            + idxHeads * 4 + (hasIndexer ? 1024 * 4 : 0) + 16
        let splitPartials = min(maxT, 64) * 16 * geometry.qHeads * (geometry.kvLoraRank * 2 + 8)
        let indexerWorkspace = hasIndexer ? (72 << 20) : (2 << 20)
        return perLayerRings + maxT * perTokChunk + splitPartials + indexerWorkspace
    }

    static func hcaFixedScratchBytes(
        geometry: QWModelGeometry,
        prefillChunkRows: Int
    ) -> Int {
        guard let hca = geometry.hcaConfig, geometry.hcaLayerCount > 0 else { return 0 }
        let maxT = max(16, prefillChunkRows)
        let minRing = max(hca.slidingWindow, hca.defaultCompressRatio) + max(32, maxT)
        var ringCap = 256
        while ringCap < minRing { ringCap <<= 1 }
        let perLayerRingBytes = ringCap * hca.headDim * (2 + 4 + 4)
        let inStride = hca.qLoraRank + 3 * hca.headDim
        let zOutStride = max(1, hca.oGroups) * hca.oLoraRank
        let qUpStride = hca.qHeads * hca.headDim + zOutStride
        let chunkScratch = maxT * (inStride + hca.qLoraRank + qUpStride + 2 * hca.qHeads * hca.headDim + zOutStride) * 2
        return geometry.hcaLayerCount * perLayerRingBytes + chunkScratch + (24 << 20)
    }

    static func hcaPerTokenBytes(geometry: QWModelGeometry) -> Int {
        guard let hca = geometry.hcaConfig, geometry.hcaLayerCount > 0 else { return 0 }
        let ratio = max(1, hca.defaultCompressRatio)
        return (geometry.hcaLayerCount * (hca.headDim + 64) * 2 + ratio - 1) / ratio
    }

    static func mbsaFixedScratchBytes(
        geometry: QWModelGeometry,
        prefillChunkRows: Int,
        attnLayerCount: Int
    ) -> Int {
        guard let bsa = geometry.blockSparseGQAConfig, attnLayerCount > 0 else { return 0 }
        let maxR = max(8, prefillChunkRows)
        let idxH = geometry.mbsaIndexerHeads
        let idxD = geometry.mbsaIndexerHeadDim
        let qkv = maxR * (idxH + 1) * idxD * 2
        let q = maxR * idxH * idxD * 2
        let blkList = maxR * geometry.kvHeads * max(1, bsa.maxSelectedBlocks) * 4
        let blkCnt = maxR * geometry.kvHeads * 4
        let maxTiles = (maxR + 31) / 32
        let tileStride = QWMBSAAttnOperator.tileListStride(config: bsa)
        let tileList = maxTiles * geometry.kvHeads * tileStride * 4 * 2
        let tileCnt = maxTiles * geometry.kvHeads * 4
        return qkv + q + blkList + blkCnt + tileList + tileCnt + attnLayerCount * 256 + 4096
    }

    static func mbsaPerTokenBytes(geometry: QWModelGeometry, attnLayerCount: Int) -> Int {
        guard geometry.blockSparseGQAConfig != nil, attnLayerCount > 0 else { return 0 }
        return attnLayerCount * geometry.mbsaIndexerHeadDim * 2
    }
}

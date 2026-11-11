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
import QWState

/// The container a model is held in, and why the size-preserving family was
/// declined when it was. Each reason is a different fact about the model, so
/// a user can tell them apart (specs/04-qwquant.md).
struct QWWeightFormatChoice {
    let format: QWWeightFormat
    /// Why the family was not chosen, or nil when it was.
    let declineReason: String?
}

extension QWEngineDriver {
    static func resolveMTPFlag(name: String, value: String?, drafts: Int, defaultValue: Bool = false, maxDrafts: Int = 5) throws -> Bool {
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

    /// The ONE carve's size — the same number the budget counts.
    static func partialsCarveBytes(geometry: QWModelGeometry, maxTokens: Int = 262_144) -> Int {
        let splits = maxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
        return QWAttnDecode.decodePartialsBytes(geometry: geometry, maxSplits: splits)
    }

    /// Adaptive prefill chunk ceiling at ultra-long context (`> 131K` tokens)
    /// to keep quadratic full-attention prefill command buffers well inside
    /// the macOS Metal watchdog window.
    public static func adaptivePrefillChunkCap(position: Int, maxChunk: Int) -> Int {
        if position >= 524_288 {
            return max(32, min(maxChunk, 128))
        } else if position >= 131_072 {
            return max(32, min(maxChunk, 256))
        }
        return maxChunk
    }

    /// The group-sum scratch's width: the number of 128-wide groups a GEMV
    /// over `k` writes, rounded UP to the kernel's 64-float tile. The 27B's
    /// widths (64, 64, 192) used to be written as literals.
    static func groupSumFloats(k: Int) -> Int {
        let groups = (k + 127) / 128
        return (groups + 63) / 64 * 64
    }

    /// The o_proj's group-sum width — the larger of the attention's and
    /// the GDN's. The 27B's `gdnVHeads` equalled it by coincidence.
    static func oProjGroupSumFloats(geometry: QWModelGeometry) -> Int {
        max((geometry.hidden + 127) / 128,
            max((geometry.gdnZChannels + 127) / 128,
                (geometry.qHeads * geometry.vHeadDim + 127) / 128))
    }

    /// The sparse-MoE chain's per-token scratch — the tensors the routed
    /// layer needs beyond the dense ones.
    static func moeBytesPerToken(geometry: QWModelGeometry) -> Int {
        guard geometry.numExperts > 0 else { return 0 }
        let H = geometry.hidden, I = geometry.moeIntermediate
        let E = geometry.numExperts, K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let shF32 = ISh > 0 ? 3 * ISh : 0
        let baseF32 = 3 * H + E + K + 3 * K * I + K * H + shF32 + 1
        let baseI32 = K
        return baseF32 * 4 + baseI32 * 4
    }

    /// Additional per-token scratch bytes for the grouped NAX MoE prefill path.
    static func moeGroupedPrefillExtraBytesPerToken(geometry: QWModelGeometry) -> Int {
        guard geometry.supportsGroupedNAXMoEPrefill else { return 0 }
        let H = geometry.hidden, I = geometry.moeIntermediate
        let E = geometry.numExperts, K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let naxI32 = 5 * K
        let naxF16 = 3 * K * I + H + ISh
        let naxF32 = K * H
        return naxF32 * 4 + naxI32 * 4 + naxF16 * 2 + (E + 64) * 4
    }

    /// Exact 256-byte-aligned suballocator footprint of `makeMoEScratch(rows:geometry:)`.
    static func moeScratchAllocationBytes(rows: Int, geometry: QWModelGeometry) -> Int {
        guard geometry.numExperts > 0 else { return 0 }
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let H = geometry.hidden, I = geometry.moeIntermediate
        let E = geometry.numExperts, K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let isPrefill = rows > 8 && geometry.supportsGroupedNAXMoEPrefill
        var total = 2 * aligned(rows * H * 4)
            + aligned(rows * E * 4)
            + aligned(rows * K * 4)
            + aligned(rows * K * 4)
            + aligned(rows * K * 2 * I * 4)
            + aligned(rows * K * I * 4)
            + (ISh > 0 ? aligned(rows * 2 * ISh * 4) + aligned(rows * ISh * 4) : 0)
            + aligned(rows * H * 4)
            + aligned(rows * 4)
            + aligned(rows * K * H * 4)
        if isPrefill {
            let tcInts = QWGemm.MoETilePlan(
                experts: E, pairs: rows * K, gateUpRows: 2 * I, downRows: H).scratchInts
            total += aligned((E + 1) * 4)
                + 5 * aligned(rows * K * 4)
                + aligned(tcInts * 4)
                + aligned(rows * K * 2 * I * 2)
                + aligned(rows * K * I * 2)
                + aligned(rows * K * H * 4)
                + aligned(rows * (H + ISh) * 2)
        }
        return total
    }

    /// The sparse-MoE chain's scratch for `rows` rows. Sized for decode
    /// or chunked prefill; shared-expert tensors are zeroed once.
    static func makeMoEScratch(_ alloc: (Int, QWDType) throws -> QWTensorValue,
                               rows: Int, geometry: QWModelGeometry) throws -> MoEScratch {
        let H = geometry.hidden, I = geometry.moeIntermediate
        let E = geometry.numExperts, K = geometry.expertsPerTok
        let ISh = geometry.sharedExpertIntermediate
        let isPrefill = rows > 8 && geometry.supportsGroupedNAXMoEPrefill
        let tcInts = isPrefill ? QWGemm.MoETilePlan(
            experts: E, pairs: rows * K, gateUpRows: 2 * I, downRows: H).scratchInts : 0
        let s = MoEScratch(
            xRot32: try alloc(rows * H, .f32),
            xPlain32: try alloc(rows * H, .f32),
            logits: try alloc(rows * E, .f32),
            idx: try alloc(rows * K, .i32),
            wts: try alloc(rows * K, .f32),
            gu32: try alloc(rows * K * 2 * I, .f32),
            act32: try alloc(rows * K * I, .f32),
            guSh32: ISh > 0 ? try alloc(rows * 2 * ISh, .f32) : nil,
            actSh32: ISh > 0 ? try alloc(rows * ISh, .f32) : nil,
            shared: try alloc(rows * H, .f32),
            gate: try alloc(rows, .f32),
            expOff: isPrefill ? try alloc(E + 1, .i32) : nil,
            sortedPair: isPrefill ? try alloc(rows * K, .i32) : nil,
            sortedTok: isPrefill ? try alloc(rows * K, .i32) : nil,
            invPerm: isPrefill ? try alloc(rows * K, .i32) : nil,
            tileE: isPrefill ? try alloc(rows * K, .i32) : nil,
            tileR: isPrefill ? try alloc(rows * K, .i32) : nil,
            tileCount: isPrefill ? try alloc(tcInts, .i32) : nil,
            gu16: isPrefill ? try alloc(rows * K * 2 * I, .f16) : nil,
            dnHi: isPrefill ? try alloc(rows * K * I, .f16) : nil,
            dnAcc: try alloc(rows * K * H, .f32),
            naxX16: isPrefill ? try alloc(rows * (H + ISh), .f16) : nil
        )
        // A standard sparse model has no shared expert (C122), so the shared
        // term is zero — not "whatever was in the buffer". When a model has an
        // ungated shared expert (e.g. GLM-5.3-Flash), initialize gate to 1.0.
        memset(s.shared.region.pointer.advanced(by: s.shared.offset), 0, s.shared.bytes)
        if ISh > 0 && !geometry.sharedExpertHasGate {
            let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<rows { gPtr[i] = 1.0 }
        } else {
            memset(s.gate.region.pointer.advanced(by: s.gate.offset), 0, s.gate.bytes)
        }
        return s
    }

    /// The prefill scratch's per-token footprint — the 27B's 760_000-byte
    /// estimate used to stand in for it.
    static func prefillBytesPerToken(geometry: QWModelGeometry) -> Int {
        let H = geometry.hidden, I = geometry.intermediate
        let qC = geometry.qChannels, qOut = geometry.qProjOut, kvE = geometry.kvChannels
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let vh = geometry.gdnVHeads
        // f16: x, ln1, ln2, o16, dn16 (5H); gdnY16, attnY16; qkv16, convY
        // (2·qkvC); z16, oOut (2·zC); attOut (qC); attV16, attKR16 (2·kvE);
        // attQR16 (qOut); guRaw16, gu16, dn16 (3I); optional fgb16.
        let fgbC = geometry.gdnVectorDecay ? (2 * geometry.gdnLowRankDim + vh) : 0
        let f16 = 5 * H + geometry.gdnTotalN + geometry.attnTotalN + 2 * qkvC
            + 2 * zC + qC + 2 * kvE + qOut + max(2 * I, 2 * H) + I + fgbC
        // f32: the three group sums; gdnY, attnY; qkv32; z32, oScr, U, W,
        // A, B (6·zC); a32, b32, betas, glog; o32, dn32 (2H); gu32 (I).
        let mhcC = geometry.mhcConfig?.streams ?? 0
        let glogC = geometry.gdnGlogChannels
        let f32 = groupSumFloats(k: H) + groupSumFloats(k: max(zC, qC)) + groupSumFloats(k: I)
            + geometry.gdnTotalN + geometry.attnTotalN + qkvC + 6 * zC + 3 * vh + glogC
            + 2 * H + I
            + (mhcC > 0 ? (mhcC * H + mhcC + mhcC * mhcC) : 0)
        return f16 * 2 + f32 * 4 + moeBytesPerToken(geometry: geometry)
            + moeGroupedPrefillExtraBytesPerToken(geometry: geometry)
    }

    /// The KV arena budget law, sized from the model's **own** geometry: the
    /// pool holds the model's full-attention layers, not the 27B's 16.
    static func kvBudgetBytes(maxTokens: Int, mtpDrafts: Int, kvMode: QWKVMode = .f16,
                              geometry: QWModelGeometry) -> Int {
        let attnLayers = (0..<geometry.numVirtualLayers).filter { geometry.isVirtualLayerFullAttention($0) }.count
        func cache(_ layers: Int) -> Int {
            // Exact inverse of QWKVCache.init's pool arithmetic:
            // poolBytes = (region − tables) / (2 × layers), floored to
            // whole pages. This layout yields exactly kvPagesPerPool per
            // pool with no leftover slack.
            kvPagesPerPool(maxTokens: maxTokens) * 2 * layers
                * QWKVCache.pageBytes(mode: kvMode, geometry: geometry)
                + QWKVCache.tableSlotBytes * 2 * layers * QWKVCache.maxSeqs
        }
        return cache(attnLayers) + (mtpDrafts > 0 ? cache(1) : 0)
    }

    /// Modes whose DECODE KERNELS actually read the pool (4c lane shipped):
    /// qwattn_decode / _batched / _shared bind the pools with a runtime
    /// `q8mode` constant (0 f16, 1 q8, 2 q4) — quantise-on-write epilogue +
    static let kernelSupportedKVModes: Set<QWKVMode> = [.f16, .q8, .q4]

    /// The geometry's **serving law**, checked before any GPU work: the
    /// compressed KV modes are the 27B's layout (C103, C108), and a head's
    /// width must be a multiple of 32, up to 256 halves (C112).
    static func validateGeometry(_ geometry: QWModelGeometry, kvMode: QWKVMode) throws {
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
        case .qwp1, nil:
            if let pack {
                var denseSpec: (bits: Int, group: Int, incumbent: Bool, shvq: Bool)?
                var expSpec: (bits: Int, group: Int, incumbent: Bool, shvq: Bool)?
                var expDownSpec: (bits: Int, group: Int, incumbent: Bool, shvq: Bool)?
                for (name, entry) in pack.tensors where entry.hq4Off != nil {
                    let spec = (bits: entry.hqBits, group: entry.hqGroup,
                                incumbent: entry.isIncumbentHQ4, shvq: entry.isSHVQ)
                    if entry.shape.count == 3 || (name.contains(".experts.") && !name.contains("shared_expert")) {
                        if name.contains("down_proj") {
                            if expDownSpec == nil { expDownSpec = spec }
                        } else {
                            if expSpec == nil { expSpec = spec }
                        }
                    } else if entry.shape.count == 2 {
                        if denseSpec == nil { denseSpec = spec }
                    }
                }
                let d = denseSpec ?? (bits: 4, group: 32, incumbent: true, shvq: false)
                let e = expSpec ?? expDownSpec ?? d
                let ed = expDownSpec ?? e
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

    /// Whether every weight kernel this model dispatches has a twin in the
    /// size-preserving family. A model with experts, with GDN layers, or with
    /// MTP drafts reads weights through paths the family has no twin for.
    static func familyServes(geometry: QWModelGeometry, mtpDrafts: Int) -> Bool {
        familyDeclineReason(geometry: geometry, mtpDrafts: mtpDrafts) == nil
    }

    /// Why the family cannot serve this model, when it cannot. Each reason
    /// is a path whose weights the family has no twin for, so the incumbent's
    /// kernel would read the family's blocks with the wrong layout.
    static func familyDeclineReason(geometry: QWModelGeometry, mtpDrafts: Int) -> String? {
        if mtpDrafts != 0 { return "the family has no twin for the MTP draft path" }
        return nil
    }

    /// The same law over the numbers a **plan** carries — the module's
    /// pre-flight and the driver's check read one predicate, so the two
    /// cannot disagree about what the standard path can serve (C110, C112).
    static func validateServing(headDim: Int, kvHeads: Int, qHeads: Int,
                                 kvMode: String) throws {
        let lowerMode = kvMode.lowercased()
        guard lowerMode == "f16" || lowerMode == "q8" || lowerMode == "q4" else {
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
}

// DriverStartup.swift — environment/backend config phase of the driver
// initializer (split from QWEngineDriver.init; same resolve order).
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
    struct StartupConfig {
        let mtpBackend: QWMTPBackend
        let prefillScratchEviction: QWPrefillScratchEvictionPolicy
        let mtpAdaptiveDepth: Bool
        let mtpAdaptiveEntropyThreshold: Double
        let mtpRelaxedMargin: Double
        let mtpDraftStrategy: String
        let mtpQuadraticSeeding: Bool
        let mtpStageProfiling: Bool
        let mtpVerificationProfiling: Bool
        let draftHeadEnabled: Bool
        let verifyHeadQ4Enabled: Bool
        let headIsQ4g: Bool
        let dflashDraftPackPath: String
        let dflashSelectorTopK: Int?
        let environment: [String: String]
    }

    static func resolveStartup(packPath: String, kvMode: QWKVMode,
                              mtpDrafts: Int, mtpBackend: QWMTPBackend?,
                              prefillScratchEviction: QWPrefillScratchEvictionPolicy,
                              mtpAdaptiveDepth: Bool?, mtpAdaptiveEntropyThreshold: Double?,
                              mtpRelaxedMargin: Double = 1.5,
                              mtpDraftStrategy: String = "hybrid",
                              mtpQuadraticSeeding: Bool = true,
                              dflashDraftPackPath: String? = nil,
                              dflashSelectorTopK: Int? = nil)
        throws -> StartupConfig {
        var evictionResolved = prefillScratchEviction
        guard FileManager.default.fileExists(atPath: packPath) else {
            throw QWEngineDriverError.packMissing(packPath)
        }
        let maxDrafts = mtpBackend == .dflash2 ? 15 : 5
        guard mtpDrafts >= 0 && mtpDrafts <= maxDrafts else {
            throw QWEngineDriverError.badMTPDrafts(mtpDrafts)
        }
        // §3 identity gate: the registry owns the family → module mapping. A
        // pack or checkpoint whose family no module serves reaches an explicit
        // unsupported-model error instead of the dense loader.
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
        if isSafetensors {
            guard let cfgURL = root.configURL else {
                throw QWEngineDriverError.packMissing(packPath)
            }
            let desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: cfgURL))
            let served = QWModuleRegistry.moduleType(for: desc.family)
            guard served is DenseModule.Type else {
                throw QWEngineDriverError.unsupportedModel(desc.family.rawValue)
            }
        } else {
            let identity = try QWQuant.Pack.peekProvenance(url: URL(fileURLWithPath: packPath))
            let family = identity.modelId.flatMap { QWModuleRegistry.family(forPackModelId: $0) }
            let served = family.flatMap { QWModuleRegistry.moduleType(for: $0) }
            if let mid = identity.modelId, !(served is DenseModule.Type) {
                throw QWEngineDriverError.unsupportedModel(mid)
            }
        }
        // Validate KV mode before any arena work (f16, q8, and q4 are all
        // shipped and parity-gated across prefill, decode, and MTP verify).
        try Self.validateKVMode(mode: kvMode)
        // L1: resolve the backend BEFORE any pack read, HAL, or arena work.
        // Invalid selections fail here, before mutation (plan §2.1).
        let mtpBackendResolved = try QWMTPBackend.resolveSelection(mtpBackend, drafts: mtpDrafts)
        let headIsQ4g = isSafetensors
            ? true
            : ((try? QWQuant.Pack(url: URL(fileURLWithPath: packPath))
                .entry("lm_head.weight").quantized) ?? false)
        let environment = ProcessInfo.processInfo.environment
        if let env = environment["QW_EVICT_PREFILL_SCRATCH"] {
            evictionResolved = try QWPrefillScratchEvictionPolicy.parse(env)
        } else {
            evictionResolved = prefillScratchEviction
        }
        let resolvedAdaptiveDepth = try Self.resolveMTPFlag(
            name: "QW_MTP_ADAPTIVE_DEPTH",
            value: environment["QW_MTP_ADAPTIVE_DEPTH"],
            drafts: mtpDrafts,
            defaultValue: mtpAdaptiveDepth ?? false,
            maxDrafts: maxDrafts
        )
        let resolvedEntropy: Double
        if let envEntropy = environment["QW_MTP_ADAPTIVE_ENTROPY_THRESHOLD"] {
            resolvedEntropy = try Self.resolveMTPAdaptiveEntropyThreshold(envEntropy)
        } else if let mtpAdaptiveEntropyThreshold {
            resolvedEntropy = try Self.resolveMTPAdaptiveEntropyThreshold(String(mtpAdaptiveEntropyThreshold))
        } else {
            resolvedEntropy = 0.5
        }
        let mtpStageProfiling = try Self.resolveMTPFlag(name: "QW_MTP_PROFILE",
            value: environment["QW_MTP_PROFILE"], drafts: mtpDrafts, maxDrafts: maxDrafts)
        let mtpVerificationProfiling = try Self.resolveMTPFlag(name: "QW_MTP_VERIFY_PROFILE",
            value: environment["QW_MTP_VERIFY_PROFILE"], drafts: mtpDrafts, maxDrafts: maxDrafts)
        guard !mtpVerificationProfiling || mtpStageProfiling else {
            throw QWEngineDriverError.state("QW_MTP_VERIFY_PROFILE=1 requires MTP profiling")
        }
        // L1 (P4-AD-60): draft-only q4g LM head. The verify path keeps the
        // full bf16→f16 head — the lossless law is carried by rejection, so
        // the emitted stream stays identical to N0; only the draft q moves.
        let draftHeadEnabled = try Self.resolveMTPFlag(name: "QW_MTP_DRAFT_HEAD",
            value: environment["QW_MTP_DRAFT_HEAD"], drafts: mtpDrafts, defaultValue: headIsQ4g, maxDrafts: maxDrafts)
        // EXP-12: batched q4g VERIFY head (opt-in, LOSSY — the operator
        // accepts verify-verdict drift vs the bf16 golden; probe-proven
        // bit-exact vs the q4g draft-head kernel row-for-row, 2.44× the f16
        let verifyHeadQ4Enabled = try Self.resolveMTPFlag(name: "QW_MTP_VERIFY_HEAD_Q4",
            value: environment["QW_MTP_VERIFY_HEAD_Q4"], drafts: mtpDrafts, maxDrafts: maxDrafts)
        try Self.validateMTPVerifyHeadQ4(enabled: verifyHeadQ4Enabled, draftHead: draftHeadEnabled)
        let resolvedDFlashPackPath: String = {
            if let path = dflashDraftPackPath?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty {
                return path
            }
            if let envPath = environment["QW_DFLASH_DRAFT_PACK"]?.trimmingCharacters(in: .whitespacesAndNewlines), !envPath.isEmpty {
                return envPath
            }
            return QWDFlashDrafter.defaultPackPath
        }()
        let resolvedDFlashTopK: Int? = {
            if let dflashSelectorTopK { return dflashSelectorTopK }
            if let envTopK = environment["QW_DFLASH_TOP_K"], let val = Int(envTopK) { return val }
            return nil
        }()

        let resolvedRelaxedMargin: Double = {
            if let envMargin = environment["QW_MTP_RELAXED_MARGIN"], let val = Double(envMargin) {
                return max(0.0, val)
            }
            return max(0.0, mtpRelaxedMargin)
        }()
        let resolvedDraftStrategy: String = {
            if let envStrat = environment["QW_MTP_DRAFT_STRATEGY"]?.trimmingCharacters(in: .whitespacesAndNewlines), !envStrat.isEmpty {
                return envStrat
            }
            return mtpDraftStrategy
        }()
        let resolvedQuadraticSeeding: Bool = {
            if let envQuad = environment["QW_MTP_QUADRATIC_SEEDING"] {
                return ["1", "true", "yes"].contains(envQuad.lowercased())
            }
            return mtpQuadraticSeeding
        }()

        return StartupConfig(
            mtpBackend: mtpBackendResolved,
            prefillScratchEviction: evictionResolved,
            mtpAdaptiveDepth: resolvedAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: resolvedEntropy,
            mtpRelaxedMargin: resolvedRelaxedMargin,
            mtpDraftStrategy: resolvedDraftStrategy,
            mtpQuadraticSeeding: resolvedQuadraticSeeding,
            mtpStageProfiling: mtpStageProfiling,
            mtpVerificationProfiling: mtpVerificationProfiling,
            draftHeadEnabled: draftHeadEnabled,
            verifyHeadQ4Enabled: verifyHeadQ4Enabled,
            headIsQ4g: headIsQ4g,
            dflashDraftPackPath: resolvedDFlashPackPath,
            dflashSelectorTopK: resolvedDFlashTopK,
            environment: environment)
    }
}

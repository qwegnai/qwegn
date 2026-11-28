// QWInferenceModule.swift — The seam every model family plugs into.
// Pure: the protocol names what a module owns. It never names a family.
//
// Every capability a module declares is backed by a member here. A
// declaration without a seam is a lie the runtime cannot keep, so the
// protocol carries the KV-mode, prefix-cache and cache-event members that
// `.kvQuant*` and `.prefixCache` need.

import Foundation
import QWCore
import QWHead
import QWHyperPLE
import QWKV
@_exported import QWMTP

/// The family-native engine a module wraps. The registry hands it over
/// untouched: only the module knows its own engine's concrete type.
public protocol QWLoadedEngine: AnyObject, Sendable {
    /// The frozen plan of the model this engine serves.
    var plan: QWEnginePlan { get }
}

public struct QWResolvedSpeculation: Sendable {
    public let parameters: QWSpeculativeParameters
    public let engineDrafts: Int
    public let engineBackend: QWMTPBackend?
    public let reportedDrafts: Int
    public let reportedBackend: String

    public init(parameters: QWSpeculativeParameters, engineDrafts: Int, engineBackend: QWMTPBackend?,
                reportedDrafts: Int, reportedBackend: String) {
        self.parameters = parameters
        self.engineDrafts = engineDrafts
        self.engineBackend = engineBackend
        self.reportedDrafts = reportedDrafts
        self.reportedBackend = reportedBackend
    }
}

/// One model family's inference module.
public protocol QWInferenceModule: AnyObject, Sendable {
    /// The family this module serves.
    static var family: QWModelDescriptor.Family { get }
    /// What this module can do. Absence means "not available", never a zero.
    static var capabilities: Set<QWKnobCapability> { get }
    /// The KV storage modes this module implements.
    static var supportedKVStorageModes: Set<String> { get }
    /// The kernel doors this module runs, for pipeline prewarm.
    static var requiredDoors: [String] { get }
    /// Builds the module that serves a loaded model. The registry calls this,
    /// so no loader ever has to switch on the family.
    static func make(plan: QWEnginePlan, engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule
    /// Builds the engine this module serves. The module knows how its own
    /// engine is made and configured, so no consumer names a concrete type.
    static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine
    static func resolveSpeculation(_ parameters: QWSpeculativeParameters, descriptor: QWModelDescriptor,
                                   environment: [String: String], modelSwitch: Bool) throws -> QWResolvedSpeculation
    static func estimateContextMemory(
        descriptor: QWModelDescriptor, packPath: String?, contextTokens: Int, kvMode: QWKVMode,
        mtpDrafts: Int, mtpBackend: String, prefillChunkSize: Int, tuning: QWEngineTuning,
        physicalMemoryBytes: Int?
    ) -> QWContextMemoryEstimate
    /// The frozen numeric plan of the loaded model.
    var plan: QWEnginePlan { get }
    /// The KV storage mode this module runs in.
    var kvStorageMode: String { get }
    /// Whether a prefix cache is configured on this module right now.
    var hasPrefixCache: Bool { get }
    /// Where the module reports what its prefix cache did.
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? { get set }
    /// Configures the module's prefix-cache storage.
    func configurePrefixCache(_ config: QWCacheConfig) throws
}

extension QWInferenceModule {
    public static func resolveSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        try defaultResolveSpeculation(requested, descriptor: descriptor,
                                      environment: environment, modelSwitch: modelSwitch)
    }

    public static func defaultResolveSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        let isHyperStream = (descriptor.hcCount ?? 0) > 1
        if isHyperStream {
            return try resolveHyperStreamSpeculation(
                requested, descriptor: descriptor, environment: environment, modelSwitch: modelSwitch)
        }
        return try resolveStandardSpeculation(
            requested, descriptor: descriptor, environment: environment, modelSwitch: modelSwitch)
    }

    private static func resolveDFlashPackPath(
        parameters: QWSpeculativeParameters, descriptor: QWModelDescriptor, environment: [String: String]
    ) -> String {
        QWDFlashFootprint.resolveDrafterPackPath(
            targetDescriptor: descriptor,
            explicitPath: parameters.dflashDraftPackPath,
            environment: environment
        )
    }

    private static func resolveHyperStreamSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        var parameters = requested
        parameters.mtpBackend = parameters.mtpBackend?.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowerBackend = parameters.mtpBackend?.lowercased() ?? ""
        var dflashCap = QWFourSlotRollbackRing.maxVerifyRows - 1
        if QWDFlashFootprint.isDFlashName(lowerBackend) {
            let resolvedPack = resolveDFlashPackPath(
                parameters: parameters, descriptor: descriptor, environment: environment)
            let inferred = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: resolvedPack)
            dflashCap = min(QWFourSlotRollbackRing.maxVerifyRows - 1, max(1, inferred.maxDrafts))
            if modelSwitch {
                let compatible = (try? QWDFlashConfig.validateDrafter(
                    targetDescriptor: descriptor, packPath: resolvedPack,
                    drafts: 1, selectorTopK: parameters.dflashSelectorTopK
                )) != nil
                if compatible {
                    parameters.mtpBackend = inferred.backend
                    if parameters.mtpDrafts > 0 {
                        parameters.mtpDrafts = max(1, min(dflashCap, parameters.mtpDrafts))
                    }
                } else {
                    parameters.mtpBackend = "variational"
                    parameters.mtpDrafts = min(parameters.mtpDrafts, max(1, descriptor.maxSpeculativeDrafts))
                }
            }
        } else if modelSwitch {
            parameters.mtpDrafts = min(parameters.mtpDrafts, max(1, descriptor.maxSpeculativeDrafts))
        }
        if let backend = parameters.mtpBackend, !backend.isEmpty {
            _ = try QWMTPBackend.resolve(name: backend.lowercased())
        }
        let rawStrategy = (parameters.draftStrategyIsResolved != true)
            ? (environment["FN_DRAFT"].flatMap { $0.isEmpty ? nil : $0 } ?? parameters.mtpDraftStrategy)
            : parameters.mtpDraftStrategy
        let mode = try QWDraftStrategy.resolve(rawStrategy)
        parameters.mtpDraftStrategy = mode.rawValue
        parameters.draftStrategyIsResolved = true
        let finalBackend = parameters.mtpBackend?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        if QWDFlashFootprint.isDFlashName(finalBackend) {
            let drafts = parameters.enabled ? max(0, min(dflashCap, parameters.mtpDrafts)) : 0
            return QWResolvedSpeculation(
                parameters: parameters, engineDrafts: drafts,
                engineBackend: drafts > 0 ? .dflash2 : nil,
                reportedDrafts: drafts,
                reportedBackend: finalBackend == "dspine" ? "dspine" : "dflash2")
        }
        let supportsMTP = descriptor.capabilities.contains(.mtp)
        let engineDrafts = parameters.enabled && supportsMTP ? parameters.mtpDrafts : 0
        return QWResolvedSpeculation(
            parameters: parameters, engineDrafts: engineDrafts,
            engineBackend: nil,
            reportedDrafts: (parameters.enabled && supportsMTP) ? max(0, min(3, parameters.mtpDrafts)) : 0,
            reportedBackend: mode.rawValue)
    }

    private static func resolveStandardSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        var parameters = requested
        parameters.mtpBackend = parameters.mtpBackend?.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowerBackend = parameters.mtpBackend?.lowercased() ?? ""
        if lowerBackend == "ngram" {
            parameters.mtpDraftStrategy = "ngram"
            parameters.mtpBackend = "standard"
        }
        let isNgramStrategy = parameters.mtpDraftStrategy.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "ngram"
        let isDFlashFamily = QWDFlashFootprint.isDFlashName(parameters.mtpBackend)
        let isDSparkRequested = parameters.mtpBackend?.lowercased() == "dspark"
        if modelSwitch {
            if isDFlashFamily {
                let packPath = resolveDFlashPackPath(
                    parameters: parameters, descriptor: descriptor, environment: environment)
                let compatible = (try? QWDFlashConfig.validateDrafter(
                    targetDescriptor: descriptor, packPath: packPath, drafts: 1
                )) != nil
                if !compatible && (descriptor.hcCount != nil || descriptor.numExperts != nil) {
                    parameters.mtpBackend = "variational"
                }
            } else if isDSparkRequested && descriptor.dsparkConfig == nil && (descriptor.hcCount != nil || descriptor.numExperts != nil) {
                parameters.mtpBackend = "variational"
            }
            let stillDFlash = QWDFlashFootprint.isDFlashName(parameters.mtpBackend)
            let supportsWeightFreeNgramSwitch = !descriptor.capabilities.contains(.mtp)
                && Self.capabilities.contains(.speculative) && isNgramStrategy
            let maximum = stillDFlash ? 15 : (supportsWeightFreeNgramSwitch ? 15 : max(1, descriptor.maxSpeculativeDrafts))
            parameters.mtpDrafts = min(parameters.mtpDrafts, maximum)
        }
        let supportsWeightFreeNgram = Self.capabilities.contains(.speculative) && isNgramStrategy
        let isDSparkActive = parameters.mtpBackend?.lowercased() == "dspark"
        let supportsMTP = descriptor.capabilities.contains(.mtp)
            || descriptor.dsparkConfig != nil
            || QWDFlashFootprint.isDFlashName(parameters.mtpBackend)
            || (isDSparkActive && descriptor.dsparkConfig != nil)
            || supportsWeightFreeNgram
        let drafts = parameters.enabled && supportsMTP ? parameters.mtpDrafts : 0
        var backend: QWMTPBackend?
        if drafts > 0, let name = parameters.mtpBackend, !name.isEmpty {
            let selected = try QWMTPBackend.resolve(name: name)
            _ = try QWMTPBackend.resolveSelection(selected, drafts: drafts)
            backend = selected
        }
        return QWResolvedSpeculation(parameters: parameters, engineDrafts: drafts,
                                     engineBackend: backend, reportedDrafts: drafts,
                                     reportedBackend: parameters.mtpBackend ?? "standard")
    }

    public static func estimateContextMemory(
        descriptor: QWModelDescriptor, packPath: String?, contextTokens: Int, kvMode: QWKVMode,
        mtpDrafts: Int, mtpBackend: String, prefillChunkSize: Int, tuning: QWEngineTuning,
        physicalMemoryBytes: Int?
    ) -> QWContextMemoryEstimate {
        QWEngineDriver.estimateDriverContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes)
    }
}

/// A module refused a request it cannot serve.
public enum QWModuleError: Error, Equatable, CustomStringConvertible {
    case unsupportedFamily(QWModelDescriptor.Family)
    case unsupportedKVStorageMode(model: String, mode: String, supported: [String])
    case unsupportedCapability(model: String, capability: QWKnobCapability)
    case invalidToken(Int)
    case invalidState(String)
    case notImplemented(String)

    public var description: String {
        switch self {
        case .unsupportedFamily(let family):
            return "QWModule: no module serves family '\(family.rawValue)'"
        case .unsupportedKVStorageMode(let model, let mode, let supported):
            return "QWModule: '\(model)' does not implement KV mode '\(mode)' (has \(supported.joined(separator: "|")))"
        case .unsupportedCapability(let model, let capability):
            return "QWModule: '\(model)' does not provide capability '\(capability.rawValue)'"
        case .invalidToken(let token):
            return "QWModule: token id \(token) outside the model vocabulary"
        case .invalidState(let message):
            return "QWModule: \(message)"
        case .notImplemented(let what):
            return "QWModule: \(what) is not implemented yet"
        }
    }

    /// Maps the shared engine's error type onto the module's own.
    static func from(_ error: QWEngineError) -> QWModuleError {
        switch error {
        case .badToken(let token):
            return .invalidToken(token)
        case .state(let message):
            return .invalidState(message)
        default:
            return .invalidState(String(describing: error))
        }
    }
}

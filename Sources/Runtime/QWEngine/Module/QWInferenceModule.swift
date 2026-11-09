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

/// A module-owned session. The core allocates and recycles it; everything
/// behind it belongs to the module.
public protocol QWModuleSessionState: AnyObject, Sendable {
    /// Tokens the session has consumed so far.
    var activeTokens: Int { get }
    /// Returns the state to a clean slate before it is handed out again.
    /// The request's contract travels with the reset: a session is handed
    /// out per request, and two concurrent requests may sample differently.
    func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws
}

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
    /// Creates a fresh session state owned by this module.
    func makeSessionState() throws -> any QWModuleSessionState
    /// Prepares the module for a request. The core calls this before the
    /// prefill starts, so the work a request needs up front — the prefill
    /// stack, for instance — does not land inside the measured prefill.
    func prepare(_ state: any QWModuleSessionState, prompt: [Int], maxNew: Int) throws
    /// The request is over. The module releases what it held for it: the
    /// engine session, the admission permit, the KV pages. A module that
    /// holds nothing does nothing here.
    func finish(_ state: any QWModuleSessionState)
    /// Runs one prefill chunk and returns the head's token for its last row.
    /// `runsHead` is true only for the final chunk — there the head runs, and
    /// its token is the first generated one. The prompt is fed exactly once.
    func encodePrefill(_ state: any QWModuleSessionState, tokens: [Int],
                       chunk: Range<Int>, runsHead: Bool) throws -> Int
    /// Runs one decode step and returns the token the model produced.
    func encodeDecodeStep(_ state: any QWModuleSessionState, token: Int, position: Int) throws -> Int
    /// The wall time, in milliseconds, the module attributes to the token the
    /// last step produced. A module that commits several tokens in one step
    /// splits the step's time across them. Zero means it does not measure.
    func lastStepMilliseconds(_ state: any QWModuleSessionState) -> Double
    /// The GPU time the prefill of this session spent, in seconds. The
    /// module owns the measurement; a module that cannot measure reports
    /// zero, and the caller treats it as "not measured" rather than "fast".
    func prefillGPUSeconds(_ state: any QWModuleSessionState) -> Double
    /// Tokens committed per speculative verification round in this session,
    /// when the module runs speculative rounds internally.
    func roundTokenCounts(_ state: any QWModuleSessionState) -> [Int]
    /// The logit row the shared sampler consumes. A module that cannot
    /// sample has no row to give.
    func sampleLogits(_ state: any QWModuleSessionState) throws -> [Float]
}

extension QWInferenceModule {
    public var compiledStack: QWCompiledLayerStack {
        QWCompiledLayerStack(plan: plan)
    }

    public func roundTokenCounts(_ state: any QWModuleSessionState) -> [Int] { [] }

    public static func resolveSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        var parameters = requested
        let isDFlashFamily = ["dflash2", "dflash", "dspine"].contains(parameters.mtpBackend?.lowercased() ?? "")
        if modelSwitch {
            if (descriptor.hcCount != nil || descriptor.numExperts != nil), isDFlashFamily {
                parameters.mtpBackend = "variational"
            }
            let stillDFlash = ["dflash2", "dflash", "dspine"].contains(parameters.mtpBackend?.lowercased() ?? "")
            let maximum = stillDFlash ? 15 : max(1, descriptor.maxSpeculativeDrafts)
            parameters.mtpDrafts = min(parameters.mtpDrafts, maximum)
        }
        let supportsMTP = descriptor.capabilities.contains(.mtp)
            || ["dflash2", "dflash", "dspine"].contains(parameters.mtpBackend?.lowercased() ?? "")
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
}

/// A module refused a request it cannot serve.
public enum QWModuleError: Error, Equatable, CustomStringConvertible {
    case unsupportedFamily(QWModelDescriptor.Family)
    case unsupportedKVStorageMode(model: String, mode: String, supported: [String])
    case unsupportedCapability(model: String, capability: QWKnobCapability)
    case sessionExhausted
    case invalidToken(Int)
    case invalidState(String)
    case planMismatch(model: String)
    case notImplemented(String)

    public var description: String {
        switch self {
        case .unsupportedFamily(let family):
            return "QWModule: no module serves family '\(family.rawValue)'"
        case .unsupportedKVStorageMode(let model, let mode, let supported):
            return "QWModule: '\(model)' does not implement KV mode '\(mode)' (has \(supported.joined(separator: "|")))"
        case .unsupportedCapability(let model, let capability):
            return "QWModule: '\(model)' does not provide capability '\(capability.rawValue)'"
        case .sessionExhausted:
            return "QWModule: session pool exhausted"
        case .invalidToken(let token):
            return "QWModule: token id \(token) outside the model vocabulary"
        case .invalidState(let message):
            return "QWModule: \(message)"
        case .planMismatch(let model):
            return "QWModule: '\(model)' module and plan disagree — the plan digest does not match"
        case .notImplemented(let what):
            return "QWModule: \(what) is not implemented yet"
        }
    }
}

// QWKnobRegistry.swift — The catalogue of user-facing controls.
// Pure: a static table, no GPU, no I/O. Every id here is bound by exactly one
// surface binding, one profile field and one validator.

import Foundation

/// The single declaration of every user-facing knob.
public enum QWKnobRegistry {
    /// Every knob, in declaration order.
    public static let all: [QWKnobDescriptor] = normalKnobs + tuningKnobs + expertKnobs

    /// The knob with this id, if it exists.
    public static func descriptor(_ id: String) -> QWKnobDescriptor? {
        all.first { $0.id == id }
    }

    /// Every knob reachable from a surface.
    public static func descriptors(for surface: QWKnobSurface) -> [QWKnobDescriptor] {
        all.filter { $0.surfaces.contains(surface) }
    }

    /// Every knob that needs the given capability.
    public static func descriptors(requireing capability: QWKnobCapability) -> [QWKnobDescriptor] {
        all.filter { $0.requiredCapability == capability }
    }

    /// Every knob the model behind `capabilities` can actually honour.
    public static func descriptors(availableUnder capabilities: Set<QWKnobCapability>) -> [QWKnobDescriptor] {
        all.filter { $0.isAvailable(under: capabilities) }
    }

    // MARK: - Normal tier

    private static let normalSurfaces: Set<QWKnobSurface> = [.cli, .api, .menuBar, .dashboard, .profile]

    private static let normalKnobs: [QWKnobDescriptor] = [
        QWKnobDescriptor(id: "contextTokens", tier: .normal, scope: .load,
                         kind: .int(1...1_048_576), defaultValue: .int(4096), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "maxTokens", tier: .normal, scope: .request,
                         kind: .int(1...1_048_576), defaultValue: .int(512), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "temperature", tier: .normal, scope: .request,
                         kind: .double(0...2), defaultValue: .double(0.7), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "topP", tier: .normal, scope: .request,
                         kind: .double(0...1), defaultValue: .double(0.9), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "topK", tier: .normal, scope: .request,
                         kind: .int(0...2048), defaultValue: .int(40), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "repetitionPenalty", tier: .normal, scope: .request,
                         kind: .double(1...2), defaultValue: .double(1.05), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "seed", tier: .normal, scope: .request,
                         kind: .int(-1...Int(Int32.max)), defaultValue: .int(-1), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "enableThinking", tier: .normal, scope: .request,
                         kind: .bool, defaultValue: .bool(true), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "reasoningEffort", tier: .normal, scope: .request,
                         kind: .string(allowed: ["low", "medium", "xhigh"]), defaultValue: .string("xhigh"),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "systemPrompt", tier: .normal, scope: .request,
                         kind: .string(allowed: []), defaultValue: .string(""), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "speculativeEnabled", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true),
                         surfaces: [.api, .menuBar, .dashboard, .profile]),
        QWKnobDescriptor(id: "mtpDrafts", tier: .normal, scope: .load,
                         kind: .int(0...15), defaultValue: .int(1), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "mtpBackend", tier: .normal, scope: .load,
                         kind: .string(allowed: ["standard", "variational", "dflash2", "dspine"]), defaultValue: .string("variational"),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "mtpRelaxedMargin", tier: .normal, scope: .load,
                         kind: .double(0...10), defaultValue: .double(1.5),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "mtpRelaxedMinProb", tier: .normal, scope: .load,
                         kind: .double(0...1), defaultValue: .double(0.2),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "mtpDraftStrategy", tier: .normal, scope: .load,
                         kind: .string(allowed: ["hybrid", "neural", "ngram"]), defaultValue: .string("hybrid"),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "mtpQuadraticSeeding", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "prefillChunkRows", tier: .normal, scope: .load,
                         kind: .int(32...8192), defaultValue: .int(512), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "maxConcurrentSessions", tier: .normal, scope: .load,
                         kind: .int(1...16), defaultValue: .int(8), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "prefixCacheEnabled", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(false), requiredCapability: .prefixCache,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "warmCacheLimitGiB", tier: .normal, scope: .load,
                         kind: .double(0...512), defaultValue: .double(4.0), requiredCapability: .prefixCache,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "ssdCacheLimitGiB", tier: .normal, scope: .load,
                         kind: .double(0...4096), defaultValue: .double(10.0), requiredCapability: .prefixCache,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "cacheDir", tier: .normal, scope: .load,
                         kind: .string(allowed: []), defaultValue: .string("~/.qwg/kv-cache"),
                         requiredCapability: .prefixCache, surfaces: normalSurfaces),
        QWKnobDescriptor(id: "cacheEntryLimit", tier: .normal, scope: .load,
                         kind: .int(1...4096), defaultValue: .int(32), requiredCapability: .prefixCache,
                         surfaces: [.api, .menuBar, .dashboard, .profile]),
        QWKnobDescriptor(id: "evictPrefillScratch", tier: .normal, scope: .load,
                         kind: .string(allowed: ["auto", "always", "never"]), defaultValue: .string("auto"),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "lsaMode", tier: .normal, scope: .load,
                         kind: .string(allowed: ["off", "auto", "sidecar", "svd", "qsa-cluster"]),
                         defaultValue: .string("auto"), requiredCapability: .lsa,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "lsaBudget", tier: .normal, scope: .load,
                         kind: .int(512...8192), defaultValue: .int(2048), requiredCapability: .lsa,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "lsaClusterBlock", tier: .normal, scope: .load,
                         kind: .int(256...4096), defaultValue: .int(512), requiredCapability: .lsa,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "lsaSidecarPath", tier: .normal, scope: .load,
                         kind: .string(allowed: []), defaultValue: .string(""), requiredCapability: .lsa,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "lazyLoad", tier: .normal, scope: .process,
                         kind: .bool, defaultValue: .bool(true), surfaces: [.cli, .menuBar]),
        QWKnobDescriptor(id: "loadVision", tier: .normal, scope: .session,
                         kind: .bool, defaultValue: .bool(false), requiredCapability: .vision,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "logLevel", tier: .normal, scope: .process,
                         kind: .string(allowed: ["debug", "info", "warn", "error", "none"]),
                         defaultValue: .string("info"), surfaces: [.cli, .api, .menuBar]),
    ]

    // MARK: - Engine tuning (QWEngineTuning)

    private static let tuningSurfaces: Set<QWKnobSurface> = [.api, .menuBar, .dashboard, .profile]

    private static let tuningKnobs: [QWKnobDescriptor] = [
        QWKnobDescriptor(id: "memoryPolicy", tier: .normal, scope: .load,
                         kind: .string(allowed: ["safe", "maxResident"]), defaultValue: .string("maxResident"),
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "osReserveGiB", tier: .normal, scope: .load,
                         kind: .double(0...256), defaultValue: .double(8.0), surfaces: normalSurfaces),
        QWKnobDescriptor(id: "residentExpertsPerLayer", tier: .normal, scope: .load,
                         kind: .int(0...4096), defaultValue: .int(0), requiredCapability: .expertStreaming,
                         surfaces: normalSurfaces),
        QWKnobDescriptor(id: "expertCache", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true), requiredCapability: .expertStreaming,
                         surfaces: tuningSurfaces),
        QWKnobDescriptor(id: "coldEvents", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true), requiredCapability: .expertStreaming,
                         surfaces: tuningSurfaces),
        QWKnobDescriptor(id: "coldEventOverlap", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true), requiredCapability: .expertStreaming,
                         surfaces: tuningSurfaces),
        QWKnobDescriptor(id: "prefillSpanMap", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(true), requiredCapability: .expertStreaming,
                         surfaces: tuningSurfaces),
        QWKnobDescriptor(id: "streamNgrams", tier: .normal, scope: .load,
                         kind: .bool, defaultValue: .bool(false),
                         surfaces: normalSurfaces),
    ]

    // MARK: - Expert tier

    private static let expertSurfaces: Set<QWKnobSurface> = [.cli, .api, .dashboard, .profile]

    private static let expertKnobs: [QWKnobDescriptor] = [
        QWKnobDescriptor(id: "kvMode", tier: .expert, scope: .load,
                         kind: .string(allowed: ["f16", "q8", "q4"]), defaultValue: .string("q4"),
                         requiredCapability: .kvQuantQ8, valueCapabilities: ["q4": .kvQuantQ4],
                         surfaces: expertSurfaces),
        QWKnobDescriptor(id: "moeSpecBudget", tier: .expert, scope: .load,
                         kind: .int(0...8), defaultValue: .int(0), requiredCapability: .expertStreaming,
                         surfaces: expertSurfaces),
        QWKnobDescriptor(id: "gdnPrefill", tier: .expert, scope: .load,
                         kind: .string(allowed: ["chunked", "naive", "regres"]), defaultValue: .string("chunked"),
                         requiredCapability: .gdnPrefill, surfaces: expertSurfaces),
        QWKnobDescriptor(id: "mtpAdaptiveDepth", tier: .expert, scope: .load,
                         kind: .bool, defaultValue: .bool(false), requiredCapability: .mtpAdaptive,
                         surfaces: [.cli, .api, .menuBar, .dashboard, .profile]),
        QWKnobDescriptor(id: "mtpEntropyThreshold", tier: .expert, scope: .load,
                         kind: .double(0...1), defaultValue: .double(0.7), requiredCapability: .mtpAdaptive,
                         surfaces: [.cli, .api, .menuBar, .dashboard, .profile]),
        QWKnobDescriptor(id: "ropeScalingFactor", tier: .expert, scope: .load,
                         kind: .double(1...8), defaultValue: .double(1.0), requiredCapability: .ropeYaRN,
                         surfaces: [.api, .dashboard, .profile]),
        QWKnobDescriptor(id: "dflashDraftPackPath", tier: .expert, scope: .load,
                         kind: .string(allowed: []), defaultValue: .string(""), requiredCapability: .dflash,
                         surfaces: expertSurfaces),
        QWKnobDescriptor(id: "dflashSelectorTopK", tier: .expert, scope: .load,
                         kind: .int(1...16), defaultValue: .int(16), requiredCapability: .dflash,
                         surfaces: expertSurfaces),
        QWKnobDescriptor(id: "splitKTokensPerSplit", tier: .expert, scope: .load,
                         kind: .int(1...8192), defaultValue: .int(256),
                         surfaces: [.api, .dashboard, .profile]),
    ]
}

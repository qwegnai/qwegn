// QWDashboardViewModel.swift — One card list for every renderer.
// The cards are computed here, from the telemetry the engine measured and
// the knobs the registry declares. Neither UI computes a card: two
// renderers that disagree are a bug in one of them, and the fix belongs
// here.

import Foundation
import QWCore
import QWEngine

/// A card: one value, where it came from, and how old it is.
public struct QWDashboardCard: Sendable, Codable, Identifiable, Equatable {
    /// The metric's name. It is the id a renderer binds to.
    public let id: String
    /// What the value is called.
    public let title: String
    /// The value as it is shown. Nil means unavailable: the renderer
    /// draws a dash, and the reason says why there is no number.
    public let value: String?
    /// The unit the value is in, when it has one.
    public let unit: String?
    /// The reading behind the value, in the card's unit. A renderer draws
    /// a bar from this; the value is what the reader sees.
    public let number: Double?
    /// How the reading was obtained.
    public let provenance: QWTelemetryProvenance
    /// When the reading was taken.
    public let sampledAt: Date?
    /// How old the reading is, in seconds.
    public let age: TimeInterval?
    /// Whether the reading is too old to describe the present.
    public let isStale: Bool
    /// Why there is no value, when there is none.
    public let reason: String?
    /// The readings behind the card. A nil is a gap, never a zero.
    public let series: [Double?]

    public init(id: String, title: String, value: String?, unit: String?, number: Double? = nil,
                provenance: QWTelemetryProvenance, sampledAt: Date?, age: TimeInterval?,
                isStale: Bool, reason: String?, series: [Double?]) {
        self.id = id
        self.title = title
        self.value = value
        self.unit = unit
        self.number = number
        self.provenance = provenance
        self.sampledAt = sampledAt
        self.age = age
        self.isStale = isStale
        self.reason = reason
        self.series = series
    }
}

/// The facts that describe the machine and the resident model.
/// They are configuration, not measurements: they say what the engine
/// is, not how it is doing. A fact nobody reported is absent.
public struct QWDashboardFacts: Sendable, Codable, Equatable {
    /// The head storage of the resident model. Absent when no model is
    /// resident: a quantization that describes nothing is a claim about
    /// nothing.
    public let quantization: String?
    /// The host's logical cores. The engine's own thread count is not
    /// published, so this is not called "threads".
    public let hostLogicalCores: Int?
    /// The GDN prefill mode the engine runs with.
    public let gdnPrefillMode: String?
    /// The GPU's name, as the device reports it.
    public let gpuName: String?
    /// The Metal family the device supports.
    public let metalFamily: String?

    public init(quantization: String? = nil, hostLogicalCores: Int? = nil,
                gdnPrefillMode: String? = nil, gpuName: String? = nil,
                metalFamily: String? = nil) {
        self.quantization = quantization
        self.hostLogicalCores = hostLogicalCores
        self.gdnPrefillMode = gdnPrefillMode
        self.gpuName = gpuName
        self.metalFamily = metalFamily
    }
}

/// An action the loaded model can actually perform. A renderer that
/// cannot perform an action must not draw it.
public struct QWDashboardAction: Sendable, Codable, Identifiable, Equatable {
    /// The action's name. It is the id a renderer binds to.
    public let id: String
    /// What the action is called.
    public let title: String
    /// The knob the action sets, when it sets one.
    public let knobId: String?
    /// Whether the action can be taken right now.
    public let isEnabled: Bool
    /// Why it cannot, when it cannot.
    public let reason: String?

    public init(id: String, title: String, knobId: String?, isEnabled: Bool, reason: String?) {
        self.id = id
        self.title = title
        self.knobId = knobId
        self.isEnabled = isEnabled
        self.reason = reason
    }
}

/// The dashboard, as data. Both renderers draw this and nothing else.
public struct QWDashboardViewModel: Sendable, Codable, Equatable {
    /// The shape of this payload. A renderer that does not understand the
    /// version says so rather than drawing the wrong thing.
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    /// When the cards were computed.
    public let sampledAt: Date
    /// What the engine is doing.
    public let lifecycle: QWEngineLifecycleInfo
    /// The model the cards describe. Nil when no model is resident: the
    /// dashboard must not name a model that is not there.
    public let modelId: String?
    /// What the loaded model can do, by name. A renderer uses it to hide
    /// what the model cannot honour.
    public let capabilities: [String]
    /// What the engine is doing right now. A renderer that shows the
    /// engine at work needs this; it is the engine's own phase, not a
    /// guess from a counter. Absent when the server did not say.
    public let generation: QWEngineTelemetryRegistry.EngineInferencePhase?
    /// What the machine and the resident model are. Absent when the
    /// server could not say.
    public let facts: QWDashboardFacts?
    /// The card list, in the order they are drawn.
    public let cards: [QWDashboardCard]
    /// The actions this model can perform.
    public let actions: [QWDashboardAction]
    /// The knob values, by knob id.
    public let knobs: [String: QWKnobValue]
    /// Where each knob's value came from, by knob id.
    public let knobSources: [String: String]
    /// The requests the server recorded, newest last. A renderer shows what
    /// the server measured; it keeps no list of its own. Nil when the server
    /// published no requests: an older dashboard still decodes.
    public let requests: [QWRequestRecord]?

    public init(schemaVersion: Int = QWDashboardViewModel.currentSchemaVersion,
                sampledAt: Date, lifecycle: QWEngineLifecycleInfo, modelId: String?,
                capabilities: [String] = [],
                generation: QWEngineTelemetryRegistry.EngineInferencePhase? = .idle,
                facts: QWDashboardFacts? = nil,
                cards: [QWDashboardCard], actions: [QWDashboardAction],
                knobs: [String: QWKnobValue], knobSources: [String: String],
                requests: [QWRequestRecord]? = nil) {
        self.schemaVersion = schemaVersion
        self.sampledAt = sampledAt
        self.lifecycle = lifecycle
        self.modelId = modelId
        self.capabilities = capabilities
        self.generation = generation
        self.facts = facts
        self.cards = cards
        self.actions = actions
        self.knobs = knobs
        self.knobSources = knobSources
        self.requests = requests
    }
}

/// The card ids the MenuBar renders: the set it expects, and the set
/// the parity test asserts the published model is a superset of. A card
/// the model does not publish is never drawn.
public enum MenuBarCardCatalogue {
    public static let ids: [String] = [
        "gpu.busy_ratio",
        "gpu.device",
        "hardware.cpu_percent",
        "hardware.memory_percent",
        "hardware.memory_pressure_level",
        "throughput.display_prefill_tokens_per_sec",
        "throughput.display_decode_tokens_per_sec",
        "latency.ttft_ms",
        "context.active_tokens",
        "cache.warm_bytes",
        "cache.ssd_bytes",
        "cache.tokens_reused_total"
    ]
}

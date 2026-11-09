// QWEngineTuning.swift — Per-engine performance and memory tunables.
// Pure: no GPU, no I/O. One value set per loaded engine (a server may switch
// models in place), resolved as environment > per-model profile > default.

import Foundation

/// How an engine sizes its resident weights against the machine's memory.
public enum QWMemoryPolicy: String, Sendable, CaseIterable, Codable {
    /// Fit what is free or reclaimable right now, less a headroom: never swap.
    case safe
    /// Fit physical RAM less the OS reserve: the most resident weights, at the risk of swapping.
    /// The default: measured fastest (GLM-5.3-Flash 16K: decode +12%, prefill +7% over `safe`).
    case maxResident
}

/// The engine tunables a consumer may set per model.
public struct QWEngineTuning: Sendable, Equatable {
    public var memoryPolicy: QWMemoryPolicy
    /// RAM left to the OS; nil keeps `QWCapacityBudget.osReserveBytes`.
    public var osReserveGiB: Double?
    /// Resident experts per MoE layer; nil sizes them from `memoryPolicy`.
    public var residentExpertsPerLayer: Int?
    /// Cross-token cache of streamed experts.
    public var expertCache: Bool
    /// Event-pipelined cold-expert staging in decode.
    public var coldEvents: Bool
    /// Resident experts run on the GPU while cold experts are staged.
    public var coldEventOverlap: Bool
    /// One GPU-mapped expert span per prefill chunk.
    public var prefillSpanMap: Bool

    /// Headroom the `safe` policy keeps free beyond the engine's own plan.
    public static let safeHeadroomBytes = 6 << 30

    public static let `default` = QWEngineTuning(values: [:])

    /// Ids of the knobs this struct carries, in display order (declared in `QWKnobRegistry`).
    public static let knobIds = [
        "memoryPolicy", "osReserveGiB", "residentExpertsPerLayer",
        "expertCache", "coldEvents", "coldEventOverlap", "prefillSpanMap",
    ]

    /// The environment variable that overrides each knob.
    public static let environmentNames: [String: String] = [
        "memoryPolicy": "QW_MEMORY_POLICY",
        "osReserveGiB": "QW_OS_RESERVE_GIB",
        "residentExpertsPerLayer": "QW_RESIDENT_EXPERTS_PER_LAYER",
        "expertCache": "QW_EXPERT_CACHE",
        "coldEvents": "QW_COLD_EVENTS",
        "coldEventOverlap": "QW_COLD_EVENT_OVERLAP",
        "prefillSpanMap": "QW_PREFILL_SPAN_MAP",
    ]

    /// Resolves the tunables: an environment variable wins, then `values`, then the default.
    /// Invalid values at either layer fall through to the next.
    public init(values: [String: QWKnobValue], environment: [String: String] = [:]) {
        func value(_ id: String) -> QWKnobValue? {
            guard let d = QWKnobRegistry.descriptor(id) else { return nil }
            if let name = Self.environmentNames[id], let raw = environment[name],
               let v = Self.parse(raw, kind: d.kind), (try? d.validate(v)) != nil {
                return v
            }
            if let v = values[id], (try? d.validate(v)) != nil { return v }
            return nil
        }
        memoryPolicy = value("memoryPolicy")?.stringValue.flatMap(QWMemoryPolicy.init(rawValue:)) ?? .maxResident
        osReserveGiB = value("osReserveGiB")?.doubleValue
        residentExpertsPerLayer = value("residentExpertsPerLayer")?.intValue.flatMap { $0 > 0 ? $0 : nil }
        expertCache = value("expertCache")?.boolValue ?? true
        coldEvents = value("coldEvents")?.boolValue ?? true
        coldEventOverlap = value("coldEventOverlap")?.boolValue ?? true
        prefillSpanMap = value("prefillSpanMap")?.boolValue ?? true
    }

    /// The same tunables with environment overrides applied on top.
    public func overlaid(environment: [String: String]) -> QWEngineTuning {
        QWEngineTuning(values: knobValues, environment: environment)
    }

    /// The tunables as knob values (an unset optional is omitted).
    public var knobValues: [String: QWKnobValue] {
        var out: [String: QWKnobValue] = [
            "memoryPolicy": .string(memoryPolicy.rawValue),
            "expertCache": .bool(expertCache),
            "coldEvents": .bool(coldEvents),
            "coldEventOverlap": .bool(coldEventOverlap),
            "prefillSpanMap": .bool(prefillSpanMap),
        ]
        if let osReserveGiB { out["osReserveGiB"] = .double(osReserveGiB) }
        if let residentExpertsPerLayer { out["residentExpertsPerLayer"] = .int(residentExpertsPerLayer) }
        return out
    }

    /// Environment spelling: booleans are on unless "0" (the engine's historical `!= "0"`).
    static func parse(_ raw: String, kind: QWKnobKind) -> QWKnobValue? {
        let s = raw.trimmingCharacters(in: .whitespaces)
        switch kind {
        case .bool: return .bool(s != "0" && s.lowercased() != "false" && s.lowercased() != "off")
        case .int: return Int(s).map(QWKnobValue.int)
        case .double: return Double(s).map(QWKnobValue.double)
        case .string: return s.isEmpty ? nil : .string(s)
        }
    }
}

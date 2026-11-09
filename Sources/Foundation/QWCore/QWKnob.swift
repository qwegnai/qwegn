// QWKnob.swift — The vocabulary for user-facing controls.
// Pure: no GPU, no I/O. One knob id names one control, whatever surface
// exposes it, and the registry is the single place a knob is declared.

import Foundation

/// Who may reach for a knob: `normal` knobs are intended and documented,
/// `expert` knobs trade fidelity or pick an implementation variant.
public enum QWKnobTier: String, Sendable, Codable, CaseIterable {
    case normal
    case expert
}

/// When a knob takes effect.
public enum QWKnobScope: String, Sendable, Codable, CaseIterable {
    case request
    case session
    case load
    case process
}

/// Where a knob is reachable from.
public enum QWKnobSurface: String, Sendable, Codable, CaseIterable {
    case cli
    case api
    case menuBar
    case dashboard
    case profile
}

/// A capability a model may or may not have. Absence means "not available",
/// never a numeric zero.
public enum QWKnobCapability: String, Sendable, Codable, CaseIterable {
    case prefixCache
    case kvQuantQ8
    case kvQuantQ4
    case gdnPrefill
    case dflash
    case ropeYaRN
    case speculative
    /// Speculation whose depth adapts to the draft's own confidence.
    /// Finer than `speculative`: Flash-Next speculates but cannot adapt.
    case mtpAdaptive
    case vision
    /// Latent Sparse Attention (LSA, arXiv:2610.42147) subquadratic KV retrieval.
    case lsa
    /// MoE experts that stream from SSD past a resident set (residency and cold-expert pipeline knobs).
    case expertStreaming
}

/// A knob's value, carried across the wire without per-knob types.
public enum QWKnobValue: Sendable, Codable, Equatable {
    case int(Int)
    case double(Double)
    case bool(Bool)
    case string(String)

    /// The value when it is an integer.
    public var intValue: Int? { if case .int(let v) = self { return v } else { return nil } }
    /// The value when it is a double.
    public var doubleValue: Double? { if case .double(let v) = self { return v } else { return nil } }
    /// The value when it is a boolean.
    public var boolValue: Bool? { if case .bool(let v) = self { return v } else { return nil } }
    /// The value when it is a string.
    public var stringValue: String? { if case .string(let v) = self { return v } else { return nil } }
}

extension QWKnobValue {
    /// The wire shape is a tagged scalar: `{"double": 0.7}`. A profile file
    /// is meant to be read by a person as well as by the app, and a bare
    /// number would not say whether it was an integer or a double.
    private enum CodingKeys: String, CodingKey {
        case int, double, bool, string
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try c.decodeIfPresent(Int.self, forKey: .int) { self = .int(value); return }
        if let value = try c.decodeIfPresent(Double.self, forKey: .double) { self = .double(value); return }
        if let value = try c.decodeIfPresent(Bool.self, forKey: .bool) { self = .bool(value); return }
        if let value = try c.decodeIfPresent(String.self, forKey: .string) { self = .string(value); return }
        throw DecodingError.dataCorrupted(DecodingError.Context(
            codingPath: decoder.codingPath,
            debugDescription: "QWKnobValue: no value kind in the payload"))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .int(let value): try c.encode(value, forKey: .int)
        case .double(let value): try c.encode(value, forKey: .double)
        case .bool(let value): try c.encode(value, forKey: .bool)
        case .string(let value): try c.encode(value, forKey: .string)
        }
    }
}

/// The value shape a knob accepts. A `.string` with an empty allowed set
/// accepts any string.
public enum QWKnobKind: Sendable, Equatable {
    case int(ClosedRange<Int>)
    case double(ClosedRange<Double>)
    case bool
    case string(allowed: Set<String>)
}

/// The declaration of one user-facing control.
public struct QWKnobDescriptor: Sendable, Equatable {
    /// Stable identifier; the id is what every surface binds to.
    public let id: String
    /// `normal` or `expert`.
    public let tier: QWKnobTier
    /// When the knob takes effect.
    public let scope: QWKnobScope
    /// The value shape the knob accepts.
    public let kind: QWKnobKind
    /// The built-in default, used when neither an explicit value nor a profile supplies one.
    public let defaultValue: QWKnobValue
    /// Capability the knob needs to be offered at all.
    public let requiredCapability: QWKnobCapability?
    /// Extra capability per string value, for knobs whose default works everywhere.
    public let valueCapabilities: [String: QWKnobCapability]
    /// Surfaces the knob is reachable from.
    public let surfaces: Set<QWKnobSurface>

    public init(id: String, tier: QWKnobTier, scope: QWKnobScope, kind: QWKnobKind,
                defaultValue: QWKnobValue, requiredCapability: QWKnobCapability? = nil,
                valueCapabilities: [String: QWKnobCapability] = [:],
                surfaces: Set<QWKnobSurface>) {
        self.id = id
        self.tier = tier
        self.scope = scope
        self.kind = kind
        self.defaultValue = defaultValue
        self.requiredCapability = requiredCapability
        self.valueCapabilities = valueCapabilities
        self.surfaces = surfaces
    }
}

/// A knob value was refused.
public enum QWKnobError: Error, Equatable, CustomStringConvertible {
    case unknownKnob(String)
    case invalidValue(knob: String, value: String, expected: String)
    case capabilityUnavailable(knob: String, model: String, required: QWKnobCapability)
    case tierRestricted(knob: String, tier: QWKnobTier)
    case scopeViolation(knob: String, declared: QWKnobScope, at: QWKnobScope)
    case staleProfile(model: String, digest: String)

    public var description: String {
        switch self {
        case .unknownKnob(let id):
            return "QWKnob: unknown knob '\(id)'"
        case .invalidValue(let knob, let value, let expected):
            return "QWKnob: '\(knob)' rejected value '\(value)' (expected \(expected))"
        case .capabilityUnavailable(let knob, let model, let required):
            return "QWKnob: '\(knob)' needs capability '\(required.rawValue)' which '\(model)' lacks"
        case .tierRestricted(let knob, let tier):
            return "QWKnob: '\(knob)' is \(tier.rawValue)-tier and not exposed on this surface"
        case .scopeViolation(let knob, let declared, let at):
            return "QWKnob: '\(knob)' is \(declared.rawValue)-scope, set at \(at.rawValue) scope"
        case .staleProfile(let model, let digest):
            return "QWKnob: profile for '\(model)' was written against digest '\(digest)'"
        }
    }
}

extension QWKnobDescriptor {
    /// Rejects a value that does not fit the knob's kind.
    public func validate(_ value: QWKnobValue) throws {
        switch (kind, value) {
        case let (.int(range), .int(v)):
            guard range.contains(v) else {
                throw QWKnobError.invalidValue(knob: id, value: "\(v)", expected: "\(range)")
            }
        case let (.double(range), .double(v)):
            guard range.contains(v) else {
                throw QWKnobError.invalidValue(knob: id, value: "\(v)", expected: "\(range)")
            }
        case (.bool, .bool):
            break
        case let (.string(allowed), .string(v)):
            guard allowed.isEmpty || allowed.contains(v) else {
                throw QWKnobError.invalidValue(knob: id, value: v,
                                               expected: allowed.sorted().joined(separator: "|"))
            }
        default:
            throw QWKnobError.invalidValue(knob: id, value: "type mismatch", expected: "\(kind)")
        }
    }

    /// Whether the model behind `capabilities` can be offered this knob.
    public func isAvailable(under capabilities: Set<QWKnobCapability>) -> Bool {
        guard let required = requiredCapability else { return true }
        return capabilities.contains(required)
    }

    /// Rejects a value the model cannot honour.
    public func validate(_ value: QWKnobValue, capabilities: Set<QWKnobCapability>,
                         model: String) throws {
        try validate(value)
        if let v = value.stringValue, let required = valueCapabilities[v],
           !capabilities.contains(required) {
            throw QWKnobError.capabilityUnavailable(knob: id, model: model, required: required)
        }
    }
}

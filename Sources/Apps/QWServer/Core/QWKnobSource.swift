// QWKnobSource.swift — Where a value came from, in a closed set.
// The resolver writes free-form labels ("cli (--config /path)"). They are
// normalised here, at the boundary, so no consumer ever has to match on a
// sentence and no consumer can invent a sixth origin.

import Foundation
import QWCore

/// Where a knob's value came from.
public enum QWKnobSource: String, Sendable, Codable, CaseIterable {
    /// The user named it: a command-line flag.
    case explicit
    /// The model's profile: the settings file, or the pack's own config.
    case profile
    /// Nobody named it: the built-in default applies.
    case `default`
    /// An environment variable. Deprecated: the command line reads no knob
    /// from the environment, and the server warns when one is set.
    case deprecatedEnv

    /// Normalises one of the resolver's free-form labels.
    public static func normalise(_ label: String) -> QWKnobSource {
        let text = label.lowercased()
        if text.hasPrefix("cli") { return .explicit }
        if text.hasPrefix("env") { return .deprecatedEnv }
        // The settings file, the pack's own config and the model config
        // are all a profile: a file that says what the model should do.
        if text.hasPrefix("config-file") || text.hasPrefix("default-file")
            || text.hasPrefix("pack-derived") || text.hasPrefix("model-config")
            || text.hasPrefix("auto-derived") || text.hasPrefix("discovered") {
            return .profile
        }
        return .default
    }
}

extension QWKnobSource {
    /// The resolver's setting key for the knobs whose key differs from
    /// their id. The settings file and the knob registry are two
    /// vocabularies for the same values; this is where they meet.
    public static let settingKeyByKnobId: [String: String] = [
        "contextTokens": "maxContext",
        "speculativeEnabled": "speculativeDecodingEnabled",
        "prefillChunkRows": "prefillChunkSize",
        "warmCacheLimitGiB": "warmRAM",
        "ssdCacheLimitGiB": "ssdCache",
        "ropeScalingFactor": "ropeScaling",
        // The prefix cache is on when the warm cache has room: the limit
        // is the setting that decides it.
        "prefixCacheEnabled": "warmRAM"
    ]

    /// The knob id a resolver key names, when it names one.
    public static func knobId(forSettingKey key: String) -> String? {
        if let match = settingKeyByKnobId.first(where: { $0.value == key }) { return match.key }
        return QWKnobRegistry.all.contains { $0.id == key } ? key : nil
    }

    /// The source of each knob, by knob id. A knob the summary does not
    /// mention was never named: the default applies. A knob with no value
    /// has no source — a source without a value is a claim about nothing.
    public static func sources(from summary: [String: String],
                                for knobIds: [String]) -> [String: String] {
        var out: [String: String] = [:]
        for id in knobIds {
            let key = settingKeyByKnobId[id] ?? id
            let label = summary[key]
            out[id] = label.map { normalise($0).rawValue } ?? QWKnobSource.default.rawValue
        }
        return out
    }
}

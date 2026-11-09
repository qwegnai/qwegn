// QWEngineLifecycle.swift — The engine's lifecycle, as the outside sees it.
// The phase, the moment it began, the model it holds and why it failed:
// the whole truth, and nothing dressed up.

import Foundation
import QWCore

/// The phase the engine's lifecycle is in. A reader asks what the server
/// is doing and gets the truth: the phase is not a guess.
public enum QWEngineLifecyclePhase: String, Sendable, Codable, CaseIterable {
    case unloaded, loading, loaded, unloading, failed
}

/// What the engine is doing, and since when.
public struct QWEngineLifecycleInfo: Codable, Sendable, Equatable {
    /// The phase the engine is in.
    public let phase: QWEngineLifecyclePhase
    /// When the engine entered the phase.
    public let since: Date?
    /// The loaded model, when one is loaded.
    public let model: String?
    /// Why the phase is what it is — the failure, when there is one.
    public let detail: String?

    public init(phase: QWEngineLifecyclePhase, since: Date?, model: String?, detail: String?) {
        self.phase = phase
        self.since = since
        self.model = model
        self.detail = detail
    }

    private enum CodingKeys: String, CodingKey {
        case phase, since, model, detail
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(phase, forKey: .phase)
        try container.encodeIfPresent(Self.iso8601(since), forKey: .since)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encodeIfPresent(detail, forKey: .detail)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        phase = try container.decode(QWEngineLifecyclePhase.self, forKey: .phase)
        since = try container.decodeIfPresent(String.self, forKey: .since).flatMap(Self.date(from:))
        model = try container.decodeIfPresent(String.self, forKey: .model)
        detail = try container.decodeIfPresent(String.self, forKey: .detail)
    }

    /// The moment as an ISO-8601 instant — the form the wire contract
    /// shows, so a reader never has to guess what the number means.
    static func iso8601(_ date: Date?) -> String? {
        guard let date else { return nil }
        return date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true))
    }

    /// Reads an ISO-8601 instant, with or without a fractional part.
    static func date(from string: String) -> Date? {
        let fractional = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
        if let date = try? fractional.parse(string) { return date }
        let plain = Date.ISO8601FormatStyle(includingFractionalSeconds: false)
        return try? plain.parse(string)
    }
}

/// The engine's state, read in one look so a response cannot contradict
/// itself.
public struct QWEngineState: Sendable {
    /// The phase, the moment it began, the model and the reason.
    public let lifecycle: QWEngineLifecycleInfo
    /// Whether a model is resident right now.
    public let isLoaded: Bool
    /// The pack that is resident, when one is.
    public let loadedModelName: String?
    /// What the loaded model can do. A model that is not loaded can do
    /// nothing: the dashboard must not describe a model that is not there.
    public let capabilities: [QWKnobCapability]

    public init(lifecycle: QWEngineLifecycleInfo, isLoaded: Bool, loadedModelName: String?,
                capabilities: [QWKnobCapability] = []) {
        self.lifecycle = lifecycle
        self.isLoaded = isLoaded
        self.loadedModelName = loadedModelName
        self.capabilities = capabilities
    }
}


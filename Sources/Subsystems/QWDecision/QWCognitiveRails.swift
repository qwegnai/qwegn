// QWCognitiveRails.swift — Pure-Swift System-1 cognitive rail domain types & orchestration entrypoint.

import Foundation
import QWCore

public enum QWGuardVerdict: String, Codable, Sendable, Equatable {
    case allow
    case confirm
    case deny
}

public struct QWGuardResult: Codable, Sendable, Equatable {
    public let decision: QWGuardVerdict
    public let confidence: Double
    public let reason: String
    public let redLineTriggered: Bool
    public let modelDecision: String?
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case decision, confidence, reason
        case redLineTriggered = "red_line_triggered"
        case modelDecision = "model_decision"
        case latencyMs = "latency_ms"
    }

    public init(
        decision: QWGuardVerdict,
        confidence: Double,
        reason: String,
        redLineTriggered: Bool = false,
        modelDecision: String? = nil,
        latencyMs: Double = 0
    ) {
        self.decision = decision
        self.confidence = confidence
        self.reason = reason
        self.redLineTriggered = redLineTriggered
        self.modelDecision = modelDecision
        self.latencyMs = latencyMs
    }
}

public struct QWRerankPassageInput: Codable, Sendable, Equatable {
    public let id: String
    public let text: String
    public let metadata: [String: String]?

    public init(id: String, text: String, metadata: [String: String]? = nil) {
        self.id = id
        self.text = text
        self.metadata = metadata
    }
}

public struct QWRerankScreenItem: Codable, Sendable, Equatable {
    public let id: String
    public let text: String
    public let relevance: Double
    public let injection: Double
    public let kept: Bool
    public let droppedReason: String?

    enum CodingKeys: String, CodingKey {
        case id, text, relevance, injection, kept
        case droppedReason = "dropped_reason"
    }
}

public struct QWRerankScreenResult: Codable, Sendable, Equatable {
    public let items: [QWRerankScreenItem]
    public let droppedCount: Int
    public let mode: String
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case items, mode
        case droppedCount = "dropped_count"
        case latencyMs = "latency_ms"
    }
}

public struct QWCompactionTurnInput: Codable, Sendable, Equatable {
    public let index: Int
    public let role: String
    public let text: String
    public let pinned: Bool

    enum CodingKeys: String, CodingKey {
        case index, role, text, pinned
    }

    public init(index: Int, role: String, text: String, pinned: Bool = false) {
        self.index = index
        self.role = role
        self.text = text
        self.pinned = pinned
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.index = try c.decode(Int.self, forKey: .index)
        self.role = try c.decodeIfPresent(String.self, forKey: .role) ?? "user"
        self.text = try c.decodeIfPresent(String.self, forKey: .text) ?? ""
        self.pinned = try c.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
    }
}

public enum QWCompactionAction: String, Codable, Sendable, Equatable {
    case keep
    case summarize
    case drop
}

public struct QWCompactionDecision: Codable, Sendable, Equatable {
    public let index: Int
    public let action: QWCompactionAction
    public let confidence: Double
    public let pinnedByRail: Bool

    enum CodingKeys: String, CodingKey {
        case index, action, confidence
        case pinnedByRail = "pinned_by_rail"
    }
}

public struct QWCompactionResult: Codable, Sendable, Equatable {
    public let decisions: [QWCompactionDecision]
    public let fallbackUsed: Bool
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case decisions
        case fallbackUsed = "fallback_used"
        case latencyMs = "latency_ms"
    }
}

public struct QWSkillDescriptor: Codable, Sendable, Equatable {
    public let name: String
    public let description: String
    public let path: String?
    public let existsOnDisk: Bool

    enum CodingKeys: String, CodingKey {
        case name, description, path
        case existsOnDisk = "exists_on_disk"
    }

    public init(name: String, description: String, path: String? = nil, existsOnDisk: Bool = true) {
        self.name = name
        self.description = description
        self.path = path
        self.existsOnDisk = existsOnDisk
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        self.path = try c.decodeIfPresent(String.self, forKey: .path)
        self.existsOnDisk = try c.decodeIfPresent(Bool.self, forKey: .existsOnDisk) ?? true
    }
}

public struct QWSkillSelectionResult: Codable, Sendable, Equatable {
    public let selected: [String]
    public let shortlisted: [String]
    public let needsSkill: Bool
    public let needsSkillConfidence: Double
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case selected, shortlisted
        case needsSkill = "needs_skill"
        case needsSkillConfidence = "needs_skill_confidence"
        case latencyMs = "latency_ms"
    }
}

public enum QWSearchSufficiencyAction: String, Codable, Sendable, Equatable {
    case answer
    case searchMore = "search_more"
    case proposeQueries = "propose_queries"
    case unknown
}

public struct QWSearchSufficiencyResult: Codable, Sendable, Equatable {
    public let action: QWSearchSufficiencyAction
    public let confidence: Double
    public let conflictDetected: Bool
    public let proposedQueries: [String]
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case action, confidence
        case conflictDetected = "conflict_detected"
        case proposedQueries = "proposed_queries"
        case latencyMs = "latency_ms"
    }
}

public struct QWSuperviseInput: Codable, Sendable, Equatable {
    public let taskId: String
    public let exitCode: Int?
    public let secondsSinceOutput: Double
    public let tailOutput: String
    public let consecutiveAlerts: Int

    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case exitCode = "exit_code"
        case secondsSinceOutput = "seconds_since_output"
        case tailOutput = "tail_output"
        case consecutiveAlerts = "consecutive_alerts"
    }

    public init(
        taskId: String,
        exitCode: Int? = nil,
        secondsSinceOutput: Double = 0,
        tailOutput: String = "",
        consecutiveAlerts: Int = 0
    ) {
        self.taskId = taskId
        self.exitCode = exitCode
        self.secondsSinceOutput = secondsSinceOutput
        self.tailOutput = tailOutput
        self.consecutiveAlerts = consecutiveAlerts
    }
}

public enum QWSuperviseVerdict: String, Codable, Sendable, Equatable {
    case `continue`
    case done
    case alert
    case escalate
}

public struct QWSuperviseResult: Codable, Sendable, Equatable {
    public let verdict: QWSuperviseVerdict
    public let confidence: Double
    public let reason: String
    public let nextConsecutiveAlerts: Int
    public let factOverride: Bool
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case verdict, confidence, reason
        case nextConsecutiveAlerts = "next_consecutive_alerts"
        case factOverride = "fact_override"
        case latencyMs = "latency_ms"
    }
}

public enum QWRouteMode: String, Codable, Sendable, Equatable {
    case on
    case shadow
    case off
}

public struct QWRoutePolicy: Codable, Sendable, Equatable {
    public let mode: QWRouteMode
    public let stickyContextTokens: Int
    public let confidenceFloor: Double
    public let costlyRiskThreshold: Double

    enum CodingKeys: String, CodingKey {
        case mode
        case stickyContextTokens = "sticky_context_tokens"
        case confidenceFloor = "confidence_floor"
        case costlyRiskThreshold = "costly_risk_threshold"
    }

    public init(
        mode: QWRouteMode = .on,
        stickyContextTokens: Int = 32_768,
        confidenceFloor: Double = 0.70,
        costlyRiskThreshold: Double = 0.60
    ) {
        self.mode = mode
        self.stickyContextTokens = stickyContextTokens
        self.confidenceFloor = confidenceFloor
        self.costlyRiskThreshold = costlyRiskThreshold
    }
}

public struct QWRouteTurnInput: Codable, Sendable, Equatable {
    public let prompt: String
    public let contextTokens: Int
    public let currentTier: String
    public let hasTools: Bool
    public let policy: QWRoutePolicy

    enum CodingKeys: String, CodingKey {
        case prompt
        case contextTokens = "context_tokens"
        case currentTier = "current_tier"
        case hasTools = "has_tools"
        case policy
    }

    public init(
        prompt: String,
        contextTokens: Int = 0,
        currentTier: String = "balanced",
        hasTools: Bool = false,
        policy: QWRoutePolicy = QWRoutePolicy()
    ) {
        self.prompt = prompt
        self.contextTokens = contextTokens
        self.currentTier = currentTier
        self.hasTools = hasTools
        self.policy = policy
    }
}

public struct QWRouteTurnResult: Codable, Sendable, Equatable {
    public let recommendedTier: String
    public let recommendedEffort: String
    public let difficulty: String
    public let domain: String
    public let costlyRisk: Double
    public let confidence: Double
    public let stickyGuardApplied: Bool
    public let mode: QWRouteMode
    public let wouldSwitch: Bool
    public let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case recommendedTier = "recommended_tier"
        case recommendedEffort = "recommended_effort"
        case difficulty, domain, confidence, mode
        case costlyRisk = "costly_risk"
        case stickyGuardApplied = "sticky_guard_applied"
        case wouldSwitch = "would_switch"
        case latencyMs = "latency_ms"
    }
}

/// Pure-Swift orchestration layer applying deterministic code rails around System-1 decisions.
public struct QWCognitiveRails: Sendable {
    public typealias DecideFn = @Sendable (QWDecisionRequest) async throws -> QWDecisionResponse
    public typealias DecideBatchFn = @Sendable ([QWDecisionRequest]) async throws -> [QWDecisionResponse]

    let decideFn: DecideFn?
    let decideBatchFn: DecideBatchFn?

    public init(runner: QWDecisionRunner?) {
        if let runner {
            self.decideFn = { req in try await runner.decide(req) }
            self.decideBatchFn = { reqs in try await runner.decideBatch(reqs) }
        } else {
            self.decideFn = nil
            self.decideBatchFn = nil
        }
    }

    public init(
        decideFn: DecideFn?,
        decideBatchFn: DecideBatchFn? = nil
    ) {
        self.decideFn = decideFn
        if let decideBatchFn {
            self.decideBatchFn = decideBatchFn
        } else if let decideFn {
            self.decideBatchFn = { reqs in
                var out: [QWDecisionResponse] = []
                out.reserveCapacity(reqs.count)
                for req in reqs {
                    out.append(try await decideFn(req))
                }
                return out
            }
        } else {
            self.decideBatchFn = nil
        }
    }
}

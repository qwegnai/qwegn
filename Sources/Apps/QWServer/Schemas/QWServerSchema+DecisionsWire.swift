// QWServerSchema+DecisionsWire.swift — Wire response DTOs for /v1/decisions, /v1/systemone, and /v1/qwegn.

import Foundation
import QWCore
import QWDecision
import QWEngine
import QWToken

public struct QWDecisionHTTPResponse: Codable, Sendable {
    public let answers: [QWDecisionAnswer]
    public let usage: QWDecisionUsage
    public let chain: String
    public let model: String
    public let backend: String

    public init(envelope: QWDecisionExecutionEnvelope) {
        self.answers = envelope.response.answers
        self.usage = envelope.response.usage
        self.chain = envelope.response.chain
        self.model = envelope.modelName
        self.backend = envelope.backendKind
    }
}

public struct QWDecisionBatchHTTPResponse: Codable, Sendable {
    public let results: [QWDecisionHTTPResponse]
}

public struct QWSystemOneUsage: Codable, Sendable {
    public let inputTokens: Int
    public let outputTokens: Int

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    public init(from usage: QWDecisionUsage) {
        self.inputTokens = usage.promptTokens
        self.outputTokens = usage.outputTokens
    }
}

public struct QWSystemOneHTTPResponse: Encodable, Sendable {
    public let model: String
    public let answers: [String: QWJson]
    public let usage: QWSystemOneUsage

    public init(envelope: QWDecisionExecutionEnvelope, meta: [QWDecisionQuestionWireMeta]) {
        self.model = envelope.modelName
        self.usage = QWSystemOneUsage(from: envelope.response.usage)
        var out: [String: QWJson] = [:]
        out.reserveCapacity(envelope.response.answers.count)
        let metaByName = Dictionary(uniqueKeysWithValues: meta.map { ($0.name, $0) })
        for ans in envelope.response.answers {
            out[ans.name] = Self.encodeAnswer(ans, legend: metaByName[ans.name]?.scoreLegend ?? [:])
        }
        self.answers = out
    }

    private static func encodeAnswer(_ ans: QWDecisionAnswer, legend: [String: QWJson]) -> QWJson {
        var probPairs: [(String, QWJson)] = []
        probPairs.reserveCapacity(ans.probs.count)
        for p in ans.probs {
            probPairs.append((p.key, .double(p.value)))
        }
        switch ans.kind {
        case .noul:
            let pYes = ans.probs.first(where: {
                let low = $0.key.lowercased()
                return low == "yes" || low == "true"
            })?.value ?? ((ans.noul == true) ? ans.confidence : (1.0 - ans.confidence))
            return .object([
                ("type", .string("noul")),
                ("noul", .double(pYes)),
            ])
        case .choice:
            let chosen = ans.choice ?? ans.probs.max(by: { $0.value < $1.value })?.key ?? ""
            return .object([
                ("type", .string("choice")),
                ("choice", .string(chosen)),
                ("confidence", .double(ans.confidence)),
                ("probabilities", .object(probPairs)),
            ])
        case .multiChoice:
            var fields: [(String, QWJson)] = [
                ("type", .string("multiple")),
                ("selected_choices", .array(ans.selectedChoices.map { .string($0) })),
                ("confidence", .double(ans.confidence)),
                ("probabilities", .object(probPairs)),
            ]
            if let c = ans.choice { fields.append(("choice", .string(c))) }
            return .object(fields)
        case .score:
            let scorePairs: [(String, QWJson)] = ans.probs.enumerated().map { idx, p in
                let key = Int(p.key) != nil ? p.key : "\(idx)"
                return (key, .double(p.value))
            }
            var resolvedLegend = legend
            if resolvedLegend.isEmpty {
                for (idx, p) in ans.probs.enumerated() {
                    let key = Int(p.key) != nil ? p.key : "\(idx)"
                    resolvedLegend[key] = .string(p.key)
                }
            }
            let legendPairs = resolvedLegend.sorted {
                (Int($0.key) ?? 0) < (Int($1.key) ?? 0)
            }
            return .object([
                ("type", .string("score")),
                ("score", .double(ans.score ?? 0.0)),
                ("confidence", .double(ans.confidence)),
                ("legend", .object(legendPairs)),
                ("probabilities", .object(scorePairs)),
            ])
        case .box:
            let b = ans.boxes.first ?? QWDecisionBox(top: 0, bottom: 0, left: 0, right: 0, found: 0, confidence: 0)
            let boxObj: QWJson = .object([
                ("top", .double(b.top)),
                ("bottom", .double(b.bottom)),
                ("left", .double(b.left)),
                ("right", .double(b.right)),
            ])
            let conf: QWJson = .double(ans.boxes.first?.confidence ?? ans.foundProb ?? ans.confidence)
            return .object([
                ("type", .string("box")),
                ("box", boxObj),
                ("confidence", conf),
                ("found", .bool(ans.found ?? !ans.boxes.isEmpty)),
                ("found_prob", .double(ans.foundProb ?? b.found)),
            ])
        case .boxes:
            let boxList: [QWJson] = ans.boxes.map { b in
                .object([
                    ("top", .double(b.top)),
                    ("bottom", .double(b.bottom)),
                    ("left", .double(b.left)),
                    ("right", .double(b.right)),
                    ("found", .double(b.found)),
                    ("confidence", .double(b.confidence)),
                ])
            }
            return .object([
                ("type", .string("boxes")),
                ("boxes", .array(boxList)),
                ("confidence", .double(ans.confidence)),
                ("found", .bool(ans.found ?? !ans.boxes.isEmpty)),
                ("at_least", .bool(ans.atLeast)),
            ])
        case .read:
            var fields: [(String, QWJson)] = [
                ("type", .string("read")),
                ("found", .bool(ans.found ?? false)),
                ("confidence", .double(ans.confidence)),
            ]
            if let fp = ans.foundProb { fields.append(("found_prob", .double(fp))) }
            if let txt = ans.text { fields.append(("text", .string(txt))) }
            if let num = ans.number { fields.append(("number", .double(num))) }
            if let dt = ans.date { fields.append(("date", .string(dt))) }
            if let jp = ans.jsonPath { fields.append(("json_path", .string(jp))) }
            if let sp = ans.charSpan ?? ans.span {
                fields.append(("span", .object([("start", .int(sp.lowerBound)), ("end", .int(sp.upperBound))])))
            }
            return .object(fields)
        case .estimate:
            var fields: [(String, QWJson)] = [
                ("type", .string("estimate")),
                ("estimate", .double(ans.estimate ?? 0.0)),
                ("confidence", .double(ans.confidence)),
            ]
            if let iv = ans.interval {
                fields.append(("interval", .array(iv.map { .double($0) })))
            }
            return .object(fields)
        }
    }
}

public struct QWSystemOneBatchHTTPResponse: Encodable, Sendable {
    public let results: [QWSystemOneHTTPResponse]
}

public struct QWOpenAIDecisionHTTPResponse: Encodable, Sendable {
    public let model: String
    public let answers: [QWJson]
    public let usage: QWSystemOneUsage

    public init(envelope: QWDecisionExecutionEnvelope, meta: [QWDecisionQuestionWireMeta]) {
        self.model = envelope.modelName
        self.usage = QWSystemOneUsage(from: envelope.response.usage)
        let metaByName = Dictionary(uniqueKeysWithValues: meta.map { ($0.name, $0) })
        self.answers = envelope.response.answers.map { ans in
            let incoming = metaByName[ans.name]?.incomingType ?? ans.kind.rawValue
            switch ans.kind {
            case .noul:
                let pYes = ans.probs.first(where: {
                    let low = $0.key.lowercased()
                    return low == "yes" || low == "true"
                })?.value ?? ((ans.noul == true) ? ans.confidence : (1.0 - ans.confidence))
                let wireType = (incoming == "predicate") ? "predicate" : "noul"
                var fields: [(String, QWJson)] = [
                    ("name", .string(ans.name)),
                    ("type", .string(wireType)),
                    ("probability", .double(pYes)),
                ]
                if wireType == "noul" {
                    fields.append(("noul", .double(pYes)))
                }
                return .object(fields)
            case .choice, .score, .multiChoice:
                let entries: [QWJson] = ans.probs.map { p in
                    .object([
                        ("value", .string(p.key)),
                        ("probability", .double(p.value)),
                    ])
                }
                var fields: [(String, QWJson)] = [
                    ("name", .string(ans.name)),
                    ("type", .string(ans.kind.rawValue)),
                    ("probabilities", .array(entries)),
                ]
                if let c = ans.choice { fields.append(("choice", .string(c))) }
                if let s = ans.score { fields.append(("score", .double(s))) }
                return .object(fields)
            default:
                return QWSystemOneHTTPResponse(envelope: envelope, meta: meta).answers[ans.name] ?? .null
            }
        }
    }
}

/// Wire payload for `POST /v1/qwegn` (`qwg_ask`) enforcing the 60,000-character rail (`ADR-008`).
public struct QWQwegnAskHTTPRequest: Codable, Sendable {
    public static let maxContextCharacters = 60_000

    public let model: String?
    public let context: QWJson?
    public let state: QWJson?
    public let question: String?
    public let instructions: String?
    public let kind: String?
    public let mode: String?
    public let command: String?
    public let options: [String]?
    public let labels: [String]?
    public let criteria: QWJson?
    public let questions: QWJson?
    public let chain: String?
    public let guardMode: Bool?
    public let scoreReadout: String?
    public let images: [String]?

    enum CodingKeys: String, CodingKey {
        case model, context, state, question, instructions, kind, mode, command
        case options, labels, criteria, questions, chain, images
        case guardMode = "guard"
        case scoreReadout = "score_readout"
    }

    public func toDecisionRequest() throws -> QWDecisionRequest {
        let resolvedState: QWDecisionState
        if let cmd = command, state == nil, context == nil {
            resolvedState = .text("About to execute command: \(cmd)")
        } else {
            resolvedState = try QWDecisionHTTPRequest.decodeState(state ?? context)
        }

        let charCount: Int = switch resolvedState {
        case .empty: 0
        case .text(let s): s.count
        case .json(let d): String(decoding: d, as: UTF8.self).count
        }
        guard charCount <= Self.maxContextCharacters else {
            throw QWDecisionError.invalidQuestion(
                "state length \(charCount) exceeds \(Self.maxContextCharacters)-character Qwegn Ask rail"
            )
        }

        let readout: QWScoreReadout
        if let rawR = scoreReadout?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !rawR.isEmpty {
            guard let parsed = QWScoreReadout(rawValue: rawR) else {
                throw QWDecisionError.invalidQuestion("unsupported score_readout '\(rawR)'")
            }
            readout = parsed
        } else {
            readout = .mean
        }

        let qList: [QWDecisionQuestion]
        if let questions, !questions.isUndefined, questions != .null {
            qList = try QWDecisionHTTPRequest.decodeQuestions(questions, defaultScoreReadout: readout)
        } else if guardMode == true || mode?.lowercased() == "guard" || (command != nil && question == nil && instructions == nil) {
            let guardOptions = options ?? ["allow", "confirm", "deny"]
            let guardCriteria = [
                "allow": "safe to run without operator confirmation",
                "confirm": "potentially risky, ask operator before executing",
                "deny": "destructive or policy-violating command",
            ]
            let criteria = guardOptions.map { key in
                QWDecisionCriterion(key: key, description: guardCriteria[key])
            }
            let promptText = question ?? instructions ?? "How should the harness treat this command?"
            qList = [QWDecisionQuestion(
                name: "guard",
                kind: .choice,
                instructions: promptText,
                criteria: criteria,
                chain: chain ?? "guardrail"
            )]
        } else {
            let promptText = question ?? instructions ?? ""
            guard !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw QWDecisionError.invalidQuestion("Qwegn Ask request requires 'question', 'questions', or 'command'")
            }
            let single = QWDecisionHTTPQuestion(
                name: "answer",
                kind: kind,
                instructions: promptText,
                criteria: criteria,
                options: options,
                labels: labels,
                scoreReadout: scoreReadout,
                chain: chain
            )
            qList = [try single.toDecisionQuestion(fallbackName: "answer", defaultScoreReadout: readout)]
        }

        let decodedImages = try QWDecisionHTTPRequest.decodeHTTPImages(images)
        return QWDecisionRequest(
            state: resolvedState,
            questions: qList,
            truncation: nil,
            chain: chain ?? "default",
            images: decodedImages
        )
    }
}

public struct QWQwegnAskHTTPResponse: Codable, Sendable {
    public let answer: QWJson
    public let decision: String?
    public let confidence: Double
    public let probabilities: [String: Double]
    public let probs: [QWDecisionProbability]
    public let answers: [String: QWDecisionAnswer]
    public let usage: QWDecisionUsage
    public let chain: String
    public let model: String
    public let backend: String

    public init(envelope: QWDecisionExecutionEnvelope) {
        let primary = envelope.response.answers.first
        let probPairs = primary?.probs ?? []
        var probDict: [String: Double] = [:]
        for p in probPairs { probDict[p.key] = p.value }

        let summaryValue: QWJson
        let decisionLabel: String?
        if let primary {
            switch primary.kind {
            case .noul:
                let b = primary.noul ?? false
                summaryValue = .bool(b)
                decisionLabel = b ? "yes" : "no"
            case .choice, .multiChoice:
                if let c = primary.choice {
                    summaryValue = .string(c)
                    decisionLabel = c
                } else {
                    summaryValue = .array(primary.selectedChoices.map { .string($0) })
                    decisionLabel = primary.selectedChoices.first
                }
            case .score:
                let val = primary.score ?? 0
                summaryValue = .double(val)
                decisionLabel = nil
            case .estimate:
                let val = primary.estimate ?? 0
                summaryValue = .double(val)
                decisionLabel = nil
            case .read:
                if let num = primary.number {
                    summaryValue = .double(num)
                } else {
                    summaryValue = .string(primary.date ?? primary.text ?? "")
                }
                decisionLabel = primary.text ?? primary.date
            case .box, .boxes:
                summaryValue = .string("\(primary.boxes.count) box(es)")
                decisionLabel = primary.boxes.isEmpty ? "none" : "detected"
            }
        } else {
            summaryValue = .null
            decisionLabel = nil
        }

        var keyed: [String: QWDecisionAnswer] = [:]
        for ans in envelope.response.answers { keyed[ans.name] = ans }

        self.answer = summaryValue
        self.decision = decisionLabel
        self.confidence = primary?.confidence ?? 0
        self.probabilities = probDict
        self.probs = probPairs
        self.answers = keyed
        self.usage = envelope.response.usage
        self.chain = envelope.response.chain
        self.model = envelope.modelName
        self.backend = envelope.backendKind
    }
}

public struct QWDecisionLoadHTTPRequest: Codable, Sendable {
    public let packPath: String?
    public let model: String?
    public let computeUnit: String?
    public let loadVision: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case packPath = "pack_path"
        case computeUnit = "compute_unit"
        case loadVision = "load_vision"
    }
}

public struct QWDecisionErrorPayload: Codable, Sendable {
    public struct Detail: Codable, Sendable {
        public let message: String
        public let type: String
        public let code: Int
        public let allowedLabels: [String]?

        enum CodingKeys: String, CodingKey {
            case message, type, code
            case allowedLabels = "allowed_labels"
        }
    }

    public let error: Detail
    public let allowedLabels: [String]?

    enum CodingKeys: String, CodingKey {
        case error
        case allowedLabels = "allowed_labels"
    }

    public init(message: String, type: String = "invalid_request_error", code: Int, allowedLabels: [String]? = nil) {
        self.error = Detail(message: message, type: type, code: code, allowedLabels: allowedLabels)
        self.allowedLabels = allowedLabels
    }
}

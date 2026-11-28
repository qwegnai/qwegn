// QWDecisionRunner.swift — Pure-host single-pass orchestrator combining framing, backend, calibration, and decoding.
import Foundation
import QWCore
import QWToken

public struct QWRerankPairScore: Sendable, Equatable {
    public let index: Int
    public let relevanceScore: Double

    public init(index: Int, relevanceScore: Double) {
        self.index = index
        self.relevanceScore = relevanceScore
    }
}

public final class QWDecisionRunner: Sendable {
    public let tokenizer: QWTokenizer
    public let backend: any QWDecisionExecutionBackend
    public let calibration: QWCalibrationProfile

    public init(
        tokenizer: QWTokenizer,
        backend: any QWDecisionExecutionBackend,
        calibration: QWCalibrationProfile = QWCalibrationProfile()
    ) {
        self.tokenizer = tokenizer
        self.backend = backend
        self.calibration = calibration.mergingInlineTemperatures(backend.decisionConfig.inlineTemperatures)
    }

    public func decide(_ request: QWDecisionRequest) async throws -> QWDecisionResponse {
        guard !request.questions.isEmpty else {
            throw QWDecisionError.invalidQuestion("at least one question is required")
        }
        let built = try QWSequenceFramingEngine.buildBatch(
            request: request,
            config: backend.decisionConfig,
            tokenizer: tokenizer,
            maxContextLength: backend.maxContextLength
        )
        let backendOut = try await backend.evaluate(built.request)
        guard backendOut.slotOutputs.count == built.request.slots.count else {
            throw QWDecisionError.backendFailed(
                "backend emitted \(backendOut.slotOutputs.count) slot outputs for \(built.request.slots.count) requested slots"
            )
        }
        let answers = try QWTypedAnswerDecoder.decode(
            built: built,
            output: backendOut,
            calibration: calibration,
            chain: request.chain
        )
        return QWDecisionResponse(
            answers: answers,
            usage: QWDecisionUsage(
                promptTokens: max(built.promptTokens, backendOut.promptTokens),
                outputTokens: 0,
                truncatedTokens: built.truncatedTokens,
                latencyMs: backendOut.latencyMs
            ),
            chain: request.chain
        )
    }

    public func decideBatch(_ requests: [QWDecisionRequest]) async throws -> [QWDecisionResponse] {
        guard !requests.isEmpty else { return [] }
        if requests.count == 1 {
            return [try await decide(requests[0])]
        }
        var builtList: [QWBuiltDecisionBatch] = []
        builtList.reserveCapacity(requests.count)
        var allSequences: [QWDecisionSequenceInput] = []
        var allSlots: [QWDecisionSlotTarget] = []
        var candSet = Set<Int>()
        var slotCounts: [Int] = []
        slotCounts.reserveCapacity(requests.count)

        for req in requests {
            guard !req.questions.isEmpty else {
                throw QWDecisionError.invalidQuestion("at least one question is required")
            }
            let built = try QWSequenceFramingEngine.buildBatch(
                request: req,
                config: backend.decisionConfig,
                tokenizer: tokenizer,
                maxContextLength: backend.maxContextLength
            )
            let seqOffset = allSequences.count
            allSequences.append(contentsOf: built.request.sequences)
            for slot in built.request.slots {
                allSlots.append(QWDecisionSlotTarget(
                    sequenceIndex: slot.sequenceIndex + seqOffset,
                    tokenPosition: slot.tokenPosition,
                    candidateTokenGroups: slot.candidateTokenGroups,
                    isPointerSlot: slot.isPointerSlot,
                    pointerEndPosition: slot.pointerEndPosition
                ))
            }
            candSet.formUnion(built.request.uniqueCandidateTokenIds)
            slotCounts.append(built.request.slots.count)
            builtList.append(built)
        }

        let combinedReq = QWDecisionBackendBatchRequest(
            sequences: allSequences,
            slots: allSlots,
            uniqueCandidateTokenIds: candSet.sorted()
        )
        let combinedOut = try await backend.evaluate(combinedReq)
        guard combinedOut.slotOutputs.count == allSlots.count else {
            throw QWDecisionError.backendFailed(
                "backend emitted \(combinedOut.slotOutputs.count) slot outputs for \(allSlots.count) requested slots"
            )
        }

        var responses: [QWDecisionResponse] = []
        responses.reserveCapacity(requests.count)
        var slotCursor = 0
        for (idx, built) in builtList.enumerated() {
            let count = slotCounts[idx]
            let slice = Array(combinedOut.slotOutputs[slotCursor..<(slotCursor + count)])
            slotCursor += count
            let subOut = QWDecisionBackendOutput(
                slotOutputs: slice,
                promptTokens: built.promptTokens,
                latencyMs: combinedOut.latencyMs
            )
            let answers = try QWTypedAnswerDecoder.decode(
                built: built,
                output: subOut,
                calibration: calibration,
                chain: requests[idx].chain
            )
            responses.append(QWDecisionResponse(
                answers: answers,
                usage: QWDecisionUsage(
                    promptTokens: built.promptTokens,
                    outputTokens: 0,
                    truncatedTokens: built.truncatedTokens,
                    latencyMs: combinedOut.latencyMs
                ),
                chain: requests[idx].chain
            ))
        }
        return responses
    }

    public func rerank(
        query: String,
        documents: [String],
        chain: String = "rerank"
    ) async throws -> (scores: [QWRerankPairScore], usage: QWDecisionUsage) {
        guard !documents.isEmpty else {
            return ([], QWDecisionUsage(promptTokens: 0, outputTokens: 0, truncatedTokens: 0, latencyMs: 0))
        }
        let cfg = backend.decisionConfig
        if case .sequencePair = cfg.framing,
           case .pooledClassifier(_, let numLabels, let id2label) = cfg.head {
            let built = try QWSequenceFramingEngine.buildNativeRerankBatch(
                query: query,
                documents: documents,
                config: cfg,
                tokenizer: tokenizer,
                maxContextLength: backend.maxContextLength
            )
            let out = try await backend.evaluate(built.batch)
            guard out.slotOutputs.count == documents.count else {
                throw QWDecisionError.backendFailed(
                    "rerank backend emitted \(out.slotOutputs.count) slot outputs for \(documents.count) documents"
                )
            }
            let temp = QWCalibrationEngine.temperature(
                profile: calibration, kind: .noul, cardinality: max(2, numLabels), chain: chain
            )
            let knots = calibration.tables["rerank"] ?? calibration.tables["noul"]
            let posIdx = Self.resolvePositiveRerankClassIndex(numLabels: numLabels, id2label: id2label)
            let scores: [QWRerankPairScore] = out.slotOutputs.enumerated().map { idx, slotOut in
                let rawProb = Self.decodePooledRelevanceProb(
                    slotOut: slotOut, numLabels: numLabels, positiveIndex: posIdx, temperature: temp
                )
                let cal = min(1.0, max(0.0, QWCalibrationEngine.through(rawProb, knots: knots)))
                return QWRerankPairScore(index: idx, relevanceScore: QWCalibrationEngine.roundPy(cal))
            }
            let usage = QWDecisionUsage(
                promptTokens: max(built.promptTokens, out.promptTokens),
                outputTokens: 0,
                truncatedTokens: built.truncatedTokens,
                latencyMs: out.latencyMs
            )
            return (scores, usage)
        }

        let requests = documents.map { doc in
            QWDecisionRequest(
                state: .text(doc),
                questions: [
                    QWDecisionQuestion(
                        name: "relevant",
                        kind: .noul,
                        instructions: "Is this document relevant to the query: \(query)"
                    ),
                ],
                truncation: .autoState,
                chain: chain
            )
        }
        let responses = try await decideBatch(requests)
        let totalPrompt = responses.reduce(0) { $0 + $1.usage.promptTokens }
        let totalTrunc = responses.reduce(0) { $0 + $1.usage.truncatedTokens }
        let maxLat = responses.map(\.usage.latencyMs).max() ?? 0.0
        let scores: [QWRerankPairScore] = responses.enumerated().map { idx, resp in
            let firstAns = resp.answers.first
            let probs = firstAns?.probs ?? []
            let fallback: Double = (firstAns?.noul == true) ? (firstAns?.confidence ?? 0.5) : 0.0
            let pYes = probs.first(where: { $0.key == "yes" })?.value ?? fallback
            return QWRerankPairScore(index: idx, relevanceScore: pYes)
        }
        return (scores, QWDecisionUsage(promptTokens: totalPrompt, outputTokens: 0, truncatedTokens: totalTrunc, latencyMs: maxLat))
    }

    private static func resolvePositiveRerankClassIndex(numLabels: Int, id2label: [Int: String]) -> Int {
        guard numLabels >= 2 else { return 0 }
        let positiveNames: Set<String> = ["entailment", "relevant", "yes", "true", "positive", "1", "label_1"]
        for (cid, raw) in id2label where cid >= 0 && cid < numLabels {
            let norm = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if positiveNames.contains(norm) { return cid }
        }
        return numLabels == 2 ? 1 : (numLabels - 1)
    }

    private static func decodePooledRelevanceProb(
        slotOut: QWDecisionSlotOutput,
        numLabels: Int,
        positiveIndex: Int,
        temperature: Double
    ) -> Double {
        let safeTemp = max(1e-4, temperature)
        guard let pooled = slotOut.pooledLogits, !pooled.isEmpty else {
            let m = Double(slotOut.markerLogits?.first ?? 0)
            return 1.0 / (1.0 + exp(-m / safeTemp))
        }
        if pooled.count == 1 || numLabels <= 1 {
            return 1.0 / (1.0 + exp(-Double(pooled[0]) / safeTemp))
        }
        let logits = pooled.prefix(max(2, numLabels)).map(Double.init)
        let probs = QWCalibrationEngine.calibratedSoftmax(logits: logits, temperature: safeTemp)
        let target = min(max(0, positiveIndex), probs.count - 1)
        return probs[target]
    }
}

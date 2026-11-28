// QWSequenceFramingEngine+Encoders.swift — Option-marker & sequence-pair/cross-encoder framing extensions.
import Foundation
import QWCore
import QWToken

extension QWSequenceFramingEngine {
    internal static func buildOptionMarkersBatch(
        request: QWDecisionRequest,
        stateRaw: String,
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        budgetLimit: Int,
        layout: QWDecisionModelConfig.MarkerLayout,
        markerToken: String,
        bosToken: String?,
        defaultReadout: QWScoreReadout
    ) throws -> QWBuiltDecisionBatch {
        let escapedState = QWPlainEscapedText(
            original: stateRaw, escaped: escapeMarkerTags(stateRaw),
            boundaryLo: Array(0...stateRaw.utf8.count), boundaryHi: Array(0...stateRaw.utf8.count)
        )
        let stateIds = try tokenizer.encode(escapedState.escaped, addSpecialTokens: false)
        let markerId = tokenizer.tokenId(for: markerToken) ?? tokenizer.tokenId(for: "[MASK]") ?? tokenizer.tokenId(for: "<|mask|>") ?? 0
        let bosId = bosToken.flatMap { tokenizer.tokenId(for: $0) } ?? tokenizer.tokenId(for: "[CLS]") ?? tokenizer.tokenId(for: "<s>") ?? 1
        let sepId = tokenizer.tokenId(for: "[SEP]") ?? tokenizer.tokenId(for: "<decide>") ?? 2

        var sequences: [QWDecisionSequenceInput] = []
        var slots: [QWDecisionSlotTarget] = []
        var bindings: [QWBoundQuestion] = []
        var truncatedTotal = 0
        var totalPromptTokens = 0

        for (qIdx, q) in request.questions.enumerated() {
            let safeIns = escapeMarkerTags(q.instructions)
            let qHeadIds = try tokenizer.encode("\(q.kind.rawValue) question: \(safeIns)", addSpecialTokens: false)
            let optTexts = renderOptionTexts(for: q).map { escapeMarkerTags($0) }
            let optBudget = max(96, min(optTexts.count * 24 + 32, budgetLimit / 2))
            let perOpt = max(2, (optBudget - 3 * optTexts.count) / max(1, optTexts.count))

            var qBlock: [Int] = [sepId] + Array(qHeadIds.prefix(max(16, optBudget)))
            var relMarkers: [Int] = []
            for ot in optTexts {
                let otIds = try tokenizer.encode(" " + ot, addSpecialTokens: false)
                if layout == .stateQuestionOptions {
                    if let optOpen = tokenizer.tokenId(for: "<opt>") { qBlock.append(optOpen) }
                    relMarkers.append(qBlock.count)
                    qBlock.append(markerId)
                    qBlock.append(contentsOf: otIds.prefix(perOpt))
                    if let optClose = tokenizer.tokenId(for: "</opt>") { qBlock.append(optClose) }
                } else {
                    relMarkers.append(qBlock.count)
                    qBlock.append(markerId)
                    qBlock.append(contentsOf: otIds.prefix(perOpt))
                }
            }
            qBlock.append(sepId)

            let room = max(0, budgetLimit - 2 - qBlock.count)
            let policy = request.truncation ?? .autoState
            if policy == .error && stateIds.count > room {
                throw QWDecisionError.contextOverflow(tokens: 2 + qBlock.count + stateIds.count, limit: budgetLimit)
            }
            let keptState = Array(stateIds.prefix(room))
            truncatedTotal = max(truncatedTotal, stateIds.count - keptState.count)

            var ids: [Int] = [bosId]
            let markers: [Int]
            let stateEndPos: Int
            if layout == .stateQuestionOptions {
                ids.append(contentsOf: keptState)
                stateEndPos = ids.count
                let base = ids.count
                markers = relMarkers.map { base + $0 }
                ids.append(contentsOf: qBlock)
            } else {
                let base = ids.count
                markers = relMarkers.map { base + $0 }
                ids.append(contentsOf: qBlock)
                stateEndPos = ids.count
                ids.append(contentsOf: keptState)
                ids.append(sepId)
            }
            let seq = QWDecisionSequenceInput(
                tokenIds: ids, topology: config.topology, stateEndPosition: stateEndPos,
                markerPositions: markers, chain: q.chain ?? request.chain, images: request.images
            )
            sequences.append(seq)
            totalPromptTokens += ids.count

            let slotIdx = slots.count
            slots.append(QWDecisionSlotTarget(sequenceIndex: qIdx, tokenPosition: markers.last ?? (ids.count - 1)))
            let kind: QWSlotBindingKind
            switch q.kind {
            case .noul: kind = .noul
            case .score: kind = .score(labels: q.criteria.map(\.key))
            default: kind = .categorical(labels: q.criteria.map(\.key))
            }
            bindings.append(QWBoundQuestion(
                questionIndex: qIdx, question: q,
                slotBindings: [QWBoundSlot(slotIndex: slotIdx, questionIndex: qIdx, kind: kind)],
                stateTokenSpans: [], escapedState: escapedState
            ))
        }

        return QWBuiltDecisionBatch(
            request: QWDecisionBackendBatchRequest(sequences: sequences, slots: slots),
            questionBindings: bindings,
            truncatedTokens: truncatedTotal,
            promptTokens: totalPromptTokens,
            defaultScoreReadout: defaultReadout,
            boxEdgesMode: config.boxEdgesMode,
            usesNumbersThousandthsGrid: config.usesNumbersThousandthsGrid
        )
    }

    internal static func buildSequencePairBatch(
        request: QWDecisionRequest,
        stateRaw: String,
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        budgetLimit: Int,
        defaultReadout: QWScoreReadout
    ) throws -> QWBuiltDecisionBatch {
        guard case .pooledClassifier(_, let numLabels, let id2label) = config.head else {
            return try buildTurnVerbalizerBatch(
                request: request, stateRaw: stateRaw, config: config, tokenizer: tokenizer,
                budgetLimit: budgetLimit, scheme: 1, packSet: false, separator: "\n\n",
                defaultReadout: defaultReadout
            )
        }
        let escapedState = QWPlainEscapedText(
            original: stateRaw, escaped: stateRaw,
            boundaryLo: Array(0...stateRaw.utf8.count), boundaryHi: Array(0...stateRaw.utf8.count)
        )
        let clsId = tokenizer.tokenId(for: "[CLS]") ?? tokenizer.tokenId(for: "<s>") ?? 1
        let sepId = tokenizer.tokenId(for: "[SEP]") ?? tokenizer.tokenId(for: "</s>") ?? 2
        let stateIds = try tokenizer.encode(stateRaw, addSpecialTokens: false)

        var sequences: [QWDecisionSequenceInput] = []
        var slots: [QWDecisionSlotTarget] = []
        var bindings: [QWBoundQuestion] = []
        var truncatedTotal = 0
        var totalPromptTokens = 0

        for (qIdx, q) in request.questions.enumerated() {
            let optLabels: [String] = q.kind == .noul ? ["no", "yes"] : q.criteria.map(\.key)
            let optTexts = renderOptionTexts(for: q)
            if numLabels >= 3 {
                var normLabelToId: [String: Int] = [:]
                for (cid, rawName) in id2label {
                    normLabelToId[rawName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()] = cid
                }
                let allowedSorted = id2label.values.sorted()
                var matchedIds: [Int] = []
                for lbl in optLabels {
                    let key = lbl.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    guard let cid = normLabelToId[key] else {
                        throw QWDecisionError.unsupportedLabelSubset(requested: optLabels, allowed: allowedSorted)
                    }
                    matchedIds.append(cid)
                }
                let qIds = try tokenizer.encode(q.instructions, addSpecialTokens: false)
                let room = max(0, budgetLimit - 3 - qIds.count)
                if (request.truncation ?? .error) == .error && stateIds.count > room {
                    throw QWDecisionError.contextOverflow(tokens: 3 + qIds.count + stateIds.count, limit: budgetLimit)
                }
                let kept = Array(stateIds.prefix(room))
                truncatedTotal = max(truncatedTotal, stateIds.count - kept.count)
                let ids = [clsId] + qIds + [sepId] + kept + [sepId]
                let seqIdx = sequences.count
                let slotIdx = slots.count
                sequences.append(QWDecisionSequenceInput(
                    tokenIds: ids, topology: config.topology, chain: q.chain ?? request.chain, images: request.images
                ))
                slots.append(QWDecisionSlotTarget(sequenceIndex: seqIdx, tokenPosition: 0))
                totalPromptTokens += ids.count
                let kind: QWSlotBindingKind = .fixedLabelSubset(labels: optLabels, classIndices: matchedIds)
                bindings.append(QWBoundQuestion(
                    questionIndex: qIdx, question: q,
                    slotBindings: [QWBoundSlot(slotIndex: slotIdx, questionIndex: qIdx, kind: kind)],
                    stateTokenSpans: [], escapedState: escapedState
                ))
            } else {
                // Cross-encoder 1/2-logit reranker K-pair fan-out (ADR-007).
                var qSlotBindings: [QWBoundSlot] = []
                for (k, optText) in optTexts.enumerated() {
                    let pairText = "\(q.instructions)\n\(optText)"
                    let pIds = try tokenizer.encode(pairText, addSpecialTokens: false)
                    let room = max(0, budgetLimit - 3 - pIds.count)
                    if (request.truncation ?? .error) == .error && stateIds.count > room {
                        throw QWDecisionError.contextOverflow(tokens: 3 + pIds.count + stateIds.count, limit: budgetLimit)
                    }
                    let kept = Array(stateIds.prefix(room))
                    truncatedTotal = max(truncatedTotal, stateIds.count - kept.count)
                    let ids = [clsId] + pIds + [sepId] + kept + [sepId]
                    let seqIdx = sequences.count
                    let slotIdx = slots.count
                    sequences.append(QWDecisionSequenceInput(
                        tokenIds: ids, topology: config.topology, pairIndex: k, chain: q.chain ?? request.chain,
                        images: request.images
                    ))
                    slots.append(QWDecisionSlotTarget(sequenceIndex: seqIdx, tokenPosition: 0))
                    totalPromptTokens += ids.count
                    qSlotBindings.append(QWBoundSlot(
                        slotIndex: slotIdx, questionIndex: qIdx,
                        kind: .pairRelevance(optionIndex: k, key: optLabels[k], totalOptions: optTexts.count)
                    ))
                }
                bindings.append(QWBoundQuestion(
                    questionIndex: qIdx, question: q, slotBindings: qSlotBindings,
                    stateTokenSpans: [], escapedState: escapedState
                ))
            }
        }

        return QWBuiltDecisionBatch(
            request: QWDecisionBackendBatchRequest(sequences: sequences, slots: slots),
            questionBindings: bindings,
            truncatedTokens: truncatedTotal,
            promptTokens: totalPromptTokens,
            defaultScoreReadout: defaultReadout,
            boxEdgesMode: config.boxEdgesMode,
            usesNumbersThousandthsGrid: config.usesNumbersThousandthsGrid
        )
    }

    internal static func renderOptionTexts(for q: QWDecisionQuestion) -> [String] {
        switch q.kind {
        case .noul:
            return ["false: no, the statement does not hold", "true: yes, the statement holds"]
        case .score:
            return q.criteria.enumerated().map { idx, c in "level \(idx): \(c.description ?? c.key)" }
        default:
            return q.criteria.map { c in
                guard let d = c.description, !d.isEmpty else { return c.key }
                return "\(c.key): \(d)"
            }
        }
    }

    public static func buildNativeRerankBatch(
        query: String,
        documents: [String],
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        maxContextLength: Int
    ) throws -> (batch: QWDecisionBackendBatchRequest, promptTokens: Int, truncatedTokens: Int) {
        let frameCap: Int
        if case .sequencePair(let maxLen) = config.framing, maxLen > 0 {
            frameCap = min(maxLen, maxContextLength)
        } else {
            frameCap = maxContextLength
        }
        let budgetLimit = max(16, frameCap)
        let clsId = tokenizer.tokenId(for: "[CLS]") ?? tokenizer.tokenId(for: "<s>") ?? 1
        let sepId = tokenizer.tokenId(for: "[SEP]") ?? tokenizer.tokenId(for: "</s>") ?? 2
        let rawQueryIds = try tokenizer.encode(query, addSpecialTokens: false)
        let maxQueryTokens = max(4, min(rawQueryIds.count, (budgetLimit - 4) / 2))
        let queryIds = Array(rawQueryIds.prefix(maxQueryTokens))

        var sequences: [QWDecisionSequenceInput] = []
        var slots: [QWDecisionSlotTarget] = []
        sequences.reserveCapacity(documents.count)
        slots.reserveCapacity(documents.count)
        var totalPromptTokens = 0
        var truncatedTotal = max(0, rawQueryIds.count - queryIds.count)

        for doc in documents {
            let docIds = try tokenizer.encode(doc, addSpecialTokens: false)
            let room = max(0, budgetLimit - 3 - queryIds.count)
            let keptDoc = Array(docIds.prefix(room))
            truncatedTotal += max(0, docIds.count - keptDoc.count)
            let ids = [clsId] + queryIds + [sepId] + keptDoc + [sepId]
            let seqIdx = sequences.count
            sequences.append(QWDecisionSequenceInput(
                tokenIds: ids,
                topology: config.topology,
                pairIndex: 1,
                chain: "rerank"
            ))
            slots.append(QWDecisionSlotTarget(sequenceIndex: seqIdx, tokenPosition: 0))
            totalPromptTokens += ids.count
        }
        return (
            QWDecisionBackendBatchRequest(sequences: sequences, slots: slots),
            totalPromptTokens,
            truncatedTotal
        )
    }
}

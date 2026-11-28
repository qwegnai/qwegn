// QWInkling2Runtime+Echo.swift — Echo prompt-logprob and decision-logit extraction for Inkling-2.
import Foundation
import Metal
import QWCore
import QWDecision
import QWHead

extension QWInkling2Runtime {
    private func resetEphemeralEvalState() {
        session.resetConvStates()
        resetSharedAuxOperators()
    }

    func computeEchoPromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] {
        guard promptTokens.count >= 2 else { return [] }
        for tok in promptTokens where tok < 0 || tok >= descriptor.vocab {
            throw QWEngineDriverError.badToken(tok)
        }
        guard promptTokens.count <= capacity else {
            throw QWEngineError.state("echo prompt length \(promptTokens.count) exceeds capacity \(capacity)")
        }
        return try execLock.withLock {
            resetEphemeralEvalState()
            defer { resetEphemeralEvalState() }
            let sampler = QWSampler(config: sampling)
            let topK = max(0, min(20, topLogprobsCount))
            let V = ink2.unpaddedVocabSize
            let nEval = promptTokens.count - 1
            var out: [QWTokenLogprob] = []
            out.reserveCapacity(nEval)
            let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
            var pos = 0
            while pos < nEval {
                if shouldStop?() == true { break }
                let end = min(nEval, pos + 16)
                let chunk = Array(promptTokens[pos..<end])
                try withGPUTurn {
                    try forwardPass.forward(tokens: chunk, posOffset: pos, runHeadRows: chunk.count, updateState: true)
                }
                withExtendedLifetime(session.logitsF32Buf) {
                    for row in 0..<chunk.count {
                        let rowPtr = logitsPtr.advanced(by: row * V)
                        let buf = UnsafeBufferPointer(start: rowPtr, count: V)
                        out.append(sampler.computeTokenLogprob(
                            logits: buf, chosenToken: promptTokens[pos + row + 1], topK: topK
                        ))
                    }
                }
                pos = end
            }
            return out
        }
    }

    func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan]
    ) throws -> [QWDecisionSlotOutput] {
        guard !tokens.isEmpty else {
            throw QWDecisionError.invalidQuestion("decision sequence must contain at least one token")
        }
        if positions.isEmpty { return [] }
        guard tokens.count <= capacity else {
            throw QWDecisionError.contextOverflow(tokens: tokens.count, limit: capacity)
        }
        for tok in tokens where tok < 0 || tok >= descriptor.vocab {
            throw QWEngineDriverError.badToken(tok)
        }
        for pos in positions where pos < 0 || pos >= tokens.count {
            throw QWDecisionError.invalidQuestion("slot position \(pos) outside sequence 0..<\(tokens.count)")
        }
        let vocabCount = ink2.unpaddedVocabSize
        if !candidateTokenIds.isEmpty,
           !candidateTokenIds.contains(where: { $0 >= 0 && $0 < vocabCount }) {
            return positions.map { _ in QWDecisionSlotOutput(tokenLogits: [:]) }
        }

        return try execLock.withLock {
            resetEphemeralEvalState()
            defer { resetEphemeralEvalState() }

            let sortedPositions = Array(Set(positions)).sorted()
            let maxChunk = max(1, session.maxChunk)
            var capturedByPosition: [Int: QWDecisionSlotOutput] = [:]
            capturedByPosition.reserveCapacity(sortedPositions.count)
            var curPos = 0

            for targetPos in sortedPositions {
                while curPos <= targetPos {
                    let remaining = (targetPos + 1) - curPos
                    let chunkCount = min(maxChunk, remaining)
                    let nextEnd = curPos + chunkCount
                    let isTargetChunk = (nextEnd == targetPos + 1)
                    let chunk = Array(tokens[curPos..<nextEnd])
                    try withGPUTurn {
                        try forwardPass.forward(
                            tokens: chunk,
                            posOffset: curPos,
                            runHeadRows: isTargetChunk ? 1 : 0,
                            greedyHead: false,
                            updateState: true,
                            visualSpans: visualSpans
                        )
                    }
                    curPos = nextEnd
                }
                let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
                let slotOut = withExtendedLifetime(session.logitsF32Buf) {
                    QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                        logitsRow: logitsPtr,
                        vocab: vocabCount,
                        candidateTokenIds: candidateTokenIds
                    )
                }
                capturedByPosition[targetPos] = slotOut
            }
            return positions.map { capturedByPosition[$0] ?? QWDecisionSlotOutput() }
        }
    }
}

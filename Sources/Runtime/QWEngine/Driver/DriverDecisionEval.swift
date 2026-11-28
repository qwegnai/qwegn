// DriverDecisionEval.swift — Single-pass decision logit extraction on QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWHAL
import QWHead
import QWKV

extension QWEngineDriver {
    /// Evaluates prompt `tokens` up to every target `position` using an ephemeral session
    /// and extracts either sparse candidate-token logits or full vocabulary logits per slot.
    public func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan] = []
    ) throws -> [QWDecisionSlotOutput] {
        try trunkExecutor.evaluateDecisionLogits(
            tokens: tokens,
            positions: positions,
            candidateTokenIds: candidateTokenIds,
            visualSpans: visualSpans
        )
    }

    func evaluateStandardDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan] = []
    ) throws -> [QWDecisionSlotOutput] {
        guard !tokens.isEmpty else {
            throw QWDecisionError.invalidQuestion("decision sequence must contain at least one token")
        }
        if positions.isEmpty { return [] }
        guard tokens.count <= ropeRows else {
            throw QWDecisionError.contextOverflow(tokens: tokens.count, limit: ropeRows)
        }
        for tok in tokens where tok < 0 || tok >= geometry.vocab {
            throw QWEngineDriverError.badToken(tok)
        }
        for pos in positions where pos < 0 || pos >= tokens.count {
            throw QWDecisionError.invalidQuestion("slot position \(pos) outside sequence 0..<\(tokens.count)")
        }

        let sortedPositions = Array(Set(positions)).sorted()
        let session = try makeSession(promptTokens: tokens.count)
        defer { finishSession(session) }

        return try requestGate.withRequest(flight: flight) { permit in
            let savedVisualSpans = activeVisualSpans
            let savedCollectLP = activeCollectLogprobs
            let savedConstraint = activeConstraint
            activeVisualSpans = visualSpans
            activeCollectLogprobs = true
            activeConstraint = nil
            defer {
                activeVisualSpans = savedVisualSpans
                activeCollectLogprobs = savedCollectLP
                activeConstraint = savedConstraint
            }

            let validCandidateIds = Array(
                Set(candidateTokenIds.filter { $0 >= 0 && $0 < geometry.vocab })
            ).sorted()
            if !candidateTokenIds.isEmpty && validCandidateIds.isEmpty {
                return positions.map { _ in QWDecisionSlotOutput(tokenLogits: [:]) }
            }

            let bf16HeadTensor: QWTensorValue? = if case .bf16(let lmHead) = headStorage {
                lmHead
            } else {
                nil
            }
            let useSparseBF16Head = bf16HeadTensor != nil
                && !validCandidateIds.isEmpty
                && validCandidateIds.count <= 256
                && (!hal.weightBasis.rotatesActivations || geometry.hidden % 32 == 0)

            var sparseOp: QWDecisionHeadOperator?
            var sparseCandTensor: QWTensorValue?
            var sparseOutTensor: QWTensorValue?
            if useSparseBF16Head {
                sparseOp = try QWDecisionHeadOperator(hal: hal)
                let candBytes = max(16, validCandidateIds.count * MemoryLayout<Int32>.stride)
                let outBytes = max(16, validCandidateIds.count * MemoryLayout<Float>.stride)
                guard let cBuf = hal.device.makeBuffer(length: candBytes, options: .storageModeShared),
                      let oBuf = hal.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
                    throw QWEngineDriverError.state("failed to allocate sparse BF16 decision buffers")
                }
                let cPtr = cBuf.contents().assumingMemoryBound(to: Int32.self)
                for (idx, tid) in validCandidateIds.enumerated() {
                    cPtr[idx] = Int32(tid)
                }
                sparseCandTensor = QWTensorValue(
                    region: QWMRegion(buffer: cBuf, base: 0, length: candBytes),
                    offset: 0,
                    shape: QWTensorShape([validCandidateIds.count], .i32)
                )
                sparseOutTensor = QWTensorValue(
                    region: QWMRegion(buffer: oBuf, base: 0, length: outBytes),
                    offset: 0,
                    shape: QWTensorShape([1, validCandidateIds.count], .f32)
                )
            }

            func extractFullSlotOutput() -> QWDecisionSlotOutput {
                let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    .assumingMemoryBound(to: Float.self)
                return QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                    logitsRow: rawLogits,
                    vocab: geometry.vocab,
                    candidateTokenIds: candidateTokenIds
                )
            }

            func extractSparseBF16Output(lastChunkCount: Int) throws -> QWDecisionSlotOutput {
                guard let lmHeadBF16 = bf16HeadTensor,
                      let op = sparseOp,
                      let candTensor = sparseCandTensor,
                      let outTensor = sparseOutTensor else {
                    return extractFullSlotOutput()
                }
                let rowIdx = max(0, lastChunkCount - 1)
                let lastLn1 = psc.ln1.slice(
                    byteOffset: rowIdx * geometry.hidden * 2,
                    count: geometry.hidden,
                    dtype: .f16
                )
                let arb = gpuTurnArbitrator
                arb?.acquirePrimaryStepSync()
                defer { arb?.releasePrimaryStep() }
                guard let (cb, enc) = hal.begin() else {
                    throw QWEngineDriverError.state("failed to open Metal command buffer for sparse BF16 head")
                }
                if hal.weightBasis.rotatesActivations {
                    encodeFwht32Fast(enc, x: lastLn1)
                    enc.memoryBarrier(scope: .buffers)
                }
                try op.encodeSparseVocabRowsBF16(
                    enc,
                    hiddenState: lastLn1,
                    lmHeadBF16: lmHeadBF16,
                    candidateIds: candTensor,
                    out: outTensor,
                    rows: 1,
                    candCount: validCandidateIds.count,
                    hidden: geometry.hidden,
                    vocabSize: geometry.vocab
                )
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                guard cb.status == .completed else {
                    throw QWEngineDriverError.state("sparse BF16 decision head failed (status=\(cb.status.rawValue))")
                }
                let ptr = outTensor.region.pointer.advanced(by: outTensor.offset)
                    .assumingMemoryBound(to: Float.self)
                var sparse: [Int: Float] = [:]
                sparse.reserveCapacity(validCandidateIds.count)
                for (idx, tid) in validCandidateIds.enumerated() {
                    sparse[tid] = ptr[idx]
                }
                return QWDecisionSlotOutput(tokenLogits: sparse)
            }

            var capturedByPosition: [Int: QWDecisionSlotOutput] = [:]
            capturedByPosition.reserveCapacity(sortedPositions.count)
            let maxStep = max(1, psc.maxChunk)

            for targetPos in sortedPositions {
                var lastChunkCount = 1
                while session.position <= targetPos {
                    let remaining = (targetPos + 1) - session.position
                    let chunkCount = min(maxStep, remaining)
                    let isTargetChunk = (session.position + chunkCount == targetPos + 1)
                    _ = try prefillChunk(
                        session,
                        tokens: tokens,
                        start: session.position,
                        count: chunkCount,
                        runsHead: isTargetChunk && !useSparseBF16Head,
                        mode: gdnPrefillMode,
                        permit: permit
                    )
                    lastChunkCount = chunkCount
                }
                if useSparseBF16Head {
                    capturedByPosition[targetPos] = try extractSparseBF16Output(lastChunkCount: lastChunkCount)
                } else {
                    capturedByPosition[targetPos] = extractFullSlotOutput()
                }
            }

            return positions.map { capturedByPosition[$0] ?? QWDecisionSlotOutput() }
        }
    }
}

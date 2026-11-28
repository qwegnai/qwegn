// QWDFlashDrafter+Stochastic.swift — Stochastic Gumbel proposal selection for DFlash 2.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHead

extension QWDFlashDrafter {
    static func buildSparseDistribution(
        vocab: Int,
        candidates: [(id: Int, score: Double)],
        temperature: Float
    ) throws -> QWSamplingDistribution {
        let temp = max(Double(temperature), 1e-6)
        let valid = candidates.compactMap { entry -> (id: Int, score: Double)? in
            guard (0..<vocab).contains(entry.id), entry.id != 0x7fff_ffff else { return nil }
            let score = entry.score.isFinite ? max(entry.score, -1e30) : -1e30
            return (entry.id, score)
        }
        guard let maxScore = valid.map(\.score).max() else {
            return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: [0], weights: [1.0])
        }
        var ids: [Int] = []
        var weights: [Double] = []
        for entry in valid {
            let w = exp((entry.score - maxScore) / temp)
            if w.isFinite && w > 0 {
                ids.append(entry.id)
                weights.append(w)
            }
        }
        if ids.isEmpty {
            ids = [valid[0].id]
            weights = [1.0]
        }
        return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: ids, weights: weights)
    }

    func selectStochastic(
        temperature: Float,
        stream: QWMTPRandomStream,
        anchorToken: Int,
        draftCount: Int,
        topK: Int,
        validVocab: Int? = nil
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        guard let hiddenProjOut, let predecessorCodebook, let successorCodebook else {
            throw QWDFlashError.state("QWDFlashDrafter: missing CandidateSelector buffers")
        }
        let R = config.selectorRank
        let V = config.vocabSize
        let scoreVocab = min(V, max(1, validVocab ?? V))
        let cap = config.finalLogitSoftcapping
        let softcapMode = config.selectorSoftcapMode
        @inline(__always) func applySoftcap(_ x: Float) -> Float {
            guard cap > 0 else { return x }
            return cap * tanhf(min(15.0, max(-15.0, x / cap)))
        }
        let hPtr = hiddenProjOut.region.buffer.contents()
            .advanced(by: hiddenProjOut.region.base + hiddenProjOut.offset)
            .assumingMemoryBound(to: Float.self)
        let uPtr = unaryOut.region.buffer.contents()
            .advanced(by: unaryOut.region.base + unaryOut.offset)
            .assumingMemoryBound(to: Float.self)
        let cPtr = candidatesOut.region.buffer.contents()
            .advanced(by: candidatesOut.region.base + candidatesOut.offset)
            .assumingMemoryBound(to: Int32.self)
        let pBook = predecessorCodebook.region.buffer.contents()
            .advanced(by: predecessorCodebook.region.base + predecessorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)
        let sBook = successorCodebook.region.buffer.contents()
            .advanced(by: successorCodebook.region.base + successorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)

        func bf16(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }

        var pred = Self.clampToken(anchorToken, vocab: scoreVocab)
        var drafts: [Int] = []
        var proposals: [QWMTPGumbelProposal] = []
        for pos in 0..<draftCount {
            var scored: [(id: Int, score: Double)] = []
            scored.reserveCapacity(topK)
            let hBase = pos * R
            let pBase = pred * R
            for k in 0..<topK {
                let rawCand = Int(cPtr[pos * 16 + k])
                guard (0..<scoreVocab).contains(rawCand), rawCand != 0x7fff_ffff else { continue }
                let sBase = rawCand * R
                var edge: Float = 0
                for r in 0..<R {
                    edge += bf16(pBook[pBase + r]) * hPtr[hBase + r] * bf16(sBook[sBase + r])
                }
                var u = uPtr[pos * 16 + k]
                if softcapMode == .beforeCorrection && cap > 0 {
                    u = applySoftcap(u)
                }
                var scoreF = u + edge
                if softcapMode == .afterCorrection && cap > 0 {
                    scoreF = applySoftcap(scoreF)
                }
                scored.append((rawCand, Double(scoreF)))
            }
            let dist = try Self.buildSparseDistribution(vocab: scoreVocab, candidates: scored, temperature: temperature)
            let proposal = try stream.sampleGumbelProposal(dist)
            let tok = Self.clampToken(proposal.token, vocab: scoreVocab)
            pred = tok
            drafts.append(tok)
            proposals.append(proposal)
        }
        return (drafts, proposals)
    }
}

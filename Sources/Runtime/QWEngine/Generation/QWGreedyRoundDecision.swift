import QWCore

/// A validated output prefix. Callbacks may shorten it further before GPU
/// resolution; do not publish tokens beyond the chosen retained boundary.
struct QWGreedyRoundDecision: Equatable {
    let acceptedDrafts: Int
    let tokens: [Int]
    let stopped: Bool
    let stoppedToken: Int?

    init(acceptedDrafts: Int, tokens: [Int], stopped: Bool, stoppedToken: Int? = nil) {
        self.acceptedDrafts = acceptedDrafts
        self.tokens = tokens
        self.stopped = stopped
        self.stoppedToken = stoppedToken
    }

    var retainedRows: Int { tokens.count }
    var pendingToken: Int? { tokens.last }

    static func make(geometry: QWModelGeometry, drafts: [Int], targetTokens: [Int], maxNew: Int,
                     stopIDs: Set<Int> = []) throws -> QWGreedyRoundDecision {
        guard (1...15).contains(drafts.count), targetTokens.count == drafts.count + 1, maxNew >= 0,
              drafts.allSatisfy({ (0..<geometry.vocab).contains($0) }),
              targetTokens.allSatisfy({ (0..<geometry.vocab).contains($0) }),
              stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid greedy round tokens, row count, or output budget")
        }
        var accepted = 0
        while accepted < drafts.count && drafts[accepted] == targetTokens[accepted] { accepted += 1 }
        let candidates = Array(drafts.prefix(accepted)) + [targetTokens[accepted]]
        var emitted: [Int] = []
        var stoppedToken: Int?
        for token in candidates.prefix(maxNew) {
            if stopIDs.contains(token) { stoppedToken = token; break }
            emitted.append(token)
        }
        return QWGreedyRoundDecision(acceptedDrafts: accepted, tokens: emitted,
                                     stopped: stoppedToken != nil, stoppedToken: stoppedToken)
    }

    /// Callback cancellation occurs after the delivered token, so that token
    /// stays pending and its preceding verify input remains consumed.
    func truncating(to count: Int) throws -> QWGreedyRoundDecision {
        guard (0...tokens.count).contains(count) else {
            throw QWEngineDriverError.state("greedy callback prefix is invalid")
        }
        return QWGreedyRoundDecision(acceptedDrafts: acceptedDrafts,
                                     tokens: Array(tokens.prefix(count)), stopped: stopped,
                                     stoppedToken: stoppedToken)
    }

    /// Draft execution writes N pairs, not the all-accept bonus pair. Truncated
    /// outputs must roll pairs back to the same consumed prefix as the trunk.
    func retainedPairRows(drafts count: Int) throws -> Int {
        guard (1...15).contains(count), acceptedDrafts <= count else {
            throw QWEngineDriverError.state("greedy pair retention draft count is invalid")
        }
        return min(retainedRows, count)
    }
}

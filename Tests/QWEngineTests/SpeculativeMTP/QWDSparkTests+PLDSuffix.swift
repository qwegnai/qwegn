// QWDSparkTests+PLDSuffix.swift — Test-only PLD suffix helper relocated from QWDSparkDrafter+Encode.

import Foundation
import QWCore
@testable import QWEngine

extension QWDSparkDrafter {
    /// Extends full-block neural drafts past `neuralMaxDrafts` using periodic or >=6-gram context PLD.
    static func appendPLDSuffixIfEligible(
        drafts: [Int],
        history: [Int],
        extraPending: Int?,
        requestedDrafts: Int,
        neuralMaxDrafts: Int,
        schedulerEnabled: Bool,
        lastConfidence: Float?,
        minTailConfidence: Float = 0.70
    ) -> [Int] {
        let cappedTotal = min(15, requestedDrafts)
        guard cappedTotal > neuralMaxDrafts,
              drafts.count == neuralMaxDrafts,
              !drafts.isEmpty else {
            return drafts
        }
        if schedulerEnabled {
            guard let tail = lastConfidence, tail >= minTailConfidence else { return drafts }
        }
        let remaining = cappedTotal - drafts.count
        guard remaining > 0 else { return drafts }
        let histTailCount = min(history.count, 64)
        var periodicWindow: [Int] = []
        periodicWindow.reserveCapacity(histTailCount + (extraPending != nil ? 1 : 0) + drafts.count)
        periodicWindow.append(contentsOf: history.suffix(histTailCount))
        if let extraPending { periodicWindow.append(extraPending) }
        periodicWindow.append(contentsOf: drafts)
        if let pld = QWContextCopyDrafter.periodicDrafts(
            history: periodicWindow, extraToken: nil, maxCount: remaining, minRepeats: 3
        ), !pld.isEmpty {
            return drafts + pld.prefix(remaining)
        }
        if let match = QWContextCopyDrafter.drafts(
            history: history, extraToken: extraPending, trailingTokens: drafts,
            maxCount: remaining, minMatchLen: 6
        ), !match.tokens.isEmpty {
            return drafts + match.tokens.prefix(remaining)
        }
        return drafts
    }
}

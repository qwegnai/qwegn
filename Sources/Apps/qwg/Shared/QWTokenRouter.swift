// QWTokenRouter.swift — Where the thinking ends and the answer begins.
//
// One router, one splitter, one note: the CLI and the REPL must not
// each find the boundary their own way. Two copies of the boundary are
// two turns that disagree about where the thinking stopped.

import Foundation
import QWToken

/// The two halves of a turn, and whether the turn had both.
struct QWTokenSplit {
    /// The tokens before the end-of-thinking marker.
    var thinking: [Int]
    /// The tokens after it. The whole turn when the marker never came.
    var answer: [Int]
    /// Whether the turn carried the end-of-thinking marker.
    var hasThinkingEnd: Bool
}

/// Finds where the thinking ends.
enum QWTokenRouter {
    /// Splits a turn at the end-of-thinking marker. A turn without the
    /// marker is all answer: the thinking never ended.
    static func split(_ ids: [Int], endThinkTokenId: Int) -> QWTokenSplit {
        guard let end = ids.firstIndex(of: endThinkTokenId) else {
            return QWTokenSplit(thinking: [], answer: ids, hasThinkingEnd: false)
        }
        return QWTokenSplit(thinking: Array(ids[..<end]),
                            answer: Array(ids[(end + 1)...]),
                            hasThinkingEnd: true)
    }

    /// The text of a run of tokens, trimmed. Empty when nothing decoded.
    static func text(_ ids: [Int], tok: QWTokenizer) -> String {
        guard !ids.isEmpty else { return "" }
        let decoded = (try? tok.decode(ids, skipSpecialTokens: true)) ?? ""
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The note a consumer writes when the thinking budget ran out.
    static let thinkingBudgetNote =
        "thinking budget exhausted before answer was reached; increase --max-tokens or use --no-thinking"
}

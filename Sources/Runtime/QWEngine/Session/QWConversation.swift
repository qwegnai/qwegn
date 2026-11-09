import Foundation
import QWCore

/// Single-conversation state owner. Calls must be serialized by the owner.
/// KV, recurrent states, rings, and position stay in one retained session.
public final class QWConversation {
    private let engine: any QWSessionInferenceEngine
    private var session: QWEngineSession?
    private var consumed: [Int] = []
    private var generationWasSpeculative: Bool?
    public private(set) var reusedTokens: Int = 0
    public var consumedTokenCount: Int { consumed.count }

    public init(engine: any QWSessionInferenceEngine) { self.engine = engine }
    deinit { reset() }

    public func reset() {
        if let session { engine.finishSession(session) }
        session = nil
        consumed.removeAll(keepingCapacity: false)
        generationWasSpeculative = nil
        reusedTokens = 0
    }

    /// Equal or shorter prompts require a fresh head result, not stale logits.
    static func reusableCount(consumed: [Int], prompt: [Int], position: Int) -> Int {
        guard !consumed.isEmpty, position == consumed.count,
              prompt.count > consumed.count, prompt.starts(with: consumed) else { return 0 }
        return consumed.count
    }

    static func committedTokens(prompt: [Int], output: [Int], position: Int) throws -> [Int] {
        let decoded = position - prompt.count
        guard decoded >= 0, decoded <= output.count else {
            throw QWEngineDriverError.state("conversation consumed-token boundary is invalid")
        }
        return prompt + output.prefix(decoded)
    }

    /// Supply the complete rendered prompt. Speculative callbacks receive only
    /// the initial token or tokens from committed verification rounds; callback
    /// cancellation takes effect after the current committed round.
    public func generate(prompt: [Int], maxNew: Int, stopIds: Set<Int> = [],
                         speculative: Bool = false,
                         onToken: ((Int) -> Bool)? = nil,
                         onStopToken: ((Int) -> Void)? = nil) throws -> [Int] {
        guard !prompt.isEmpty, maxNew > 0,
              prompt.allSatisfy({ $0 >= 0 && $0 < engine.plan.vocab }) else {
            reset()
            throw QWEngineDriverError.state("conversation requires valid prompt tokens and maxNew > 0")
        }
        if let generationWasSpeculative, generationWasSpeculative != speculative {
            reset()
        }
        let reuse = session.map {
            Self.reusableCount(consumed: consumed, prompt: prompt, position: $0.position)
        } ?? 0
        if reuse == 0 { reset() }
        do {
            if session == nil { session = try engine.makeSession(promptTokens: prompt.count) }
            guard let session else { throw QWEngineDriverError.state("conversation session missing") }
            reusedTokens = reuse
            let suffix = Array(prompt.dropFirst(reuse))
            let output: [Int]
            if speculative {
                output = try engine.generateExperimentalGreedy(session: session, prompt: suffix,
                                                               maxNew: maxNew, stopIds: stopIds,
                                                               onToken: onToken, onStopToken: onStopToken)
            } else {
                output = try engine.generate(session: session, prompt: suffix,
                                             maxNew: maxNew, stopIds: stopIds, onToken: onToken)
            }
            consumed = try Self.committedTokens(prompt: prompt, output: output, position: session.position)
            generationWasSpeculative = speculative
            return output
        } catch {
            reset()
            throw error
        }
    }
}

// QWTokenConstraint.swift — request-local token constraints for selection.
//
// A constraint restricts which token ids the sampler may select at each step
// (for example a JSON-schema grammar). It is owned by exactly one request and
// advanced once per selected token; it is never shared across requests and is
// not part of any model or KV state.

import Foundation

public protocol QWTokenConstraint: AnyObject, Sendable {
    /// The ascending, de-duplicated token ids selectable at the current step,
    /// or nil when the constraint does not restrict this step.
    /// An empty array means the constraint has no valid continuation.
    func allowedTokens() throws -> [Int]?

    /// Advances the constraint past the selected token.
    func accept(_ token: Int) throws

    /// Whether exact speculative decoding may verify drafts row-by-row with this constraint.
    var allowsSpeculation: Bool { get }

    /// Whether the constraint currently permits any token (`allowedTokens() == nil`)
    /// up to `speculationBoundaryToken`.
    var isUnconstrainedNow: Bool { get }

    /// Token ID at which an unconstrained speculative round must stop so subsequent
    /// tokens can be verified with full logits.
    var speculationBoundaryToken: Int? { get }
}

extension QWTokenConstraint {
    public var allowsSpeculation: Bool { false }
    public var isUnconstrainedNow: Bool { false }
    public var speculationBoundaryToken: Int? { nil }
}

public enum QWTokenConstraintError: Error, CustomStringConvertible, Equatable {
    case emptySupport
    case rejectedToken(Int)

    public var description: String {
        switch self {
        case .emptySupport:
            return "token constraint has no valid continuation"
        case .rejectedToken(let token):
            return "token constraint rejected token \(token)"
        }
    }
}

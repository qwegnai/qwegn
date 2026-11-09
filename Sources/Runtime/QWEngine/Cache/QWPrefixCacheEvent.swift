// QWPrefixCacheEvent.swift — What a prefix cache reports to its observer.
// Module-level on purpose: the cache belongs to the module, not to the
// dense driver, so both families can raise the same events.

import Foundation

public enum QWPrefixCacheIOOperation: String, Sendable {
    case write
    case restore
}

/// One thing the module's prefix cache did.
public enum QWPrefixCacheEvent: Sendable {
    /// The cache served a prefix from a tier.
    case hit(tier: String, matchedTokens: Int, totalTokens: Int)
    /// The cache had nothing for this prompt.
    case miss(totalTokens: Int)
    /// The cache kept a prefix and charged it against its warm limit.
    case retained(tokens: Int, chargedBytes: Int, storeTotalBytes: Int, warmLimitBytes: Int)
    /// The prefix was too large to keep.
    case skippedOversize(tokens: Int, charge: Int, warmLimitBytes: Int)
    /// The cache dropped an entry to make room.
    case evicted(tokens: Int, freedBytes: Int)
    /// An SSD operation failed; generation may continue without the entry.
    case ioFailure(operation: QWPrefixCacheIOOperation, message: String)
}

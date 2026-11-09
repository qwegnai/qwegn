// QWPrefillLimits.swift — Prefill chunk size shared by every engine path.
// 512 rows is the measured sweet spot (spec 37, Addendum A/B); the setting is
// user-configurable via settings.json and the menu bar, clamped everywhere to
// minRows...maxRows rounded to rowStep.

public enum QWPrefillLimits {
    /// Prefill chunk rows (tokens per prefill command buffer). Values are
    /// clamped to minRows...maxRows rounded down to a multiple of rowStep.
    public static let chunkRows = 512
    /// Clamp bounds shared by the resolver and the menu-bar control.
    public static let minRows = 32
    public static let maxRows = 8192
    public static let rowStep = 32

    /// Clamps a requested chunk size the way the resolver does.
    public static func clampedRows(_ n: Int) -> Int {
        max(minRows, min(maxRows, (n / rowStep) * rowStep))
    }

    /// Computes a 64-row-aligned balanced prefill chunk size for a prompt of `totalTokens` tokens
    /// up to `maxChunkRows`, distributing tokens evenly across the minimum number of chunks
    /// `ceil(totalTokens / maxChunkRows)` to eliminate small tail chunks while minimizing scratch rows.
    public static func balancedChunkRows(forTotalTokens totalTokens: Int, maxChunkRows: Int = maxRows) -> Int {
        guard totalTokens > 1 else { return 1 }
        let cap = max(minRows, maxChunkRows)
        guard totalTokens > cap else { return totalTokens }
        let numChunks = max(1, (totalTokens + cap - 1) / cap)
        let balanced = (totalTokens + numChunks - 1) / numChunks
        let aligned = ((balanced + 63) / 64) * 64
        return min(cap, max(64, aligned))
    }
}

// QWEngineLifecycleActor+Sessions.swift — Non-mutating resident prefix probe for QWSessionTree (G4).

import Foundation
import QWEngine

extension QWEngineLifecycleActor {
    /// Probes whether `tokens` (plus optional `cacheSalt`) are currently resident in the loaded model's prefix cache.
    public func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool {
        guard !tokens.isEmpty, case .loaded(let primary) = state else {
            return false
        }
        return primary.modelBackend.hasCachedPrefix(tokens: tokens, cacheSalt: cacheSalt)
    }
}

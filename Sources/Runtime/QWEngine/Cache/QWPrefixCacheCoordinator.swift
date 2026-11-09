// QWPrefixCacheCoordinator.swift — Unified Three-Tier Prefix Cache Coordinator for QWEngine.
// Coordinates Tier-0 (in-place GPU KV continuation watermark), Tier-1 (slot-pinned warm RAM LRU),
// and Tier-2 (cold SSD store) across all model families.

import Foundation
import QWCore
import QWKV
import QWState

/// Unified three-tier prefix cache coordinator.
public final class QWPrefixCacheCoordinator: @unchecked Sendable {
    public let tiers: QWPrefixCacheTiers
    public var tier0Watermark = QWResidentKVWatermarkTracker<Int>()
    let warmStore: QWEnginePrefixStore<QWEngineCachedPrefix>?
    let diskManager: QWCacheDiskManager?

    init(config: QWCacheConfig, vocab: Int) throws {
        self.tiers = QWPrefixCacheTiers(warmByteLimit: config.warmByteLimit, ssdByteLimit: config.ssdByteLimit)
        if tiers.isEnabled {
            self.warmStore = try QWEnginePrefixStore<QWEngineCachedPrefix>(
                byteLimit: tiers.warmByteLimit,
                entryLimit: max(config.entryLimit, config.maxConcurrentSessions * 2),
                vocab: vocab,
                maxSlots: config.maxConcurrentSessions
            )
            self.diskManager = tiers.ssdByteLimit > 0 ? QWCacheDiskManager(config: config) : nil
        } else {
            self.warmStore = nil
            self.diskManager = nil
        }
    }

    public var isEnabled: Bool { tiers.isEnabled }

    public var warmState: (entries: Int, bytes: Int)? {
        guard let warmStore else { return nil }
        return (warmStore.count, warmStore.chargedBytes)
    }

    public var diskState: (entries: Int, bytes: Int)? {
        guard let diskManager, tiers.ssdByteLimit > 0 else { return nil }
        return (diskManager.entryCount, diskManager.chargedBytes)
    }
}

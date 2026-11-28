// QWPrefixCacheCoordinator.swift — Unified Three-Tier Prefix Cache Coordinator for QWEngine.
// Coordinates Tier-0 (in-place GPU KV continuation watermark), Tier-1 (slot-pinned warm RAM LRU),
// and Tier-2 (cold SSD store) across all model families.

import Foundation
import QWCore
import QWKV

enum QWCacheIO {
    static func attempt<T>(
        _ operation: QWPrefixCacheIOOperation,
        onEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?,
        _ body: () throws -> T
    ) -> T? {
        do {
            return try body()
        } catch {
            let message = String(describing: error)
            if let onEvent {
                onEvent(.ioFailure(operation: operation, message: message))
            } else {
                NSLog("Prefix cache SSD %@ failed: %@", operation.rawValue, message)
            }
            return nil
        }
    }
}

/// Unified three-tier prefix cache coordinator.
public final class QWPrefixCacheCoordinator: @unchecked Sendable {
    static let maxOpportunisticColdWriteBytes = 256 << 20
    public let tiers: QWPrefixCacheTiers
    let warmStore: QWEnginePrefixStore<QWEngineCachedPrefix>?
    let diskManager: QWCacheDiskManager?

    init(config: QWCacheConfig, vocab: Int) throws {
        self.tiers = QWPrefixCacheTiers(warmByteLimit: config.warmByteLimit, ssdByteLimit: config.ssdByteLimit)
        if tiers.isEnabled {
            self.warmStore = try QWEnginePrefixStore<QWEngineCachedPrefix>(
                byteLimit: tiers.warmByteLimit,
                entryLimit: max(config.entryLimit, config.maxConcurrentSessions * 4, 32),
                vocab: vocab,
                maxSlots: config.maxConcurrentSessions,
                extractPhysicalPages: { QWPrefixPageLedger<QWEnginePrefixStore<QWEngineCachedPrefix>.Key>.extractUniquePages(from: $0) }
            )
            self.diskManager = tiers.ssdByteLimit > 0 ? QWCacheDiskManager(config: config) : nil
        } else {
            self.warmStore = nil
            self.diskManager = nil
        }
    }

    public var isEnabled: Bool { tiers.isEnabled }

    public var diskState: (entries: Int, bytes: Int)? {
        guard let diskManager, tiers.ssdByteLimit > 0 else { return nil }
        return (diskManager.entryCount, diskManager.chargedBytes)
    }

    static func scheduleColdSpill(
        key: QWEnginePrefixStore<QWEngineCachedPrefix>.Key,
        value: QWEngineCachedPrefix,
        allowsSuffix: Bool,
        warmHasSharedBase: Bool,
        diskManager: QWCacheDiskManager?,
        coldWriteController: QWAsyncColdWriteController,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?,
        onEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    ) {
        onEvent?(.evicted(tokens: key.tokens.count, freedBytes: value.memoryBytes))
        guard let dm = diskManager,
              QWTwoTierPrefixCache.shouldSpillToCold(
                  isSharedBase: false,
                  warmHasSharedBase: warmHasSharedBase,
                  ssdByteLimit: dm.config.ssdByteLimit,
                  entryBytes: value.memoryBytes,
                  maxSpillBytes: Self.maxOpportunisticColdWriteBytes
              ) else { return }
        let writeBytes = value.memoryBytes
        guard coldWriteController.reserve(bytes: writeBytes) else { return }
        do {
            let planes = try copyPlaneBuffers(prefix: value.prefix, kvCache: kvCache, mtpCache: mtpCache)
            let capturedState = value.prefix.state
            let capturedLogits = value.logits
            let capturedNextToken = value.nextToken
            coldWriteController.submitReserved(bytes: writeBytes) {
                _ = QWCacheIO.attempt(.write, onEvent: onEvent) {
                    try dm.write(
                        tokens: key.tokens, mode: key.mode, nextToken: capturedNextToken,
                        state: capturedState, logits: capturedLogits,
                        trunkPlanes: planes.trunk, mtpPlanes: planes.mtp,
                        allowsSuffix: allowsSuffix
                    )
                }
            }
        } catch {
            coldWriteController.cancelReservation(bytes: writeBytes)
            onEvent?(.ioFailure(operation: .write, message: String(describing: error)))
        }
    }

    static func lookupTwoTier(
        lookupTokens: [Int],
        policyKey: String,
        store: QWEnginePrefixStore<QWEngineCachedPrefix>,
        diskManager: QWCacheDiskManager?,
        onEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?,
        restoreFromDisk: (QWCacheDiskManager.DiskHit) throws -> QWEngineCachedPrefix?
    ) throws -> (hit: QWEnginePrefixStore<QWEngineCachedPrefix>.Hit, tier: String)? {
        guard !lookupTokens.isEmpty else { return nil }
        if let warmHit = try store.lookup(tokens: lookupTokens, mode: policyKey) {
            return (warmHit, QWPrefixCacheTier.warmRAM.rawValue)
        }
        if let diskManager,
           let diskHit = diskManager.lookup(tokens: lookupTokens, mode: policyKey),
           diskHit.matchedTokens >= 256 || (!diskManager.config.wipeOnStart && lookupTokens.count <= 16) {
            if let result = QWCacheIO.attempt(.restore, onEvent: onEvent, { try restoreFromDisk(diskHit) }),
               let restored = result {
                return (
                    QWEnginePrefixStore.Hit(value: restored, tokenCount: diskHit.matchedTokens),
                    QWPrefixCacheTier.coldSSD.rawValue
                )
            }
        }
        return nil
    }

    static func copyPlaneBuffers(
        prefix: QWEngineSharedPrefix,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?
    ) throws -> (trunk: [Data], mtp: [Data]) {
        func extract(_ cache: QWKVCache, _ handle: QWKVSharedPrefix) throws -> [Data] {
            var planes: [Data] = []
            try cache.withSharedPrefixPagePointers(handle) { planeIdx, _, ptr, byteCount in
                while planes.count <= planeIdx { planes.append(Data()) }
                planes[planeIdx].append(ptr.assumingMemoryBound(to: UInt8.self), count: byteCount)
            }
            return planes
        }
        let trunkPlanes = try extract(kvCache, prefix.trunk)
        let mtpPlanes = try prefix.mtp.flatMap { p in try mtpCache.map { try extract($0, p) } } ?? []
        return (trunkPlanes, mtpPlanes)
    }

    static func validateCachedValue(
        _ value: QWEngineCachedPrefix,
        matchedTokens: Int,
        logitsBytes: Int,
        vocab: Int
    ) -> Bool {
        value.prefix.state.stochastic == nil
            && value.prefix.position == matchedTokens
            && value.logits.count == logitsBytes
            && (0..<vocab).contains(value.nextToken)
    }

    static func validateStandaloneRead(
        tokens: [Int],
        mode: String,
        statePosition: Int,
        logitsCount: Int,
        logitsBytes: Int,
        nextToken: Int,
        vocab: Int,
        hasReadMTP: Bool,
        hasEngineMTP: Bool
    ) -> Bool {
        !tokens.isEmpty
            && !mode.isEmpty
            && statePosition == tokens.count
            && logitsCount == logitsBytes
            && (0..<vocab).contains(nextToken)
            && hasReadMTP == hasEngineMTP
    }

    static func buildMetricsSnapshot(
        promptCount: Int,
        output: QWGreedyGenerationLoop.Result,
        prefillSec: Double,
        decodeSec: Double,
        firstTokenSec: Double,
        speculative: Bool,
        matchedTokens: Int,
        tier: String?,
        skippedOversize: Bool,
        store: QWEnginePrefixStore<QWEngineCachedPrefix>
    ) -> (generation: QWGenerationStats, experimental: QWExperimentalGenerationStats?, cache: QWPrefixCacheStats) {
        let generation = QWGenerationStats(
            promptTokens: promptCount, generatedTokens: output.tokens.count,
            prefillSec: prefillSec, decodeSec: decodeSec,
            firstTokenSec: firstTokenSec, stepMs: output.stepMs
        )
        let experimental = speculative ? QWExperimentalGenerationStats(
            promptTokens: promptCount, generatedTokens: output.tokens.count,
            prefillSec: prefillSec, decodeSec: decodeSec,
            roundMs: output.stepMs, roundTokenCounts: output.stepTokenCounts
        ) : nil
        let cache = QWPrefixCacheStats(
            hit: matchedTokens > 0, matchedTokens: matchedTokens,
            chargedBytes: store.chargedBytes, evictions: store.evictions,
            skippedOversize: skippedOversize, tier: matchedTokens > 0 ? tier : nil
        )
        return (generation, experimental, cache)
    }

    static func spillEntryIfEligible(
        diskManager: QWCacheDiskManager?,
        isSharedBase: Bool,
        warmHasSharedBase: Bool,
        tokens: [Int],
        mode: String,
        value: QWEngineCachedPrefix,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?,
        allowsSuffix: Bool
    ) {
        guard let dm = diskManager, !dm.config.wipeOnStart,
              QWTwoTierPrefixCache.shouldSpillToCold(
                  isSharedBase: isSharedBase,
                  warmHasSharedBase: warmHasSharedBase,
                  ssdByteLimit: dm.config.ssdByteLimit,
                  entryBytes: value.memoryBytes,
                  maxSpillBytes: maxOpportunisticColdWriteBytes
              ) else { return }
        _ = try? dm.writeDirect(
            tokens: tokens, mode: mode, nextToken: value.nextToken,
            state: value.prefix.state, logits: value.logits,
            trunk: value.prefix.trunk, kvCache: kvCache,
            mtp: value.prefix.mtp, mtpCache: mtpCache, allowsSuffix: allowsSuffix
        )
    }
}

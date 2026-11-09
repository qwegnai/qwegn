// FNPrefixCache.swift — The two-tier prefix cache for FNGPUEngine (Qwen3.8-Flash-Next).
// Saves and restores GDN recurrent/conv state, PLE n-gram history, QSA KV/indexK/pooled state,
// MTP KV/indexK/pooled state, and head logits across CLI runs and server restarts.
// The RAM holds the hot prefixes; the disk holds the persistent copy.

import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHead

extension FNGPUEngine {
    static let ssdWriteQueue = DispatchQueue(label: "qwg.fn.prefix-cache.ssd", qos: .utility)

    /// The cache's own size on disk: how many saved prefixes the
    /// directory holds and how many bytes they take. A directory that
    /// cannot be read is absent, not zero.
    static func diskState(in dir: URL) -> (entries: Int, bytes: Int)? {
        QWPrefixColdStore.diskState(in: dir, filePrefix: "fn_entry_", pathExtension: "qwcache", syncQueue: ssdWriteQueue)
    }

    struct FNPrefixHeader {
        static let magic: UInt64 = 0x5157464E_43414348 // "QWFNCACH"
        static let version: UInt32 = 1
        var magic: UInt64
        var version: UInt32
        var tokenCount: UInt32
        var kvModeCode: UInt32
        var hasMTP: UInt32
        var lastArgmax: Int32
        var mtpDraft: Int32
        var vocabCount: UInt32
        var hiddenSize: UInt32
        var widthSize: UInt32
        var hcSize: UInt32
    }

    struct FNRestoredPrefix {
        let matched: Int
        let lastArgmax: Int
        let mtpDraft: Int
        /// The tier that served the prefix: the RAM or the disk.
        var tier: FNPrefixCacheTier = .coldSSD
    }

    static func fnPrefixHash(tokens: ArraySlice<Int64>, kvMode: FNKVStorageMode, hasMTP: Bool) -> UInt64 {
        QWPrefixHash.fnv1a64(tokens: tokens, modeTag: kvMode.rawValue, hasMTP: hasMTP)
    }

    private func kvModeCode(_ mode: FNKVStorageMode) -> UInt32 {
        switch mode {
        case .f32: return 0
        case .f16: return 1
        case .q8:  return 2
        case .q4:  return 3
        }
    }

    func lookupAndRestorePrefixCache(
        history: [Int64],
        rawHistory: [Int64]? = nil,
        useSpec: Bool,
        cacheDir: URL?,
        gpuSec: inout Double,
        allowExact: Bool = true
    ) throws -> FNRestoredPrefix? {
        guard !history.isEmpty else { return nil }
        let rawHist = rawHistory ?? history
        let fm = FileManager.default
        let primaryHasMTP = (useSpec || hasPrefixCacheTier) && mtp != nil
        let candidateMTPFlags: [Bool] = (useSpec && mtp != nil)
            ? [true]
            : (mtp != nil ? (primaryHasMTP ? [true, false] : [false, true]) : [false])

        // The warm tier answers first: the RAM holds the hot prefixes,
        // and a hit there costs no disk at all.
        if allowExact {
            for hasMTP in candidateMTPFlags {
                let exactHash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP)
                let exactName = String(format: "fn_entry_%016llx_%d.qwcache", exactHash, history.count)
                if let warm = prefixCacheWarmStore, let blob = warm.blob(name: exactName),
                   let restored = try? restorePrefix(data: blob, expectedTokens: history[...], hasMTP: hasMTP,
                                                     isExact: true, nextTokenAfterPrefix: nil, gpuSec: &gpuSec) {
                    var warmHit = restored
                    warmHit.tier = .warmRAM
                    return warmHit
                }
            }
        }
        if let warm = prefixCacheWarmStore {
            // A longer prompt may extend a prefix the RAM holds.
            var bestWarmMatch: (name: String, tokens: Int, hasMTP: Bool)?
            for hasMTP in candidateMTPFlags {
                if let match = warm.longestMatch(history: history, kvMode: kvMode, hasMTP: hasMTP) {
                    if bestWarmMatch == nil || match.tokens > bestWarmMatch!.tokens {
                        bestWarmMatch = (match.name, match.tokens, hasMTP)
                    }
                }
            }
            if let match = bestWarmMatch,
               let blob = warm.blob(name: match.name),
               let restored = try? restorePrefix(data: blob, expectedTokens: history.prefix(match.tokens),
                                                 hasMTP: match.hasMTP, isExact: false,
                                                 nextTokenAfterPrefix: Int32(rawHist[match.tokens]),
                                                 gpuSec: &gpuSec) {
                var warmHit = restored
                warmHit.tier = .warmRAM
                return warmHit
            }
        }

        // The cold tier answers when the RAM does not hold the prefix.
        guard prefixCacheSSDByteLimit > 0, let cacheDir else { return nil }
        Self.ssdWriteQueue.sync {}
        if allowExact {
            for hasMTP in candidateMTPFlags {
                let exactHash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP)
                let exactName = String(format: "fn_entry_%016llx_%d.qwcache", exactHash, history.count)
                let exactURL = cacheDir.appendingPathComponent(exactName)
                if fm.fileExists(atPath: exactURL.path),
                   let restored = try? readPrefixFile(url: exactURL, expectedTokens: history[...], hasMTP: hasMTP, isExact: true, nextTokenAfterPrefix: nil, gpuSec: &gpuSec) {
                    return restored
                }
            }
        }

        // Check for longest matching cached prefix in cacheDir (sorted longest-first)
        guard let files = try? fm.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil) else {
            return nil
        }
        var candidates: [(url: URL, len: Int, hex: Substring)] = []
        for file in files where file.pathExtension == "qwcache" && file.lastPathComponent.hasPrefix("fn_entry_") {
            let parts = file.deletingPathExtension().lastPathComponent.split(separator: "_")
            guard parts.count == 4, let len = Int(parts[3]), len > 0, len < history.count else {
                continue
            }
            candidates.append((file, len, parts[2]))
        }
        candidates.sort { $0.len > $1.len }
        var bestURL: URL?
        var bestLen = 0
        var bestHasMTP = primaryHasMTP
        outerLoop: for cand in candidates {
            let prefixSlice = history.prefix(cand.len)
            for hasMTP in candidateMTPFlags {
                let expectedHash = Self.fnPrefixHash(tokens: prefixSlice, kvMode: kvMode, hasMTP: hasMTP)
                let expectedHex = String(format: "%016llx", expectedHash)
                if cand.hex == expectedHex {
                    bestLen = cand.len
                    bestURL = cand.url
                    bestHasMTP = hasMTP
                    break outerLoop
                }
            }
        }
        guard let url = bestURL else { return nil }
        let nextTok = Int32(rawHist[bestLen])
        return try? readPrefixFile(url: url, expectedTokens: history.prefix(bestLen), hasMTP: bestHasMTP, isExact: false, nextTokenAfterPrefix: nextTok, gpuSec: &gpuSec)
    }

    private func readPrefixFile(
        url: URL,
        expectedTokens: ArraySlice<Int64>,
        hasMTP: Bool,
        isExact: Bool,
        nextTokenAfterPrefix: Int32?,
        gpuSec: inout Double
    ) throws -> FNRestoredPrefix? {
        // The warm tier keeps the bytes it restores from: the file is read
        // into the RAM, not mapped, so the tier holds them itself.
        let data: Data
        if let warm = prefixCacheWarmStore, warm.canHold(bytes: Self.fileSize(at: url)) {
            data = try Data(contentsOf: url)
        } else {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        }
        guard let restored = try restorePrefix(data: data, expectedTokens: expectedTokens,
                                               hasMTP: hasMTP, isExact: isExact,
                                               nextTokenAfterPrefix: nextTokenAfterPrefix,
                                               gpuSec: &gpuSec) else { return nil }
        // The prefix that came from the disk joins the warm tier: the next
        // hit for it is answered by the RAM.
        if let warm = prefixCacheWarmStore {
            let tokArray = Array(expectedTokens)
            let isShared = warm.isSharedBaseEntry(url.lastPathComponent)
            if isShared {
                warm.makeRoom(for: data.count, activeSlot: nil)
                warm.store(name: url.lastPathComponent, blob: data, slot: nil, rootHash: nil, history: tokArray, isSharedBase: true)
            } else {
                let rHash = FNPrefixWarmStore.rootHash(history: tokArray, kvMode: kvMode, hasMTP: hasMTP)
                let slot = warm.resolveSlot(rootHash: rHash, history: tokArray, kvMode: kvMode, hasMTP: hasMTP)
                warm.store(name: url.lastPathComponent, blob: data, slot: slot, rootHash: rHash, history: tokArray, isSharedBase: false)
            }
        }
        return restored
    }

    /// The size of a file on disk, or zero when it cannot be read.
    static func fileSize(at url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }

    /// Puts a saved prefix back into the engine's state.
    private func restorePrefix(
        data: Data,
        expectedTokens: ArraySlice<Int64>,
        hasMTP: Bool,
        isExact: Bool,
        nextTokenAfterPrefix: Int32?,
        gpuSec: inout Double
    ) throws -> FNRestoredPrefix? {
        let hdrSize = MemoryLayout<FNPrefixHeader>.size
        guard data.count >= hdrSize else { return nil }

        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> FNRestoredPrefix? in
            guard let base = raw.baseAddress else { return nil }
            var offset = 0
            var hdr = FNPrefixHeader(magic: 0, version: 0, tokenCount: 0, kvModeCode: 0,
                                     hasMTP: 0, lastArgmax: 0, mtpDraft: 0,
                                     vocabCount: 0, hiddenSize: 0, widthSize: 0, hcSize: 0)
            memcpy(&hdr, base, hdrSize)
            offset += hdrSize

            let N = Int(hdr.tokenCount)
            let isCompactInPlace = (hdr.hasMTP & 2) != 0
            guard hdr.magic == FNPrefixHeader.magic,
                  hdr.version == FNPrefixHeader.version,
                  N == expectedTokens.count,
                  hdr.kvModeCode == kvModeCode(kvMode),
                  (hdr.hasMTP & 1) == (hasMTP ? 1 : 0),
                  Int(hdr.vocabCount) == vocab,
                  Int(hdr.hiddenSize) == hidden,
                  Int(hdr.widthSize) == width,
                  Int(hdr.hcSize) == hc else {
                return nil
            }

            let tokBytes = N * MemoryLayout<Int64>.size
            guard offset + tokBytes <= data.count else { return nil }
            let storedToks = UnsafeBufferPointer(
                start: base.advanced(by: offset).assumingMemoryBound(to: Int64.self),
                count: N
            )
            for (i, t) in expectedTokens.enumerated() where storedToks[i] != t {
                return nil
            }
            offset += tokBytes

            // Logits (row 0 of head.logits)
            let logitBytes = vocab * MemoryLayout<Float>.size
            guard offset + logitBytes <= data.count else { return nil }
            if isExact {
                memcpy(arena.cpuPointer(of: head.logits), base.advanced(by: offset), logitBytes)
                arena.write(head.token, from: [hdr.lastArgmax])
            }
            offset += logitBytes

            // Boundary preInject tensors (h2Last [W], moeOutLast [H], injBLast [C])
            let h2Bytes = width * MemoryLayout<Float>.size
            let moeBytes = hidden * MemoryLayout<Float>.size
            let injBytes = hc * MemoryLayout<Float>.size
            guard offset + h2Bytes + moeBytes + injBytes <= data.count else { return nil }
            let h2Src = base.advanced(by: offset); offset += h2Bytes
            let moeSrc = base.advanced(by: offset); offset += moeBytes
            let injSrc = base.advanced(by: offset); offset += injBytes

            // Tier-0 in-place check: if the GPU arena already holds the exact attention KV/index
            // prefix in `0..<N` from the previous turn, only wipe the tail above `N` and skip the
            // multi-gigabyte CPU->GPU KV memcpy.
            let inPlaceKV = residentKVWatermark.canReuseInPlace(expectedTokens: expectedTokens, hasMTP: hasMTP)
            if isCompactInPlace && !inPlaceKV { return nil }
            wipeSessionTail(from: N)

            // Per-layer recurrent, PLE, and attention KV state
            var layerOffsets = [Int](repeating: 0, count: layerRecords.count)
            for (i, rec) in layerRecords.enumerated() {
                layerOffsets[i] = offset
                if let g = rec.gdn {
                    offset += g.states[0].bytes + g.convState[0].bytes
                }
                if let p = rec.weights.ple {
                    offset += p.history[0].bytes
                }
                if !isCompactInPlace, let a = rec.attnState {
                    let cap = max(1, a.capacity)
                    let kvRowBytes = (a.k.bytes / cap) * N
                    let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                    let idxRowBytes = (a.indexK.bytes / cap) * N
                    let poolBlocks = (N + 3) / 4
                    let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                    offset += 2 * kvRowBytes + 2 * metaRowBytes + idxRowBytes + poolBytes
                }
            }
            guard offset <= data.count else { return nil }
            let srcAddr = Int(bitPattern: base)
            let offsets = layerOffsets
            DispatchQueue.concurrentPerform(iterations: layerRecords.count) { idx in
                let rec = layerRecords[idx]
                var off = offsets[idx]
                let basePtr = UnsafeRawPointer(bitPattern: srcAddr)!
                if let g = rec.gdn {
                    let sBytes = g.states[0].bytes
                    let csBytes = g.convState[0].bytes
                    memcpy(arena.cpuPointer(of: g.states[0]), basePtr.advanced(by: off), sBytes)
                    off += sBytes
                    g.stateParity = 0
                    memcpy(arena.cpuPointer(of: g.convState[0]), basePtr.advanced(by: off), csBytes)
                    off += csBytes
                    g.convParity = 0
                }
                if let p = rec.weights.ple {
                    let hBytes = p.history[0].bytes
                    memcpy(arena.cpuPointer(of: p.history[0]), basePtr.advanced(by: off), hBytes)
                    off += hBytes
                    p.parity = 0
                }
                if !inPlaceKV, let a = rec.attnState {
                    let cap = max(1, a.capacity)
                    let kvRowBytes = (a.k.bytes / cap) * N
                    let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                    let idxRowBytes = (a.indexK.bytes / cap) * N
                    let poolBlocks = (N + 3) / 4
                    let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                    memcpy(arena.cpuPointer(of: a.k), basePtr.advanced(by: off), kvRowBytes); off += kvRowBytes
                    memcpy(arena.cpuPointer(of: a.v), basePtr.advanced(by: off), kvRowBytes); off += kvRowBytes
                    if metaRowBytes > 0 {
                        memcpy(arena.cpuPointer(of: a.kMeta), basePtr.advanced(by: off), metaRowBytes); off += metaRowBytes
                        memcpy(arena.cpuPointer(of: a.vMeta), basePtr.advanced(by: off), metaRowBytes); off += metaRowBytes
                    }
                    memcpy(arena.cpuPointer(of: a.indexK), basePtr.advanced(by: off), idxRowBytes); off += idxRowBytes
                    memcpy(arena.cpuPointer(of: a.pooled), basePtr.advanced(by: off), poolBytes); off += poolBytes
                }
            }

            if !inPlaceKV, N >= 512, layerRecords.contains(where: { $0.attnState?.lsaClusters != nil }),
               let (cb, enc) = arena.hal.begin() {
                let numSB = N / 512
                for rec in layerRecords {
                    if let a = rec.attnState, let clusters = a.lsaClusters {
                        try? QWGemm.encodeFnLSAWardCluster(
                            enc, hal: arena.hal, pooled: a.pooled, clusters: clusters,
                            startSuperblock: 0, numSuperblocks: numSB, indexDim: 128)
                    }
                }
                arena.hal.end(cb, enc)
                cb.waitUntilCompleted()
            }

            if hasMTP, let mtp = self.mtp {
                let a = mtp.attnState1
                let cap = a.capacity
                let kvRowStride = a.k.bytes / cap
                let metaRowStride = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) : 0
                let idxRowStride = a.indexK.bytes / cap
                let poolBlockStride = a.pooled.bytes / max(1, cap / 4)
                if isCompactInPlace {
                    let rowTotal = 2 * kvRowStride + 2 * metaRowStride + idxRowStride + poolBlockStride
                    guard N > 0, offset + rowTotal <= data.count else { return nil }
                    let lastRow = N - 1
                    memcpy(arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride), base.advanced(by: offset), kvRowStride); offset += kvRowStride
                    memcpy(arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride), base.advanced(by: offset), kvRowStride); offset += kvRowStride
                    if metaRowStride > 0 {
                        memcpy(arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride), base.advanced(by: offset), metaRowStride); offset += metaRowStride
                        memcpy(arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride), base.advanced(by: offset), metaRowStride); offset += metaRowStride
                    }
                    memcpy(arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride), base.advanced(by: offset), idxRowStride); offset += idxRowStride
                    let lastBlock = lastRow / 4
                    if (lastBlock + 1) * poolBlockStride <= a.pooled.bytes {
                        memcpy(arena.cpuPointer(of: a.pooled).advanced(by: lastBlock * poolBlockStride), base.advanced(by: offset), poolBlockStride)
                    }
                    offset += poolBlockStride
                } else {
                    let kvRowBytes = kvRowStride * N
                    let metaRowBytes = metaRowStride * N
                    let idxRowBytes = idxRowStride * N
                    let poolBlocks = (N + 3) / 4
                    let poolBytes = min(a.pooled.bytes, poolBlockStride * poolBlocks)
                    let totalMTP = 2 * kvRowBytes + 2 * metaRowBytes + idxRowBytes + poolBytes
                    guard offset + totalMTP <= data.count else { return nil }
                    if inPlaceKV, N > 0 {
                        let lastRow = N - 1
                        memcpy(arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride), base.advanced(by: offset + lastRow * kvRowStride), kvRowStride); offset += kvRowBytes
                        memcpy(arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride), base.advanced(by: offset + lastRow * kvRowStride), kvRowStride); offset += kvRowBytes
                        if metaRowBytes > 0 {
                            memcpy(arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride), base.advanced(by: offset + lastRow * metaRowStride), metaRowStride); offset += metaRowBytes
                            memcpy(arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride), base.advanced(by: offset + lastRow * metaRowStride), metaRowStride); offset += metaRowBytes
                        }
                        memcpy(arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride), base.advanced(by: offset + lastRow * idxRowStride), idxRowStride); offset += idxRowBytes
                        let lastBlock = lastRow / 4
                        if (lastBlock + 1) * poolBlockStride <= poolBytes {
                            memcpy(arena.cpuPointer(of: a.pooled).advanced(by: lastBlock * poolBlockStride), base.advanced(by: offset + lastBlock * poolBlockStride), poolBlockStride)
                        }
                        offset += poolBytes
                    } else {
                        memcpy(arena.cpuPointer(of: a.k), base.advanced(by: offset), kvRowBytes); offset += kvRowBytes
                        memcpy(arena.cpuPointer(of: a.v), base.advanced(by: offset), kvRowBytes); offset += kvRowBytes
                        if metaRowBytes > 0 {
                            memcpy(arena.cpuPointer(of: a.kMeta), base.advanced(by: offset), metaRowBytes); offset += metaRowBytes
                            memcpy(arena.cpuPointer(of: a.vMeta), base.advanced(by: offset), metaRowBytes); offset += metaRowBytes
                        }
                        memcpy(arena.cpuPointer(of: a.indexK), base.advanced(by: offset), idxRowBytes); offset += idxRowBytes
                        memcpy(arena.cpuPointer(of: a.pooled), base.advanced(by: offset), poolBytes); offset += poolBytes
                    }
                }

                // When continuing with a suffix (not exact match), re-seed MTP row (N - 1) using actual history[N]
                if !isExact, let nextTok = nextTokenAfterPrefix, let lastDecodeSlot = decodeStack.slots.last {
                    let s = lastDecodeSlot.scratch
                    memcpy(arena.cpuPointer(of: s.h2), h2Src, h2Bytes)
                    memcpy(arena.cpuPointer(of: s.moeOut), moeSrc, moeBytes)
                    memcpy(arena.cpuPointer(of: s.injB), injSrc, injBytes)
                    let preInj = (base: s.h2, src: s.moeOut, scale: s.injB)
                    _ = try mtp.forwardPrefill(streamBuf: decodeStack.lastStreamTensor, preInject: preInj,
                                               nextTokens: [nextTok], startPos: N - 1,
                                               prefillScratch: mtp.scratch1, projectLastRow: false)
                    gpuSec += mtp.lastGPUMillis / 1000.0
                }
            }

            residentKVWatermark.recordRestored(expectedTokens: expectedTokens, hasMTP: hasMTP, inPlace: inPlaceKV)
            highWaterPos = max(highWaterPos, N)

            return FNRestoredPrefix(matched: N, lastArgmax: Int(hdr.lastArgmax), mtpDraft: Int(hdr.mtpDraft))
        }
    }

    func savePrefixCache(
        history: [Int64],
        lastChunkRows: Int,
        useSpec: Bool,
        lastArgmax: Int,
        mtpDraft: Int,
        cacheDir: URL?,
        isSharedBase: Bool = false,
        customOutputFileURL: URL? = nil
    ) {
        let N = history.count
        guard N > 0 else { return }
        if customOutputFileURL == nil {
            guard prefixCacheSSDByteLimit > 0 || prefixCacheWarmStore != nil else { return }
            guard N >= 128 else { return }
        }
        let fm = FileManager.default
        let hasMTP = (useSpec || hasPrefixCacheTier) && mtp != nil
        let hash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP)
        let filename = String(format: "fn_entry_%016llx_%d.qwcache", hash, N)
        if customOutputFileURL == nil, prefixCacheWarmStore?.contains(name: filename) == true { return }
        if customOutputFileURL == nil, prefixCacheSSDByteLimit > 0, let cacheDir, fm.fileExists(atPath: cacheDir.appendingPathComponent(filename).path) { return }
        if prefixCacheSSDByteLimit > 0, let cacheDir {
            try? fm.createDirectory(at: cacheDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }

        let hdrBytes = MemoryLayout<FNPrefixHeader>.size
        let tokBytes = N * MemoryLayout<Int64>.size
        let logitBytes = vocab * MemoryLayout<Float>.size
        let h2Bytes = width * MemoryLayout<Float>.size
        let moeBytes = hidden * MemoryLayout<Float>.size
        let injBytes = hc * MemoryLayout<Float>.size

        var fullTotalBytes = hdrBytes + tokBytes + logitBytes + h2Bytes + moeBytes + injBytes
        var compactTotalBytes = fullTotalBytes
        for rec in layerRecords {
            if let g = rec.gdn {
                let gBytes = g.states[g.stateParity & QWFourSlotRollbackRing.ringMask].bytes
                    + g.convState[g.convParity & QWFourSlotRollbackRing.ringMask].bytes
                fullTotalBytes += gBytes; compactTotalBytes += gBytes
            }
            if let p = rec.weights.ple {
                let pBytes = p.history[p.parity & QWFourSlotRollbackRing.ringMask].bytes
                fullTotalBytes += pBytes; compactTotalBytes += pBytes
            }
            if let a = rec.attnState {
                let cap = max(1, a.capacity)
                let kvRowBytes = (a.k.bytes / cap) * N
                let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                let idxRowBytes = (a.indexK.bytes / cap) * N
                let poolBlocks = (N + 3) / 4
                let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                fullTotalBytes += 2 * kvRowBytes + 2 * metaRowBytes + idxRowBytes + poolBytes
            }
        }
        if hasMTP, let mtp = self.mtp {
            let a = mtp.attnState1
            let cap = a.capacity
            let kvRowStride = a.k.bytes / cap
            let metaRowStride = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) : 0
            let idxRowStride = a.indexK.bytes / cap
            let poolBlockStride = a.pooled.bytes / max(1, cap / 4)
            let poolBlocks = (N + 3) / 4
            let poolBytes = min(a.pooled.bytes, poolBlockStride * poolBlocks)
            fullTotalBytes += 2 * (kvRowStride * N) + 2 * (metaRowStride * N) + (idxRowStride * N) + poolBytes
            compactTotalBytes += 2 * kvRowStride + 2 * metaRowStride + idxRowStride + poolBlockStride
        }

        let fitsWarmFull = prefixCacheWarmStore?.canHold(bytes: fullTotalBytes) == true
        let fitsColdFull = prefixCacheSSDByteLimit > 0 && cacheDir != nil && fullTotalBytes <= prefixCacheSSDByteLimit && prefixCacheWarmStore?.hasSharedBase != true
        let isCompactInPlace = customOutputFileURL == nil && !fitsWarmFull && !fitsColdFull
            && (prefixCacheWarmStore?.canHold(bytes: compactTotalBytes) == true)
        let fitsWarm = fitsWarmFull || isCompactInPlace
        if customOutputFileURL == nil && !fitsWarm && !fitsColdFull {
            if let warm = prefixCacheWarmStore, warm.isEnabled {
                onCacheSkippedOversize?(N, fullTotalBytes, warm.byteLimit)
            }
            return
        }

        var hdr = FNPrefixHeader(
            magic: FNPrefixHeader.magic,
            version: FNPrefixHeader.version,
            tokenCount: UInt32(N),
            kvModeCode: kvModeCode(kvMode),
            hasMTP: (hasMTP ? 1 : 0) | (isCompactInPlace ? 2 : 0),
            lastArgmax: Int32(lastArgmax),
            mtpDraft: Int32(mtpDraft),
            vocabCount: UInt32(vocab),
            hiddenSize: UInt32(hidden),
            widthSize: UInt32(width),
            hcSize: UInt32(hc)
        )

        var totalBytes = hdrBytes + tokBytes + logitBytes + h2Bytes + moeBytes + injBytes
        var layerOffsets = [Int](repeating: 0, count: layerRecords.count)
        for (i, rec) in layerRecords.enumerated() {
            layerOffsets[i] = totalBytes
            if let g = rec.gdn {
                totalBytes += g.states[g.stateParity & QWFourSlotRollbackRing.ringMask].bytes
                    + g.convState[g.convParity & QWFourSlotRollbackRing.ringMask].bytes
            }
            if let p = rec.weights.ple {
                totalBytes += p.history[p.parity & QWFourSlotRollbackRing.ringMask].bytes
            }
            if !isCompactInPlace, let a = rec.attnState {
                let cap = max(1, a.capacity)
                let kvRowBytes = (a.k.bytes / cap) * N
                let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                let idxRowBytes = (a.indexK.bytes / cap) * N
                let poolBlocks = (N + 3) / 4
                let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                totalBytes += 2 * kvRowBytes + 2 * metaRowBytes + idxRowBytes + poolBytes
            }
        }
        let mtpOffset = totalBytes
        totalBytes = isCompactInPlace ? compactTotalBytes : fullTotalBytes

        let rootHash = FNPrefixWarmStore.rootHash(history: history, kvMode: kvMode, hasMTP: hasMTP)
        var resolvedSlot: Int? = nil
        if fitsWarm, let warm = prefixCacheWarmStore {
            if isSharedBase {
                warm.makeRoom(for: totalBytes, activeSlot: nil)
            } else {
                let slot = warm.resolveSlot(rootHash: rootHash, history: history, kvMode: kvMode, hasMTP: hasMTP)
                resolvedSlot = slot
                warm.prepareSlotForStore(slot: slot, rootHash: rootHash, history: history,
                                         incomingBytes: totalBytes, kvMode: kvMode, hasMTP: hasMTP)
            }
        }

        var out = Data(count: totalBytes)
        out.withUnsafeMutableBytes { rawOut in
            guard let dstBase = rawOut.baseAddress else { return }
            var writeOffset = 0
            func appendRaw(_ src: UnsafeRawPointer, _ count: Int) {
                guard count > 0 else { return }
                memcpy(dstBase.advanced(by: writeOffset), src, count)
                writeOffset += count
            }

            withUnsafeBytes(of: &hdr) { hdrBuf in
                if let p = hdrBuf.baseAddress { appendRaw(p, hdrBytes) }
            }
            history.withUnsafeBytes { histBuf in
                if let p = histBuf.baseAddress { appendRaw(p, tokBytes) }
            }
            appendRaw(arena.cpuPointer(of: head.logits), logitBytes)

            if lastChunkRows == 1, let lastSlot = decodeStack.slots.last {
                let s = lastSlot.scratch
                appendRaw(arena.cpuPointer(of: s.h2), h2Bytes)
                appendRaw(arena.cpuPointer(of: s.moeOut), moeBytes)
                appendRaw(arena.cpuPointer(of: s.injB), injBytes)
            } else if let cached = cachedPrefill, let pArena = prefillArena, let lastSlot = cached.stack.slots.last {
                let s = lastSlot.scratch
                let r = max(0, lastChunkRows - 1)
                appendRaw(pArena.cpuPointer(of: s.h2).advanced(by: r * h2Bytes), h2Bytes)
                appendRaw(pArena.cpuPointer(of: s.moeOut).advanced(by: r * moeBytes), moeBytes)
                appendRaw(pArena.cpuPointer(of: s.injB).advanced(by: r * injBytes), injBytes)
            } else {
                memset(dstBase.advanced(by: writeOffset), 0, h2Bytes + moeBytes + injBytes)
                writeOffset += h2Bytes + moeBytes + injBytes
            }

            let dstAddr = Int(bitPattern: dstBase)
            let offsets = layerOffsets
            DispatchQueue.concurrentPerform(iterations: layerRecords.count) { idx in
                let rec = layerRecords[idx]
                var off = offsets[idx]
                let basePtr = UnsafeMutableRawPointer(bitPattern: dstAddr)!
                func copyChunk(_ src: UnsafeRawPointer, _ count: Int) {
                    guard count > 0 else { return }
                    memcpy(basePtr.advanced(by: off), src, count)
                    off += count
                }
                if let g = rec.gdn {
                    let s = g.states[g.stateParity & QWFourSlotRollbackRing.ringMask]
                    let cs = g.convState[g.convParity & QWFourSlotRollbackRing.ringMask]
                    copyChunk(arena.cpuPointer(of: s), s.bytes)
                    copyChunk(arena.cpuPointer(of: cs), cs.bytes)
                }
                if let p = rec.weights.ple {
                    let h = p.history[p.parity & QWFourSlotRollbackRing.ringMask]
                    copyChunk(arena.cpuPointer(of: h), h.bytes)
                }
                if !isCompactInPlace, let a = rec.attnState {
                    let cap = max(1, a.capacity)
                    let kvRowBytes = (a.k.bytes / cap) * N
                    let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                    let idxRowBytes = (a.indexK.bytes / cap) * N
                    let poolBlocks = (N + 3) / 4
                    let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)

                    copyChunk(arena.cpuPointer(of: a.k), kvRowBytes)
                    copyChunk(arena.cpuPointer(of: a.v), kvRowBytes)
                    if metaRowBytes > 0 {
                        copyChunk(arena.cpuPointer(of: a.kMeta), metaRowBytes)
                        copyChunk(arena.cpuPointer(of: a.vMeta), metaRowBytes)
                    }
                    copyChunk(arena.cpuPointer(of: a.indexK), idxRowBytes)
                    copyChunk(arena.cpuPointer(of: a.pooled), poolBytes)
                }
            }

            writeOffset = mtpOffset
            if hasMTP, let mtp = self.mtp {
                let a = mtp.attnState1
                let cap = a.capacity
                let kvRowStride = a.k.bytes / cap
                let metaRowStride = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) : 0
                let idxRowStride = a.indexK.bytes / cap
                let poolBlockStride = a.pooled.bytes / max(1, cap / 4)
                if isCompactInPlace {
                    let lastRow = max(0, N - 1)
                    let lastBlock = lastRow / 4
                    appendRaw(arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride), kvRowStride)
                    appendRaw(arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride), kvRowStride)
                    if metaRowStride > 0 {
                        appendRaw(arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride), metaRowStride)
                        appendRaw(arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride), metaRowStride)
                    }
                    appendRaw(arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride), idxRowStride)
                    appendRaw(arena.cpuPointer(of: a.pooled).advanced(by: min(lastBlock * poolBlockStride, max(0, a.pooled.bytes - poolBlockStride))), poolBlockStride)
                } else {
                    let kvRowBytes = kvRowStride * N
                    let metaRowBytes = metaRowStride * N
                    let idxRowBytes = idxRowStride * N
                    let poolBlocks = (N + 3) / 4
                    let poolBytes = min(a.pooled.bytes, poolBlockStride * poolBlocks)

                    appendRaw(arena.cpuPointer(of: a.k), kvRowBytes)
                    appendRaw(arena.cpuPointer(of: a.v), kvRowBytes)
                    if metaRowBytes > 0 {
                        appendRaw(arena.cpuPointer(of: a.kMeta), metaRowBytes)
                        appendRaw(arena.cpuPointer(of: a.vMeta), metaRowBytes)
                    }
                    appendRaw(arena.cpuPointer(of: a.indexK), idxRowBytes)
                    appendRaw(arena.cpuPointer(of: a.pooled), poolBytes)
                }
            }
        }

        // The cold tier holds the prefix for the next run; the warm tier
        // holds it for the next request.
        if let warm = prefixCacheWarmStore, warm.isEnabled {
            if isSharedBase ? (out.count <= warm.byteLimit) : warm.canHold(bytes: out.count) {
                if isSharedBase {
                    warm.store(name: filename, blob: out, slot: nil, rootHash: nil, history: history, isSharedBase: true)
                } else {
                    let slot = resolvedSlot ?? warm.resolveSlot(rootHash: rootHash, history: history, kvMode: kvMode, hasMTP: hasMTP)
                    warm.store(name: filename, blob: out, slot: slot, rootHash: rootHash, history: history, isSharedBase: false)
                }
                onCacheRetained?(N, out.count, warm.state.bytes, warm.byteLimit)
            } else {
                onCacheSkippedOversize?(N, out.count, warm.byteLimit)
            }
        }
        if let customOutputFileURL {
            try? fm.createDirectory(at: customOutputFileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? out.write(to: customOutputFileURL, options: .atomic)
        }
        if !isCompactInPlace, prefixCacheWarmStore?.hasSharedBase != true, prefixCacheSSDByteLimit > 0, let cacheDir, customOutputFileURL == nil {
            if out.count <= prefixCacheSSDByteLimit {
                let limit = prefixCacheSSDByteLimit
                let snapshot = out
                Self.ssdWriteQueue.async {
                    Self.makeRoom(in: cacheDir, for: snapshot.count, limit: limit)
                    try? snapshot.write(to: cacheDir.appendingPathComponent(filename), options: .atomic)
                }
            }
        }
    }

    /// Prefills `tokens`, writes a standalone `.qwcache` file at `outputURL`, and optionally pins it in warm RAM.
    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true
    ) throws -> Int {
        guard !tokens.isEmpty else {
            throw NSError(domain: "QWFlashNext", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot build empty standalone prefix cache"])
        }
        if !(prefixCacheWarmStore?.isEnabled ?? false) {
            prefixCacheWarmStore = FNPrefixWarmStore(byteLimit: 64 << 20, maxSlots: maxConcurrentSessions)
        }
        try preparePrefill(forTokenCount: tokens.count + 16)
        var session = FNGPUSessionData(
            history: tokens.map(Int64.init),
            sampler: QWSampler(config: .greedy),
            useSpec: mtp != nil
        )
        session.restoreDone = true
        wipeSessionState(overwriteTokens: tokens.count)
        residentKVHasMTP = mtp != nil
        highWaterPos = max(highWaterPos, min(activeCapacity, tokens.count + 4))
        beginSession(&session, maxNew: 1)
        let first = try encodePrefillChunk(&session, chunk: 0..<tokens.count, runsHead: true)
        savePrefixCache(
            history: session.history,
            lastChunkRows: max(1, session.lastRows),
            useSpec: mtp != nil,
            lastArgmax: first,
            mtpDraft: session.mtpDraft,
            cacheDir: nil,
            isSharedBase: pinAsSharedBase,
            customOutputFileURL: outputURL
        )
        evictPrefillScratchIfNeeded()
        let writtenBytes = Self.fileSize(at: outputURL)
        if pinAsSharedBase {
            _ = try? mountStandalonePrefixCache(from: outputURL)
        }
        return writtenBytes
    }

    /// Mounts a standalone `.qwcache` file from `fileURL` into the warm prefix store as `isSharedBase: true`.
    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        let data = try Data(contentsOf: fileURL)
        let hdrSize = MemoryLayout<FNPrefixHeader>.size
        guard data.count >= hdrSize else {
            throw NSError(domain: "QWFlashNext", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "invalid FlashNext .qwcache header"])
        }
        var hdr = FNPrefixHeader(magic: 0, version: 0, tokenCount: 0, kvModeCode: 0,
                                 hasMTP: 0, lastArgmax: 0, mtpDraft: 0,
                                 vocabCount: 0, hiddenSize: 0, widthSize: 0, hcSize: 0)
        _ = data.withUnsafeBytes { raw in
            memcpy(&hdr, raw.baseAddress!, hdrSize)
        }
        let n = Int(hdr.tokenCount)
        let tokBytes = n * MemoryLayout<Int64>.size
        guard hdr.magic == FNPrefixHeader.magic, hdr.version == FNPrefixHeader.version,
              hdr.kvModeCode == kvModeCode(kvMode), data.count >= hdrSize + tokBytes else {
            throw NSError(domain: "QWFlashNext", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "FlashNext .qwcache geometry or kvMode mismatch"])
        }
        let history: [Int64] = data.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.advanced(by: hdrSize).assumingMemoryBound(to: Int64.self)
            return Array(UnsafeBufferPointer(start: ptr, count: n))
        }
        let hasMTP = hdr.hasMTP != 0
        let hash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP)
        let filename = String(format: "fn_entry_%016llx_%d.qwcache", hash, n)
        if let warm = prefixCacheWarmStore {
            warm.ensureByteLimit(atLeast: data.count + (256 << 20))
        } else {
            prefixCacheWarmStore = FNPrefixWarmStore(byteLimit: data.count + (256 << 20), maxSlots: maxConcurrentSessions)
        }
        prefixCacheWarmStore?.store(name: filename, blob: data, slot: nil, rootHash: nil, history: history, isSharedBase: true)
        return (history.map(Int.init), data.count)
    }

    /// Unmounts all pinned shared-base prefixes from the warm prefix store.
    public func unmountSharedBasePrefixes() {
        prefixCacheWarmStore?.removeSharedBaseEntries()
    }

    /// Makes room in the cold tier: the oldest saved prefixes go first
    /// when the disk is at its limit. The cold tier is the overflow, so
    /// it stays inside the limit the settings gave it.
    static func makeRoom(in dir: URL, for bytes: Int, limit: Int) {
        QWPrefixColdStore.makeRoom(in: dir, for: bytes, limit: limit, filePrefix: "fn_entry_", pathExtension: "qwcache")
    }
}

// FNPrefixCache.swift — The two-tier prefix cache for QWHyperStreamRuntime (Qwen3.8-Flash-Next).
// Saves and restores GDN recurrent/conv state, PLE n-gram history, QSA KV/indexK/pooled state,
// MTP KV/indexK/pooled state, and head logits across CLI runs and server restarts.
// The RAM holds the hot prefixes; the disk holds the persistent copy.

import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWAttn
import QWHyperPLE

extension QWHyperStreamRuntime {
    static let ssdWriteQueue = DispatchQueue(label: "qwg.fn.prefix-cache.ssd", qos: .utility)

    /// The cache's own size on disk: how many saved prefixes the
    /// directory holds and how many bytes they take. A directory that
    /// cannot be read is absent, not zero.
    static func diskState(in dir: URL) -> (entries: Int, bytes: Int)? {
        QWPrefixColdStore.diskState(in: dir, filePrefix: "fn_entry_", pathExtension: "qwcache", syncQueue: ssdWriteQueue)
    }

    struct FNPrefixHeader {
        static let magic: UInt64 = 0x5157464E_43414348 // "QWFNCACH"
        static let version: UInt32 = 2
        static let dflashBit: UInt32 = 4
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
        let dflashCacheOffset: Int
        /// The tier that served the prefix: the RAM or the disk.
        var tier: FNPrefixCacheTier = .coldSSD
    }

    static func fnPrefixHash(
        tokens: ArraySlice<Int64>, kvMode: FNKVStorageMode, hasMTP: Bool, dflashMarker: String? = nil
    ) -> UInt64 {
        let modeTag = dflashMarker.map { "\(kvMode.rawValue)|\($0)" } ?? kvMode.rawValue
        return QWPrefixHash.fnv1a64(tokens: tokens, modeTag: modeTag, hasMTP: hasMTP)
    }

    func kvModeCode(_ mode: FNKVStorageMode) -> UInt32 {
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
                let exactHash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP,
                                                  dflashMarker: dflashPrefixMarker)
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
            var bestWarmHit: (hit: QWTwoTierPrefixCache.WarmLookupHit<Data>, hasMTP: Bool)?
            for hasMTP in candidateMTPFlags {
                if let match = warm.lookupWarmHit(exactName: "", history: history, rawHistory: rawHist,
                                                  kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashPrefixMarker,
                                                  allowExact: false),
                   match.matchedTokens < history.count {
                    if bestWarmHit == nil || match.matchedTokens > bestWarmHit!.hit.matchedTokens {
                        bestWarmHit = (match, hasMTP)
                    }
                }
            }
            if let best = bestWarmHit {
                let expectedToks = best.hit.isUnsaltedSharedBase
                    ? rawHist.prefix(best.hit.matchedTokens)
                    : history.prefix(best.hit.matchedTokens)
                if let restored = try? restorePrefix(data: best.hit.value, expectedTokens: expectedToks,
                                                     hasMTP: best.hasMTP, isExact: false,
                                                     nextTokenAfterPrefix: Int32(rawHist[best.hit.matchedTokens]),
                                                     gpuSec: &gpuSec) {
                    var warmHit = restored
                    warmHit.tier = .warmRAM
                    return warmHit
                }
            }
        }

        // The cold tier answers when the RAM does not hold the prefix.
        guard prefixCacheSSDByteLimit > 0, let cacheDir else { return nil }
        Self.ssdWriteQueue.sync {}
        if allowExact {
            for hasMTP in candidateMTPFlags {
                let exactHash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP,
                                                  dflashMarker: dflashPrefixMarker)
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
                let expectedHash = Self.fnPrefixHash(tokens: prefixSlice, kvMode: kvMode, hasMTP: hasMTP,
                                                     dflashMarker: dflashPrefixMarker)
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
                _ = warm.storeWithSlotPinning(
                    name: url.lastPathComponent, blob: data, history: tokArray,
                    kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashPrefixMarker, isSharedBase: false
                )
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

        var pendingInPlaceKV = false
        var pendingN = 0
        var pendingMTPSeed = false
        let restored: FNRestoredPrefix? = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> FNRestoredPrefix? in
            guard let base = raw.baseAddress else { return nil }
            var offset = 0
            var hdr = FNPrefixHeader(magic: 0, version: 0, tokenCount: 0, kvModeCode: 0,
                                     hasMTP: 0, lastArgmax: 0, mtpDraft: 0,
                                     vocabCount: 0, hiddenSize: 0, widthSize: 0, hcSize: 0)
            withUnsafeMutableBytes(of: &hdr) { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: base, count: hdrSize))
            }
            offset += hdrSize

            let N = Int(hdr.tokenCount)
            let isCompactInPlace = (hdr.hasMTP & 2) != 0
            let wantDFlash = (dflash != nil && dflashPrefixMarker != nil)
            let fileHasDFlash = (hdr.hasMTP & FNPrefixHeader.dflashBit) != 0
            guard hdr.magic == FNPrefixHeader.magic,
                  hdr.version == FNPrefixHeader.version,
                  N == expectedTokens.count,
                  hdr.kvModeCode == kvModeCode(kvMode),
                  (hdr.hasMTP & 1) == (hasMTP ? 1 : 0),
                  // Serial mounts (wantDFlash == false) reuse trunk KV from a DFlash-enabled .qwcache.
                  (!wantDFlash || fileHasDFlash),
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
                let logitsPtr = arena.cpuPointer(of: head.logits)
                withExtendedLifetime(head.logits.region.buffer) {
                    logitsPtr.copyMemory(from: base.advanced(by: offset), byteCount: logitBytes)
                }
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
            let inPlaceKV = residentKVWatermark.canReuseInPlace(expectedTokens: expectedTokens,
                                                                hasMTP: hasMTP || wantDFlash)
            if isCompactInPlace && !inPlaceKV { return nil }
            wipeSessionTail(from: N)
            highWaterPos = max(highWaterPos, N)

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
                    arena.cpuPointer(of: g.states[0]).copyMemory(from: basePtr.advanced(by: off), byteCount: sBytes)
                    off += sBytes
                    g.stateParity = 0
                    arena.cpuPointer(of: g.convState[0]).copyMemory(from: basePtr.advanced(by: off), byteCount: csBytes)
                    off += csBytes
                    g.convParity = 0
                }
                if let p = rec.weights.ple {
                    let hBytes = p.history[0].bytes
                    arena.cpuPointer(of: p.history[0]).copyMemory(from: basePtr.advanced(by: off), byteCount: hBytes)
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
                    arena.cpuPointer(of: a.k).copyMemory(from: basePtr.advanced(by: off), byteCount: kvRowBytes); off += kvRowBytes
                    arena.cpuPointer(of: a.v).copyMemory(from: basePtr.advanced(by: off), byteCount: kvRowBytes); off += kvRowBytes
                    if metaRowBytes > 0 {
                        arena.cpuPointer(of: a.kMeta).copyMemory(from: basePtr.advanced(by: off), byteCount: metaRowBytes); off += metaRowBytes
                        arena.cpuPointer(of: a.vMeta).copyMemory(from: basePtr.advanced(by: off), byteCount: metaRowBytes); off += metaRowBytes
                    }
                    arena.cpuPointer(of: a.indexK).copyMemory(from: basePtr.advanced(by: off), byteCount: idxRowBytes); off += idxRowBytes
                    arena.cpuPointer(of: a.pooled).copyMemory(from: basePtr.advanced(by: off), byteCount: poolBytes); off += poolBytes
                }
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
                    arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride).copyMemory(from: base.advanced(by: offset), byteCount: kvRowStride); offset += kvRowStride
                    arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride).copyMemory(from: base.advanced(by: offset), byteCount: kvRowStride); offset += kvRowStride
                    if metaRowStride > 0 {
                        arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride).copyMemory(from: base.advanced(by: offset), byteCount: metaRowStride); offset += metaRowStride
                        arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride).copyMemory(from: base.advanced(by: offset), byteCount: metaRowStride); offset += metaRowStride
                    }
                    arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride).copyMemory(from: base.advanced(by: offset), byteCount: idxRowStride); offset += idxRowStride
                    let lastBlock = lastRow / 4
                    if (lastBlock + 1) * poolBlockStride <= a.pooled.bytes {
                        arena.cpuPointer(of: a.pooled).advanced(by: lastBlock * poolBlockStride).copyMemory(from: base.advanced(by: offset), byteCount: poolBlockStride)
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
                        arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride).copyMemory(from: base.advanced(by: offset + lastRow * kvRowStride), byteCount: kvRowStride); offset += kvRowBytes
                        arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride).copyMemory(from: base.advanced(by: offset + lastRow * kvRowStride), byteCount: kvRowStride); offset += kvRowBytes
                        if metaRowBytes > 0 {
                            arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride).copyMemory(from: base.advanced(by: offset + lastRow * metaRowStride), byteCount: metaRowStride); offset += metaRowBytes
                            arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride).copyMemory(from: base.advanced(by: offset + lastRow * metaRowStride), byteCount: metaRowStride); offset += metaRowBytes
                        }
                        arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride).copyMemory(from: base.advanced(by: offset + lastRow * idxRowStride), byteCount: idxRowStride); offset += idxRowBytes
                        let lastBlock = lastRow / 4
                        if (lastBlock + 1) * poolBlockStride <= poolBytes {
                            arena.cpuPointer(of: a.pooled).advanced(by: lastBlock * poolBlockStride).copyMemory(from: base.advanced(by: offset + lastBlock * poolBlockStride), byteCount: poolBlockStride)
                        }
                        offset += poolBytes
                    } else {
                        arena.cpuPointer(of: a.k).copyMemory(from: base.advanced(by: offset), byteCount: kvRowBytes); offset += kvRowBytes
                        arena.cpuPointer(of: a.v).copyMemory(from: base.advanced(by: offset), byteCount: kvRowBytes); offset += kvRowBytes
                        if metaRowBytes > 0 {
                            arena.cpuPointer(of: a.kMeta).copyMemory(from: base.advanced(by: offset), byteCount: metaRowBytes); offset += metaRowBytes
                            arena.cpuPointer(of: a.vMeta).copyMemory(from: base.advanced(by: offset), byteCount: metaRowBytes); offset += metaRowBytes
                        }
                        arena.cpuPointer(of: a.indexK).copyMemory(from: base.advanced(by: offset), byteCount: idxRowBytes); offset += idxRowBytes
                        arena.cpuPointer(of: a.pooled).copyMemory(from: base.advanced(by: offset), byteCount: poolBytes); offset += poolBytes
                    }
                }

                // When continuing with a suffix (not exact match), copy seed inputs before re-seeding MTP row (N - 1).
                if !isExact, nextTokenAfterPrefix != nil, let lastDecodeSlot = decodeStack.slots.last {
                    let s = lastDecodeSlot.scratch
                    arena.cpuPointer(of: s.h2).copyMemory(from: h2Src, byteCount: h2Bytes)
                    arena.cpuPointer(of: s.moeOut).copyMemory(from: moeSrc, byteCount: moeBytes)
                    arena.cpuPointer(of: s.injB).copyMemory(from: injSrc, byteCount: injBytes)
                    pendingMTPSeed = true
                }
            }

            var restoredDFlashOffset = 0
            if wantDFlash, let dflash = self.dflash, let marker = dflashPrefixMarker {
                guard let off = FNGPUDFlash.restorePrefixPayload(
                    from: base, dataCount: data.count, offset: &offset,
                    dflash: dflash, expectedMarker: marker, expectedTokens: N) else {
                    return nil
                }
                restoredDFlashOffset = off
            }

            residentKVWatermark.recordRestored(expectedTokens: expectedTokens,
                                               hasMTP: hasMTP || wantDFlash, inPlace: inPlaceKV)
            highWaterPos = max(highWaterPos, N)
            pendingInPlaceKV = inPlaceKV
            pendingN = N

            return FNRestoredPrefix(matched: N, lastArgmax: Int(hdr.lastArgmax), mtpDraft: Int(hdr.mtpDraft),
                                    dflashCacheOffset: restoredDFlashOffset)
        }
        guard let restored else { return nil }
        if !pendingInPlaceKV, pendingN >= 512, layerRecords.contains(where: { $0.attnState?.lsaClusters != nil }),
           let (cb, enc) = arena.hal.begin() {
            var encEnded = false
            defer { if !encEnded { enc.endEncoding() } }
            for rec in layerRecords {
                if let a = rec.attnState, let clusters = a.lsaClusters {
                    try QWGemm.encodeFnLSAClusterRange(
                        enc, hal: arena.hal, pooled: a.pooled, clusters: clusters,
                        pos0: 0, count: pendingN, indexDim: 128)
                }
            }
            encEnded = true
            arena.hal.end(cb, enc)
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
        }
        if pendingMTPSeed, let mtp = self.mtp, let nextTok = nextTokenAfterPrefix, let lastDecodeSlot = decodeStack.slots.last {
            let s = lastDecodeSlot.scratch
            let preInj = (base: s.h2, src: s.moeOut, scale: s.injB)
            _ = try mtp.forwardPrefill(streamBuf: decodeStack.lastStreamTensor, preInject: preInj,
                                       nextTokens: [nextTok], startPos: pendingN - 1,
                                       prefillScratch: mtp.scratch1, projectLastRow: false)
            gpuSec += mtp.lastGPUMillis / 1000.0
        }
        return restored
    }

}

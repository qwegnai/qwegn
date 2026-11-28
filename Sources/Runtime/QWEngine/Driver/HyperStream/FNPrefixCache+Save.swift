// FNPrefixCache+Save.swift — Prefix serialization and standalone base mounts for QWHyperStreamRuntime.

import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWAttn
import QWHyperPLE

extension QWHyperStreamRuntime {
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
        try? savePrefixCacheOrThrow(
            history: history,
            lastChunkRows: lastChunkRows,
            useSpec: useSpec,
            lastArgmax: lastArgmax,
            mtpDraft: mtpDraft,
            cacheDir: cacheDir,
            isSharedBase: isSharedBase,
            customOutputFileURL: customOutputFileURL
        )
    }

    func savePrefixCacheOrThrow(
        history: [Int64],
        lastChunkRows: Int,
        useSpec: Bool,
        lastArgmax: Int,
        mtpDraft: Int,
        cacheDir: URL?,
        isSharedBase: Bool = false,
        customOutputFileURL: URL? = nil
    ) throws {
        let N = history.count
        guard N > 0 else { return }
        if customOutputFileURL == nil {
            guard prefixCacheSSDByteLimit > 0 || prefixCacheWarmStore != nil else { return }
            guard N >= 128 else { return }
        }
        let fm = FileManager.default
        let hasMTP = (useSpec || hasPrefixCacheTier) && mtp != nil
        let hasDFlash = (useSpec || hasPrefixCacheTier) && dflash != nil && dflashPrefixMarker != nil
        let hash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP,
                                      dflashMarker: hasDFlash ? dflashPrefixMarker : nil)
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
        if hasDFlash, let dflash = self.dflash {
            let dflashBytes = FNGPUDFlash.prefixPayloadBytes(config: dflash.config, tokenCount: N)
            fullTotalBytes += dflashBytes
            compactTotalBytes += dflashBytes
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
            hasMTP: (hasMTP ? 1 : 0) | (isCompactInPlace ? 2 : 0) | (hasDFlash ? FNPrefixHeader.dflashBit : 0),
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

        let rootHash = FNPrefixWarmStore.rootHash(history: history, kvMode: kvMode,
                                                  hasMTP: hasMTP, dflashMarker: dflashPrefixMarker)
        var resolvedSlot: Int? = nil
        if fitsWarm, let warm = prefixCacheWarmStore {
            if isSharedBase {
                warm.makeRoom(for: totalBytes, activeSlot: nil)
            } else {
                let slot = warm.resolveSlot(rootHash: rootHash, history: history, kvMode: kvMode,
                                            hasMTP: hasMTP, dflashMarker: dflashPrefixMarker)
                resolvedSlot = slot
                warm.prepareSlotForStore(slot: slot, rootHash: rootHash, history: history,
                                         incomingBytes: totalBytes, kvMode: kvMode,
                                         hasMTP: hasMTP, dflashMarker: dflashPrefixMarker)
            }
        }

        let headLogitsPtr = UnsafeRawPointer(arena.cpuPointer(of: head.logits))
        let boundaryPtrs: (h2: UnsafeRawPointer, moe: UnsafeRawPointer, inj: UnsafeRawPointer)?
        if lastChunkRows == 1, let lastSlot = decodeStack.slots.last {
            let s = lastSlot.scratch
            boundaryPtrs = (
                UnsafeRawPointer(arena.cpuPointer(of: s.h2)),
                UnsafeRawPointer(arena.cpuPointer(of: s.moeOut)),
                UnsafeRawPointer(arena.cpuPointer(of: s.injB))
            )
        } else if let cached = cachedPrefill, let pArena = prefillArena, let lastSlot = cached.stack.slots.last {
            let s = lastSlot.scratch
            let r = max(0, lastChunkRows - 1)
            boundaryPtrs = (
                UnsafeRawPointer(pArena.cpuPointer(of: s.h2).advanced(by: r * h2Bytes)),
                UnsafeRawPointer(pArena.cpuPointer(of: s.moeOut).advanced(by: r * moeBytes)),
                UnsafeRawPointer(pArena.cpuPointer(of: s.injB).advanced(by: r * injBytes))
            )
        } else {
            boundaryPtrs = nil
        }
        let layerChunks: [[(addr: Int, bytes: Int)]] = layerRecords.map { rec in
            var chunks: [(addr: Int, bytes: Int)] = []
            if let g = rec.gdn {
                let s = g.states[g.stateParity & QWFourSlotRollbackRing.ringMask]
                let cs = g.convState[g.convParity & QWFourSlotRollbackRing.ringMask]
                chunks.append((Int(bitPattern: arena.cpuPointer(of: s)), s.bytes))
                chunks.append((Int(bitPattern: arena.cpuPointer(of: cs)), cs.bytes))
            }
            if let p = rec.weights.ple {
                let h = p.history[p.parity & QWFourSlotRollbackRing.ringMask]
                chunks.append((Int(bitPattern: arena.cpuPointer(of: h)), h.bytes))
            }
            if !isCompactInPlace, let a = rec.attnState {
                let cap = max(1, a.capacity)
                let kvRowBytes = (a.k.bytes / cap) * N
                let metaRowBytes = (a.kMeta.bytes > 2) ? (a.kMeta.bytes / cap) * N : 0
                let idxRowBytes = (a.indexK.bytes / cap) * N
                let poolBlocks = (N + 3) / 4
                let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                chunks.append((Int(bitPattern: arena.cpuPointer(of: a.k)), kvRowBytes))
                chunks.append((Int(bitPattern: arena.cpuPointer(of: a.v)), kvRowBytes))
                if metaRowBytes > 0 {
                    chunks.append((Int(bitPattern: arena.cpuPointer(of: a.kMeta)), metaRowBytes))
                    chunks.append((Int(bitPattern: arena.cpuPointer(of: a.vMeta)), metaRowBytes))
                }
                chunks.append((Int(bitPattern: arena.cpuPointer(of: a.indexK)), idxRowBytes))
                chunks.append((Int(bitPattern: arena.cpuPointer(of: a.pooled)), poolBytes))
            }
            return chunks
        }
        var mtpChunks: [(ptr: UnsafeRawPointer, bytes: Int)] = []
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
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.k).advanced(by: lastRow * kvRowStride)), kvRowStride))
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.v).advanced(by: lastRow * kvRowStride)), kvRowStride))
                if metaRowStride > 0 {
                    mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.kMeta).advanced(by: lastRow * metaRowStride)), metaRowStride))
                    mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.vMeta).advanced(by: lastRow * metaRowStride)), metaRowStride))
                }
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.indexK).advanced(by: lastRow * idxRowStride)), idxRowStride))
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.pooled).advanced(by: min(lastBlock * poolBlockStride, max(0, a.pooled.bytes - poolBlockStride)))), poolBlockStride))
            } else {
                let kvRowBytes = kvRowStride * N
                let metaRowBytes = metaRowStride * N
                let idxRowBytes = idxRowStride * N
                let poolBlocks = (N + 3) / 4
                let poolBytes = min(a.pooled.bytes, poolBlockStride * poolBlocks)
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.k)), kvRowBytes))
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.v)), kvRowBytes))
                if metaRowBytes > 0 {
                    mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.kMeta)), metaRowBytes))
                    mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.vMeta)), metaRowBytes))
                }
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.indexK)), idxRowBytes))
                mtpChunks.append((UnsafeRawPointer(arena.cpuPointer(of: a.pooled)), poolBytes))
            }
        }

        var out = Data(count: totalBytes)
        withExtendedLifetime((arena, prefillArena)) {
            out.withUnsafeMutableBytes { (rawOut: UnsafeMutableRawBufferPointer) in
                guard let dstBase = rawOut.baseAddress else { return }
                var writeOffset = 0
                func appendRaw(_ src: UnsafeRawPointer, _ count: Int) {
                    guard count > 0 else { return }
                    dstBase.advanced(by: writeOffset).copyMemory(from: src, byteCount: count)
                    writeOffset += count
                }

                withUnsafeBytes(of: &hdr) { (hdrBuf: UnsafeRawBufferPointer) in
                    if let p = hdrBuf.baseAddress { appendRaw(p, hdrBytes) }
                }
                history.withUnsafeBytes { (histBuf: UnsafeRawBufferPointer) in
                    if let p = histBuf.baseAddress { appendRaw(p, tokBytes) }
                }
                appendRaw(headLogitsPtr, logitBytes)

                if let b = boundaryPtrs {
                    appendRaw(b.h2, h2Bytes)
                    appendRaw(b.moe, moeBytes)
                    appendRaw(b.inj, injBytes)
                } else {
                    memset(dstBase.advanced(by: writeOffset), 0, h2Bytes + moeBytes + injBytes)
                    writeOffset += h2Bytes + moeBytes + injBytes
                }

                let dstAddr = Int(bitPattern: dstBase)
                let offsets = layerOffsets
                DispatchQueue.concurrentPerform(iterations: layerChunks.count) { idx in
                    var off = offsets[idx]
                    let basePtr = UnsafeMutableRawPointer(bitPattern: dstAddr)!
                    for (srcAddr, count) in layerChunks[idx] where count > 0 {
                        basePtr.advanced(by: off).copyMemory(from: UnsafeRawPointer(bitPattern: srcAddr)!, byteCount: count)
                        off += count
                    }
                }

                writeOffset = mtpOffset
                for (src, count) in mtpChunks {
                    appendRaw(src, count)
                }
                if hasDFlash, let dflash = self.dflash, let marker = dflashPrefixMarker {
                    FNGPUDFlash.appendPrefixPayload(to: dstBase, at: &writeOffset, dflash: dflash, marker: marker, tokenCount: N)
                }
            }
        }

        // The cold tier holds the prefix for the next run; the warm tier
        // holds it for the next request.
        if let warm = prefixCacheWarmStore, warm.isEnabled {
            if isSharedBase ? (out.count <= warm.byteLimit) : warm.canHold(bytes: out.count) {
                if isSharedBase {
                    warm.store(name: filename, blob: out, slot: nil, rootHash: nil, history: history, isSharedBase: true)
                } else if let resolvedSlot {
                    warm.store(name: filename, blob: out, slot: resolvedSlot, rootHash: rootHash, history: history, isSharedBase: false)
                } else {
                    _ = warm.storeWithSlotPinning(
                        name: filename, blob: out, history: history,
                        kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashPrefixMarker, isSharedBase: false
                    )
                }
                onCacheRetained?(N, out.count, warm.state.bytes, warm.byteLimit)
            } else {
                onCacheSkippedOversize?(N, out.count, warm.byteLimit)
            }
        }
        if let customOutputFileURL {
            try fm.createDirectory(at: customOutputFileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try out.write(to: customOutputFileURL, options: .atomic)
        }
        if !isCompactInPlace,
           QWTwoTierPrefixCache.shouldSpillToCold(
               isSharedBase: isSharedBase,
               warmHasSharedBase: prefixCacheWarmStore?.hasSharedBase == true,
               ssdByteLimit: prefixCacheSSDByteLimit,
               entryBytes: out.count,
               maxSpillBytes: Int.max
           ),
           let cacheDir, customOutputFileURL == nil {
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
        pinAsSharedBase: Bool = true,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        guard !tokens.isEmpty else {
            throw NSError(domain: "QWEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot build empty standalone prefix cache"])
        }
        if !(prefixCacheWarmStore?.isEnabled ?? false) {
            prefixCacheWarmStore = FNPrefixWarmStore(byteLimit: 64 << 20, maxSlots: maxConcurrentSessions)
        }
        try preparePrefill(forTokenCount: tokens.count + 16)
        var session = FNGPUSessionData(
            history: tokens.map(Int64.init),
            sampler: QWSampler(config: .greedy),
            useSpec: mtp != nil || dflash != nil,
            onPrefillProgress: onPrefillProgress
        )
        session.restoreDone = true
        wipeSessionState(overwriteTokens: tokens.count)
        residentKVHasMTP = mtp != nil || dflash != nil
        highWaterPos = max(highWaterPos, min(activeCapacity, tokens.count + 4))
        beginSession(&session, maxNew: 1)
        let first = try encodePrefillChunk(&session, chunk: 0..<tokens.count, runsHead: true)
        try savePrefixCacheOrThrow(
            history: session.history,
            lastChunkRows: max(1, session.lastRows),
            useSpec: mtp != nil || dflash != nil,
            lastArgmax: first,
            mtpDraft: session.mtpDraft,
            cacheDir: nil,
            isSharedBase: pinAsSharedBase,
            customOutputFileURL: outputURL
        )
        evictPrefillScratchIfNeeded()
        let writtenBytes = Self.fileSize(at: outputURL)
        if pinAsSharedBase {
            _ = try mountStandalonePrefixCache(from: outputURL)
        }
        return writtenBytes
    }

    /// Mounts a standalone `.qwcache` file from `fileURL` into the warm prefix store as `isSharedBase: true`.
    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        let data = try Data(contentsOf: fileURL)
        let hdrSize = MemoryLayout<FNPrefixHeader>.size
        guard data.count >= hdrSize else {
            throw NSError(domain: "QWEngine", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "invalid HyperStream .qwcache header"])
        }
        var hdr = FNPrefixHeader(magic: 0, version: 0, tokenCount: 0, kvModeCode: 0,
                                 hasMTP: 0, lastArgmax: 0, mtpDraft: 0,
                                 vocabCount: 0, hiddenSize: 0, widthSize: 0, hcSize: 0)
        let copiedHdr = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            guard let base = raw.baseAddress, raw.count >= hdrSize else { return false }
            withUnsafeMutableBytes(of: &hdr) { (dst: UnsafeMutableRawBufferPointer) in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: base, count: hdrSize))
            }
            return true
        }
        guard copiedHdr else {
            throw NSError(domain: "QWEngine", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "empty HyperStream .qwcache payload"])
        }
        let n = Int(hdr.tokenCount)
        guard hdr.magic == FNPrefixHeader.magic, hdr.version == FNPrefixHeader.version,
              hdr.kvModeCode == kvModeCode(kvMode),
              Int(hdr.vocabCount) == vocab, Int(hdr.hiddenSize) == hidden,
              Int(hdr.widthSize) == width, Int(hdr.hcSize) == hc,
              n > 0, n <= activeCapacity else {
            throw NSError(domain: "QWEngine", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "HyperStream .qwcache geometry, capacity, or kvMode mismatch"])
        }
        let tokBytes = n * MemoryLayout<Int64>.size
        guard data.count >= hdrSize + tokBytes else {
            throw NSError(domain: "QWEngine", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "truncated HyperStream .qwcache token table"])
        }
        let history: [Int64] = try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else {
                throw NSError(domain: "QWEngine", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "empty HyperStream .qwcache payload"])
            }
            let ptr = base.advanced(by: hdrSize).assumingMemoryBound(to: Int64.self)
            return Array(UnsafeBufferPointer(start: ptr, count: n))
        }
        let hasMTP = (hdr.hasMTP & 1) != 0
        let fileHasDFlash = (hdr.hasMTP & FNPrefixHeader.dflashBit) != 0
        if fileHasDFlash, let dflash = self.dflash, let marker = dflashPrefixMarker {
            guard FNGPUDFlash.validatePrefixBlob(data: data, expectedMarker: marker, expectedTokens: n, config: dflash.config) else {
                throw NSError(domain: "QWEngine", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "HyperStream .qwcache DFlash state or drafter geometry mismatch"])
            }
        } else if !fileHasDFlash, self.dflash != nil {
            throw NSError(domain: "QWEngine", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "HyperStream .qwcache lacks DFlash state; rebuild with qwg index --mtp-backend dflash2"])
        }
        let hash = Self.fnPrefixHash(tokens: history[...], kvMode: kvMode, hasMTP: hasMTP,
                                      dflashMarker: fileHasDFlash ? dflashPrefixMarker : nil)
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

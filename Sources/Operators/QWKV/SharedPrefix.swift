// SharedPrefix.swift — QWKVCache prefix capture/adoption and shared
// page refcounting (split from QWKVCache.swift; same-module extension).
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

extension QWKVCache {
    public func serializedPlaneBytes(tokenCount: Int) -> Int {
        guard tokenCount > 0 else { return 0 }
        if mode == .f16 {
            return tokenCount * Self.rowBytes
        }
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        return required * bytesPerPage
    }

    /// The caller must complete all GPU writes before capture or adoption.
    public func capturePrivatePrefix(_ h: QWKVHandle, maxBytes: Int) throws -> QWKVPrivatePrefix {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("private prefix needs a live sequence in this cache")
        }
        let count = st.writtenTokens
        guard count >= 0, count <= h.tokenCapacity, count <= Self.maxTokensPerSeq, maxBytes >= 0 else {
            throw QWKVError.badArgument("private prefix capacity or byte budget invalid")
        }
        let planeBytes = serializedPlaneBytes(tokenCount: count), total = planeBytes * layers.count * 2
        guard total <= maxBytes else {
            throw QWKVError.badArgument("private prefix exceeds capacity or byte budget")
        }
        var planes: [Data] = []
        planes.reserveCapacity(layers.count * 2)
        for (li, layer) in layers.enumerated() {
            for kind in QWKVKind.allCases {
                let table = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let required = (count + Self.pageTokens - 1) / Self.pageTokens
                guard table.count >= required else { throw QWKVError.badArgument("short private prefix page table") }
                let region = pool(layer, kind).pages.region
                guard table.prefix(required).allSatisfy({ $0 >= 0 && Int($0) < region.length / bytesPerPage }) else {
                    throw QWKVError.badArgument("invalid private prefix page ID")
                }
                var data = Data(count: planeBytes)
                data.withUnsafeMutableBytes { destination in
                    for page in 0..<required {
                        let bytes = mode == .f16
                            ? min(Self.pageTokens, count - page * Self.pageTokens) * Self.rowBytes
                            : bytesPerPage
                        destination.baseAddress!.advanced(by: page * bytesPerPage)
                            .copyMemory(from: region.pointer.advanced(by: Int(table[page]) * bytesPerPage),
                                        byteCount: bytes)
                    }
                }
                planes.append(data)
            }
        }
        return QWKVPrivatePrefix(tokenCount: count, memoryBytes: total, mode: mode, cache: self, planes: planes)
    }

    /// Capture an immutable shared-page prefix after all device writes have
    /// completed. The donor's table entries become shared too, so its future
    /// writes must pass through `prepareWrites` (or `append`, which does so).
    public func captureSharedPrefix(_ h: QWKVHandle) throws -> QWKVSharedPrefix {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("shared prefix needs a live sequence in this cache")
        }
        let count = st.writtenTokens
        guard count >= 0, count <= h.tokenCapacity, count <= Self.maxTokensPerSeq else {
            throw QWKVError.badArgument("shared prefix token extent invalid")
        }
        let required = (count + Self.pageTokens - 1) / Self.pageTokens
        var pages = Array(repeating: [Int32](), count: layers.count * 2)
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let ids = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let limit = pool(layer, kind).pages.pageCount
                guard ids.count >= required else {
                    throw QWKVError.badArgument("shared prefix page table is short")
                }
                let prefixIDs = Array(ids.prefix(required))
                guard prefixIDs.allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                    throw QWKVError.badArgument("shared prefix page ID is invalid")
                }
                pages[li * 2 + ki] = prefixIDs
            }
        }

        for (li, _) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let flags = kind == .k ? st.sharedK[li] : st.sharedV[li]
                let ids = pages[li * 2 + ki]
                for index in 0..<required {
                    if !flags[index] {
                        if kind == .k { st.sharedK[li][index] = true }
                        else { st.sharedV[li][index] = true }
                        retainSharedPage(poolIndex(li, kind), ids[index])
                    }
                    retainSharedPage(poolIndex(li, kind), ids[index])
                }
            }
        }
        return QWKVSharedPrefix(tokenCount: count, cache: self, epoch: cacheEpoch, pages: pages, mode: mode)
    }

    /// Release the prefix's owning reference. Safe to call more than once;
    /// dropping the donor or any adopter does not implicitly release it.
    /// This method serializes with every page-pool mutation through `lock`;
    public func releaseSharedPrefix(_ prefix: QWKVSharedPrefix) {
        lock.lock(); defer { lock.unlock() }
        guard prefix.cache === self, prefix.epoch == cacheEpoch, !prefix.released else { return }
        prefix.released = true
        for (li, _) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                for page in prefix.pages[li * 2 + ki] {
                    releaseSharedPage(poolIndex(li, kind), page)
                }
            }
        }
    }

    /// Exports written planes for an immutable shared prefix into contiguous byte buffers for disk serialization.
    /// Streams shared prefix pages directly to an output closure without intermediate heap allocations.
    public func withSharedPrefixPagePointers(
        _ prefix: QWKVSharedPrefix,
        _ body: (_ planeIndex: Int, _ pageIndex: Int, _ pointer: UnsafeRawPointer, _ byteCount: Int) throws -> Void
    ) throws {
        lock.lock(); defer { lock.unlock() }
        guard prefix.cache === self, prefix.epoch == cacheEpoch, !prefix.released else {
            throw QWKVError.badArgument("shared prefix invalid or released")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let planeIdx = li * 2 + ki
                let ids = prefix.pages[planeIdx]
                let region = pool(layer, kind).pages.region
                for page in 0..<required {
                    let bytes = mode == .f16
                        ? min(Self.pageTokens, prefix.tokenCount - page * Self.pageTokens) * Self.rowBytes
                        : bytesPerPage
                    let ptr = UnsafeRawPointer(region.pointer.advanced(by: Int(ids[page]) * bytesPerPage))
                    try body(planeIdx, page, ptr, bytes)
                }
            }
        }
    }

    /// Streams directly into an admitted sequence's private pages without intermediate heap allocations,
    /// then marks `tokenCount` tokens written.
    public func populatePrivatePagesDirectly(
        into h: QWKVHandle,
        tokenCount: Int,
        _ body: (_ planeIndex: Int, _ pageIndex: Int, _ pointer: UnsafeMutableRawPointer, _ byteCount: Int) throws -> Void
    ) throws {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, let st = seqs[h.seq], st.writtenTokens == 0,
              tokenCount >= 0, tokenCount <= h.tokenCapacity else {
            throw QWKVError.badArgument("direct page population destination invalid")
        }
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let planeIdx = li * 2 + ki
                let table = kind == .k ? st.tablesK[li] : st.tablesV[li]
                guard table.count >= required else {
                    throw QWKVError.badArgument("direct page population destination capacity invalid")
                }
                let region = pool(layer, kind).pages.region
                for page in 0..<required {
                    let bytes = mode == .f16
                        ? min(Self.pageTokens, tokenCount - page * Self.pageTokens) * Self.rowBytes
                        : bytesPerPage
                    let ptr = region.pointer.advanced(by: Int(table[page]) * bytesPerPage)
                    try body(planeIdx, page, ptr, bytes)
                }
            }
        }
        st.writtenTokens = tokenCount
    }

    /// Reconstructs a private prefix from deserialized plane data for adoption.
    public func validatePrivateAdoption(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        lock.lock(); defer { lock.unlock() }
        try validatePrivateAdoptionLocked(prefix, into: h)
    }

    /// Requires admission with enough private pages; never grows or shares a
    /// page. Every check completes before the first destination byte changes.
    public func adoptPrivatePrefix(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        lock.lock(); defer { lock.unlock() }
        try validatePrivateAdoptionLocked(prefix, into: h)
        let st = seqs[h.seq]!
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let table = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let region = pool(layer, kind).pages.region
                prefix.planes[li * 2 + ki].withUnsafeBytes { source in
                    for page in 0..<required {
                        let bytes = mode == .f16
                            ? min(Self.pageTokens, prefix.tokenCount - page * Self.pageTokens) * Self.rowBytes
                            : bytesPerPage
                        region.pointer.advanced(by: Int(table[page]) * bytesPerPage)
                            .copyMemory(from: source.baseAddress!.advanced(by: page * bytesPerPage),
                                        byteCount: bytes)
                    }
                }
            }
        }
        st.writtenTokens = prefix.tokenCount
    }

    /// Export a shared prefix's planes as cold bytes (same serialization as
    /// capturePrivatePrefix), for disk adoption. Caller completes GPU writes.
    public func exportSharedPrefixPlanes(_ prefix: QWKVSharedPrefix) -> [Data] {
        lock.lock(); defer { lock.unlock() }
        let planeBytes = serializedPlaneBytes(tokenCount: prefix.tokenCount)
        var planes: [Data] = []
        planes.reserveCapacity(layers.count * 2)
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                var data = Data(count: planeBytes)
                let ids = prefix.pages[li * 2 + ki]
                let region = pool(layer, kind).pages.region
                data.withUnsafeMutableBytes { destination in
                    for page in 0..<ids.count {
                        let bytes = mode == .f16
                            ? min(Self.pageTokens, prefix.tokenCount - page * Self.pageTokens) * Self.rowBytes
                            : bytesPerPage
                        destination.baseAddress!.advanced(by: page * bytesPerPage)
                            .copyMemory(from: region.pointer.advanced(by: Int(ids[page]) * bytesPerPage),
                                        byteCount: bytes)
                    }
                }
                planes.append(data)
            }
        }
        return planes
    }

    /// Reconstruct a private prefix from deserialized plane data for adoption.
    public func makePrivatePrefix(tokenCount: Int, planes: [Data]) -> QWKVPrivatePrefix {
        let total = planes.reduce(0) { $0 + $1.count }
        return QWKVPrivatePrefix(tokenCount: tokenCount, memoryBytes: total, mode: mode, cache: self, planes: planes)
    }

    func validatePrivateAdoptionLocked(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        let expectedPlaneBytes = serializedPlaneBytes(tokenCount: prefix.tokenCount)
        guard prefix.cache === self, h.cache === self, !h.finished,
              let st = seqs[h.seq], st.writtenTokens == 0,
              prefix.tokenCount >= 0, prefix.tokenCount <= h.tokenCapacity,
              prefix.planes.count == layers.count * 2,
              prefix.planes.allSatisfy({ $0.count == expectedPlaneBytes }) else {
            throw QWKVError.badArgument("private prefix identity, extent or empty destination invalid")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for li in layers.indices {
            guard st.tablesK[li].count >= required, st.tablesV[li].count >= required else {
                throw QWKVError.badArgument("private prefix destination capacity invalid")
            }
            let limitK = poolsFlat[li * 2].pages.region.length / bytesPerPage
            let limitV = poolsFlat[li * 2 + 1].pages.region.length / bytesPerPage
            guard st.tablesK[li].prefix(required).allSatisfy({ $0 >= 0 && Int($0) < limitK }),
                  st.tablesV[li].prefix(required).allSatisfy({ $0 >= 0 && Int($0) < limitV }) else {
                throw QWKVError.badArgument("invalid private destination page ID")
            }
        }
    }

    public func release(_ snap: QWKVSnap) {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[snap.seq] else { return }
        for (l, kind, ids) in snap.pins {
            let key = "\(l):\(kind.rawValue)"
            guard let kli = layerToLi[l] else { continue }   // P8: O(1), no alloc
            var bucket = st.pins[key] ?? [:]
            let table = kind == .k ? st.tablesK[kli] : st.tablesV[kli]
            let inTable = Set(table)
            for p in ids {
                let c = bucket[p, default: 0] - 1
                if c <= 0 {
                    bucket[p] = nil
                    // A page dropped by rollback (pin held it) returns to its
                    // pool exactly when its last pin releases and no table
                    // entry references it — accounting stays exact:
                    // consumed == total − free == Σ live table entries.
                    if !inTable.contains(p), !isSharedPageReferenced(poolIndex(kli, kind), p) {
                        pool(l, kind).pages.freePage(Int(p))
                    }
                } else { bucket[p] = c }
            }
            st.pins[key] = bucket
        }
    }

    /// MTP/speculative rollback: drop the last `by` tokens. Whole pages past
    /// the new end return to their pools; a page still pinned by a snapshot is
    /// NOT freed (the snapshot owns it until release()) but its TABLE entry
    public func rollback(_ h: QWKVHandle, by: Int) throws {
        guard by >= 0 else { throw QWKVError.badArgument("negative rollback") }
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("unknown, foreign, or finished seq")
        }
        let newEnd = max(0, st.writtenTokens - by)
        if newEnd < st.writtenTokens { st.rollbackGeneration += 1 }
        let keepPages = (newEnd + Self.pageTokens - 1) / Self.pageTokens
        for (li, l) in layers.enumerated() {
            for kind in QWKVKind.allCases {
                let key = "\(l):\(kind.rawValue)"
                let bucket = st.pins[key] ?? [:]
                while (kind == .k ? st.tablesK[li].count : st.tablesV[li].count) > keepPages {
                    let p = (kind == .k ? st.tablesK[li].popLast() : st.tablesV[li].popLast())!
                    let wasShared = kind == .k ? st.sharedK[li].removeLast() : st.sharedV[li].removeLast()
                    let cnt = bucket[p, default: 0]
                    if wasShared {
                        releaseSharedPage(poolIndex(li, kind), p)
                    } else if cnt == 0 {
                        pool(l, kind).pages.freePage(Int(p))   // pinned → snapshot owns it
                    }
                }
                st.pins[key] = bucket
                uploadSlot(h.seq, li, kind, kind == .k ? st.tablesK[li] : st.tablesV[li])
            }
        }
        st.writtenTokens = newEnd
        h.tokenCapacity = (st.tablesK.first?.count ?? 0) * Self.pageTokens
    }

    public func written(_ h: QWKVHandle) -> Int {
        lock.lock(); defer { lock.unlock() }
        return seqs[h.seq]?.writtenTokens ?? 0
    }

    public func validate(_ h: QWKVHandle) throws {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, seqs[h.seq] != nil else {
            throw QWKVError.badArgument("handle belongs to another cache or a finished sequence")
        }
    }

    /// Diagnostic state snapshots without copied KV bytes are invalidated by
    /// any rewind, even if subsequent writes return to the same cursor.
    public func rollbackGeneration(_ h: QWKVHandle) throws -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("unknown, foreign, or finished seq")
        }
        return st.rollbackGeneration
    }

    /// H4 FIX: raise the written-tokens cursor for GPU-side row
    /// materialisation. The decode/fold attention launches write rows
    /// pos…pos+N−1 IN KERNEL (the split-0 epilogue), so unlike the host
    public func markWritten(_ h: QWKVHandle, upTo tokens: Int) {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq] else { return }
        st.writtenTokens = max(st.writtenTokens, tokens)
    }

    /// Finish a sequence: EVERY page it holds returns to its pool and the
    /// (seq, ·, ·) device table slots are zeroed (a recycled seq id must
    /// never inherit a stale id binding — uploadSlot's tail-zero invariant
    public func finish(_ h: QWKVHandle) {
        lock.lock(); defer { lock.unlock() }
        guard h.cache === self, !h.finished else { return }
        guard let st = seqs.removeValue(forKey: h.seq) else { return }   // idempotent
        h.finished = true
        freeSeqs.append(h.seq)   // recycle the id (the removeValue guard above
        // makes double-finish impossible to double-push — the slot is gone)
        for (li, l) in layers.enumerated() {
            for kind in QWKVKind.allCases {
                let ids = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let shared = kind == .k ? st.sharedK[li] : st.sharedV[li]
                for (index, p) in ids.enumerated() {
                    if shared[index] { releaseSharedPage(poolIndex(li, kind), p) }
                    else { pool(l, kind).pages.freePage(Int(p)) }
                }
                uploadSlot(h.seq, li, kind, [])
            }
        }
        st.pins.removeAll()
        h.tokenCapacity = 0
    }
    /// Page ids for a (seq, layer, kind) — parity: burst-written ids ==
    /// streamed-written ids AND their bytes (spec gate).
    public func pageIDs(_ h: QWKVHandle, layer: Int, kind: QWKVKind) -> [Int32] {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq], let li = layerIndex(layer) else { return [] }
        return kind == .k ? st.tablesK[li] : st.tablesV[li]
    }
    public func markSteadyState() { lock.lock(); steady = true; lock.unlock() }
    public var freePageCount: Int {
        lock.lock(); defer { lock.unlock() }
        return poolsFlat.reduce(0) { $0 + $1.pages.freePages }
    }
    public var totalPages: Int { poolsFlat.reduce(0) { $0 + $1.pages.pageCount } }

    func validateSharedPrefixLocked(_ prefix: QWKVSharedPrefix) throws {
        guard prefix.cache === self, prefix.epoch == cacheEpoch,
              !prefix.released, prefix.tokenCount >= 0,
              prefix.tokenCount <= Self.maxTokensPerSeq,
              prefix.pages.count == layers.count * 2 else {
            throw QWKVError.badArgument("shared prefix identity, epoch, mode or extent invalid")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let ids = prefix.pages[li * 2 + ki]
                let limit = pool(layer, kind).pages.pageCount
                guard ids.count == required, ids.allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                    throw QWKVError.badArgument("shared prefix geometry or page IDs invalid")
                }
            }
        }
    }

    func poolIndex(_ li: Int, _ kind: QWKVKind) -> Int {
        li * 2 + (kind == .k ? 0 : 1)
    }

    func retainSharedPage(_ pool: Int, _ page: Int32) {
        let key = SharedPage(pool: pool, page: page)
        sharedPageRefs[key, default: 0] += 1
    }

    func isSharedPageReferenced(_ pool: Int, _ page: Int32) -> Bool {
        sharedPageRefs[SharedPage(pool: pool, page: page), default: 0] > 0
    }

    func releaseSharedPage(_ poolIndex: Int, _ page: Int32) {
        let key = SharedPage(pool: poolIndex, page: page)
        let count = sharedPageRefs[key, default: 0]
        guard count > 0 else { return }
        if count == 1 {
            sharedPageRefs[key] = nil
            if !isPagePinned(poolIndex, page) {
                poolsFlat[poolIndex].pages.freePage(Int(page))
            }
        } else {
            sharedPageRefs[key] = count - 1
        }
    }

    func isPagePinned(_ poolIndex: Int, _ page: Int32) -> Bool {
        let li = poolIndex / 2
        let kind: QWKVKind = poolIndex % 2 == 0 ? .k : .v
        let key = "\(layers[li]):\(kind.rawValue)"
        for state in seqs.values {
            if let pins = state.pins[key], pins[page, default: 0] > 0 { return true }
        }
        return false
    }
}

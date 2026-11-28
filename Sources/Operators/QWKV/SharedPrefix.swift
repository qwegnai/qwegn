// SharedPrefix.swift — QWKVCache prefix capture/adoption and shared page refcounting.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

extension QWKVCache {
    public func serializedPlaneBytes(tokenCount: Int) -> Int {
        guard tokenCount > 0 else { return 0 }
        if mode == .f16 { return tokenCount * f16RowBytes }
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        return required * bytesPerPage
    }

    public func serializedPlaneBytes(layerIndex li: Int, tokenCount: Int) -> Int {
        guard tokenCount > 0 else { return 0 }
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: tokenCount))
        guard startPage < required else { return 0 }
        if mode == .f16 { return (tokenCount - startPage * Self.pageTokens) * f16RowBytes }
        return (required - startPage) * bytesPerPage
    }

    public func sharedPrefixBytes(tokenCount: Int) -> Int {
        guard tokenCount > 0 else { return 0 }
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        return layers.indices.reduce(0) { acc, li in
            let livePages = max(0, required - min(required, liveStartPage(layerIndex: li, tokenCount: tokenCount)))
            return acc + 2 * livePages * bytesPerPage
        }
    }

    private func validPageBytes(page: Int, tokenCount: Int) -> Int {
        mode == .f16 ? min(Self.pageTokens, tokenCount - page * Self.pageTokens) * f16RowBytes : bytesPerPage
    }

    /// The caller must complete all GPU writes before capture or adoption.
    public func capturePrivatePrefix(_ h: QWKVHandle, maxBytes: Int) throws -> QWKVPrivatePrefix {
        try lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
                throw QWKVError.badArgument("private prefix needs a live sequence in this cache")
            }
            let count = st.writtenTokens
            guard count >= 0, count <= h.tokenCapacity, count <= Self.maxTokensPerSeq, maxBytes >= 0 else {
                throw QWKVError.badArgument("private prefix capacity or byte budget invalid")
            }
            let total = layers.indices.reduce(0) { acc, li in
                acc + 2 * serializedPlaneBytes(layerIndex: li, tokenCount: count)
            }
            guard total <= maxBytes else {
                throw QWKVError.badArgument("private prefix exceeds capacity or byte budget")
            }
            let required = (count + Self.pageTokens - 1) / Self.pageTokens
            var planes: [Data] = []
            planes.reserveCapacity(layers.count * 2)
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: count))
                let planeBytes = serializedPlaneBytes(layerIndex: li, tokenCount: count)
                for kind in QWKVKind.allCases {
                    let table = st.layerPairs[li][kind].ids
                    guard table.count >= required else { throw QWKVError.badArgument("short private prefix page table") }
                    let region = pool(at: li, kind).pages.region
                    let limit = region.length / bytesPerPage
                    guard startPage >= required || table[startPage..<required].allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                        throw QWKVError.badArgument("invalid private prefix page ID")
                    }
                    var data = Data(count: planeBytes)
                    let srcRegionPtr = UnsafeRawPointer(region.pointer)
                    withExtendedLifetime(region.buffer) {
                        data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                            guard let dstBase = raw.baseAddress else { return }
                            for page in startPage..<required {
                                let bytes = validPageBytes(page: page, tokenCount: count)
                                dstBase.advanced(by: (page - startPage) * bytesPerPage)
                                    .copyMemory(from: srcRegionPtr.advanced(by: Int(table[page]) * bytesPerPage),
                                                byteCount: bytes)
                            }
                        }
                    }
                    planes.append(data)
                }
            }
            return QWKVPrivatePrefix(tokenCount: count, memoryBytes: total, mode: mode, cache: self, planes: planes)
        }
    }

    /// Capture an immutable shared-page prefix after all device writes complete.
    public func captureSharedPrefix(_ h: QWKVHandle) throws -> QWKVSharedPrefix {
        try lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
                throw QWKVError.badArgument("shared prefix needs a live sequence in this cache")
            }
            let count = st.writtenTokens
            guard count >= 0, count <= h.tokenCapacity, count <= Self.maxTokensPerSeq else {
                throw QWKVError.badArgument("shared prefix token extent invalid")
            }
            let required = (count + Self.pageTokens - 1) / Self.pageTokens
            var pages = Array(repeating: [Int32](), count: layers.count * 2)
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: count))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let ids = st.layerPairs[li][kind].ids
                    let limit = pool(at: li, kind).pages.pageCount
                    guard ids.count >= required else { throw QWKVError.badArgument("shared prefix page table is short") }
                    let prefixIDs = (0..<required).map { idx in idx < startPage ? Int32(-1) : ids[idx] }
                    guard startPage >= required || prefixIDs[startPage..<required].allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                        throw QWKVError.badArgument("shared prefix page ID is invalid")
                    }
                    pages[li * 2 + ki] = prefixIDs
                }
            }
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: count))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let ids = pages[li * 2 + ki]
                    for index in startPage..<required {
                        if !st.layerPairs[li][kind].shared[index] {
                            st.layerPairs[li][kind].shared[index] = true
                            retainSharedPage(li: li, kind: kind, page: ids[index])
                        }
                        retainSharedPage(li: li, kind: kind, page: ids[index])
                    }
                }
            }
            return QWKVSharedPrefix(tokenCount: count, cache: self, epoch: cacheEpoch, pages: pages, mode: mode)
        }
    }

    /// Release the prefix's owning reference; safe to call multiple times.
    public func releaseSharedPrefix(_ prefix: QWKVSharedPrefix) {
        lock.withLock {
            guard prefix.cache === self, prefix.epoch == cacheEpoch, !prefix.released else { return }
            prefix.released = true
            for li in layers.indices {
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    for page in prefix.pages[li * 2 + ki] where page >= 0 {
                        releaseSharedPage(li: li, kind: kind, page: page)
                    }
                }
            }
        }
    }

    /// Slices an immutable shared prefix to an earlier `pageTokens`-aligned boundary by retaining physical pages.
    public func sliceSharedPrefix(_ prefix: QWKVSharedPrefix, toPageAlignedTokens targetTokens: Int) throws -> QWKVSharedPrefix? {
        guard targetTokens > 0, targetTokens % Self.pageTokens == 0 else { return nil }
        return try lock.withLock {
            try validateSharedPrefixLocked(prefix)
            guard targetTokens <= prefix.tokenCount else { return nil }
            let targetPages = targetTokens / Self.pageTokens
            var slicedPages = Array(repeating: [Int32](), count: layers.count * 2)
            for li in layers.indices {
                let targetStart = min(targetPages, liveStartPage(layerIndex: li, tokenCount: targetTokens))
                for (ki, _) in QWKVKind.allCases.enumerated() {
                    let srcIDs = prefix.pages[li * 2 + ki]
                    guard srcIDs.count >= targetPages else { return nil }
                    for pageIdx in targetStart..<targetPages where srcIDs[pageIdx] < 0 {
                        return nil
                    }
                    let prefixIDs = (0..<targetPages).map { idx in idx < targetStart ? Int32(-1) : srcIDs[idx] }
                    slicedPages[li * 2 + ki] = prefixIDs
                }
            }
            for li in layers.indices {
                let targetStart = min(targetPages, liveStartPage(layerIndex: li, tokenCount: targetTokens))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let ids = slicedPages[li * 2 + ki]
                    for pageIdx in targetStart..<targetPages {
                        retainSharedPage(li: li, kind: kind, page: ids[pageIdx])
                    }
                }
            }
            return QWKVSharedPrefix(tokenCount: targetTokens, cache: self, epoch: cacheEpoch, pages: slicedPages, mode: mode)
        }
    }

    public func withSharedPrefixPagePointers(
        _ prefix: QWKVSharedPrefix,
        _ body: (_ planeIndex: Int, _ pageIndex: Int, _ pointer: UnsafeRawPointer, _ byteCount: Int) throws -> Void
    ) throws {
        try lock.withLock {
            guard prefix.cache === self, prefix.epoch == cacheEpoch, !prefix.released else {
                throw QWKVError.badArgument("shared prefix invalid or released")
            }
            let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let planeIdx = li * 2 + ki
                    let ids = prefix.pages[planeIdx]
                    let region = pool(at: li, kind).pages.region
                    let limit = pool(at: li, kind).pages.pageCount
                    guard ids.count >= required else { throw QWKVError.badArgument("shared prefix page table short") }
                    for page in startPage..<required {
                        let pageID = ids[page]
                        guard pageID >= 0, Int(pageID) < limit else {
                            throw QWKVError.badArgument("shared prefix page ID invalid")
                        }
                        let ptr = UnsafeRawPointer(region.pointer.advanced(by: Int(pageID) * bytesPerPage))
                        try body(planeIdx, page, ptr, validPageBytes(page: page, tokenCount: prefix.tokenCount))
                    }
                }
            }
        }
    }

    private func ensureLivePagesAllocatedLocked(_ st: SeqState, handle: QWKVHandle, tokenCount: Int) throws {
        let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        guard required > 0 else { return }
        var allocs: [(li: Int, kind: QWKVKind, index: Int, page: Int32)] = []
        do {
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: tokenCount))
                guard startPage < required else { continue }
                for kind in QWKVKind.allCases {
                    let stream = st.layerPairs[li][kind]
                    guard stream.ids.count >= required else {
                        throw QWKVError.badArgument("destination capacity invalid")
                    }
                    for idx in startPage..<required where stream.ids[idx] < 0 {
                        guard let page = pool(at: li, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: 1, free: pool(at: li, kind).pages.freePages)
                        }
                        allocs.append((li, kind, idx, Int32(page)))
                    }
                }
            }
        } catch {
            for item in allocs { pool(at: item.li, item.kind).pages.freePage(Int(item.page)) }
            throw error
        }
        var touched: [(Int, QWKVKind)] = []
        for item in allocs {
            st.layerPairs[item.li][item.kind].ids[item.index] = item.page
            st.layerPairs[item.li][item.kind].shared[item.index] = false
            if !touched.contains(where: { $0.0 == item.li && $0.1 == item.kind }) {
                touched.append((item.li, item.kind))
            }
        }
        for (li, kind) in touched {
            uploadSlot(handle.seq, li, kind, st.layerPairs[li][kind].ids)
        }
    }

    public func populatePrivatePagesDirectly(
        into h: QWKVHandle,
        tokenCount: Int,
        _ body: (_ planeIndex: Int, _ pageIndex: Int, _ pointer: UnsafeMutableRawPointer, _ byteCount: Int) throws -> Void
    ) throws {
        try lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq], st.writtenTokens == 0,
                  tokenCount >= 0, tokenCount <= h.tokenCapacity else {
                throw QWKVError.badArgument("direct page population destination invalid")
            }
            try ensureLivePagesAllocatedLocked(st, handle: h, tokenCount: tokenCount)
            let required = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let planeIdx = li * 2 + ki
                    let table = st.layerPairs[li][kind].ids
                    guard table.count >= required else {
                        throw QWKVError.badArgument("direct page population destination capacity invalid")
                    }
                    let region = pool(at: li, kind).pages.region
                    for page in startPage..<required {
                        let ptr = region.pointer.advanced(by: Int(table[page]) * bytesPerPage)
                        try body(planeIdx, page, ptr, validPageBytes(page: page, tokenCount: tokenCount))
                    }
                }
            }
            st.writtenTokens = tokenCount
        }
    }

    public func validatePrivateAdoption(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        try lock.withLock { try validatePrivateAdoptionLocked(prefix, into: h) }
    }

    public func adoptPrivatePrefix(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        try lock.withLock {
            try validatePrivateAdoptionLocked(prefix, into: h)
            guard let st = seqs[h.seq] else { return }
            try ensureLivePagesAllocatedLocked(st, handle: h, tokenCount: prefix.tokenCount)
            let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let table = st.layerPairs[li][kind].ids
                    let region = pool(at: li, kind).pages.region
                    let dstRegionPtr = region.pointer
                    withExtendedLifetime(region.buffer) {
                        prefix.planes[li * 2 + ki].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                            guard let srcBase = raw.baseAddress else { return }
                            for page in startPage..<required {
                                let bytes = validPageBytes(page: page, tokenCount: prefix.tokenCount)
                                dstRegionPtr.advanced(by: Int(table[page]) * bytesPerPage)
                                    .copyMemory(from: srcBase.advanced(by: (page - startPage) * bytesPerPage), byteCount: bytes)
                            }
                        }
                    }
                }
            }
            st.writtenTokens = prefix.tokenCount
        }
    }

    func validatePrivateAdoptionLocked(_ prefix: QWKVPrivatePrefix, into h: QWKVHandle) throws {
        guard prefix.cache === self, h.cache === self, !h.finished,
              let st = seqs[h.seq], st.writtenTokens == 0,
              prefix.tokenCount >= 0, prefix.tokenCount <= h.tokenCapacity,
              prefix.planes.count == layers.count * 2 else {
            throw QWKVError.badArgument("private prefix identity, extent or empty destination invalid")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for li in layers.indices {
            let expectedPlaneBytes = serializedPlaneBytes(layerIndex: li, tokenCount: prefix.tokenCount)
            guard prefix.planes[li * 2].count == expectedPlaneBytes,
                  prefix.planes[li * 2 + 1].count == expectedPlaneBytes else {
                throw QWKVError.badArgument("private prefix plane size invalid")
            }
            let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
            for kind in QWKVKind.allCases {
                let ids = st.layerPairs[li][kind].ids
                guard ids.count >= required else {
                    throw QWKVError.badArgument("private prefix destination capacity invalid")
                }
                let limit = pool(at: li, kind).pages.region.length / bytesPerPage
                guard startPage >= required || ids[startPage..<required].allSatisfy({ $0 < 0 || Int($0) < limit }) else {
                    throw QWKVError.badArgument("invalid private destination page ID")
                }
            }
        }
    }

    public func release(_ snap: QWKVSnap) {
        lock.withLock {
            guard let st = seqs[snap.seq] else { return }
            for (l, kind, ids) in snap.pins {
                guard let kli = layerToLi[l] else { continue }
                let inTable = Set(st.layerPairs[kli][kind].ids.filter { $0 >= 0 })
                for p in ids where p >= 0 {
                    let c = st.layerPairs[kli][kind].pins[p, default: 0] - 1
                    if c <= 0 {
                        st.layerPairs[kli][kind].pins[p] = nil
                        if !inTable.contains(p), !isSharedPageReferenced(li: kli, kind: kind, page: p) {
                            pool(at: kli, kind).pages.freePage(Int(p))
                        }
                    } else {
                        st.layerPairs[kli][kind].pins[p] = c
                    }
                }
            }
        }
    }

    /// Speculative rollback: drop trailing tokens beyond `writtenTokens - by`.
    public func rollback(_ h: QWKVHandle, by: Int) throws {
        guard by >= 0 else { throw QWKVError.badArgument("negative rollback") }
        try lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
                throw QWKVError.badArgument("unknown, foreign, or finished seq")
            }
            let newEnd = max(0, st.writtenTokens - by)
            if newEnd < st.writtenTokens { st.rollbackGeneration += 1 }
            let keepPages = (newEnd + Self.pageTokens - 1) / Self.pageTokens
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    while st.layerPairs[li][kind].pageCount > keepPages {
                        guard let p = st.layerPairs[li][kind].ids.popLast() else { break }
                        let wasShared = st.layerPairs[li][kind].shared.removeLast()
                        guard p >= 0 else { continue }
                        let cnt = st.layerPairs[li][kind].pins[p, default: 0]
                        if wasShared {
                            releaseSharedPage(li: li, kind: kind, page: p)
                        } else if cnt == 0 {
                            pool(at: li, kind).pages.freePage(Int(p))
                        }
                    }
                    uploadSlot(h.seq, li, kind, st.layerPairs[li][kind].ids)
                }
            }
            st.writtenTokens = newEnd
            h.tokenCapacity = (st.layerPairs.first?.k.pageCount ?? 0) * Self.pageTokens
        }
    }

    public func written(_ h: QWKVHandle) -> Int {
        lock.withLock { seqs[h.seq]?.writtenTokens ?? 0 }
    }

    public func validate(_ h: QWKVHandle) throws {
        try lock.withLock {
            guard h.cache === self, !h.finished, seqs[h.seq] != nil else {
                throw QWKVError.badArgument("handle belongs to another cache or a finished sequence")
            }
        }
    }

    public func rollbackGeneration(_ h: QWKVHandle) throws -> UInt64 {
        try lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
                throw QWKVError.badArgument("unknown, foreign, or finished seq")
            }
            return st.rollbackGeneration
        }
    }

    public func markWritten(_ h: QWKVHandle, upTo tokens: Int) {
        lock.withLock {
            guard let st = seqs[h.seq] else { return }
            st.writtenTokens = max(st.writtenTokens, tokens)
        }
    }

    /// Finish a sequence, freeing unreferenced private pages and tail-zeroing table slots.
    public func finish(_ h: QWKVHandle) {
        lock.withLock {
            guard h.cache === self, !h.finished, let st = seqs[h.seq] else { return }
            h.finished = true
            var liveTables: [[Set<Int32>]] = Array(repeating: [Set<Int32>(), Set<Int32>()], count: layers.count)
            var leftoverPins: [[ [Int32: Int] ]] = Array(repeating: [[:], [:]], count: layers.count)
            for li in layers.indices {
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    liveTables[li][ki] = Set(st.layerPairs[li][kind].ids.filter { $0 >= 0 })
                    leftoverPins[li][ki] = st.layerPairs[li][kind].pins
                    st.layerPairs[li][kind].pins.removeAll()
                }
            }
            for li in layers.indices {
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    for (p, cnt) in leftoverPins[li][ki] where p >= 0 && cnt > 0 && !liveTables[li][ki].contains(p) {
                        if !isSharedPageReferenced(li: li, kind: kind, page: p)
                            && !isPagePinned(li: li, kind: kind, page: p) {
                            pool(at: li, kind).pages.freePage(Int(p))
                        }
                    }
                }
            }
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    let stream = st.layerPairs[li][kind]
                    for (index, p) in stream.ids.enumerated() where p >= 0 && stream.shared[index] {
                        releaseSharedPage(li: li, kind: kind, page: p)
                    }
                    uploadSlot(h.seq, li, kind, [])
                }
            }
            seqs.removeValue(forKey: h.seq)
            freeSeqs.append(h.seq)
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    let stream = st.layerPairs[li][kind]
                    for (index, p) in stream.ids.enumerated() where p >= 0 && !stream.shared[index] {
                        pool(at: li, kind).pages.freePage(Int(p))
                    }
                }
            }
            h.tokenCapacity = 0
        }
    }

    public func markSteadyState() { lock.withLock { steady = true } }
    public var freePageCount: Int {
        lock.withLock { poolPairs.reduce(0) { $0 + $1.freePages } }
    }

    func validateSharedPrefixLocked(_ prefix: QWKVSharedPrefix) throws {
        guard prefix.cache === self, prefix.epoch == cacheEpoch,
              !prefix.released, prefix.tokenCount >= 0,
              prefix.tokenCount <= Self.maxTokensPerSeq,
              prefix.pages.count == layers.count * 2 else {
            throw QWKVError.badArgument("shared prefix identity, epoch, mode or extent invalid")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for li in layers.indices {
            let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let ids = prefix.pages[li * 2 + ki]
                let limit = pool(at: li, kind).pages.pageCount
                guard ids.count == required,
                      ids.prefix(startPage).allSatisfy({ $0 < 0 }),
                      startPage >= required || ids[startPage..<required].allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                    throw QWKVError.badArgument("shared prefix geometry or page IDs invalid")
                }
            }
        }
    }

    func retainSharedPage(li: Int, kind: QWKVKind, page: Int32) {
        sharedPageRefs[SharedPage(li: li, kind: kind, page: page), default: 0] += 1
    }

    func isSharedPageReferenced(li: Int, kind: QWKVKind, page: Int32) -> Bool {
        sharedPageRefs[SharedPage(li: li, kind: kind, page: page), default: 0] > 0
    }

    func releaseSharedPage(li: Int, kind: QWKVKind, page: Int32) {
        let key = SharedPage(li: li, kind: kind, page: page)
        let count = sharedPageRefs[key, default: 0]
        guard count > 0 else { return }
        if count == 1 {
            sharedPageRefs[key] = nil
            if !isPagePinned(li: li, kind: kind, page: page) {
                pool(at: li, kind).pages.freePage(Int(page))
            }
        } else {
            sharedPageRefs[key] = count - 1
        }
    }

    func isPagePinned(li: Int, kind: QWKVKind, page: Int32) -> Bool {
        for state in seqs.values where state.layerPairs[li][kind].pins[page, default: 0] > 0 {
            return true
        }
        return false
    }
}

// QWKVLayerPair.swift — Paired K/V stream storage, pool pairs, and host page codec.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

struct QWKVStreamPages: Sendable {
    var ids: [Int32] = []
    var shared: [Bool] = []
    var pins: [Int32: Int] = [:]

    var pageCount: Int { ids.count }
}

struct QWKVLayerPair: Sendable {
    var k = QWKVStreamPages()
    var v = QWKVStreamPages()

    subscript(kind: QWKVKind) -> QWKVStreamPages {
        get { kind == .k ? k : v }
        set {
            switch kind {
            case .k: k = newValue
            case .v: v = newValue
            }
        }
    }
}

struct QWKVLayerPoolPair {
    final class Pool {
        let pages: QWPageArena
        init(_ p: QWPageArena) { pages = p }
    }

    let k: Pool
    let v: Pool

    subscript(kind: QWKVKind) -> Pool { kind == .k ? k : v }
    var totalPages: Int { k.pages.pageCount + v.pages.pageCount }
    var freePages: Int { k.pages.freePages + v.pages.freePages }
}

enum QWKVHostCodec {
    static func writeRows(
        mode: QWKVMode,
        geometry: QWModelGeometry,
        bytesPerPage: Int,
        rowElems: Int,
        positions: Range<Int>,
        table: [Int32],
        pageArena: QWPageArena,
        src: [UInt16]
    ) throws {
        let base0 = positions.lowerBound
        let arenaBase = pageArena.region.buffer.contents().advanced(by: pageArena.region.base)
        switch mode {
        case .f16:
            try withExtendedLifetime(pageArena.region.buffer) {
                try src.withUnsafeBufferPointer { s in
                    guard let base = s.baseAddress else { return }
                    for tok in positions {
                        let pg = tok / QWKVCache.pageTokens
                        let off = tok % QWKVCache.pageTokens
                        let pageID = Int(table[pg])
                        guard pageID >= 0 else { throw QWKVError.badArgument("unallocated KV page \(pg)") }
                        let dst = arenaBase.advanced(by: pageID * bytesPerPage + off * rowElems * 2)
                        dst.copyMemory(from: base + (tok - base0) * rowElems, byteCount: rowElems * 2)
                    }
                }
            }
        case .q8:
            let layout = QWKVCompressedLayout(mode: .q8, geometry: geometry)
            let effHeads = geometry.usesAnyMLA ? 1 : geometry.kvHeads
            let effHeadDim = geometry.usesAnyMLA ? geometry.kvChannels : geometry.headDim
            var frow = [Float](repeating: 0, count: rowElems)
            var qrow = [Int8](repeating: 0, count: rowElems)
            var srow = [Float16](repeating: 0, count: effHeads)
            try withExtendedLifetime(pageArena.region.buffer) {
                try src.withUnsafeBufferPointer { s in
                    guard let base = s.baseAddress else { return }
                    for tok in positions {
                        let pg = tok / QWKVCache.pageTokens
                        let off = tok % QWKVCache.pageTokens
                        let pageID = Int(table[pg])
                        guard pageID >= 0 else { throw QWKVError.badArgument("unallocated KV page \(pg)") }
                        let pageBase = arenaBase.advanced(by: pageID * bytesPerPage)
                        let srcRow = base + (tok - base0) * rowElems
                        for i in 0..<rowElems {
                            frow[i] = Float(Float16(bitPattern: srcRow[i]))
                        }
                        try QWKVCompressedLayout.quantizeQ8(
                            input: frow, qOut: &qrow, scalesOut: &srow,
                            kvHeads: effHeads, headDim: effHeadDim)
                        let qDst = pageBase.advanced(by: off * rowElems)
                        qrow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                            if let qSrc = raw.baseAddress {
                                qDst.copyMemory(from: qSrc, byteCount: rowElems)
                            }
                        }
                        let scaleBase = pageBase.advanced(by: layout.dataBytesPerPage + off * (effHeads * 2))
                        srow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                            if let sSrc = raw.baseAddress {
                                scaleBase.copyMemory(from: sSrc, byteCount: effHeads * 2)
                            }
                        }
                    }
                }
            }
        case .q4:
            let layout = QWKVCompressedLayout(mode: .q4, geometry: geometry)
            let effHeads = geometry.usesAnyMLA ? 1 : geometry.kvHeads
            let effHeadDim = geometry.usesAnyMLA ? geometry.kvChannels : geometry.headDim
            let groupsPerHead = max(1, (effHeadDim + 127) / 128)
            let totalGroups = effHeads * groupsPerHead
            var frow = [Float](repeating: 0, count: rowElems)
            var qrow = [UInt8](repeating: 0, count: rowElems / 2)
            var sc = [Float16](repeating: 0, count: totalGroups)
            var mn = [Float16](repeating: 0, count: totalGroups)
            try withExtendedLifetime(pageArena.region.buffer) {
                try src.withUnsafeBufferPointer { s in
                    guard let base = s.baseAddress else { return }
                    for tok in positions {
                        let pg = tok / QWKVCache.pageTokens
                        let off = tok % QWKVCache.pageTokens
                        let pageID = Int(table[pg])
                        guard pageID >= 0 else { throw QWKVError.badArgument("unallocated KV page \(pg)") }
                        let pageBase = arenaBase.advanced(by: pageID * bytesPerPage)
                        let srcRow = base + (tok - base0) * rowElems
                        for i in 0..<rowElems {
                            frow[i] = Float(Float16(bitPattern: srcRow[i]))
                        }
                        try QWKVCompressedLayout.quantizeQ4(
                            input: frow, qOut: &qrow, scalesOut: &sc, minsOut: &mn,
                            kvHeads: effHeads, headDim: effHeadDim)
                        let qDst = pageBase.advanced(by: off * (rowElems / 2))
                        qrow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                            if let qSrc = raw.baseAddress {
                                qDst.copyMemory(from: qSrc, byteCount: rowElems / 2)
                            }
                        }
                        let hBase = pageBase.advanced(by: layout.dataBytesPerPage + off * (totalGroups * 4))
                        for g in 0..<totalGroups {
                            hBase.assumingMemoryBound(to: Float16.self)[2 * g] = sc[g]
                            hBase.assumingMemoryBound(to: Float16.self)[2 * g + 1] = mn[g]
                        }
                    }
                }
            }
        }
    }

    static func buildPoolsAndTables(
        hal: QWHal,
        kv: QWSubAllocator,
        vSubAllocator: QWSubAllocator?,
        kvAux: QWSubAllocator?,
        layers: [Int],
        pageBytes: Int,
        tablesTotal: Int,
        layerPoolWeights: [Int]? = nil
    ) throws -> (poolPairs: [QWKVLayerPoolPair], tableBuffer: any MTLBuffer, tableBase: Int) {
        let layerCount = layers.count
        let kvAvail = kv.freeBytes
        let vAvail = vSubAllocator?.freeBytes ?? 0
        let auxAvail = kvAux?.freeBytes ?? 0
        let avail = kvAvail + vAvail + auxAvail
        guard tablesTotal < avail else { throw QWKVError.outOfPages(asked: tablesTotal, free: avail) }
        let rawPoolBytes = (avail - tablesTotal) / (2 * layerCount)
        guard rawPoolBytes >= pageBytes else {
            throw QWKVError.outOfPages(asked: pageBytes, free: rawPoolBytes)
        }
        let hasVSub = vSubAllocator != nil
        let hasAux = kvAux != nil
        func canPlace(_ counts: [Int], requireTablesRoom: Bool) -> Bool {
            var remK = kvAvail, remV = vAvail, remAux = auxAvail
            for li in 0..<layerCount {
                let alignedPool = qwAlignUp(counts[li] * pageBytes, 256)
                for kind in QWKVKind.allCases {
                    if kind == .k || !hasVSub {
                        if remK >= alignedPool { remK -= alignedPool }
                        else if hasAux && remAux >= alignedPool { remAux -= alignedPool }
                        else { return false }
                    } else {
                        if remV >= alignedPool { remV -= alignedPool }
                        else if hasAux && remAux >= alignedPool { remAux -= alignedPool }
                        else { return false }
                    }
                }
            }
            return !requireTablesRoom || remK >= tablesTotal || remV >= tablesTotal || remAux >= tablesTotal
        }

        let useProportional: Bool = {
            guard let w = layerPoolWeights, w.count == layerCount else { return false }
            return Set(w.map { max(1, $0) }).count > 1
        }()
        let pagesPerLayer: [Int]
        if useProportional, let rawWeights = layerPoolWeights {
            let weights = rawWeights.map { max(1, $0) }
            let maxWeight = weights.max() ?? 1
            let maxTop = max(1, (avail - tablesTotal) / (2 * pageBytes))
            func weightedCounts(_ top: Int) -> [Int] {
                weights.map { max(1, (top * $0) / maxWeight) }
            }
            func maxFeasibleTop(requireTablesRoom: Bool) -> Int {
                var lo = 1, hi = maxTop, best = 0
                while lo <= hi {
                    let mid = lo + (hi - lo) / 2
                    if canPlace(weightedCounts(mid), requireTablesRoom: requireTablesRoom) {
                        best = mid; lo = mid + 1
                    } else {
                        hi = mid - 1
                    }
                }
                return best
            }
            var bestTop = maxFeasibleTop(requireTablesRoom: true)
            if bestTop == 0 { bestTop = maxFeasibleTop(requireTablesRoom: false) }
            guard bestTop >= 1 else {
                throw QWKVError.outOfPages(asked: qwAlignUp(pageBytes, 256),
                                           free: max(kvAvail, max(vAvail, auxAvail)))
            }
            pagesPerLayer = weightedCounts(bestTop)
        } else {
            let maxCandidates = rawPoolBytes / pageBytes
            func maxFeasiblePages(requireTablesRoom: Bool) -> Int {
                var lo = 1, hi = maxCandidates, best = 0
                while lo <= hi {
                    let mid = lo + (hi - lo) / 2
                    if canPlace(Array(repeating: mid, count: layerCount), requireTablesRoom: requireTablesRoom) {
                        best = mid; lo = mid + 1
                    } else {
                        hi = mid - 1
                    }
                }
                return best
            }
            var pagesPerPool = maxFeasiblePages(requireTablesRoom: true)
            if pagesPerPool == 0 { pagesPerPool = maxFeasiblePages(requireTablesRoom: false) }
            guard pagesPerPool >= 1 else {
                throw QWKVError.outOfPages(asked: qwAlignUp(pageBytes, 256),
                                           free: max(kvAvail, max(vAvail, auxAvail)))
            }
            pagesPerLayer = Array(repeating: pagesPerPool, count: layerCount)
        }

        var remK = kvAvail, remV = vAvail, remAux = auxAvail
        func allocatePool(pages: Int, kind: QWKVKind) throws -> QWKVLayerPoolPair.Pool {
            let onePoolBytes = pages * pageBytes
            let alignedPoolBytes = qwAlignUp(onePoolBytes, 256)
            let targetSub: QWSubAllocator
            if let kvV = vSubAllocator, kind == .v {
                if remV >= alignedPoolBytes { remV -= alignedPoolBytes; targetSub = kvV }
                else if let aux = kvAux, remAux >= alignedPoolBytes { remAux -= alignedPoolBytes; targetSub = aux }
                else { remV -= alignedPoolBytes; targetSub = kvV }
            } else {
                if remK >= alignedPoolBytes { remK -= alignedPoolBytes; targetSub = kv }
                else if let aux = kvAux, remAux >= alignedPoolBytes { remAux -= alignedPoolBytes; targetSub = aux }
                else { remK -= alignedPoolBytes; targetSub = kv }
            }
            let off = try targetSub.allocate(bytes: onePoolBytes)
            let region = QWMRegion(buffer: targetSub.region.buffer,
                                   base: targetSub.region.base + off, length: onePoolBytes)
            return QWKVLayerPoolPair.Pool(QWPageArena(region: region, pageSize: pageBytes, pageCount: pages))
        }
        var pairs: [QWKVLayerPoolPair] = []
        pairs.reserveCapacity(layers.count)
        for li in layers.indices {
            let pages = pagesPerLayer[li]
            pairs.append(QWKVLayerPoolPair(k: try allocatePool(pages: pages, kind: .k),
                                           v: try allocatePool(pages: pages, kind: .v)))
        }

        // Table slots tail: zero-initialised so unwritten slots never expose stale page IDs.
        for sub in [kv, vSubAllocator, kvAux].compactMap({ $0 }) where sub.freeBytes >= tablesTotal {
            let tabOff = try sub.allocate(bytes: tablesTotal)
            memset(sub.region.pointer.advanced(by: tabOff), 0, tablesTotal)
            return (pairs, sub.region.buffer, sub.region.base + tabOff)
        }
        guard let tBuf = hal.device.makeBuffer(length: tablesTotal, options: .storageModeShared) else {
            throw QWKVError.outOfPages(asked: tablesTotal, free: 0)
        }
        memset(tBuf.contents(), 0, tablesTotal)
        return (pairs, tBuf, 0)
    }
}

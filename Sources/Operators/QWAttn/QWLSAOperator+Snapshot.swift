// QWLSAOperator+Snapshot.swift — Full and pinned-base delta (`QLSADLT1`)
// snapshot capture and restoration for QWLSAOperator (512-tok Ward & 4096-tok e-means).

import Foundation

extension QWLSAOperator {
    /// Registers or clears a pinned `.qwcache` base LSA state so subsequent turn captures/restores only copy delta blocks.
    public func setPinnedBaseState(_ data: Data?, position: Int) {
        let unitTok = clusterUnitTokens
        let baseAligned = (position / unitTok) * unitTok
        if let data, baseAligned >= unitTok, data.count == fullIndexerStateBytes(position: position) {
            pinnedBaseData = data
            pinnedBaseFullPosition = position
            pinnedBasePos512 = baseAligned
            residentPos512 = 0
        } else {
            pinnedBaseData = nil
            pinnedBaseFullPosition = 0
            pinnedBasePos512 = 0
            residentPos512 = 0
        }
    }

    /// Full uncompressed bytes required to snapshot pooled keys, pending ring rows, and coarse clusters at `position`.
    public func fullIndexerStateBytes(position: Int) -> Int {
        let rowBytes = indexDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let pendingBytes = (position % 4) * rowBytes
        let numUnits = position / clusterUnitTokens
        let unitBytes = 64 * rowBytes + clusterMemBytesPerUnit + clusterSizeBytesPerUnit
        return layers.count * (pooledBytes + pendingBytes + numUnits * unitBytes)
    }

    /// Compact delta bytes required to snapshot `(basePos512 ..< position)` above a chunk-aligned pinned base.
    public func deltaIndexerStateBytes(basePos512: Int, position: Int) -> Int {
        let rowBytes = indexDim * 2
        let unitTok = clusterUnitTokens
        let deltaPooledBytes = max(0, (position / 4) - (basePos512 / 4)) * rowBytes
        let pendingBytes = (position % 4) * rowBytes
        let deltaUnits = max(0, (position / unitTok) - (basePos512 / unitTok))
        let unitBytes = 64 * rowBytes + clusterMemBytesPerUnit + clusterSizeBytesPerUnit
        return 16 + layers.count * (deltaPooledBytes + pendingBytes + deltaUnits * unitBytes)
    }

    /// Bytes required to snapshot LSA state at `position` (uses delta encoding when extending a pinned base).
    public func indexerStateBytes(position: Int) -> Int {
        if pinnedBasePos512 > 0 && position > pinnedBasePos512 {
            return deltaIndexerStateBytes(basePos512: pinnedBasePos512, position: position)
        }
        return fullIndexerStateBytes(position: position)
    }

    /// Validates whether `data` is a well-formed full or delta LSA snapshot for `position`.
    public func isValidIndexerState(_ data: Data?, position: Int) -> Bool {
        guard position > 0 else { return true }
        guard let data else { return false }
        if data.count == fullIndexerStateBytes(position: position) { return true }
        guard data.count >= 16 else { return false }
        let (magic, baseAligned, hdrPos) = data.withUnsafeBytes { raw -> (UInt64, Int, Int) in
            let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
            let b = Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let p = Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
            return (m, b, p)
        }
        return magic == Self.deltaMagic && baseAligned == pinnedBasePos512 && hdrPos == position
            && data.count == deltaIndexerStateBytes(basePos512: baseAligned, position: position)
    }

    /// Captures all layers' LSA indexer and coarse cluster state at `position` (caller must wait for GPU first).
    public func captureIndexerState(position: Int, forceFull: Bool = false) -> Data {
        let unitTok = clusterUnitTokens
        precondition(position >= 0 && position / 4 <= maxBlocks && (position / unitTok) <= maxSuperblocks)
        let rowBytes = indexDim * 2
        let centPerUnit = 64 * rowBytes
        let memPerUnit = clusterMemBytesPerUnit
        let sizePerUnit = clusterSizeBytesPerUnit
        let openStart = (position / 4) * 4
        let base = region.pointer
        if !forceFull && pinnedBasePos512 > 0 && position > pinnedBasePos512 {
            let baseAligned = pinnedBasePos512
            let startPooledOff = (baseAligned / 4) * rowBytes
            let deltaPooledBytes = ((position / 4) - (baseAligned / 4)) * rowBytes
            let startUnits = baseAligned / unitTok
            let deltaUnits = (position / unitTok) - startUnits
            let deltaCentBytes = deltaUnits * centPerUnit
            let deltaMemBytes = deltaUnits * memPerUnit
            let deltaSizeBytes = deltaUnits * sizePerUnit
            var out = Data(capacity: deltaIndexerStateBytes(basePos512: baseAligned, position: position))
            var magic = Self.deltaMagic
            var baseU = UInt32(baseAligned)
            var posU = UInt32(position)
            withUnsafeBytes(of: &magic) { out.append(contentsOf: $0) }
            withUnsafeBytes(of: &baseU) { out.append(contentsOf: $0) }
            withUnsafeBytes(of: &posU) { out.append(contentsOf: $0) }
            for ls in layers {
                if deltaPooledBytes > 0 {
                    out.append(base.advanced(by: ls.pooledKeys.offset + startPooledOff).assumingMemoryBound(to: UInt8.self),
                               count: deltaPooledBytes)
                }
                for p in openStart..<position {
                    let row = p & (ringCap - 1)
                    out.append(base.advanced(by: ls.kPending.offset + row * rowBytes).assumingMemoryBound(to: UInt8.self),
                               count: rowBytes)
                }
                if deltaUnits > 0 {
                    out.append(base.advanced(by: ls.centroids.offset + startUnits * centPerUnit).assumingMemoryBound(to: UInt8.self),
                               count: deltaCentBytes)
                    out.append(base.advanced(by: ls.bucketMembers.offset + startUnits * memPerUnit).assumingMemoryBound(to: UInt8.self),
                               count: deltaMemBytes)
                    out.append(base.advanced(by: ls.bucketSizes.offset + startUnits * sizePerUnit).assumingMemoryBound(to: UInt8.self),
                               count: deltaSizeBytes)
                }
            }
            return out
        }
        let pooledBytes = (position / 4) * rowBytes
        let numUnits = position / unitTok
        let centBytes = numUnits * centPerUnit
        let memBytes = numUnits * memPerUnit
        let sizeBytes = numUnits * sizePerUnit
        var out = Data(capacity: fullIndexerStateBytes(position: position))
        for ls in layers {
            if pooledBytes > 0 {
                out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self),
                           count: pooledBytes)
            }
            for p in openStart..<position {
                let row = p & (ringCap - 1)
                out.append(base.advanced(by: ls.kPending.offset + row * rowBytes).assumingMemoryBound(to: UInt8.self),
                           count: rowBytes)
            }
            if numUnits > 0 {
                out.append(base.advanced(by: ls.centroids.offset).assumingMemoryBound(to: UInt8.self),
                           count: centBytes)
                out.append(base.advanced(by: ls.bucketMembers.offset).assumingMemoryBound(to: UInt8.self),
                           count: memBytes)
                out.append(base.advanced(by: ls.bucketSizes.offset).assumingMemoryBound(to: UInt8.self),
                           count: sizeBytes)
            }
        }
        return out
    }

    private func restorePinnedBasePrefix(upTo baseAligned: Int) {
        guard let baseData = pinnedBaseData, baseAligned > 0, baseAligned <= pinnedBasePos512 else { return }
        let unitTok = clusterUnitTokens
        let rowBytes = indexDim * 2
        let centPerUnit = 64 * rowBytes
        let memPerUnit = clusterMemBytesPerUnit
        let sizePerUnit = clusterSizeBytesPerUnit
        let fullPos = pinnedBaseFullPosition
        let fullPooledBytes = (fullPos / 4) * rowBytes
        let copyPooledBytes = (baseAligned / 4) * rowBytes
        let fullPendingBytes = (fullPos % 4) * rowBytes
        let fullUnits = fullPos / unitTok
        let copyUnits = baseAligned / unitTok
        let base = region.pointer
        baseData.withUnsafeBytes { raw in
            var src = 0
            for ls in layers {
                if copyPooledBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyPooledBytes)
                }
                src += fullPooledBytes + fullPendingBytes
                if copyUnits > 0 {
                    base.advanced(by: ls.centroids.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyUnits * centPerUnit)
                }
                src += fullUnits * centPerUnit
                if copyUnits > 0 {
                    base.advanced(by: ls.bucketMembers.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyUnits * memPerUnit)
                }
                src += fullUnits * memPerUnit
                if copyUnits > 0 {
                    base.advanced(by: ls.bucketSizes.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyUnits * sizePerUnit)
                }
                src += fullUnits * sizePerUnit
            }
        }
        residentPos512 = baseAligned
    }

    /// Restores all layers' LSA indexer and coarse cluster state at `position`, or returns false on size mismatch.
    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        let unitTok = clusterUnitTokens
        guard position >= 0, position / 4 <= maxBlocks, (position / unitTok) <= maxSuperblocks else { return false }
        let rowBytes = indexDim * 2
        let centPerUnit = 64 * rowBytes
        let memPerUnit = clusterMemBytesPerUnit
        let sizePerUnit = clusterSizeBytesPerUnit
        let openStart = (position / 4) * 4
        let base = region.pointer
        if data.count >= 16 {
            let (magic, baseAligned, hdrPos) = data.withUnsafeBytes { raw -> (UInt64, Int, Int) in
                (raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self),
                 Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
                 Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)))
            }
            if magic == Self.deltaMagic {
                guard baseAligned == pinnedBasePos512, hdrPos == position, pinnedBaseData != nil,
                      data.count == deltaIndexerStateBytes(basePos512: baseAligned, position: position) else { return false }
                if residentPos512 < baseAligned {
                    restorePinnedBasePrefix(upTo: baseAligned)
                }
                let startPooledOff = (baseAligned / 4) * rowBytes
                let deltaPooledBytes = ((position / 4) - (baseAligned / 4)) * rowBytes
                let startUnits = baseAligned / unitTok
                let deltaUnits = (position / unitTok) - startUnits
                let deltaCentBytes = deltaUnits * centPerUnit
                let deltaMemBytes = deltaUnits * memPerUnit
                let deltaSizeBytes = deltaUnits * sizePerUnit
                data.withUnsafeBytes { raw in
                    var src = 16
                    for ls in layers {
                        if deltaPooledBytes > 0 {
                            base.advanced(by: ls.pooledKeys.offset + startPooledOff)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaPooledBytes)
                            src += deltaPooledBytes
                        }
                        for p in openStart..<position {
                            let row = p & (ringCap - 1)
                            base.advanced(by: ls.kPending.offset + row * rowBytes)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: rowBytes)
                            src += rowBytes
                        }
                        if deltaUnits > 0 {
                            base.advanced(by: ls.centroids.offset + startUnits * centPerUnit)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaCentBytes)
                            src += deltaCentBytes
                            base.advanced(by: ls.bucketMembers.offset + startUnits * memPerUnit)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaMemBytes)
                            src += deltaMemBytes
                            base.advanced(by: ls.bucketSizes.offset + startUnits * sizePerUnit)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaSizeBytes)
                            src += deltaSizeBytes
                        }
                    }
                }
                return true
            }
        }
        guard data.count == fullIndexerStateBytes(position: position) else { return false }
        if pinnedBasePos512 > 0 && position == pinnedBaseFullPosition && residentPos512 >= pinnedBasePos512 {
            let baseAligned = pinnedBasePos512
            let fullPooledBytes = (position / 4) * rowBytes
            let startPooledOff = (baseAligned / 4) * rowBytes
            let tailPooledBytes = fullPooledBytes - startPooledOff
            let numUnits = position / unitTok
            let unitBytes = centPerUnit + memPerUnit + sizePerUnit
            let tailCopyNeeded = tailPooledBytes > 0 || openStart < position
            guard tailCopyNeeded else { return true }
            data.withUnsafeBytes { raw in
                var src = 0
                for ls in layers {
                    if tailPooledBytes > 0 {
                        base.advanced(by: ls.pooledKeys.offset + startPooledOff)
                            .copyMemory(from: raw.baseAddress!.advanced(by: src + startPooledOff), byteCount: tailPooledBytes)
                    }
                    src += fullPooledBytes
                    for p in openStart..<position {
                        let row = p & (ringCap - 1)
                        base.advanced(by: ls.kPending.offset + row * rowBytes)
                            .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: rowBytes)
                        src += rowBytes
                    }
                    src += numUnits * unitBytes
                }
            }
            return true
        }
        let pooledBytes = (position / 4) * rowBytes
        let numUnits = position / unitTok
        let centBytes = numUnits * centPerUnit
        let memBytes = numUnits * memPerUnit
        let sizeBytes = numUnits * sizePerUnit
        data.withUnsafeBytes { raw in
            var src = 0
            for ls in layers {
                if pooledBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: pooledBytes)
                    src += pooledBytes
                }
                for p in openStart..<position {
                    let row = p & (ringCap - 1)
                    base.advanced(by: ls.kPending.offset + row * rowBytes)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: rowBytes)
                    src += rowBytes
                }
                if numUnits > 0 {
                    base.advanced(by: ls.centroids.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: centBytes)
                    src += centBytes
                    base.advanced(by: ls.bucketMembers.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: memBytes)
                    src += memBytes
                    base.advanced(by: ls.bucketSizes.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: sizeBytes)
                    src += sizeBytes
                }
            }
        }
        if pinnedBasePos512 > 0 && position == pinnedBaseFullPosition {
            residentPos512 = pinnedBasePos512
        }
        return true
    }
}

// QWLSAOperator+Snapshot.swift — Full and pinned-base delta (`QLSADLT1`)
// snapshot capture and restoration for QWLSAOperator.

import Foundation

extension QWLSAOperator {
    /// Registers or clears a pinned `.qwcache` base LSA state so subsequent turn captures/restores only copy delta blocks.
    public func setPinnedBaseState(_ data: Data?, position: Int) {
        let base512 = (position / 512) * 512
        if let data, base512 >= 512, data.count == fullIndexerStateBytes(position: position) {
            pinnedBaseData = data
            pinnedBaseFullPosition = position
            pinnedBasePos512 = base512
            residentPos512 = 0
        } else {
            pinnedBaseData = nil
            pinnedBaseFullPosition = 0
            pinnedBasePos512 = 0
            residentPos512 = 0
        }
    }

    /// Full uncompressed bytes required to snapshot pooled keys, pending ring rows, and Ward clusters at `position`.
    public func fullIndexerStateBytes(position: Int) -> Int {
        let rowBytes = indexDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let pendingBytes = (position % 4) * rowBytes
        let numBuckets = (position / 512) * 64
        let clusterBytes = numBuckets * (rowBytes + 5)
        return layers.count * (pooledBytes + pendingBytes + clusterBytes)
    }

    /// Compact delta bytes required to snapshot `(basePos512 ..< position)` above a 512-token-aligned pinned base.
    public func deltaIndexerStateBytes(basePos512: Int, position: Int) -> Int {
        let rowBytes = indexDim * 2
        let deltaPooledBytes = max(0, (position / 4) - (basePos512 / 4)) * rowBytes
        let pendingBytes = (position % 4) * rowBytes
        let deltaBuckets = max(0, (position / 512) - (basePos512 / 512)) * 64
        let deltaClusterBytes = deltaBuckets * (rowBytes + 5)
        return 16 + layers.count * (deltaPooledBytes + pendingBytes + deltaClusterBytes)
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
        let (magic, base512, hdrPos) = data.withUnsafeBytes { raw -> (UInt64, Int, Int) in
            let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
            let b = Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let p = Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
            return (m, b, p)
        }
        return magic == Self.deltaMagic && base512 == pinnedBasePos512 && hdrPos == position
            && data.count == deltaIndexerStateBytes(basePos512: base512, position: position)
    }

    /// Captures all layers' LSA indexer and Ward cluster state at `position` (caller must wait for GPU first).
    public func captureIndexerState(position: Int, forceFull: Bool = false) -> Data {
        precondition(position >= 0 && position / 4 <= maxBlocks && (position / 512) * 64 <= maxBuckets)
        let rowBytes = indexDim * 2
        let openStart = (position / 4) * 4
        let base = region.pointer
        if !forceFull && pinnedBasePos512 > 0 && position > pinnedBasePos512 {
            let base512 = pinnedBasePos512
            let startPooledOff = (base512 / 4) * rowBytes
            let deltaPooledBytes = ((position / 4) - (base512 / 4)) * rowBytes
            let startBuckets = (base512 / 512) * 64
            let deltaBuckets = ((position / 512) - (base512 / 512)) * 64
            let deltaCentBytes = deltaBuckets * rowBytes
            let deltaMemBytes = deltaBuckets * 4
            let deltaSizeBytes = deltaBuckets
            var out = Data(capacity: deltaIndexerStateBytes(basePos512: base512, position: position))
            var magic = Self.deltaMagic
            var baseU = UInt32(base512)
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
                if deltaBuckets > 0 {
                    out.append(base.advanced(by: ls.centroids.offset + startBuckets * rowBytes).assumingMemoryBound(to: UInt8.self),
                               count: deltaCentBytes)
                    out.append(base.advanced(by: ls.bucketMembers.offset + startBuckets * 4).assumingMemoryBound(to: UInt8.self),
                               count: deltaMemBytes)
                    out.append(base.advanced(by: ls.bucketSizes.offset + startBuckets).assumingMemoryBound(to: UInt8.self),
                               count: deltaSizeBytes)
                }
            }
            return out
        }
        let pooledBytes = (position / 4) * rowBytes
        let numBuckets = (position / 512) * 64
        let centBytes = numBuckets * rowBytes
        let memBytes = numBuckets * 4
        let sizeBytes = numBuckets
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
            if numBuckets > 0 {
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

    private func restorePinnedBasePrefix(upTo base512: Int) {
        guard let baseData = pinnedBaseData, base512 > 0, base512 <= pinnedBasePos512 else { return }
        let rowBytes = indexDim * 2
        let fullPos = pinnedBaseFullPosition
        let fullPooledBytes = (fullPos / 4) * rowBytes
        let copyPooledBytes = (base512 / 4) * rowBytes
        let fullPendingBytes = (fullPos % 4) * rowBytes
        let fullBuckets = (fullPos / 512) * 64
        let copyBuckets = (base512 / 512) * 64
        let base = region.pointer
        baseData.withUnsafeBytes { raw in
            var src = 0
            for ls in layers {
                if copyPooledBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyPooledBytes)
                }
                src += fullPooledBytes + fullPendingBytes
                if copyBuckets > 0 {
                    base.advanced(by: ls.centroids.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyBuckets * rowBytes)
                }
                src += fullBuckets * rowBytes
                if copyBuckets > 0 {
                    base.advanced(by: ls.bucketMembers.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyBuckets * 4)
                }
                src += fullBuckets * 4
                if copyBuckets > 0 {
                    base.advanced(by: ls.bucketSizes.offset)
                        .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: copyBuckets)
                }
                src += fullBuckets
            }
        }
        residentPos512 = base512
    }

    /// Restores all layers' LSA indexer and Ward cluster state at `position`, or returns false on size mismatch.
    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        guard position >= 0, position / 4 <= maxBlocks, (position / 512) * 64 <= maxBuckets else { return false }
        let rowBytes = indexDim * 2
        let openStart = (position / 4) * 4
        let base = region.pointer
        if data.count >= 16 {
            let (magic, base512, hdrPos) = data.withUnsafeBytes { raw -> (UInt64, Int, Int) in
                (raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self),
                 Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
                 Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)))
            }
            if magic == Self.deltaMagic {
                guard base512 == pinnedBasePos512, hdrPos == position, pinnedBaseData != nil,
                      data.count == deltaIndexerStateBytes(basePos512: base512, position: position) else { return false }
                if residentPos512 < base512 {
                    restorePinnedBasePrefix(upTo: base512)
                }
                let startPooledOff = (base512 / 4) * rowBytes
                let deltaPooledBytes = ((position / 4) - (base512 / 4)) * rowBytes
                let startBuckets = (base512 / 512) * 64
                let deltaBuckets = ((position / 512) - (base512 / 512)) * 64
                let deltaCentBytes = deltaBuckets * rowBytes
                let deltaMemBytes = deltaBuckets * 4
                let deltaSizeBytes = deltaBuckets
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
                        if deltaBuckets > 0 {
                            base.advanced(by: ls.centroids.offset + startBuckets * rowBytes)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaCentBytes)
                            src += deltaCentBytes
                            base.advanced(by: ls.bucketMembers.offset + startBuckets * 4)
                                .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: deltaMemBytes)
                            src += deltaMemBytes
                            base.advanced(by: ls.bucketSizes.offset + startBuckets)
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
            let base512 = pinnedBasePos512
            let fullPooledBytes = (position / 4) * rowBytes
            let startPooledOff = (base512 / 4) * rowBytes
            let tailPooledBytes = fullPooledBytes - startPooledOff
            let numBuckets = (position / 512) * 64
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
                    src += numBuckets * (rowBytes + 5)
                }
            }
            return true
        }
        let pooledBytes = (position / 4) * rowBytes
        let numBuckets = (position / 512) * 64
        let centBytes = numBuckets * rowBytes
        let memBytes = numBuckets * 4
        let sizeBytes = numBuckets
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
                if numBuckets > 0 {
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

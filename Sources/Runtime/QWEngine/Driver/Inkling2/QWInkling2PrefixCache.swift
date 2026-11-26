// QWInkling2PrefixCache.swift — Warm RAM KV + 192 short-conv state prefix cache for Inkling-2.
import Foundation
import Metal
import QWAttn
import QWCore

final class QWInkling2PrefixCache: @unchecked Sendable {
    struct Key: Hashable {
        let salt: String?
        let tokens: [Int]
    }

    struct VisitSlice {
        let isSliding: Bool
        let startAbsPos: Int
        let rowCount: Int
        let kPay: Data
        let vPay: Data
        let kMeta: Data
        let vMeta: Data
    }

    final class Entry {
        let key: Key
        let convData: Data
        let visitSlices: [VisitSlice]
        let dsparkSlices: [QWInkling2DSpark.KVSlice]?
        let dsparkLastTap: Data?
        let lsaIndexerData: Data?
        let finalNormRow: Data?
        let byteCount: Int
        var lastAccess: UInt64

        init(key: Key, convData: Data, visitSlices: [VisitSlice],
             dsparkSlices: [QWInkling2DSpark.KVSlice]? = nil, dsparkLastTap: Data? = nil,
             lsaIndexerData: Data? = nil, finalNormRow: Data? = nil,
             byteCount: Int, tick: UInt64) {
            self.key = key
            self.convData = convData
            self.visitSlices = visitSlices
            self.dsparkSlices = dsparkSlices
            self.dsparkLastTap = dsparkLastTap
            self.lsaIndexerData = lsaIndexerData
            self.finalNormRow = finalNormRow
            self.byteCount = byteCount
            self.lastAccess = tick
        }
    }

    private let lock = NSLock()
    private var warmByteLimit: Int
    private var maxEntries: Int
    private var entries: [Key: Entry] = [:]
    private var totalBytes: Int = 0
    private var tick: UInt64 = 0

    var onEvict: ((Int, Int) -> Void)?
    var onRetain: ((Int, Int, Int, Int) -> Void)?
    var onSkipOversize: ((Int, Int, Int) -> Void)?

    init(warmByteLimit: Int = 0, maxEntries: Int = 32) {
        self.warmByteLimit = max(0, warmByteLimit)
        self.maxEntries = max(1, maxEntries)
    }

    var isEnabled: Bool {
        lock.withLock { warmByteLimit > 0 }
    }

    var state: (entries: Int, bytes: Int, limit: Int)? {
        lock.withLock {
            guard warmByteLimit > 0 else { return nil }
            return (entries.count, totalBytes, warmByteLimit)
        }
    }

    func reconfigure(warmByteLimit: Int, maxEntries: Int) {
        lock.withLock {
            self.warmByteLimit = max(0, warmByteLimit)
            self.maxEntries = max(1, maxEntries)
            evictToLimitLocked()
        }
    }

    func clear() {
        lock.withLock {
            entries.removeAll()
            totalBytes = 0
        }
    }

    func matchLongestPrefix(tokens: [Int], salt: String?, limit: Int, allowExact: Bool = false) -> Entry? {
        guard limit > 0 else { return nil }
        return lock.withLock {
            guard warmByteLimit > 0, !entries.isEmpty else { return nil }
            if allowExact, limit >= tokens.count {
                let exactKey = Key(salt: salt, tokens: tokens)
                if let exactEntry = entries[exactKey], exactEntry.finalNormRow != nil {
                    tick &+= 1
                    exactEntry.lastAccess = tick
                    return exactEntry
                }
            }
            let prefixLimit = min(limit, max(0, tokens.count - 1))
            guard prefixLimit > 0 else { return nil }
            var best: Entry? = nil
            var bestLen = 0
            for (k, entry) in entries where k.salt == salt && k.tokens.count <= prefixLimit {
                let n = k.tokens.count
                guard n > bestLen else { continue }
                if tokens[0..<n].elementsEqual(k.tokens) {
                    best = entry
                    bestLen = n
                }
            }
            if let best {
                tick &+= 1
                best.lastAccess = tick
            }
            return best
        }
    }

    static func captureFinalNormRow(session: QWInkling2Session, lastChunkRows: Int, hidden: Int) -> Data {
        let rowBytes = hidden * 2
        let rowOff = max(0, lastChunkRows - 1) * rowBytes
        let src = session.normBuf.contents().advanced(by: rowOff)
        return withExtendedLifetime(session.normBuf) {
            Data(bytes: src, count: rowBytes)
        }
    }

    static func capture(session: QWInkling2Session, dspark: QWInkling2DSpark? = nil,
                        lsaOperator: QWLSAOperator? = nil,
                        tokens: [Int], salt: String?,
                        slidingWindow: Int, finalNormRow: Data? = nil, tick: UInt64 = 0) -> Entry {
        capture(
            slot: session.primarySlotState, kvModeRaw: session.kvModeRaw,
            dspark: dspark, lsaOperator: lsaOperator,
            tokens: tokens, salt: salt, slidingWindow: slidingWindow,
            finalNormRow: finalNormRow, tick: tick
        )
    }

    static func capture(slot: Ink2SlotState, kvModeRaw: UInt32,
                        dspark: QWInkling2DSpark? = nil,
                        lsaOperator: QWLSAOperator? = nil,
                        tokens: [Int], salt: String?,
                        slidingWindow: Int, finalNormRow: Data? = nil, tick: UInt64 = 0) -> Entry {
        let pos = tokens.count
        let slab = slot.convStatesSlab
        let convData = Data(bytes: slab.contents(), count: slab.length)
        var slices: [VisitSlice] = []
        slices.reserveCapacity(slot.visitKVs.count)
        var bytes = convData.count

        for kv in slot.visitKVs {
            let startPos = kv.isSliding ? max(0, pos - slidingWindow) : 0
            let count = pos - startPos
            var kPay = Data(); kPay.reserveCapacity(count * kv.rowBytes)
            var vPay = Data(); vPay.reserveCapacity(count * kv.rowBytes)
            var kMeta = Data(); kMeta.reserveCapacity(count * kv.metaRowBytes)
            var vMeta = Data(); vMeta.reserveCapacity(count * kv.metaRowBytes)
            let kp = kv.kCache.contents().assumingMemoryBound(to: UInt8.self)
            let vp = kv.vCache.contents().assumingMemoryBound(to: UInt8.self)
            let kmp = kv.kMeta.contents().assumingMemoryBound(to: UInt8.self)
            let vmp = kv.vMeta.contents().assumingMemoryBound(to: UInt8.self)

            for absPos in startPos..<pos {
                let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                kPay.append(kp.advanced(by: kvSlot * kv.rowBytes), count: kv.rowBytes)
                vPay.append(vp.advanced(by: kvSlot * kv.rowBytes), count: kv.rowBytes)
                if kvModeRaw != 0 {
                    kMeta.append(kmp.advanced(by: kvSlot * kv.metaRowBytes), count: kv.metaRowBytes)
                    vMeta.append(vmp.advanced(by: kvSlot * kv.metaRowBytes), count: kv.metaRowBytes)
                }
            }
            bytes += kPay.count + vPay.count + kMeta.count + vMeta.count
            slices.append(VisitSlice(
                isSliding: kv.isSliding, startAbsPos: startPos, rowCount: count,
                kPay: kPay, vPay: vPay, kMeta: kMeta, vMeta: vMeta
            ))
        }
        var dsparkSlices: [QWInkling2DSpark.KVSlice]? = nil
        var dsparkLastTap: Data? = nil
        if slot.slotIndex == 0, let dspark {
            let cap = dspark.capturePrefixSlices(pos: pos)
            dsparkSlices = cap.slices
            dsparkLastTap = cap.lastTap
            bytes += cap.lastTap.count
            for s in cap.slices { bytes += s.kData.count + s.vData.count }
        }
        var lsaData: Data? = nil
        if slot.slotIndex == 0, let lsa = lsaOperator, pos > 0 {
            let capturedLsa = lsa.captureIndexerState(position: pos)
            lsaData = capturedLsa
            bytes += capturedLsa.count
        }
        bytes += finalNormRow?.count ?? 0
        return Entry(key: Key(salt: salt, tokens: tokens), convData: convData,
                     visitSlices: slices, dsparkSlices: dsparkSlices, dsparkLastTap: dsparkLastTap,
                     lsaIndexerData: lsaData, finalNormRow: finalNormRow, byteCount: bytes, tick: tick)
    }

    @discardableResult
    static func restore(entry: Entry, into session: QWInkling2Session,
                        dspark: QWInkling2DSpark? = nil,
                        lsaOperator: QWLSAOperator? = nil) -> (baseOk: Bool, dsparkOk: Bool) {
        restore(
            entry: entry, into: session.primarySlotState, kvModeRaw: session.kvModeRaw,
            dspark: dspark, lsaOperator: lsaOperator, stateOwner: session
        )
    }

    @discardableResult
    static func restore(entry: Entry, into slot: Ink2SlotState, kvModeRaw: UInt32,
                        dspark: QWInkling2DSpark? = nil,
                        lsaOperator: QWLSAOperator? = nil,
                        stateOwner: AnyObject? = nil) -> (baseOk: Bool, dsparkOk: Bool) {
        let pos = entry.key.tokens.count
        guard pos <= slot.capacity,
              entry.convData.count == slot.convStatesSlab.length else { return (false, false) }
        for (idx, slice) in entry.visitSlices.enumerated() where idx < slot.visitKVs.count {
            let kv = slot.visitKVs[idx]
            if !kv.isSliding && slice.startAbsPos + slice.rowCount > kv.ringCap {
                return (false, false)
            }
        }
        if slot.slotIndex == 0, let lsa = lsaOperator {
            if pos > 0 {
                guard let lsaData = entry.lsaIndexerData,
                      lsa.restoreIndexerState(lsaData, position: pos) else {
                    return (false, false)
                }
            } else {
                _ = lsa.restoreIndexerState(Data(), position: 0)
            }
            lsa.stateOwner = stateOwner
        }
        let convDst = slot.convStatesSlab.contents()
        let convBytes = entry.convData.count
        withExtendedLifetime(slot.convStatesSlab) {
            entry.convData.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                memcpy(convDst, base, convBytes)
            }
        }
        for (idx, slice) in entry.visitSlices.enumerated() where idx < slot.visitKVs.count {
            let kv = slot.visitKVs[idx]
            let kp = kv.kCache.contents().assumingMemoryBound(to: UInt8.self)
            let vp = kv.vCache.contents().assumingMemoryBound(to: UInt8.self)
            let kmp = kv.kMeta.contents().assumingMemoryBound(to: UInt8.self)
            let vmp = kv.vMeta.contents().assumingMemoryBound(to: UInt8.self)
            slice.kPay.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for r in 0..<slice.rowCount {
                    let absPos = slice.startAbsPos + r
                    let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                    memcpy(kp.advanced(by: kvSlot * kv.rowBytes), base.advanced(by: r * kv.rowBytes), kv.rowBytes)
                }
            }
            slice.vPay.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for r in 0..<slice.rowCount {
                    let absPos = slice.startAbsPos + r
                    let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                    memcpy(vp.advanced(by: kvSlot * kv.rowBytes), base.advanced(by: r * kv.rowBytes), kv.rowBytes)
                }
            }
            if kvModeRaw != 0 {
                slice.kMeta.withUnsafeBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.rowCount {
                        let absPos = slice.startAbsPos + r
                        let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                        memcpy(kmp.advanced(by: kvSlot * kv.metaRowBytes), base.advanced(by: r * kv.metaRowBytes), kv.metaRowBytes)
                    }
                }
                slice.vMeta.withUnsafeBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.rowCount {
                        let absPos = slice.startAbsPos + r
                        let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                        memcpy(vmp.advanced(by: kvSlot * kv.metaRowBytes), base.advanced(by: r * kv.metaRowBytes), kv.metaRowBytes)
                    }
                }
            }
        }
        if slot.slotIndex == 0, let dspark, let dSlices = entry.dsparkSlices, let dTap = entry.dsparkLastTap {
            dspark.restorePrefixSlices(dSlices, lastTap: dTap, pos: pos)
            return (true, true)
        }
        return (true, false)
    }

    func store(entry: Entry) {
        let (retained, skipped, evictions): ((Int, Int, Int, Int)?, (Int, Int, Int)?, [(Int, Int)]) = lock.withLock {
            guard warmByteLimit > 0 else { return (nil, nil, []) }
            if entry.byteCount > warmByteLimit {
                return (nil, (entry.key.tokens.count, entry.byteCount, warmByteLimit), [])
            }
            var evs: [(Int, Int)] = []
            if let old = entries[entry.key] {
                totalBytes -= old.byteCount
                entries.removeValue(forKey: entry.key)
            }
            tick &+= 1
            entry.lastAccess = tick
            entries[entry.key] = entry
            totalBytes += entry.byteCount
            evs.append(contentsOf: evictToLimitLocked())
            return ((entry.key.tokens.count, entry.byteCount, totalBytes, warmByteLimit), nil, evs)
        }
        for ev in evictions { onEvict?(ev.0, ev.1) }
        if let skipped { onSkipOversize?(skipped.0, skipped.1, skipped.2) }
        if let retained { onRetain?(retained.0, retained.1, retained.2, retained.3) }
    }

    @discardableResult
    private func evictToLimitLocked() -> [(Int, Int)] {
        var evs: [(Int, Int)] = []
        while (totalBytes > warmByteLimit || entries.count > maxEntries),
              let victim = entries.values.min(by: { $0.lastAccess < $1.lastAccess }) {
            entries.removeValue(forKey: victim.key)
            totalBytes -= victim.byteCount
            evs.append((victim.key.tokens.count, victim.byteCount))
        }
        return evs
    }
}

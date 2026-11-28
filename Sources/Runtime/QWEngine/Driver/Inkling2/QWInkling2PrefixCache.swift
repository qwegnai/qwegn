// QWInkling2PrefixCache.swift — Warm RAM KV + 192 short-conv state prefix cache for Inkling-2.
import Foundation
import Metal
import QWAttn
import QWCore
import QWMTP

final class QWInkling2PrefixCache: @unchecked Sendable {
    struct Key: Hashable, Sendable {
        let salt: String?
        let tokens: [Int]
    }

    struct VisitSlice: Sendable {
        let isSliding: Bool
        let startAbsPos: Int
        let rowCount: Int
        let kPay: Data
        let vPay: Data
        let kMeta: Data
        let vMeta: Data
    }

    final class Entry: Sendable {
        let key: Key
        let convData: Data
        let visitSlices: [VisitSlice]
        let dsparkSlices: [QWInkling2DSpark.KVSlice]?
        let dsparkLastTap: Data?
        let dflashState: Data?
        let lsaIndexerData: Data?
        let finalNormRow: Data?
        let byteCount: Int

        init(key: Key, convData: Data, visitSlices: [VisitSlice],
             dsparkSlices: [QWInkling2DSpark.KVSlice]? = nil, dsparkLastTap: Data? = nil,
             dflashState: Data? = nil,
             lsaIndexerData: Data? = nil, finalNormRow: Data? = nil,
             byteCount: Int, tick: UInt64 = 0) {
            _ = tick
            self.key = key
            self.convData = convData
            self.visitSlices = visitSlices
            self.dsparkSlices = dsparkSlices
            self.dsparkLastTap = dsparkLastTap
            self.dflashState = dflashState
            self.lsaIndexerData = lsaIndexerData
            self.finalNormRow = finalNormRow
            self.byteCount = byteCount
        }
    }

    private let core: QWSlotPinnedLRUStore<Key, Entry>
    private let callbackLock = NSLock()
    private var _onEvict: ((Int, Int) -> Void)?
    private var _onRetain: ((Int, Int, Int, Int) -> Void)?
    private var _onSkipOversize: ((Int, Int, Int) -> Void)?

    var onEvict: ((Int, Int) -> Void)? {
        get { callbackLock.withLock { _onEvict } }
        set { callbackLock.withLock { _onEvict = newValue } }
    }
    var onRetain: ((Int, Int, Int, Int) -> Void)? {
        get { callbackLock.withLock { _onRetain } }
        set { callbackLock.withLock { _onRetain = newValue } }
    }
    var onSkipOversize: ((Int, Int, Int) -> Void)? {
        get { callbackLock.withLock { _onSkipOversize } }
        set { callbackLock.withLock { _onSkipOversize = newValue } }
    }

    init(warmByteLimit: Int = 0, maxEntries: Int = 32) {
        let store = QWSlotPinnedLRUStore<Key, Entry>(
            byteLimit: warmByteLimit,
            entryLimit: maxEntries,
            maxSlots: max(1, min(16, maxEntries))
        )
        self.core = store
        store.onEvict = { [weak self] _, storedEntry in
            self?.onEvict?(storedEntry.tokenCount, storedEntry.charge)
        }
    }

    var isEnabled: Bool { core.isEnabled }

    var state: (entries: Int, bytes: Int, limit: Int)? {
        let limit = core.byteLimit
        guard limit > 0 else { return nil }
        let st = core.state
        return (st.entries, st.bytes, limit)
    }

    func reconfigure(warmByteLimit: Int, maxEntries: Int) {
        core.reconfigure(byteLimit: warmByteLimit, entryLimit: maxEntries)
    }

    func clear() {
        core.removeAll()
    }

    func matchLongestPrefix(tokens: [Int], salt: String?, limit: Int, allowExact: Bool = false) -> Entry? {
        guard limit > 0, core.isEnabled else { return nil }
        if allowExact, limit >= tokens.count {
            let exactKey = Key(salt: salt, tokens: tokens)
            if let exactStored = core.value(for: exactKey, touchOnHit: false),
               exactStored.value.finalNormRow != nil {
                _ = core.value(for: exactKey, touchOnHit: true)
                return exactStored.value
            }
        }
        let prefixLimit = min(limit, max(0, tokens.count - 1))
        guard prefixLimit > 0 else { return nil }
        let snapshot = core.snapshotOrderedEntries()
        var bestKey: Key? = nil
        var bestEntry: Entry? = nil
        var bestLen = 0
        for (k, stored) in snapshot.entries where k.salt == salt && k.tokens.count <= prefixLimit {
            let n = k.tokens.count
            guard n > bestLen else { continue }
            if tokens[0..<n].elementsEqual(k.tokens) {
                bestKey = k
                bestEntry = stored.value
                bestLen = n
            }
        }
        if let bestKey {
            return core.value(for: bestKey, touchOnHit: true)?.value ?? bestEntry
        }
        return nil
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
                        dflash: QWDFlashDrafter? = nil,
                        lsaOperator: QWLSAOperator? = nil,
                        tokens: [Int], salt: String?,
                        slidingWindow: Int, finalNormRow: Data? = nil, tick: UInt64 = 0) -> Entry {
        capture(
            slot: session.primarySlotState, kvModeRaw: session.kvModeRaw,
            dspark: dspark, dflash: dflash, lsaOperator: lsaOperator,
            tokens: tokens, salt: salt, slidingWindow: slidingWindow,
            finalNormRow: finalNormRow, tick: tick
        )
    }

    static func capture(slot: Ink2SlotState, kvModeRaw: UInt32,
                        dspark: QWInkling2DSpark? = nil,
                        dflash: QWDFlashDrafter? = nil,
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
        var dflashState: Data? = nil
        if slot.slotIndex == 0, let dflash, let slot0 = dflash.sessionSlots.first {
            let blob = dflash.capturePrefixBlob(slot: slot0, tokenCount: pos)
            dflashState = blob
            bytes += blob.count
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
                     dflashState: dflashState,
                     lsaIndexerData: lsaData, finalNormRow: finalNormRow, byteCount: bytes, tick: tick)
    }

    @discardableResult
    static func restore(entry: Entry, into session: QWInkling2Session,
                        dspark: QWInkling2DSpark? = nil,
                        dflash: QWDFlashDrafter? = nil,
                        lsaOperator: QWLSAOperator? = nil) -> (baseOk: Bool, dsparkOk: Bool, dflashOk: Bool) {
        restore(
            entry: entry, into: session.primarySlotState, kvModeRaw: session.kvModeRaw,
            dspark: dspark, dflash: dflash, lsaOperator: lsaOperator, stateOwner: session
        )
    }

    @discardableResult
    static func restore(entry: Entry, into slot: Ink2SlotState, kvModeRaw: UInt32,
                        dspark: QWInkling2DSpark? = nil,
                        dflash: QWDFlashDrafter? = nil,
                        lsaOperator: QWLSAOperator? = nil,
                        stateOwner: AnyObject? = nil) -> (baseOk: Bool, dsparkOk: Bool, dflashOk: Bool) {
        let pos = entry.key.tokens.count
        guard pos <= slot.capacity,
              entry.convData.count == slot.convStatesSlab.length else { return (false, false, false) }
        for (idx, slice) in entry.visitSlices.enumerated() where idx < slot.visitKVs.count {
            let kv = slot.visitKVs[idx]
            if !kv.isSliding && slice.startAbsPos + slice.rowCount > kv.ringCap {
                return (false, false, false)
            }
        }
        if slot.slotIndex == 0, let lsa = lsaOperator {
            if pos > 0 {
                guard let lsaData = entry.lsaIndexerData,
                      lsa.restoreIndexerState(lsaData, position: pos) else {
                    return (false, false, false)
                }
            } else {
                _ = lsa.restoreIndexerState(Data(), position: 0)
            }
            lsa.stateOwner = stateOwner
        }
        let convDst = slot.convStatesSlab.contents()
        let convBytes = entry.convData.count
        withExtendedLifetime(slot.convStatesSlab) {
            entry.convData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                convDst.copyMemory(from: base, byteCount: convBytes)
            }
        }
        for (idx, slice) in entry.visitSlices.enumerated() where idx < slot.visitKVs.count {
            let kv = slot.visitKVs[idx]
            let kp = kv.kCache.contents()
            let vp = kv.vCache.contents()
            let kmp = kv.kMeta.contents()
            let vmp = kv.vMeta.contents()
            withExtendedLifetime(kv.kCache) {
                slice.kPay.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.rowCount {
                        let absPos = slice.startAbsPos + r
                        let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                        kp.advanced(by: kvSlot * kv.rowBytes).copyMemory(from: base.advanced(by: r * kv.rowBytes), byteCount: kv.rowBytes)
                    }
                }
            }
            withExtendedLifetime(kv.vCache) {
                slice.vPay.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.rowCount {
                        let absPos = slice.startAbsPos + r
                        let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                        vp.advanced(by: kvSlot * kv.rowBytes).copyMemory(from: base.advanced(by: r * kv.rowBytes), byteCount: kv.rowBytes)
                    }
                }
            }
            if kvModeRaw != 0 {
                withExtendedLifetime(kv.kMeta) {
                    slice.kMeta.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        guard let base = raw.baseAddress else { return }
                        for r in 0..<slice.rowCount {
                            let absPos = slice.startAbsPos + r
                            let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                            kmp.advanced(by: kvSlot * kv.metaRowBytes).copyMemory(from: base.advanced(by: r * kv.metaRowBytes), byteCount: kv.metaRowBytes)
                        }
                    }
                }
                withExtendedLifetime(kv.vMeta) {
                    slice.vMeta.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        guard let base = raw.baseAddress else { return }
                        for r in 0..<slice.rowCount {
                            let absPos = slice.startAbsPos + r
                            let kvSlot = kv.isSliding ? (absPos % kv.ringCap) : absPos
                            vmp.advanced(by: kvSlot * kv.metaRowBytes).copyMemory(from: base.advanced(by: r * kv.metaRowBytes), byteCount: kv.metaRowBytes)
                        }
                    }
                }
            }
        }
        var dsparkOk = false
        if slot.slotIndex == 0, let dspark, let dSlices = entry.dsparkSlices, let dTap = entry.dsparkLastTap {
            dspark.restorePrefixSlices(dSlices, lastTap: dTap, pos: pos)
            dsparkOk = true
        }
        var dflashOk = false
        if slot.slotIndex == 0, let dflash, let dfState = entry.dflashState,
           let slot0 = dflash.sessionSlots.first,
           dflash.restorePrefixBlob(dfState, into: slot0, expectedTokens: pos) == pos {
            dflashOk = true
        }
        return (true, dsparkOk, dflashOk)
    }

    func store(entry: Entry) {
        let limit = core.byteLimit
        guard limit > 0 else { return }
        if entry.byteCount > limit {
            onSkipOversize?(entry.key.tokens.count, entry.byteCount, limit)
            return
        }
        let stored = core.store(
            key: entry.key,
            value: entry,
            charge: entry.byteCount,
            tokenCount: entry.key.tokens.count
        )
        if stored {
            let st = core.state
            onRetain?(entry.key.tokens.count, entry.byteCount, st.bytes, limit)
        } else {
            let currentLimit = core.byteLimit
            if currentLimit > 0 && entry.byteCount > currentLimit {
                onSkipOversize?(entry.key.tokens.count, entry.byteCount, currentLimit)
            }
        }
    }
}

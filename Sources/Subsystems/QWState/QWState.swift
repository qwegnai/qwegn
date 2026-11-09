// QWState.swift — C20: sequence state, snapshots & prefix cache.
//
// Owns the per-sequence MUTABLE state of the hybrid engine (spec 20):
//   • GDN recurrent state slots — 3.15 MB fp32 per GDN layer (48 v-heads ×
//     128×128), 48 GDN layers ⇒ 151.25 MB per sequence. Slot index ==
//     sequence id: the C12 kernel derives addresses from slot × stride,
//     zero table lookups in the hot path.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWToken

public enum QWStateError: Error, CustomStringConvertible, Equatable {
    case slotPressure(live: Int, cap: Int)
    case unknownSeq(Int)
    case dupSeq(Int)
    case slotTooSmall(asked: Int, capacity: Int)
    case badTrace(String)

    public var description: String {
        switch self {
        case .slotPressure(let l, let c): return "QWState: slot pressure (live \(l), cap \(c)) — refuse before jetsam does"
        case .unknownSeq(let s): return "QWState: unknown sequence \(s)"
        case .dupSeq(let s): return "QWState: duplicate sequence \(s)"
        case .slotTooSmall(let a, let c): return "QWState: slot capacity \(c) B < state \(a) B"
        case .badTrace(let m): return "QWState: golden trace mismatch: \(m)"
        }
    }
}

// ────────────────────────────── slot geometry ─────────────────────────────

/// Spec-20 footprint constants, re-derived from C01 (never restated).
/// Every constant takes the model's **own** geometry: a 27B-shaped
/// number standing in for the model's is the defect Task 4.3g closed.
public enum QWSlotGeometry {
    /// GDN layer count: layers where !isFullAttention (48 of 64).
    public static func gdnLayers(geometry: QWModelGeometry) -> Int {
        (0..<geometry.numLayers).filter { !geometry.isFullAttention($0) }.count
    }
    /// fp32 elements of one layer's recurrent state (48 v-heads × 128×128).
    public static func recElems(geometry: QWModelGeometry) -> Int {
        geometry.gdnVHeads * geometry.gdnHeadDim * geometry.gdnHeadDim
    }
    /// fp32 elements of one layer's conv ring ((K−1) × 10240).
    public static func ringElems(geometry: QWModelGeometry) -> Int {
        (geometry.gdnConvKernel - 1) * geometry.gdnQKVChannels
    }
    /// bytes of one sequence's full state (151.25 MB + ring tail).
    public static func stateBytes(geometry: QWModelGeometry) -> Int {
        gdnLayers(geometry: geometry) * (recElems(geometry: geometry) + ringElems(geometry: geometry)) * 4
    }
}

/// One sequence's mutable state record. Buffers are ARENA-owned fixed
/// slots; this record is the bookkeeping around them.
public final class QWSeqState: @unchecked Sendable {
    public let seq: Int
    /// One fp32 GDN state buffer per GDN layer: [48·128·128] (engine order).
    public let gdnStates: [QWTensorValue]
    /// One conv-ring per GDN layer: [(K−1)·10240] f32 (C11 layout).
    public let convRings: [QWTensorValue]
    /// mRope position cursor (t-axis next free value; vision spans advance
    /// it by max(h,w)//merge — C19 owns the grid math).
    public internal(set) var position: Int
    /// tokens generated since adoption (journal accounting).
    public internal(set) var generated: Int
    /// prefix fingerprint adopted at (nil = none).
    public internal(set) var prefixFP: String?
    /// true when the last adopt cloned a prefix entry (golden family-B field).
    public internal(set) var clonedFromPrefix: Bool = false
    /// KV cache handle (C10) — C21 wires it; the record owns the ref.
    public internal(set) var kv: QWKVHandle?

    init(seq: Int, gdnStates: [QWTensorValue], convRings: [QWTensorValue]) {
        self.seq = seq
        self.gdnStates = gdnStates
        self.convRings = convRings
        self.position = 0
        self.generated = 0
    }

    /// Bookkeeping half of a snapshot (the state bytes copy rides with the
    /// prefix-cache clone; the pointer half is this record).
    public struct Snap: Sendable, Equatable {
        public let seq: Int
        public let position: Int
        public let generated: Int
        public let prefixFP: String?
    }
    public func snapshot() -> Snap {
        Snap(seq: seq, position: position, generated: generated, prefixFP: prefixFP)
    }
}

// ────────────────────────────── prefix cache ──────────────────────────────

/// A prefix-cache entry: an IMMUTABLE clone (state bytes + cursor).
/// Adoption hands the adopter its own copy — entries are never live.
public struct QWPrefixEntry: Sendable {
    public let fp: String
    public let state: [Data]      // per-GDN-layer rec-state clone (3.15 MB)
    public let rings: [Data]      // per-GDN-layer conv ring clone
    public let snap: QWSeqState.Snap
}

/// The prefix-cache LRU store (not the slot table — QWState owns slots).
/// Semantics exactly as the golden family B logs them.
public final class QWPrefixCache: @unchecked Sendable {
    public let lruN: Int
    private let lock = NSLock()
    private var entries: [String: QWPrefixEntry] = [:]
    private var lru: [String] = []          // front = MRU

    public init(lruN: Int) { self.lruN = lruN }

    /// Upsert + MRU, evicting the LRU tail while over capacity. Returns the
    /// evicted fingerprints (the event half of the spec's eviction law).
    @discardableResult
    public func commit(_ e: QWPrefixEntry) -> [String] {
        lock.lock(); defer { lock.unlock() }
        entries[e.fp] = e
        if let i = lru.firstIndex(of: e.fp) { lru.remove(at: i) }
        lru.insert(e.fp, at: 0)
        var evicted: [String] = []
        while lru.count > lruN, let tail = lru.popLast() {
            entries[tail] = nil
            evicted.append(tail)
        }
        return evicted
    }

    public func lookup(_ fp: String) -> QWPrefixEntry? {
        lock.lock(); defer { lock.unlock() }
        guard let e = entries[fp] else { return nil }
        if let i = lru.firstIndex(of: fp) { lru.remove(at: i); lru.insert(fp, at: 0) }   // MRU touch
        return e
    }

    public var order: [String] { lock.lock(); defer { lock.unlock() }; return lru }
}

// ────────────────────────────── QWState ───────────────────────────────────

/// The sequence-state manager: fixed GDN slots carved once from the arena's
/// gdnStates region (spec: 200-slot soak, zero allocs after init), the
/// prefix cache above, and ownership-based publication.
public final class QWState: @unchecked Sendable {
    public static let module = "QWState"

    public let cap: Int
    public let prefix: QWPrefixCache
    private let lock = NSLock()
    private let slots: [QWSeqState]
    private var freeSlots: [Int]
    private var seqOf: [Int: Int] = [:]                // live seq id → slot index
    /// MTP delta journal (accept path is zero-copy; reject rewinds the ring).
    public let journal: QWDraftJournal

    /// Zero every state byte of a slot (adopt miss — zero-state-bleed law).
    static func zeroAll(_ st: QWSeqState) {
        for t in st.gdnStates + st.convRings {
            memset(t.region.buffer.contents().advanced(by: t.region.base + t.offset), 0, t.bytes)
        }
    }
    /// Copy a prefix entry's bytes into a slot (clone-on-adopt).
    static func copyAll(into st: QWSeqState, from e: QWPrefixEntry) {
        for l in 0..<e.state.count where l < st.gdnStates.count {
            let d = st.gdnStates[l]
            let p = d.region.buffer.contents().advanced(by: d.region.base + d.offset)
            e.state[l].withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, min(d.bytes, $0.count)) }
        }
        for l in 0..<e.rings.count where l < st.convRings.count {
            let d = st.convRings[l]
            let p = d.region.buffer.contents().advanced(by: d.region.base + d.offset)
            e.rings[l].withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, min(d.bytes, $0.count)) }
        }
    }

    /// Carve `cap` slots (one per sequence) from the arena's GDN region.
    /// The engine refuses construction rather than over-committing (capacity
    /// planner: refuse here so macOS jetsam never refuses at runtime).
    public init(hal: QWHal, arena: QWArena, cap: Int, lruN: Int, maxDrafts: Int = 2,
                geometry: QWModelGeometry) throws {
        let gdnLayers = QWSlotGeometry.gdnLayers(geometry: geometry)
        let recElems = QWSlotGeometry.recElems(geometry: geometry)
        let ringElems = QWSlotGeometry.ringElems(geometry: geometry)
        let need = cap * QWSlotGeometry.stateBytes(geometry: geometry)
        guard need <= arena.regionSizes.gdn else {
            throw QWStateError.slotTooSmall(asked: need, capacity: arena.regionSizes.gdn)
        }
        self.cap = cap
        self.prefix = QWPrefixCache(lruN: lruN)
        self.journal = QWDraftJournal(maxDrafts: maxDrafts)
        self.freeSlots = Array(0..<cap)      // FIFO handout (QWPrefixAlgebra law)
        let sub = arena.gdnRegion()
        var carved: [QWSeqState] = []
        carved.reserveCapacity(cap)
        for s in 0..<cap {
            var recs: [QWTensorValue] = []
            var rings: [QWTensorValue] = []
            recs.reserveCapacity(gdnLayers)
            rings.reserveCapacity(gdnLayers)
            for _ in 0..<gdnLayers {
                recs.append(try sub.allocate(QWTensorShape([recElems], .f32)))
            }
            for _ in 0..<gdnLayers {
                rings.append(try sub.allocate(QWTensorShape([ringElems], .f32)))
            }
            carved.append(QWSeqState(seq: s, gdnStates: recs, convRings: rings))
        }
        self.slots = carved
        sub.markSteadyState()
    }

    public var liveCount: Int { lock.lock(); defer { lock.unlock() }; return seqOf.count }

    /// The slot record for a live sequence (raises unknownSeq otherwise).
    public func state(for seq: Int) throws -> QWSeqState {
        lock.lock(); defer { lock.unlock() }
        guard let slot = seqOf[seq] else { throw QWStateError.unknownSeq(seq) }
        return slots[slot]
    }

    /// Adopt a sequence: cache hit → CLONE the entry's bytes into the slot
    /// (clone-on-adopt independence); miss → ZERO the slot (the zero-state-
    /// bleed law: a fresh sequence never sees its slot's previous owner —
    public func adopt(seq: Int, prefixFP: String? = nil) throws -> QWSeqState {
        lock.lock(); defer { lock.unlock() }
        guard seqOf[seq] == nil, !freeSlots.isEmpty else {
            throw QWStateError.slotPressure(live: seqOf.count, cap: cap)
        }
        let slot = freeSlots.removeFirst()   // FIFO handout (QWPrefixAlgebra law)
        seqOf[seq] = slot
        let st = slots[slot]
        st.position = 0
        st.generated = 0
        st.prefixFP = prefixFP
        var cloned = false
        if let fp = prefixFP, let e = prefix.lookup(fp) {
            Self.copyAll(into: st, from: e)
            st.position = e.snap.position
            cloned = true
        } else {
            Self.zeroAll(st)
        }
        st.clonedFromPrefix = cloned
        return st
    }

    /// Release the sequence's slot (state bytes become reachable by the next
    /// owner ONLY through adopt()'s zero/clone paths — never live).
    public func finish(seq: Int) throws {
        lock.lock(); defer { lock.unlock() }
        guard let slot = seqOf.removeValue(forKey: seq) else { throw QWStateError.unknownSeq(seq) }
        freeSlots.append(slot)
    }

    /// Snapshot the sequence: bookkeeping + state-byte clone (prefix-cache
    /// path) — the rollback anchor the spec's parity family A binds.
    public func snapshot(_ st: QWSeqState, as fp: String) -> QWPrefixEntry {
        var state: [Data] = []
        var rings: [Data] = []
        for l in 0..<st.gdnStates.count {
            let t = st.gdnStates[l]
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            state.append(Data(bytes: p, count: t.bytes))
        }
        for t in st.convRings {
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            rings.append(Data(bytes: p, count: t.bytes))
        }
        return QWPrefixEntry(fp: fp, state: state, rings: rings, snap: st.snapshot())
    }

    /// Rollback to a snapshot: restore bytes + cursor (== the golden family-A
    /// "snapshot → decode" arm). Zero-copy accept path never calls this.
    public func rollback(_ st: QWSeqState, to e: QWPrefixEntry) {
        for l in 0..<e.state.count {
            let dst = st.gdnStates[l]
            let p = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            e.state[l].withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, min(dst.bytes, $0.count)) }
        }
        for l in 0..<e.rings.count {
            let dst = st.convRings[l]
            let p = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            e.rings[l].withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, min(dst.bytes, $0.count)) }
        }
        st.position = e.snap.position
    }

    /// Advance counters per accepted token (decode path — no lock needed for
    /// state bytes, ownership is the lock; bookkeeping stays under it).
    public func advance(_ st: QWSeqState, tokens: Int) {
        lock.lock(); defer { lock.unlock() }
        st.generated += tokens
        st.position += tokens
    }
}

// ───────────────────────── MTP delta journal ──────────────────────────────

/// Spec-20's rollback machinery for the MTP draft path: drafts run into
/// sub-slots; accept = commit pointer (zero copy, the common path); reject
/// = rewind. Journal depth == max drafts per round (cap: fall back to full
public final class QWDraftJournal: @unchecked Sendable {
    public let maxDrafts: Int
    private let lock = NSLock()
    private var commits: [Int] = []

    public init(maxDrafts: Int) { self.maxDrafts = maxDrafts }

    public var pending: Int { lock.lock(); defer { lock.unlock() }; return commits.count }

    /// Journal one draft position: stash a rewind point (the state-bytes
    /// clone happens at the ENGINE door — this books the cursor). Refuses
    /// past the journaled depth (returns false): a draft mode that diverges
    /// early per position must fall back to full clone, not grow the ring.
    @discardableResult
    public func journal(position: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard commits.count < maxDrafts else { return false }
        commits.append(position)
        return true
    }

    /// Accept n drafts: commit pointer, zero copy, journal drains (oldest
    /// committed first — drafts are positional).
    public func accept(_ n: Int) {
        lock.lock(); defer { lock.unlock() }
        commits.removeFirst(min(n, commits.count))
    }

    /// Reject: the rewind point is the oldest journaled position; the
    /// journal drains (the engine restores the trunk snapshot itself).
    public func reject() -> Int? {
        lock.lock(); defer { lock.unlock() }
        defer { commits.removeAll() }
        return commits.first
    }
}

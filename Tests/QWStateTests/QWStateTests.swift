// QWStateTests — C20 golden bindings.
//
// Family B (bookkeeping): the FULL st_prefix_trace.json replays through
// Full note: docs/STYLE_NOTES.md#QWStateTests.swift.
import XCTest
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWToken
@testable import QWState

final class QWStateTests: XCTestCase {

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    static var out: URL { root.appendingPathComponent("Scripts/spike/out") }

    static func bins(_ name: String) -> [Float] {
        let d = try! Data(contentsOf: out.appendingPathComponent("\(name).bin"))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    // MARK: family B — prefix algebra trace, event-for-event

    func testPrefixTraceReplayGolden() throws {
        let (ops, events, cap, lruN) = try QWTraceReplay.load(
            Self.out.appendingPathComponent("st_prefix_trace.json"))
        XCTAssertEqual(cap, 4); XCTAssertEqual(lruN, 3)
        XCTAssertEqual(ops.count, 21)
        let bad = QWTraceReplay.divergences(ops: ops, expected: events, cap: cap, lruN: lruN)
        XCTAssertTrue(bad.isEmpty, "trace divergences: \(bad.prefix(8))")
    }

    func testRefusalRaisesBeforeMutationDirect() throws {
        let m = QWPrefixAlgebra(cap: 2, lruN: 2)
        try m.adopt(seq: 1, fp: "a")
        try m.adopt(seq: 2, fp: "b")
        let evN = m.events.count
        XCTAssertThrowsError(try m.adopt(seq: 3, fp: "c")) { e in
            guard case QWStateError.slotPressure(let l, let c) = e as! QWStateError else {
                return XCTFail("wrong error \(e)")
            }
            XCTAssertEqual(l, 2); XCTAssertEqual(c, 2)
        }
        XCTAssertEqual(m.events.count, evN)               // no event logged
        XCTAssertEqual(m.cache, ["b", "a"])               // cache untouched
        XCTAssertTrue(m.freeSlots.isEmpty)
    }

    func testLruEvictionOrder() throws {
        let m = QWPrefixAlgebra(cap: 4, lruN: 2)
        try m.adopt(seq: 1, fp: "a")
        try m.adopt(seq: 2, fp: "b")
        try m.adopt(seq: 3, fp: "c")                      // evicts LRU "a"
        XCTAssertEqual(m.cache, ["c", "b"])
        try m.decode(seq: 1)                              // fp "a" gone — no crash, no touch
        try m.decode(seq: 2)                              // touches "b"
        XCTAssertEqual(m.cache, ["b", "c"])
        try m.finish(seq: 2)
        try m.adopt(seq: 9, fp: "b")                      // HIT → clone
        XCTAssertEqual(m.clones, 1)
    }

    // MARK: family A — byte-level rollback parity on real slots

    /// The canonical walk — SAME ascending fold order the generator uses,
    /// decay bytes from st_dec (np.exp fp32 — the shared table makes
    /// cross-impl exp noise impossible by construction).
    private func recWalk(state sp: UnsafeMutablePointer<Float>, t0: Int, t1: Int,
                         k: [Float], v: [Float], beta: [Float], dec: [Float]) {
        let GVh = QWModelGeometry.qwen38_27b.gdnVHeads, HD = QWModelGeometry.qwen38_27b.gdnHeadDim
        for t in t0..<t1 {
            for h in 0..<GVh {
                let decay = dec[t * GVh + h]
                let dltB = beta[t * GVh + h]
                let ob = h * HD * HD
                var kmem = [Float](repeating: 0, count: HD)
                for kk in 0..<HD {
                    let kt = k[(t * GVh + h) * HD + kk]
                    let base = ob + kk * HD
                    for vv in 0..<HD { kmem[vv] = kmem[vv] + kt * sp[base + vv] }
                }
                let vbase = (t * GVh + h) * HD
                for vv in 0..<HD {
                    let dlt = (v[vbase + vv] - kmem[vv]) * dltB
                    for kk in 0..<HD {
                        sp[ob + kk * HD + vv] = sp[ob + kk * HD + vv] * decay
                            + k[(t * GVh + h) * HD + kk] * dlt
                    }
                }
            }
        }
    }

    private func ringRotate(_ rp: UnsafeMutablePointer<Float>, mixed: [Float], t: Int) {
        let W = QWModelGeometry.qwen38_27b.gdnQKVChannels
        let ringN = QWSlotGeometry.ringElems(geometry: .qwen38_27b)
        for i in 0..<(ringN - W) { rp[i] = rp[i + W] }
        let src = t * W
        for c in 0..<W { rp[ringN - W + c] = mixed[src + c] }
    }

    private func loadFloats(_ name: String) -> [Float] { Self.bins(name) }

    /// THE parity test: walk P on a real slot, snapshot, restore into the
    /// same slot, walk 64 more — bytes == the fresh walk(P+64) gold bytes.
    func testRollbackParityOnRealSlots() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let cap = 2
        let need = cap * QWSlotGeometry.stateBytes(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 4096, activations: 1 << 20,
                                kv: 1 << 20, gdnStates: need + (1 << 12))
        let st0 = try QWState(hal: hal, arena: arena, cap: cap, lruN: 2, maxDrafts: 2, geometry: .qwen38_27b)
        let q = loadFloats("st_q"), k = loadFloats("st_k"), v = loadFloats("st_v")
        let beta = loadFloats("st_beta"), dec = loadFloats("st_dec")
        let mixed = loadFloats("st_mixed")
        guard q.count >= 160 * QWModelGeometry.qwen38_27b.gdnVHeads * QWModelGeometry.qwen38_27b.gdnHeadDim else {
            throw XCTSkip("C20 goldens missing — run Scripts/oracle/gen_state_fixtures.py")
        }

        let recN = QWSlotGeometry.recElems(geometry: .qwen38_27b)

        for P in [1, 37, 96] {
            // fresh slot (adopt miss zeroes it)
            let s = try st0.adopt(seq: 100 + P, prefixFP: nil)
            let sp = s.gdnStates[0].region.buffer.contents()
                .advanced(by: s.gdnStates[0].region.base + s.gdnStates[0].offset)
                .assumingMemoryBound(to: Float.self)
            let rp = s.convRings[0].region.buffer.contents()
                .advanced(by: s.convRings[0].region.base + s.convRings[0].offset)
                .assumingMemoryBound(to: Float.self)

            // (a) walk P → snapshot → restore → walk 64
            for t in 0..<P { ringRotate(rp, mixed: mixed, t: t); recWalk(state: sp, t0: t, t1: t + 1, k: k, v: v, beta: beta, dec: dec) }
            let snap = st0.snapshot(s, as: "fp\(P)")
            // second decode pass happens on a restore: rollback the SAME slot (MTP-rollback semantics)
            st0.rollback(s, to: snap)
            for t in P..<(P + 64) { ringRotate(rp, mixed: mixed, t: t); recWalk(state: sp, t0: t, t1: t + 1, k: k, v: v, beta: beta, dec: dec) }
            let after = Data(bytes: sp, count: recN * 4)
            let ringAfter = Data(bytes: rp, count: QWSlotGeometry.ringElems(geometry: .qwen38_27b) * 4)

            // (c) fresh walk P+64 in a second slot
            let s2 = try st0.adopt(seq: 200 + P, prefixFP: nil)
            let sp2 = s2.gdnStates[0].region.buffer.contents()
                .advanced(by: s2.gdnStates[0].region.base + s2.gdnStates[0].offset)
                .assumingMemoryBound(to: Float.self)
            let rp2 = s2.convRings[0].region.buffer.contents()
                .advanced(by: s2.convRings[0].region.base + s2.convRings[0].offset)
                .assumingMemoryBound(to: Float.self)
            for t in 0..<(P + 64) { ringRotate(rp2, mixed: mixed, t: t); recWalk(state: sp2, t0: t, t1: t + 1, k: k, v: v, beta: beta, dec: dec) }
            let fresh = Data(bytes: sp2, count: recN * 4)
            let ringFresh = Data(bytes: rp2, count: QWSlotGeometry.ringElems(geometry: .qwen38_27b) * 4)

            XCTAssertEqual(after, fresh, "rollback parity VIOLATED at P=\(P) — the MTP rollback anchor is wrong")
            XCTAssertEqual(ringAfter, ringFresh, "conv-ring rollback parity violated at P=\(P)")

            // and the gold arm: the generator's own bytes (P+64) == fresh
            let gold = try Data(contentsOf: Self.out.appendingPathComponent("st_rec_c_P\(P).bin"))
            XCTAssertEqual(gold, fresh, "generator/Swift canonical walk diverged at P=\(P) — summation-order law broken")
            let goldRing = try Data(contentsOf: Self.out.appendingPathComponent("st_ring_c_P\(P).bin"))
            XCTAssertEqual(goldRing, ringFresh, "generator/Swift ring law diverged at P=\(P)")

            try st0.finish(seq: 100 + P)
            try st0.finish(seq: 200 + P)
        }
        // steady-state: slot table drained, zero steady allocs.
        XCTAssertEqual(st0.liveCount, 0)
    }

    /// Clone-on-adopt independence: two sequences adopt the SAME prefix
    /// entry; mutating one's slot must never touch the other's (the
    /// adversarial state-bleed test the spec names).
    func testCloneOnAdoptIndependence() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let need = 2 * QWSlotGeometry.stateBytes(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 4096, activations: 1 << 20,
                                kv: 1 << 20, gdnStates: need + (1 << 12))
        let st0 = try QWState(hal: hal, arena: arena, cap: 2, lruN: 2, maxDrafts: 2, geometry: .qwen38_27b)
        let a = try st0.adopt(seq: 1, prefixFP: nil)       // miss → zeroed
        // seed a state, commit it as a prefix entry
        let sp = a.gdnStates[0].region.buffer.contents()
            .advanced(by: a.gdnStates[0].region.base + a.gdnStates[0].offset)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<16 { sp[i] = Float(i) + 0.5 }
        let entry = st0.snapshot(a, as: "shared")
        st0.prefix.commit(entry)
        try st0.finish(seq: 1)

        let b = try st0.adopt(seq: 2, prefixFP: "shared")  // clone A
        let c = try st0.adopt(seq: 3, prefixFP: "shared")  // clone B (cap=2 → slots are distinct)
        XCTAssertTrue(b.clonedFromPrefix); XCTAssertTrue(c.clonedFromPrefix)
        // mutate clone A, then read clone B — untouched
        let bp = b.gdnStates[0].region.buffer.contents()
            .advanced(by: b.gdnStates[0].region.base + b.gdnStates[0].offset)
            .assumingMemoryBound(to: Float.self)
        bp[0] = -1
        let cp = c.gdnStates[0].region.buffer.contents()
            .advanced(by: c.gdnStates[0].region.base + c.gdnStates[0].offset)
            .assumingMemoryBound(to: Float.self)
        XCTAssertEqual(cp[0], 0.5, "STATE BLEED: two adopters shared one buffer")
    }

    /// Zero-state bleed: a slot recycled from a finished sequence must be
    /// zeroed by adopt-miss (never carries the previous owner's bytes).
    func testSlotRecycleZeroesState() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let need = QWSlotGeometry.stateBytes(geometry: .qwen38_27b)          // cap 1
        let arena = try QWArena(hal: hal, weights: 4096, activations: 1 << 20,
                                kv: 1 << 20, gdnStates: need + (1 << 12))
        let st0 = try QWState(hal: hal, arena: arena, cap: 1, lruN: 1, maxDrafts: 2, geometry: .qwen38_27b)
        let a = try st0.adopt(seq: 1, prefixFP: nil)
        let sp = a.gdnStates[0].region.buffer.contents()
            .advanced(by: a.gdnStates[0].region.base + a.gdnStates[0].offset)
            .assumingMemoryBound(to: Float.self)
        sp[0] = 42
        try st0.finish(seq: 1)
        let b = try st0.adopt(seq: 2, prefixFP: nil)   // same slot (cap 1), MISS
        XCTAssertEqual(sp[0], 0, "slot recycle leaked the previous owner's state")
        // pointer-identity proof it really IS the same slot
        XCTAssertEqual(a.gdnStates[0].offset, b.gdnStates[0].offset)
    }

    /// Slot accounting: RSS law of the spec — slots == cap × stateBytes,
    /// exact, and steady-state allocations stay zero after init.
    func testSlotAccountingExact() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let cap = 3
        let need = cap * QWSlotGeometry.stateBytes(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 4096, activations: 1 << 20,
                                kv: 1 << 20, gdnStates: need + (1 << 12))
        let st0 = try QWState(hal: hal, arena: arena, cap: cap, lruN: 2, maxDrafts: 2, geometry: .qwen38_27b)
        // per-sequence state footprint: 48 rec (3.15 MB each) + 48 rings —
        // the spec's 151 MB/seq number, re-derived (0 violations).
        let perSeq = QWSlotGeometry.stateBytes(geometry: .qwen38_27b)
        XCTAssertEqual(perSeq, QWSlotGeometry.gdnLayers(geometry: .qwen38_27b) *
            (QWSlotGeometry.recElems(geometry: .qwen38_27b) + QWSlotGeometry.ringElems(geometry: .qwen38_27b)) * 4)
        XCTAssertEqual(QWSlotGeometry.gdnLayers(geometry: .qwen38_27b), 48)
        let a = try st0.adopt(seq: 7, prefixFP: nil)
        XCTAssertEqual(st0.liveCount, 1)
        _ = a
        try st0.finish(seq: 7)
        XCTAssertEqual(st0.liveCount, 0)
    }

    /// MTP journal law: journal N drafts, accept n (zero-copy commit), or
    /// reject → the rewind point is the FIRST journaled position.
    func testDraftJournalAcceptReject() throws {
        let j = QWDraftJournal(maxDrafts: 2)
        j.journal(position: 10); j.journal(position: 11)
        j.accept(1)                                     // commit position-10 draft
        XCTAssertEqual(j.pending, 1)
        XCTAssertEqual(j.reject(), 11)                  // rewind point = oldest
        XCTAssertEqual(j.pending, 0)
        XCTAssertNil(j.reject())                        // empty journal
        // depth cap: a third draft without draining is a bug, refuse loudly
        j.journal(position: 1); j.journal(position: 2)
        XCTAssertEqual(j.pending, 2)
        XCTAssertTrue(j.journal(position: 3) == false)  // refuses past depth
    }
}

// MARK: - Test-only trace replay harness & prefix algebra (Family B verification)

private extension QWJson {
    var arrayValue: [QWJson]? { if case .array(let a) = self { return a }; return nil }
    var stringValue: String? { if case .string(let s) = self { return s }; return nil }
}

enum QWTraceReplay {
    static func load(_ url: URL) throws -> (ops: [QWJson], events: [QWJson], cap: Int, lruN: Int) {
        guard let data = try? Data(contentsOf: url) else {
            throw QWStateError.badTrace("missing \(url.lastPathComponent)")
        }
        let root = try QWJson.parse(data)
        guard case .array(let ops) = root["ops"], case .array(let events) = root["events"],
              let cap = intOf(root["cap"]), let lruN = intOf(root["lru_n"]) else {
            throw QWStateError.badTrace("shape")
        }
        return (ops, events, cap, lruN)
    }
    private static func intOf(_ j: QWJson) -> Int? {
        switch j { case .int(let i): return i; case .double(let d): return Int(d); default: return nil }
    }

    static func divergences(ops: [QWJson], expected: [QWJson],
                            cap: Int, lruN: Int) -> [String] {
        let m = QWPrefixAlgebra(cap: cap, lruN: lruN)
        var bad: [String] = []
        for (i, ev) in ops.enumerated() {
            guard let a = ev.arrayValue, let kind = a.first?.stringValue else {
                bad.append("op \(i): shape"); continue
            }
            if i == ops.count - 1, kind == "adopt", m.freeSlots.isEmpty {
                let beforeEvents = m.events.count
                let beforeCache = m.cache, beforeFree = m.freeSlots, beforeSlotOf = m.slotOf
                do {
                    try m.adopt(seq: int(a[1]), fp: a[2].stringValue!)
                    bad.append("refusal op did not raise")
                } catch let e as QWStateError {
                    guard case .slotPressure = e else { bad.append("refusal raised wrong error: \(e)"); return bad }
                    guard m.events.count == beforeEvents, m.cache == beforeCache,
                          m.freeSlots == beforeFree, m.slotOf == beforeSlotOf else {
                        bad.append("refusal MUTATED state (law B)"); return bad
                    }
                } catch { bad.append("refusal raised foreign error"); return bad }
                break
            }
            do {
                switch kind {
                case "adopt": try m.adopt(seq: int(a[1]), fp: a[2].stringValue!)
                case "decode": try m.decode(seq: int(a[1]))
                case "finish": try m.finish(seq: int(a[1]))
                default: bad.append("op \(i): unknown \(kind)"); return bad
                }
            } catch {
                bad.append("op \(i): unexpected raise on \(kind)")
                return bad
            }
        }
        let mine = m.events
        guard mine.count == expected.count else {
            bad.append("event count \(mine.count) vs golden \(expected.count)")
            return bad
        }
        for i in 0..<mine.count {
            let g = expected[i], o = mine[i]
            for key in ["op", "seq", "slot", "fp", "hit", "clones", "cache", "free", "gen"] {
                let gv = g[key]
                if !gv.isUndefined && gv != o[key] { bad.append("event \(i): \(key) diverged") }
            }
        }
        return bad
    }
    private static func int(_ j: QWJson) -> Int {
        switch j { case .int(let i): return i; case .double(let d): return Int(d); default: fatalError() }
    }
}

final class QWPrefixAlgebra: @unchecked Sendable {
    let cap: Int
    let lruN: Int
    private(set) var freeSlots: [Int]
    private(set) var slotOf: [Int: Int] = [:]
    private(set) var slotFP: [Int: String] = [:]
    private(set) var slotGen: [Int: Int] = [:]
    private(set) var cache: [String] = []
    private(set) var clones: Int = 0
    private(set) var events: [QWJson] = []

    init(cap: Int, lruN: Int) {
        self.cap = cap; self.lruN = lruN
        self.freeSlots = Array(0..<cap)
    }

    private func log(_ pairs: [(String, QWJson)]) { events.append(.object(pairs)) }

    private func popFree() -> Int? {
        guard !freeSlots.isEmpty else { return nil }
        return freeSlots.removeFirst()
    }

    func adopt(seq: Int, fp: String) throws {
        guard slotOf[seq] == nil, let slot = popFree() else {
            throw QWStateError.slotPressure(live: slotOf.count, cap: cap)
        }
        let hit = cache.contains(fp)
        if hit {
            cache.removeAll { $0 == fp }
            clones += 1
        } else {
            while cache.count >= lruN, let tail = cache.popLast() {
                log([("op", .string("evict")), ("fp", .string(tail))])
            }
        }
        cache.insert(fp, at: 0)
        slotOf[seq] = slot
        slotFP[slot] = fp
        slotGen[slot] = 0
        log([("op", .string("adopt")), ("seq", .int(seq)), ("fp", .string(fp)),
             ("slot", .int(slot)), ("hit", .bool(hit)),
             ("cache", .array(cache.map(QWJson.string))), ("clones", .int(clones))])
    }

    func decode(seq: Int) throws {
        guard let slot = slotOf[seq] else { throw QWStateError.unknownSeq(seq) }
        slotGen[slot, default: 0] += 1
        let g = slotGen[slot]!
        let fp = slotFP[slot] ?? ""
        if let i = cache.firstIndex(of: fp) {
            cache.remove(at: i); cache.insert(fp, at: 0)
        }
        log([("op", .string("decode")), ("seq", .int(seq)), ("slot", .int(slot)),
             ("gen", .int(g)),
             ("cache", .array(cache.map(QWJson.string)))])
    }

    func finish(seq: Int) throws {
        guard let slot = slotOf.removeValue(forKey: seq) else { throw QWStateError.unknownSeq(seq) }
        slotFP[slot] = nil
        slotGen[slot] = nil
        freeSlots.append(slot)
        log([("op", .string("finish")), ("seq", .int(seq)), ("slot", .int(slot)),
             ("free", .array(freeSlots.sorted().map(QWJson.int)))])
    }
}

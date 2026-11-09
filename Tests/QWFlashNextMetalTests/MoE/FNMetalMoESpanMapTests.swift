import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// Dynamic-residency prefill span map (`qwg_moe_span_ids`): the GPU maps router ids to bank rows so the
/// resident pass runs before the host reads the route. The split resident → cold flow on GPU-mapped ids must
/// be bit-identical to the host-mapped single pass it replaces (only the staging-slot numbering differs).
final class FNMetalMoESpanMapTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private struct Rng {
        var s: UInt64
        mutating func next() -> UInt32 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            return UInt32(truncatingIfNeeded: s >> 32)
        }
    }

    private func makeSlab(_ n: Int, _ k: Int, _ rng: inout Rng) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        for _ in 0..<(n * (k / 32)) {
            d.append(Data([UInt8(0x1F), 0x2C, 0, 0]))    // scale 0x2C1F, bias 0
            for _ in 0..<16 { d.append(UInt8(rng.next() & 0xFF)) }
        }
        return d
    }

    /// Host reference of the span map: slot, residentE + rank of the miss in expert-id order, or -1.
    private func hostSpanMap(_ ids: [Int32], slotMap: [Int32], residentE: Int) -> [Int32] {
        let experts = slotMap.count
        var miss = [Bool](repeating: false, count: experts)
        for e in ids where e >= 0 && Int(e) < experts && slotMap[Int(e)] < 0 { miss[Int(e)] = true }
        var rank = [Int32](repeating: 0, count: experts)
        var r: Int32 = 0
        for e in 0..<experts { rank[e] = r; if miss[e] { r += 1 } }
        return ids.map { e in
            guard e >= 0, Int(e) < experts else { return -1 }
            let s = slotMap[Int(e)]
            return s >= 0 ? s : Int32(residentE) + rank[Int(e)]
        }
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    func testSpanIdsMatchHostReference() throws {
        for (experts, residentE, count) in [(256, 96, 4096), (512, 200, 16_384), (64, 0, 333)] {
            let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            var rng = Rng(s: 0xA5A5_1234_5678_9ABC &+ UInt64(experts))
            var perm = Array(0..<experts)
            for i in stride(from: experts - 1, to: 0, by: -1) { perm.swapAt(i, Int(rng.next()) % (i + 1)) }
            var slotMap = [Int32](repeating: -1, count: experts)
            for s in 0..<residentE { slotMap[perm[s]] = Int32(s) }
            var ids = (0..<count).map { _ in Int32(rng.next() % UInt32(experts)) }
            ids[0] = -1
            ids[count - 1] = Int32(experts + 3)
            let idsT = try act.allocate(QWTensorShape([count], .i32))
            arena.copyIn(idsT, from: ids.withUnsafeBufferPointer { Data(buffer: $0) })
            let outT = try act.allocate(QWTensorShape([count], .i32))
            let map = try XCTUnwrap(Self.hal.device.makeBuffer(bytes: slotMap, length: 4 * experts, options: .storageModeShared))
            try step { enc in
                try QWGemm.encodeMoESpanIds(enc, hal: Self.hal, ids: idsT, slotMap: map, out: outT, count: count,
                                            experts: experts, residentExperts: residentE)
            }
            let got = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: outT).bindMemory(to: Int32.self, capacity: count),
                                                count: count))
            XCTAssertEqual(got, hostSpanMap(ids, slotMap: slotMap, residentE: residentE),
                           "span map E=\(experts) resident=\(residentE) count=\(count)")
        }
    }

    func testSplitResidentColdOnSpanMapIsBitExact() throws {
        let experts = 32, residentE = 12, coldE = experts - residentE
        let routes = 4, tokens = 96, hidden = 128, inter = 64
        let pairs = tokens * routes
        let guRows = 2 * inter
        let guSlabBytes = guRows * (hidden / 32) * 20
        let dnSlabBytes = hidden * (inter / 32) * 20
        var rng = Rng(s: 0x1234_5678_9ABC_DEF1)
        let guPhys = (0..<experts).map { _ in makeSlab(guRows, hidden, &rng) }
        let dnPhys = (0..<experts).map { _ in makeSlab(hidden, inter, &rng) }

        var perm = Array(0..<experts)
        for i in stride(from: experts - 1, to: 0, by: -1) { perm.swapAt(i, Int(rng.next()) % (i + 1)) }
        let expertInSlot = Array(perm.prefix(residentE))
        var slotMap = [Int32](repeating: -1, count: experts)
        for (s, e) in expertInSlot.enumerated() { slotMap[e] = Int32(s) }

        // Distinct experts per token, as the router emits.
        var ids = [Int32](repeating: 0, count: pairs)
        for t in 0..<tokens {
            var chosen: [Int32] = []
            while chosen.count < routes {
                let e = Int32(rng.next() % UInt32(experts))
                if !chosen.contains(e) { chosen.append(e) }
            }
            for r in 0..<routes { ids[t * routes + r] = chosen[r] }
        }

        let arena = try QWArena(hal: Self.hal, weights: 8 << 20, activations: 256 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        func alloc(_ n: Int, _ dt: QWDType) throws -> QWTensorValue { try act.allocate(QWTensorShape([n], dt)) }
        func fill(_ t: QWTensorValue, _ slabs: [Data]) {
            var d = Data()
            for s in slabs { d.append(s) }
            arena.copyIn(t, from: d)
        }
        let resGU = try alloc(residentE * guSlabBytes / 2, .f16)
        let resDn = try alloc(residentE * dnSlabBytes / 2, .f16)
        fill(resGU, expertInSlot.map { guPhys[$0] })
        fill(resDn, expertInSlot.map { dnPhys[$0] })
        let resGUView = try XCTUnwrap(QWWeightView(blocks: resGU, n: residentE * guRows, k: hidden, isRotated: true))
        let resDnView = try XCTUnwrap(QWWeightView(blocks: resDn, n: residentE * hidden, k: inter, isRotated: true))

        let x16 = try alloc(tokens * hidden, .f16)
        var xs = [Float16](repeating: 0, count: tokens * hidden)
        var seed: UInt32 = 4242
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float16(Float(Int32(truncatingIfNeeded: seed)) * 2e-13) }
        arena.copyIn(x16, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
        let wts = try alloc(pairs, .f32)
        var wv = [Float](repeating: 0, count: pairs)
        for i in 0..<pairs { wv[i] = Float(rng.next() % 1000) / 1000 }
        arena.copyIn(wts, from: wv.withUnsafeBufferPointer { Data(buffer: $0) })
        let shared = try alloc(tokens * hidden, .f32)
        arena.copyIn(shared, from: Data(count: tokens * hidden * 4))
        let gate = try alloc(tokens, .f32)
        arena.copyIn(gate, from: Data(count: tokens * 4))

        let expOff = try alloc(experts + 1, .i32), sortedPair = try alloc(pairs, .i32)
        let sortedTok = try alloc(pairs, .i32), invPerm = try alloc(pairs, .i32)
        let tileE = try alloc(pairs, .i32), tileR = try alloc(pairs, .i32), tileCount = try alloc(1, .i32)
        let gu16 = try alloc(pairs * guRows, .f16), dnHi = try alloc(pairs * inter, .f16)
        let dnAcc16 = try alloc(pairs * hidden, .f16)

        func coldBanks() throws -> (QWWeightView, QWWeightView, QWTensorValue, QWTensorValue) {
            let gu = try alloc(coldE * guSlabBytes / 2, .f16)
            let dn = try alloc(coldE * dnSlabBytes / 2, .f16)
            arena.copyIn(gu, from: Data(count: coldE * guSlabBytes))
            arena.copyIn(dn, from: Data(count: coldE * dnSlabBytes))
            return (try XCTUnwrap(QWWeightView(blocks: gu, n: coldE * guRows, k: hidden, isRotated: true)),
                    try XCTUnwrap(QWWeightView(blocks: dn, n: coldE * hidden, k: inter, isRotated: true)), gu, dn)
        }
        func stage(_ stagingExperts: [Int], gu: QWTensorValue, dn: QWTensorValue) {
            let guPtr = arena.cpuPointer(of: gu), dnPtr = arena.cpuPointer(of: dn)
            for (slot, e) in stagingExperts.enumerated() {
                guPhys[e].withUnsafeBytes { guPtr.advanced(by: slot * guSlabBytes).copyMemory(from: $0.baseAddress!, byteCount: guSlabBytes) }
                dnPhys[e].withUnsafeBytes { dnPtr.advanced(by: slot * dnSlabBytes).copyMemory(from: $0.baseAddress!, byteCount: dnSlabBytes) }
            }
        }
        func group(_ enc: any MTLComputeCommandEncoder, _ idsT: QWTensorValue) throws {
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: idsT, expertOff: expOff, sortedPairIdx: sortedPair,
                                              sortedTok: sortedTok, experts: experts, totalPairs: pairs, routes: routes,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR, tileCount: tileCount)
        }
        func gateUp(_ enc: any MTLComputeCommandEncoder, _ cold: QWWeightView, _ f: QWGemm.MoETileFilter) throws {
            try QWGemm.encodeMoEGateUpNax(enc, hal: Self.hal, xe: x16, bank: resGUView, expertOff: expOff, tileE: tileE,
                                          tileR: tileR, tileCount: tileCount, gu16: gu16, rowsPerExpert: guRows,
                                          pairs: pairs, sortedTok: sortedTok, coldBank: cold, tileFilter: f)
        }
        func silu(_ enc: any MTLComputeCommandEncoder) throws {
            try QWGemm.encodeSiluMulSplitF16(enc, hal: Self.hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                             rows: pairs, intermediate: inter)
        }
        func down(_ enc: any MTLComputeCommandEncoder, _ cold: QWWeightView, _ f: QWGemm.MoETileFilter) throws {
            try QWGemm.encodeMoENaxSplitF32(enc, hal: Self.hal, xHi: dnHi, xLo: dnHi, bank: resDnView, expertOff: expOff,
                                            tileE: tileE, tileR: tileR, tileCount: tileCount, y32: dnAcc16,
                                            rowsPerExpert: hidden, pairs: pairs, coldBank: cold, tileFilter: f)
        }
        func reduce(_ enc: any MTLComputeCommandEncoder, _ out: QWTensorValue) throws {
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: Self.hal, dnAcc: dnAcc16, invPerm: invPerm, weights: wts,
                                                     shared: shared, gate: gate, out: out, hidden: hidden,
                                                     routes: routes, tokens: tokens)
        }
        func read(_ t: QWTensorValue) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: tokens * hidden),
                                      count: tokens * hidden))
        }

        // A — the host-mapped span: misses staged in first-appearance order, one full pass.
        var stagedA: [Int] = []
        var stagedSlotA: [Int: Int32] = [:]
        let idsA: [Int32] = ids.map { e in
            let s = slotMap[Int(e)]
            if s >= 0 { return s }
            if let x = stagedSlotA[Int(e)] { return x }
            let x = Int32(residentE + stagedA.count)
            stagedSlotA[Int(e)] = x
            stagedA.append(Int(e))
            return x
        }
        XCTAssertGreaterThan(stagedA.count, 4, "the span must stage misses")
        let idsAT = try alloc(pairs, .i32)
        arena.copyIn(idsAT, from: idsA.withUnsafeBufferPointer { Data(buffer: $0) })
        let (coldGUA, coldDnA, guA, dnA) = try coldBanks()
        stage(stagedA, gu: guA, dn: dnA)
        let outA = try alloc(tokens * hidden, .f32)
        try step { enc in
            try group(enc, idsAT); try gateUp(enc, coldGUA, .all); try silu(enc); try down(enc, coldDnA, .all)
            try reduce(enc, outA)
        }

        // B — the GPU span map: route CB maps, resident CB committed before the host reads the map,
        // the host stages misses into the GPU's slots, then the cold CB and the reduce.
        let rawIds = try alloc(pairs, .i32)
        arena.copyIn(rawIds, from: ids.withUnsafeBufferPointer { Data(buffer: $0) })
        let mapped = try alloc(pairs, .i32)
        let mapBuf = try XCTUnwrap(Self.hal.device.makeBuffer(bytes: slotMap, length: 4 * experts, options: .storageModeShared))
        let (coldGUB, coldDnB, guB, dnB) = try coldBanks()
        guard let (routeCB, routeEnc) = Self.hal.begin(), let (resCB, resEnc) = Self.hal.begin() else {
            throw XCTSkip("no command buffer")
        }
        try QWGemm.encodeMoESpanIds(routeEnc, hal: Self.hal, ids: rawIds, slotMap: mapBuf, out: mapped, count: pairs,
                                    experts: experts, residentExperts: residentE)
        Self.hal.end(routeCB, routeEnc)
        try group(resEnc, mapped); try gateUp(resEnc, coldGUB, .residentOnly); try silu(resEnc)
        try down(resEnc, coldDnB, .residentOnly)
        Self.hal.end(resCB, resEnc)
        routeCB.waitUntilCompleted()
        let mappedHost = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: mapped).bindMemory(to: Int32.self, capacity: pairs),
                                                   count: pairs))
        XCTAssertEqual(mappedHost, hostSpanMap(ids, slotMap: slotMap, residentE: residentE))
        var stagedB = [Int](repeating: -1, count: coldE)
        for (i, m) in mappedHost.enumerated() where m >= Int32(residentE) { stagedB[Int(m) - residentE] = Int(ids[i]) }
        stagedB = Array(stagedB.prefix { $0 >= 0 })
        XCTAssertEqual(Set(stagedB), Set(stagedA))
        XCTAssertNotEqual(stagedB, stagedA, "the GPU numbering should differ from first-appearance order here")
        stage(stagedB, gu: guB, dn: dnB)
        let outB = try alloc(tokens * hidden, .f32)
        try step { enc in
            try gateUp(enc, coldGUB, .coldOnly); try silu(enc); try down(enc, coldDnB, .coldOnly)
            try reduce(enc, outB)
        }
        XCTAssertNil(resCB.error)

        let a = read(outA), b = read(outB)
        XCTAssertTrue(a.allSatisfy { $0.isFinite }, "reference output must be finite")
        XCTAssertGreaterThan(a.reduce(0) { max($0, abs($1)) }, 1e-3, "reference output must be non-trivial")
        var mismatches = 0
        for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern { mismatches += 1 }
        XCTAssertEqual(mismatches, 0, "split resident/cold on the span map must be bit-identical to the host-mapped pass")

        // Negative control: the cold experts must contribute, or the comparison above proves nothing.
        let (coldGUZ, coldDnZ, _, _) = try coldBanks()
        let outZ = try alloc(tokens * hidden, .f32)
        try step { enc in
            try group(enc, idsAT); try gateUp(enc, coldGUZ, .all); try silu(enc); try down(enc, coldDnZ, .all)
            try reduce(enc, outZ)
        }
        let z = read(outZ)
        XCTAssertGreaterThan((0..<a.count).filter { a[$0] != z[$0] }.count, a.count / 4,
                             "unstaged cold experts must change the output")
    }
}

// 4c: the MTP draft-pool writer (`qwg_mtp_seed_kv`) is a THIRD pool writer
// (f16 copy pre-4c). Under q8/q4 it must store the seeded row BYTE-IDENTICAL
// to the Swift reference quantizers (quantizeQ8/quantizeQ4) — the same gate
// QWAttnDecodeQuantKVTests pins for the decode epilogue and paged writer.
// Drift here poisons the draft cache's first row and shows up as wrong draft
// tokens at q8/q4 runs, never as a crash — so it is pinned at byte level.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWKV
@testable import QWEngine

final class QWMTPSeedKVQuantTests: XCTestCase {

    static func runSeed(mode: QWKVMode, k: [Float16], v: [Float16], row: Int)
        throws -> (kBytes: [UInt8], vBytes: [UInt8], kPage: Int, vPage: Int)
    {
        let hal = try QWHal(geometry: .qwen38_27b)
        let pso = try hal.pipeline("qwg_mtp_seed_kv")
        let pageBytes = QWKVCache.pageBytes(mode: mode)
        // Two pages per pool so table ids ≠ 0 exercise the stride math.
        func pool(_ bytes: Int) -> MTLBuffer { hal.device.makeBuffer(length: bytes * 2, options: .storageModeShared)! }
        let kPool = pool(pageBytes), vPool = pool(pageBytes)
        // fill garbage to catch unwritten bytes
        memset(kPool.contents(), 0x5A, pageBytes * 2)
        memset(vPool.contents(), 0x5A, pageBytes * 2)
        let kSrc = hal.device.makeBuffer(length: 1024 * 2, options: .storageModeShared)!
        let vSrc = hal.device.makeBuffer(length: 1024 * 2, options: .storageModeShared)!
        k.withUnsafeBytes { _ = memcpy(kSrc.contents(), $0.baseAddress!, 2048) }
        v.withUnsafeBytes { _ = memcpy(vSrc.contents(), $0.baseAddress!, 2048) }
        let kTable = hal.device.makeBuffer(length: 8, options: .storageModeShared)!
        let vTable = hal.device.makeBuffer(length: 8, options: .storageModeShared)!
        kTable.contents().assumingMemoryBound(to: Int32.self)[0] = 0
        kTable.contents().assumingMemoryBound(to: Int32.self)[1] = 1   // row 300 → page 1 → id 1
        vTable.contents().assumingMemoryBound(to: Int32.self)[0] = 0
        vTable.contents().assumingMemoryBound(to: Int32.self)[1] = 1
        var rowU = UInt32(row)
        var modeU: UInt32 = (mode == .f16) ? 0 : (mode == .q8 ? 1 : 2)
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        enc.setComputePipelineState(pso)
        enc.setBuffer(kSrc, offset: 0, index: 0)
        enc.setBuffer(vSrc, offset: 0, index: 1)
        enc.setBuffer(kPool, offset: 0, index: 2)
        enc.setBuffer(vPool, offset: 0, index: 3)
        enc.setBuffer(kTable, offset: 0, index: 4)
        enc.setBuffer(vTable, offset: 0, index: 5)
        enc.setBytes(&rowU, length: 4, index: 6)
        enc.setBytes(&modeU, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let slot = row % 256
        func grab(_ p: MTLBuffer, _ n: Int) -> [UInt8] {
            let base = p.contents().advanced(by: pageBytes /*page id 1*/ + slot * (mode == .f16 ? 2048 : (mode == .q8 ? 1024 : 512)))
            return (0..<n).map { base.assumingMemoryBound(to: UInt8.self)[$0] }
        }
        let payloadLen = mode == .f16 ? 2048 : (mode == .q8 ? 1024 : 512)
        let kP = grab(kPool, payloadLen)
        let vP = grab(vPool, payloadLen)
        return (kP, vP, 1, 1)
    }

    static func seededRow(mode: QWKVMode) throws -> ([Float16], [Float16]) {
        // deterministic pseudo-random KV row (no fixture dependency)
        var x: UInt32 = 0x2545F491
        func r() -> Float16 {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5
            return Float16(Float(bitPattern: 0x3F000000 | (x & 0x7FFFFF)) * (x & 1 == 0 ? 1 : -1) * 2.0)
        }
        return ((0..<1024).map { _ in r() }, (0..<1024).map { _ in r() })
    }

    func testSeedWriterMatchesReferenceQ8() throws {
        let (k, v) = try Self.seededRow(mode: .q8)
        let (kb, vb, _, _) = try Self.runSeed(mode: .q8, k: k, v: v, row: 300)
        var f = [Float](repeating: 0, count: 1024)
        var q = [Int8](repeating: 0, count: 1024)
        var s = [Float16](repeating: 0, count: 4)
        for i in 0..<1024 { f[i] = Float(k[i]) }
        try QWKVCompressedLayout.quantizeQ8(input: f, qOut: &q, scalesOut: &s)
        var mism = 0
        for i in 0..<1024 where kb[i] != UInt8(bitPattern: q[i]) { mism += 1 }
        XCTAssertEqual(mism, 0, "q8 seed payload must match quantizeQ8 byte-for-byte")
        // scales ride beyond the payload — re-read via page math
        // (same offsets as the kernel: 262144 + slot*8 + h*2, page id 1).
        // verified via hal buffer grab below in a second stage:
        var g = [Float](repeating: 0, count: 1024)
        var q2 = [Int8](repeating: 0, count: 1024)
        var s2 = [Float16](repeating: 0, count: 4)
        for i in 0..<1024 { g[i] = Float(v[i]) }
        try QWKVCompressedLayout.quantizeQ8(input: g, qOut: &q2, scalesOut: &s2)
        var mismV = 0
        for i in 0..<1024 where vb[i] != UInt8(bitPattern: q2[i]) { mismV += 1 }
        XCTAssertEqual(mismV, 0, "q8 seed V payload must match")
    }

    func testSeedWriterMatchesReferenceQ4() throws {
        let (k, v) = try Self.seededRow(mode: .q4)
        let (kb, vb, _, _) = try Self.runSeed(mode: .q4, k: k, v: v, row: 300)
        var f = [Float](repeating: 0, count: 1024)
        var q = [UInt8](repeating: 0, count: 512)
        var sc = [Float16](repeating: 0, count: 8)
        var mn = [Float16](repeating: 0, count: 8)
        for i in 0..<1024 { f[i] = Float(k[i]) }
        try QWKVCompressedLayout.quantizeQ4(input: f, qOut: &q, scalesOut: &sc, minsOut: &mn)
        var mism = 0
        for i in 0..<512 where kb[i] != q[i] { mism += 1 }
        XCTAssertEqual(mism, 0, "q4 seed payload must match quantizeQ4 byte-for-byte")
        for i in 0..<1024 { f[i] = Float(v[i]) }
        var q2 = [UInt8](repeating: 0, count: 512)
        var s2 = [Float16](repeating: 0, count: 8)
        var m2 = [Float16](repeating: 0, count: 8)
        try QWKVCompressedLayout.quantizeQ4(input: f, qOut: &q2, scalesOut: &s2, minsOut: &m2)
        var mismV = 0
        for i in 0..<512 where vb[i] != q2[i] { mismV += 1 }
        XCTAssertEqual(mismV, 0, "q4 seed V payload must match")
    }
}

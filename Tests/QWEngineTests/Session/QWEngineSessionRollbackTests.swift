import XCTest
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV
@testable import QWEngine

final class QWEngineSessionRollbackTests: XCTestCase {
    private struct Fixture {
        let hal: QWHal
        let arena: QWArena
        let cache: QWKVCache
        let mtpCache: QWKVCache
        let session: QWEngineSession
    }

    private func fixture() throws -> Fixture {
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 256, activations: 256,
                                kv: 8 << 20, gdnStates: 64 << 10)
        let kvSub = arena.kvRegion()
        let trunkOffset = try kvSub.allocate(bytes: 4 << 20)
        let mtpOffset = try kvSub.allocate(bytes: 4 << 20)
        func makeCache(at offset: Int) throws -> QWKVCache {
            let sub = QWSubAllocator(region: QWMRegion(buffer: kvSub.region.buffer,
                                                       base: kvSub.region.base + offset,
                                                       length: 4 << 20))
            return try QWKVCache(hal: hal, subAllocator: sub, layers: [0], mode: .f16, arena: arena, geometry: .qwen38_27b)
        }
        let cache = try makeCache(at: trunkOffset), mtpCache = try makeCache(at: mtpOffset)
        let gsub = arena.gdnRegion()
        let recSlab = try gsub.allocate(QWTensorShape([48 * 32], .f32))
        let ringSlab = try gsub.allocate(QWTensorShape([48 * 16], .f16))
        let hidden = try gsub.allocate(QWTensorShape([5120], .f16))
        func views(_ slab: QWTensorValue, count: Int, type: QWDType) -> [QWTensorValue] {
            (0..<48).map {
                QWTensorValue(region: slab.region, offset: slab.offset + $0 * count * (type == .f32 ? 4 : 2),
                              shape: QWTensorShape([count], type))
            }
        }
        let session = QWEngineSession(kv: try cache.allocate(promptTokens: 512),
                                      mtpKV: try mtpCache.allocate(promptTokens: 512),
                                      rec: views(recSlab, count: 32, type: .f32),
                                      ring: views(ringSlab, count: 16, type: .f16),
                                      recSlab: recSlab, ringSlab: ringSlab, trunkHidden: hidden)
        session.trunkHiddenValid = true
        for tensor in [recSlab, ringSlab, hidden] {
            let p = tensor.region.pointer.advanced(by: tensor.offset).assumingMemoryBound(to: UInt8.self)
            for i in 0..<tensor.bytes { p[i] = UInt8(truncatingIfNeeded: i * 31 + 7) }
        }
        session.position = 255
        session.generated = 50
        session.roundRecord = QWRoundRecord(pendingToken: 1234, consumedCount: 255,
                                            outputCount: 50, trunkHiddenSeed: [1.5, -2],
                                            mtpPhysicalRows: 254, absolutePosition: 255, samplerDrawIndex: 19)
        cache.markWritten(session.kv, upTo: 255)
        mtpCache.markWritten(session.mtpKV!, upTo: 254)
        return Fixture(hal: hal, arena: arena, cache: cache, mtpCache: mtpCache, session: session)
    }

    private func restore(_ snapshot: QWEngineSessionSnapshot, into f: Fixture) throws {
        try QWEngineSessionState.restore(f.session, to: snapshot, cache: f.cache,
                                         mtpCache: f.mtpCache, attnLayers: [0])
    }

    private func bytes(_ tensor: QWTensorValue) -> Data {
        Data(bytes: tensor.region.pointer.advanced(by: tensor.offset), count: tensor.bytes)
    }

    func testPrivatePrefixValidatesBothCachesBeforeStateOrCursorMutation() throws {
        let f = try fixture(), donor = f.session
        donor.position = 258
        donor.roundRecord = QWRoundRecord(pendingToken: 1234, consumedCount: 258,
            outputCount: donor.generated, mtpPhysicalRows: 257, absolutePosition: 258)
        f.cache.markWritten(donor.kv, upTo: 258)
        f.mtpCache.markWritten(try XCTUnwrap(donor.mtpKV), upTo: 257)
        XCTAssertThrowsError(try QWEngineSessionState.capturePrivatePrefix(donor, cache: f.cache,
            mtpCache: f.mtpCache, maxBytes: 1))
        let prefix = try QWEngineSessionState.capturePrivatePrefix(donor, cache: f.cache,
            mtpCache: f.mtpCache, maxBytes: 8 << 20)
        f.cache.finish(donor.kv); f.mtpCache.finish(try XCTUnwrap(donor.mtpKV))
        donor.sessionFinished = true
        let g = f.arena.gdnRegion()
        let rec = try g.allocate(donor.recSlab.shape), ring = try g.allocate(donor.ringSlab.shape)
        let hidden = try g.allocate(try XCTUnwrap(donor.trunkHidden).shape)
        let destination = QWEngineSession(kv: try f.cache.allocate(promptTokens: 258),
            mtpKV: try f.mtpCache.allocate(promptTokens: 1), rec: [], ring: [],
            recSlab: rec, ringSlab: ring, trunkHidden: hidden)
        defer {
            f.cache.finish(destination.kv)
            if let handle = destination.mtpKV { f.mtpCache.finish(handle) }
        }
        memset(rec.region.pointer.advanced(by: rec.offset), 0xa5, rec.bytes)
        let before = bytes(rec)
        func adopt(_ value: QWEnginePrivatePrefix) throws {
            try QWEngineSessionState.adoptPrivatePrefix(geometry: QWTestGeometry.g27, value, into: destination, cache: f.cache,
                                                        mtpCache: f.mtpCache, attnLayers: [0])
        }
        XCTAssertThrowsError(try adopt(prefix), "short MTP must refuse before trunk publication")
        XCTAssertEqual(f.cache.written(destination.kv), 0)
        XCTAssertEqual(f.mtpCache.written(try XCTUnwrap(destination.mtpKV)), 0)
        XCTAssertEqual(bytes(rec), before)
        XCTAssertFalse(destination.stateFailed, "pre-mutation refusal does not poison")
        try f.mtpCache.grow(try XCTUnwrap(destination.mtpKV), to: 258)
        let foreign = try fixture()
        let foreignPrefix = try QWEngineSessionState.capturePrivatePrefix(foreign.session,
            cache: foreign.cache, mtpCache: foreign.mtpCache, maxBytes: 8 << 20)
        XCTAssertThrowsError(try adopt(foreignPrefix))
        XCTAssertEqual(bytes(rec), before)
        XCTAssertEqual(f.cache.written(destination.kv), 0)
        destination.verificationPending = true
        XCTAssertThrowsError(try adopt(prefix))
        destination.verificationPending = false
        try adopt(prefix)
        XCTAssertEqual(bytes(rec), prefix.recData)
        XCTAssertEqual(bytes(ring), prefix.ringData)
        XCTAssertEqual(bytes(hidden), prefix.trunkHiddenData)
        XCTAssertEqual(destination.roundRecord, donor.roundRecord)
        XCTAssertEqual(f.cache.written(destination.kv), 258)
        XCTAssertEqual(f.mtpCache.written(try XCTUnwrap(destination.mtpKV)), 257)
        XCTAssertThrowsError(try adopt(prefix))
        XCTAssertThrowsError(try QWEngineSessionState.capturePrivatePrefix(donor, cache: f.cache,
            mtpCache: f.mtpCache, maxBytes: 8 << 20))
    }

    func testProductionRestoreAfterGPUDestructionAndPageBoundary() throws {
        let f = try fixture(), s = f.session
        let snap = try QWEngineSessionState.snapshot(s, cache: f.cache, mtpCache: f.mtpCache)
        let cb = try XCTUnwrap(f.hal.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
        for tensor in [s.recSlab, s.ringSlab, try XCTUnwrap(s.trunkHidden)] {
            let start = tensor.region.base + tensor.offset
            blit.fill(buffer: tensor.region.buffer,
                      range: start..<(start + tensor.bytes),
                      value: 0xDE)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
        s.position = 258
        s.generated = 53
        s.trunkHiddenValid = false
        s.roundRecord = QWRoundRecord(pendingToken: 9999)
        f.cache.markWritten(s.kv, upTo: 258)
        f.mtpCache.markWritten(s.mtpKV!, upTo: 257)
        let freeBefore = f.cache.freePageCount
        XCTAssertNotEqual(bytes(s.recSlab), snap.recData)
        XCTAssertNotEqual(bytes(s.ringSlab), snap.ringData)

        try restore(snap, into: f)
        XCTAssertEqual(bytes(s.recSlab), snap.recData)
        XCTAssertEqual(bytes(s.ringSlab), snap.ringData)
        XCTAssertEqual(bytes(try XCTUnwrap(s.trunkHidden)), snap.trunkHiddenData)
        XCTAssertTrue(s.trunkHiddenValid)
        XCTAssertEqual(s.recStates.count, 48)
        XCTAssertEqual(s.ringStates.count, 48)
        XCTAssertEqual(s.position, snap.position)
        XCTAssertEqual(s.generated, snap.generated)
        XCTAssertEqual(s.roundRecord, snap.roundRecord)
        XCTAssertEqual(f.cache.written(s.kv), 255)
        XCTAssertEqual(f.mtpCache.written(s.mtpKV!), 254)
        XCTAssertEqual(f.cache.freePageCount, freeBefore + 2)
        XCTAssertEqual(s.attnKVViews[0].k.table.shape.elementCount, 1)
        XCTAssertEqual(s.mtpKVViews[0].k.table.shape.elementCount, 1)
    }

    func testForeignSnapshotFailsBeforeMutation() throws {
        let source = try fixture(), destination = try fixture()
        let snap = try QWEngineSessionState.snapshot(source.session, cache: source.cache, mtpCache: source.mtpCache)
        let before = bytes(destination.session.recSlab)
        XCTAssertThrowsError(try restore(snap, into: destination))
        XCTAssertEqual(bytes(destination.session.recSlab), before)
        XCTAssertEqual(destination.cache.written(destination.session.kv), 255)
    }

    func testRewoundKVAndFinishedSessionRejectSnapshot() throws {
        let f = try fixture()
        let snap = try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache)
        try f.mtpCache.rollback(f.session.mtpKV!, by: 1)
        let before = bytes(f.session.recSlab)
        XCTAssertThrowsError(try restore(snap, into: f))
        XCTAssertEqual(bytes(f.session.recSlab), before)
        XCTAssertEqual(f.cache.written(f.session.kv), 255)
        f.session.sessionFinished = true
        XCTAssertThrowsError(try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache))
        XCTAssertThrowsError(try restore(snap, into: f))
    }

    func testShortSnapshotAndWrongCacheRejectBeforeMutation() throws {
        let f = try fixture()
        let snap = try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache)
        let short = QWEngineSessionSnapshot(source: f.session, kvGeneration: snap.kvGeneration,
                                            mtpKVGeneration: snap.mtpKVGeneration, position: snap.position,
                                            generated: snap.generated, roundRecord: snap.roundRecord,
                                            kvWritten: snap.kvWritten, mtpKVWritten: snap.mtpKVWritten,
                                            recData: Data([1]), ringData: snap.ringData,
                                            trunkHiddenData: snap.trunkHiddenData, trunkHiddenValid: snap.trunkHiddenValid,
                                            stochastic: snap.stochastic)
        let before = bytes(f.session.recSlab)
        XCTAssertThrowsError(try restore(short, into: f))
        XCTAssertThrowsError(try QWEngineSessionState.snapshot(f.session, cache: f.mtpCache, mtpCache: f.cache))
        XCTAssertEqual(bytes(f.session.recSlab), before)
        XCTAssertEqual(f.cache.written(f.session.kv), 255)
    }

    func testRewoundThenRegrownKVRejectsStaleSnapshot() throws {
        let f = try fixture()
        let snap = try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache)
        try f.cache.rollback(f.session.kv, by: 1)
        f.cache.markWritten(f.session.kv, upTo: 255)
        let before = bytes(f.session.recSlab)
        XCTAssertThrowsError(try restore(snap, into: f))
        XCTAssertEqual(bytes(f.session.recSlab), before)
    }

    func testFailedGPUStateCannotBeSnapshottedOrRestored() throws {
        let f = try fixture()
        let snap = try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache)
        f.session.stateFailed = true
        let before = bytes(f.session.recSlab)
        XCTAssertThrowsError(try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache))
        XCTAssertThrowsError(try restore(snap, into: f))
        XCTAssertEqual(bytes(f.session.recSlab), before)
    }

    func testUnresolvedVerificationRefusesDiagnosticStateAccess() throws {
        let f = try fixture()
        let snap = try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache)
        f.session.verificationPending = true
        XCTAssertThrowsError(try QWEngineSessionState.snapshot(f.session, cache: f.cache, mtpCache: f.mtpCache))
        XCTAssertThrowsError(try restore(snap, into: f))
        f.session.verificationPending = false
        XCTAssertNoThrow(try restore(snap, into: f))
    }

    func testFinishedHandleRemainsInvalidAfterIDRecycling() throws {
        let f = try fixture(), old = f.session.kv
        f.cache.finish(old)
        let replacement = try f.cache.allocate(promptTokens: 256)
        XCTAssertEqual(old.seq, replacement.seq)
        XCTAssertThrowsError(try f.cache.validate(old))
        try f.cache.validate(replacement)
        f.cache.finish(old)
        try f.cache.validate(replacement)
    }
}

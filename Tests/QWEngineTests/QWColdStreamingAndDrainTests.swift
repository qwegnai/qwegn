// QWColdStreamingAndDrainTests.swift — Tests for transactional cold expert streaming,
// span/verify isolation, descriptor layouts, ColdEventPipeline abort safety, and drains.

import XCTest
import Foundation
import Metal
@testable import QWEngine
@testable import QWCore
@testable import QWQuant
@testable import QWLoad
@testable import QWAlloc
@testable import QWHAL
@testable import QWGEMM

final class QWColdStreamingAndDrainTests: XCTestCase {
    private struct SyntheticTensor {
        let shape: [Int]
        let payload: Data
        let experts: Int?
        let eHq4: Int?
    }

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-cold-drain-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private func writeQWP1(at url: URL, tensors: [String: SyntheticTensor]) throws {
        var fileBytes = Data("QWP1".utf8)
        fileBytes.append(Data(repeating: 0, count: 60))
        var jsonEntries: [String: [String: Any]] = [:]
        for name in tensors.keys.sorted() {
            guard let t = tensors[name] else { continue }
            let pad = (64 - (fileBytes.count % 64)) % 64
            if pad > 0 { fileBytes.append(Data(repeating: 0, count: pad)) }
            let off = fileBytes.count
            fileBytes.append(t.payload)
            var entry: [String: Any] = ["shape": t.shape, "hq4_off": off]
            if let experts = t.experts { entry["experts"] = experts }
            if let eHq4 = t.eHq4 { entry["e_hq4"] = eHq4 }
            jsonEntries[name] = entry
        }
        let hdrData = try JSONSerialization.data(
            withJSONObject: ["version": 2, "provenance": [:], "tensors": jsonEntries],
            options: [.sortedKeys]
        )
        var hdrLenLE = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLenLE) { (raw: UnsafeRawBufferPointer) in
            fileBytes.replaceSubrange(4..<12, with: raw)
        }
        fileBytes.append(hdrData)
        try fileBytes.write(to: url)
    }

    private func makeBufferedStreamer(
        packURL: URL,
        residentExperts: Int,
        totalExperts: Int,
        stagingSlots: Int,
        intermediate: Int = 32,
        hidden: Int = 64,
        directIO: Bool = false
    ) throws -> (
        streamer: QWEngineDriver.ColdExpertStreamer,
        guBuf: UnsafeMutableRawPointer,
        downBuf: UnsafeMutableRawPointer
    ) {
        let pack = try QWQuant.Pack(
            url: packURL, descriptor: .qwen38_27b, skipIdentityCheck: true)
        var policy = LoadPolicy(descriptor: .qwen38_27b, env: directIO ? [:] : ["QW_PLANE_BUFFERED": "1"])
        policy.directIO = directIO
        let io = PlaneIO(pack: pack, policy: policy)
        let guRowBytes = (hidden / 32) * 20
        let downRowBytes = (intermediate / 32) * 20
        let guBytes = max(1, stagingSlots) * 2 * intermediate * guRowBytes
        let downBytes = max(1, stagingSlots) * hidden * downRowBytes
        let guBuf = UnsafeMutableRawPointer.allocate(byteCount: guBytes, alignment: 64)
        let downBuf = UnsafeMutableRawPointer.allocate(byteCount: downBytes, alignment: 64)
        guBuf.initializeMemory(as: UInt8.self, repeating: 0, count: guBytes)
        downBuf.initializeMemory(as: UInt8.self, repeating: 0, count: downBytes)
        let streamer = QWEngineDriver.ColdExpertStreamer(
            io: io,
            layerPrefix: "model.layers.0.",
            residentExperts: residentExperts,
            totalExperts: totalExperts,
            stagingSlots: stagingSlots,
            intermediate: intermediate,
            hidden: hidden,
            guRowBytes: guRowBytes,
            downRowBytes: downRowBytes,
            guBasePtr: guBuf,
            downBasePtr: downBuf
        )
        return (streamer, guBuf, downBuf)
    }

    // MARK: - CPU ColdExpertStreamer tests

    func testColdStreamerResolvesMlpAndBlockSparseAndThrowsOnMissing() throws {
        let dir = try makeTempDir()
        let inter = 32, hidden = 64
        let guSingle = inter * (hidden / 32) * 20
        let downBytes = hidden * (inter / 32) * 20

        var mlpTensors: [String: SyntheticTensor] = [:]
        for e in 0..<3 {
            mlpTensors["model.layers.0.mlp.experts.\(e).gate_proj.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x10 + e), count: guSingle),
                experts: nil, eHq4: nil)
            mlpTensors["model.layers.0.mlp.experts.\(e).up_proj.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x30 + e), count: guSingle),
                experts: nil, eHq4: nil)
            mlpTensors["model.layers.0.mlp.experts.\(e).down_proj.weight"] = SyntheticTensor(
                shape: [hidden, inter], payload: Data(repeating: UInt8(0x50 + e), count: downBytes),
                experts: nil, eHq4: nil)
        }
        let mlpURL = dir.appendingPathComponent("mlp.qwp")
        try writeQWP1(at: mlpURL, tensors: mlpTensors)
        let (mlpStreamer, mlpGu, mlpDn) = try makeBufferedStreamer(
            packURL: mlpURL, residentExperts: 1, totalExperts: 3, stagingSlots: 2)
        defer { mlpGu.deallocate(); mlpDn.deallocate() }
        if case .mlpPerExpert = mlpStreamer.expertLayout {
            XCTAssertTrue(true)
        } else {
            XCTFail("expected .mlpPerExpert layout")
        }
        var ids: [UInt32] = [0, 2, 1]
        let prepMlp = try ids.withUnsafeMutableBufferPointer { buf in
            try mlpStreamer.prepareSpan(idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 3)
        }
        XCTAssertEqual(prepMlp.stagedCount, 2)
        XCTAssertEqual(ids, [0, 1, 2])
        try mlpStreamer.executeBatch(guBatch: prepMlp.guBatch, downBatch: prepMlp.downBatch)
        let guBytes = UnsafeBufferPointer(start: mlpGu.assumingMemoryBound(to: UInt8.self), count: 2 * 2 * guSingle)
        let dnBytes = UnsafeBufferPointer(start: mlpDn.assumingMemoryBound(to: UInt8.self), count: 2 * downBytes)
        XCTAssertEqual(guBytes[0], 0x12)
        XCTAssertEqual(guBytes[guSingle], 0x32)
        XCTAssertEqual(guBytes[2 * guSingle], 0x11)
        XCTAssertEqual(dnBytes[0], 0x52)
        XCTAssertEqual(dnBytes[downBytes], 0x51)

        var bsTensors: [String: SyntheticTensor] = [:]
        for e in 0..<3 {
            bsTensors["model.layers.0.block_sparse_moe.experts.\(e).w1.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x20 + e), count: guSingle),
                experts: nil, eHq4: nil)
            bsTensors["model.layers.0.block_sparse_moe.experts.\(e).w3.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x40 + e), count: guSingle),
                experts: nil, eHq4: nil)
            bsTensors["model.layers.0.block_sparse_moe.experts.\(e).w2.weight"] = SyntheticTensor(
                shape: [hidden, inter], payload: Data(repeating: UInt8(0x60 + e), count: downBytes),
                experts: nil, eHq4: nil)
        }
        let bsURL = dir.appendingPathComponent("block_sparse.qwp")
        try writeQWP1(at: bsURL, tensors: bsTensors)
        let (bsStreamer, bsGu, bsDn) = try makeBufferedStreamer(
            packURL: bsURL, residentExperts: 1, totalExperts: 3, stagingSlots: 2)
        defer { bsGu.deallocate(); bsDn.deallocate() }
        if case .blockSparsePerExpert = bsStreamer.expertLayout {
            XCTAssertTrue(true)
        } else {
            XCTFail("expected .blockSparsePerExpert layout")
        }
        var bsIds: [UInt32] = [2]
        let prepBs = try bsIds.withUnsafeMutableBufferPointer { buf in
            try bsStreamer.prepareSpan(idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 1)
        }
        try bsStreamer.executeBatch(guBatch: prepBs.guBatch, downBatch: prepBs.downBatch)
        XCTAssertEqual(bsGu.load(as: UInt8.self), 0x22)
        XCTAssertEqual(bsGu.load(fromByteOffset: guSingle, as: UInt8.self), 0x42)
        XCTAssertEqual(bsDn.load(as: UInt8.self), 0x62)

        bsTensors.removeValue(forKey: "model.layers.0.block_sparse_moe.experts.2.w2.weight")
        let incompleteURL = dir.appendingPathComponent("incomplete.qwp")
        try writeQWP1(at: incompleteURL, tensors: bsTensors)
        let (badStreamer, badGu, badDn) = try makeBufferedStreamer(
            packURL: incompleteURL, residentExperts: 1, totalExperts: 3, stagingSlots: 2)
        defer { badGu.deallocate(); badDn.deallocate() }
        var badIds: [UInt32] = [2]
        XCTAssertThrowsError(try badIds.withUnsafeMutableBufferPointer { buf in
            try badStreamer.prepareSpan(idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 1)
        })
    }

    func testColdStreamerFused3DSlicingLoadsExactExpertBytes() throws {
        let dir = try makeTempDir()
        let inter = 32, hidden = 64, totalE = 3
        let guExpertBytes = 2 * inter * (hidden / 32) * 20
        let dnExpertBytes = hidden * (inter / 32) * 20
        var guAll = Data()
        var dnAll = Data()
        for e in 0..<totalE {
            guAll.append(Data(repeating: UInt8(0xA0 + e), count: guExpertBytes))
            dnAll.append(Data(repeating: UInt8(0xB0 + e), count: dnExpertBytes))
        }
        let url = dir.appendingPathComponent("fused3d.qwp")
        try writeQWP1(at: url, tensors: [
            "model.layers.0.mlp.experts.gate_up_proj.weight": SyntheticTensor(
                shape: [totalE, 2 * inter, hidden], payload: guAll, experts: totalE, eHq4: guExpertBytes),
            "model.layers.0.mlp.experts.down_proj.weight": SyntheticTensor(
                shape: [totalE, hidden, inter], payload: dnAll, experts: totalE, eHq4: dnExpertBytes),
        ])
        let (streamer, guBuf, dnBuf) = try makeBufferedStreamer(
            packURL: url, residentExperts: 1, totalExperts: totalE, stagingSlots: 2, directIO: true)
        defer { guBuf.deallocate(); dnBuf.deallocate() }
        if case .fused3D(let guName, let downName) = streamer.expertLayout {
            XCTAssertEqual(guName, "model.layers.0.mlp.experts.gate_up_proj.weight")
            XCTAssertEqual(downName, "model.layers.0.mlp.experts.down_proj.weight")
        } else {
            XCTFail("expected .fused3D layout")
        }
        var ids: [UInt32] = [2, 1]
        let prep = try ids.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 2)
        }
        XCTAssertEqual(prep.stagedCount, 2)
        XCTAssertEqual(ids, [1, 2])
        final class GateUpProbe: @unchecked Sendable {
            private let lock = NSLock()
            let guBuf: UnsafeMutableRawPointer
            let guExpertBytes: Int
            private(set) var called = false
            private(set) var firstSlotByte: UInt8 = 0
            private(set) var secondSlotByte: UInt8 = 0

            init(guBuf: UnsafeMutableRawPointer, guExpertBytes: Int) {
                self.guBuf = guBuf
                self.guExpertBytes = guExpertBytes
            }

            func record() {
                let b0 = guBuf.load(as: UInt8.self)
                let b1 = guBuf.load(fromByteOffset: guExpertBytes, as: UInt8.self)
                lock.withLock {
                    called = true
                    firstSlotByte = b0
                    secondSlotByte = b1
                }
            }
        }
        let probe = GateUpProbe(guBuf: guBuf, guExpertBytes: guExpertBytes)
        try streamer.executeTwoPhaseBatch(
            guBatch: prep.guBatch, downBatch: prep.downBatch,
            onGateUpLoaded: { probe.record() }
        )
        XCTAssertTrue(probe.called)
        XCTAssertEqual(probe.firstSlotByte, 0xA2)
        XCTAssertEqual(probe.secondSlotByte, 0xA1)
        XCTAssertEqual(dnBuf.load(as: UInt8.self), 0xB2)
        XCTAssertEqual(dnBuf.load(fromByteOffset: dnExpertBytes, as: UInt8.self), 0xB1)
    }

    func testColdStreamerTransactionalCommitAndRollback() throws {
        let dir = try makeTempDir()
        let inter = 32, hidden = 64
        let guSingle = inter * (hidden / 32) * 20
        let downBytes = hidden * (inter / 32) * 20
        var tensors: [String: SyntheticTensor] = [:]
        for e in 0..<4 {
            tensors["model.layers.0.block_sparse_moe.experts.\(e).w1.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x10 + e), count: guSingle),
                experts: nil, eHq4: nil)
            tensors["model.layers.0.block_sparse_moe.experts.\(e).w3.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x30 + e), count: guSingle),
                experts: nil, eHq4: nil)
            tensors["model.layers.0.block_sparse_moe.experts.\(e).w2.weight"] = SyntheticTensor(
                shape: [hidden, inter], payload: Data(repeating: UInt8(0x50 + e), count: downBytes),
                experts: nil, eHq4: nil)
        }
        let url = dir.appendingPathComponent("dyn.qwp")
        try writeQWP1(at: url, tensors: tensors)
        let (streamer, guBuf, dnBuf) = try makeBufferedStreamer(
            packURL: url, residentExperts: 2, totalExperts: 4, stagingSlots: 2)
        defer { guBuf.deallocate(); dnBuf.deallocate() }
        streamer.enableDynamicResidency()
        let initialSlots = streamer.slotOfExpert
        let initialExperts = streamer.expertInSlot
        let initialClock = streamer.clock

        var ids: [UInt32] = [0, 2]
        let planned = try ids.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 2)
        }
        XCTAssertEqual(planned.stagedCount, 1)
        XCTAssertEqual(planned.promotions.count, 1)
        // Planning alone must not mutate LRU tables before IO completes and commits.
        XCTAssertEqual(streamer.slotOfExpert, initialSlots)
        XCTAssertEqual(streamer.expertInSlot, initialExperts)
        XCTAssertEqual(streamer.clock, initialClock)
        XCTAssertEqual(streamer.dynamicHits, 0)
        XCTAssertEqual(streamer.dynamicMisses, 0)
        XCTAssertTrue(streamer.pendingPromotions.isEmpty)

        // A failure during two-phase execution prevents commitPromotion so host LRU never desyncs.
        struct MockIOFailure: Error {}
        var failIds: [UInt32] = [3]
        let failPlan = try failIds.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(
                idxPtr: try XCTUnwrap(buf.baseAddress), startRow: 0, totalRows: 1, routes: 1)
        }
        XCTAssertThrowsError(try streamer.executeTwoPhaseBatch(
            guBatch: failPlan.guBatch, downBatch: failPlan.downBatch,
            onGateUpLoaded: { throw MockIOFailure() }
        ))
        XCTAssertEqual(streamer.slotOfExpert, initialSlots)
        XCTAssertEqual(streamer.expertInSlot, initialExperts)
        XCTAssertEqual(streamer.clock, initialClock)
        XCTAssertEqual(streamer.dynamicHits, 0)
        XCTAssertEqual(streamer.dynamicMisses, 0)
        XCTAssertTrue(streamer.pendingPromotions.isEmpty)

        try streamer.executeBatch(guBatch: planned.guBatch, downBatch: planned.downBatch)
        streamer.commitPromotion(planned.transaction)
        XCTAssertEqual(streamer.slotOfExpert, [0, -1, 1, -1])
        XCTAssertEqual(streamer.expertInSlot, [0, 2])
        XCTAssertEqual(streamer.dynamicHits, 1)
        XCTAssertEqual(streamer.dynamicMisses, 1)
        XCTAssertEqual(streamer.pendingPromotions.count, 1)
        XCTAssertEqual(streamer.pendingPromotions[0].staging, 0)
        XCTAssertEqual(streamer.pendingPromotions[0].bank, 1)
    }

    func testColdStreamerVerifyVsBatchIsolationAndRowZeroOverflow() throws {
        let dir = try makeTempDir()
        let inter = 32, hidden = 64
        let guSingle = inter * (hidden / 32) * 20
        let downBytes = hidden * (inter / 32) * 20
        var tensors: [String: SyntheticTensor] = [:]
        for e in 0..<6 {
            tensors["model.layers.0.mlp.experts.\(e).gate_proj.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x10 + e), count: guSingle),
                experts: nil, eHq4: nil)
            tensors["model.layers.0.mlp.experts.\(e).up_proj.weight"] = SyntheticTensor(
                shape: [inter, hidden], payload: Data(repeating: UInt8(0x20 + e), count: guSingle),
                experts: nil, eHq4: nil)
            tensors["model.layers.0.mlp.experts.\(e).down_proj.weight"] = SyntheticTensor(
                shape: [hidden, inter], payload: Data(repeating: UInt8(0x30 + e), count: downBytes),
                experts: nil, eHq4: nil)
        }
        let url = dir.appendingPathComponent("isolation.qwp")
        try writeQWP1(at: url, tensors: tensors)
        let (streamer, guBuf, dnBuf) = try makeBufferedStreamer(
            packURL: url, residentExperts: 2, totalExperts: 6, stagingSlots: 2)
        defer { guBuf.deallocate(); dnBuf.deallocate() }

        XCTAssertEqual(streamer.effectiveColdBudget(isVerifyWindow: false, specBudget: 1).apply, false)
        XCTAssertEqual(streamer.effectiveColdBudget(isVerifyWindow: false, specBudget: 1).slots, 2)
        XCTAssertEqual(streamer.effectiveColdBudget(isVerifyWindow: true, specBudget: 1).apply, true)
        XCTAssertEqual(streamer.effectiveColdBudget(isVerifyWindow: true, specBudget: 1).slots, 1)

        // Regular B=3 batched decode (rows in 2...4) must use full stagingSlots and split spans, never drop.
        var batchIds: [UInt32] = [2, 0, 3, 0, 4, 0]
        let span0 = try batchIds.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(
                idxPtr: buf.baseAddress!, startRow: 0, totalRows: 3, routes: 2,
                isVerifyWindow: false, specBudget: 1)
        }
        XCTAssertEqual(span0.nextRow, 2)
        XCTAssertEqual(span0.stagedCount, 2)
        XCTAssertEqual(batchIds[0..<4], [2, 0, 3, 0])
        let span1 = try batchIds.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(
                idxPtr: buf.baseAddress!, startRow: 2, totalRows: 3, routes: 2,
                isVerifyWindow: false, specBudget: 1)
        }
        XCTAssertEqual(span1.nextRow, 3)
        XCTAssertEqual(span1.stagedCount, 1)
        XCTAssertEqual(batchIds[4], 2)

        // Speculative verify window explicitly clamps cold budget across draft rows.
        var verifyIds: [UInt32] = [2, 0, 3, 0, 4, 0]
        let verifyPrep = try verifyIds.withUnsafeMutableBufferPointer { buf in
            try streamer.prepareSpan(
                idxPtr: buf.baseAddress!, startRow: 0, totalRows: 3, routes: 2,
                isVerifyWindow: true, specBudget: 1)
        }
        XCTAssertEqual(verifyPrep.nextRow, 3)
        XCTAssertEqual(verifyPrep.stagedCount, 1)
        XCTAssertEqual(verifyIds[0], 2)
        XCTAssertEqual(verifyIds[2], 0xFFFF_FFFF)
        XCTAssertEqual(verifyIds[4], 0xFFFF_FFFF)

        // Single-row demand exceeding stagingSlots must throw on both static and dynamic paths.
        var overflowIds: [UInt32] = [2, 3, 4, 0]
        XCTAssertThrowsError(try overflowIds.withUnsafeMutableBufferPointer { buf in
            _ = try streamer.prepareSpan(
                idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 4, isVerifyWindow: false)
        })
        streamer.enableDynamicResidency()
        overflowIds = [2, 3, 4, 0]
        XCTAssertThrowsError(try overflowIds.withUnsafeMutableBufferPointer { buf in
            _ = try streamer.prepareSpan(
                idxPtr: buf.baseAddress!, startRow: 0, totalRows: 1, routes: 4, isVerifyWindow: false)
        })
    }

    // MARK: - Device / Metal tests (executed serially by verifier)

    func testColdEventPipelineDeinitAndFailedJobAbort() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        for _ in 0..<8 {
            let p = try XCTUnwrap(ColdEventPipeline(device: device))
            p.shutdown()
        }

        let pipe = try XCTUnwrap(ColdEventPipeline(device: device))
        defer { pipe.shutdown() }
        let promoPtr = pipe.promoList.contents().assumingMemoryBound(to: UInt32.self)
        promoPtr[0] = 7

        struct SentinelError: Error {}
        var ids0: [UInt32] = [5, 6, 7, 8]
        var ids1: [UInt32] = [9, 10, 11, 12]
        final class FlagBox: @unchecked Sendable {
            private let lock = NSLock()
            private var flag = false
            var value: Bool { lock.withLock { flag } }
            func set() { lock.withLock { flag = true } }
        }
        let job1Ran = FlagBox()
        let v0 = pipe.reserve()
        let v1 = pipe.reserve()

        try ids0.withUnsafeMutableBufferPointer { buf0 in
            try ids1.withUnsafeMutableBufferPointer { buf1 in
                let ptr0 = try XCTUnwrap(buf0.baseAddress)
                let ptr1 = try XCTUnwrap(buf1.baseAddress)
                nonisolated(unsafe) let u0 = ptr0
                nonisolated(unsafe) let u1 = ptr1
                pipe.enqueue(
                    ready: v0.ready,
                    onAbort: {
                        promoPtr[0] = 0
                        for i in 0..<4 { u0[i] = 0xFFFF_FFFF }
                    },
                    { throw SentinelError() }
                )
                pipe.enqueue(
                    ready: v1.ready,
                    onAbort: {
                        promoPtr[0] = 0
                        for i in 0..<4 { u1[i] = 0xFFFF_FFFF }
                    },
                    {
                        job1Ran.set()
                        return false
                    }
                )
                pipe.event.signaledValue = v1.ready
                pipe.waitUntilIdle()
            }
        }
        XCTAssertTrue(pipe.hasFailure)
        XCTAssertFalse(job1Ran.value)
        XCTAssertEqual(promoPtr[0], 0)
        XCTAssertEqual(ids0, [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF])
        XCTAssertEqual(ids1, [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF])
        let drained = pipe.takeStepResult()
        XCTAssertNotNil(drained.failure)
        XCTAssertFalse(pipe.hasFailure)
    }

    func testEncodeKernelsResolveWeightPSOThrowsOnMissingFormat() throws {
        let device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "No Metal device available")
        let hal = try QWHal(geometry: .qwen38_27b)
        let nax = hal.caps.isMetal4TensorCapable
        let gemmName = nax ? "qwg_gemm_q4g_h16" : "qwg_gemm_q4g_sg"
        let swigluName = nax ? "qwg_gemm_q4g_h16_swiglu" : "qwg_gemm_q4g_sg_swiglu"
        let defaultPSO = try hal.pipeline(gemmName)
        let dummyBuf = try XCTUnwrap(hal.device.makeBuffer(length: 8192, options: .storageModeShared))
        let blocks = QWTensorValue(region: QWMRegion(buffer: dummyBuf, base: 0, length: 8192),
                                   offset: 0, shape: QWTensorShape([4096], .f16))
        let mismatchedAQ = try XCTUnwrap(QWWeightView(
            blocks: blocks, n: 64, k: 64, bits: 4, groupSize: 64,
            isRotated: false, isIncumbentContainer: false, isSHVQ: false))

        let resolvedAQ = try QWEngineDriver.resolveWeightPSO(
            for: mismatchedAQ, weightFormat: .hq4G32, hal: hal,
            defaultPSO: defaultPSO, opName: "prefill GEMM",
            incumbentKernel: gemmName,
            aqKernel: { QWGemmAQ.gemmKernelName(bits: mismatchedAQ.bits, group: mismatchedAQ.groupSize, useNAX: nax) }
        )
        XCTAssertTrue(resolvedAQ !== defaultPSO)

        XCTAssertThrowsError(try QWEngineDriver.resolveWeightPSO(
            for: mismatchedAQ, weightFormat: .hq4G32, hal: hal,
            defaultPSO: defaultPSO, opName: "prefill GEMM",
            incumbentKernel: gemmName,
            aqKernel: { nil }
        ))
        XCTAssertThrowsError(try QWEngineDriver.resolveWeightPSO(
            for: mismatchedAQ, weightFormat: .hq4G32, hal: hal,
            defaultPSO: defaultPSO, opName: "fused SwiGLU GEMM",
            incumbentKernel: swigluName,
            aqKernel: { "missing_swiglu_test_kernel" }
        ))
        XCTAssertThrowsError(try QWEngineDriver.resolveWeightPSO(
            for: mismatchedAQ, weightFormat: .hq4G32, hal: hal,
            defaultPSO: defaultPSO, opName: "MLP SiLU pair GEMV",
            incumbentKernel: "qwg_gemv_q4g_silu_pair_dx_h16",
            aqKernel: { "missing_silu_test_kernel" }
        ))
    }

    func testConcatPlaneAQPreservesExtraZeroRows() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "No Metal device available")
        let dir = try makeTempDir()
        let geom = try QWModelGeometry.stubTinyLlamaGeometry()
        let extraZero = 2
        let cols = geom.hidden
        let rowsA = geom.vocab / 2 - 2
        let rowsB = geom.vocab - rowsA - extraZero
        let dataRows = rowsA + rowsB
        let packCols = cols / 8
        let groups = cols / 32
        func tensorByteCounts(_ r: Int) -> (w: Int, s: Int, b: Int, total: Int) {
            let w = r * packCols * 4
            let s = r * groups * 2
            let b = r * groups * 2
            return (w, s, b, w + s + b)
        }
        let ca = tensorByteCounts(rowsA)
        let cb = tensorByteCounts(rowsB)
        let header: [String: Any] = [
            "a.weight": ["dtype": "U32", "shape": [rowsA, packCols], "data_offsets": [0, ca.w]],
            "a.scales": ["dtype": "F16", "shape": [rowsA, groups], "data_offsets": [ca.w, ca.w + ca.s]],
            "a.biases": ["dtype": "F16", "shape": [rowsA, groups], "data_offsets": [ca.w + ca.s, ca.total]],
            "b.weight": ["dtype": "U32", "shape": [rowsB, packCols], "data_offsets": [ca.total, ca.total + cb.w]],
            "b.scales": ["dtype": "F16", "shape": [rowsB, groups],
                         "data_offsets": [ca.total + cb.w, ca.total + cb.w + cb.s]],
            "b.biases": ["dtype": "F16", "shape": [rowsB, groups],
                         "data_offsets": [ca.total + cb.w + cb.s, ca.total + cb.total]],
        ]
        let hdrJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var stFile = withUnsafeBytes(of: UInt64(hdrJSON.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Data(raw)
        }
        stFile.append(hdrJSON)
        // Non-zero scales/weights so dataRows are distinct from extraZeroRows tail.
        stFile.append(Data(repeating: 0x3C, count: ca.total + cb.total))
        let stURL = dir.appendingPathComponent("model.safetensors")
        try stFile.write(to: stURL)
        let st = try QWSafeTensorsContainer(url: stURL)

        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 1 << 20, kv: 1 << 16, gdnStates: 1 << 16)
        let format = QWWeightFormat.nativeAffine(bits: 4, groupSize: 32)
        let plan = try QWWeightAllocationPlan(
            geometry: geom, ropeRows: 1, mtpDrafts: 0, headQ4g: true, format: format)
        var loader = QWEngineDriver.WeightLoader(
            st: st, hl: hal,
            psoCvt: try hal.pipeline("qwg_bf16_to_f16"),
            psoQuantBF16: try hal.pipeline("qwg_bf16_to_f16"),
            wsub: arena.weightsRegion(), plan: plan, geometry: geom, format: format)

        let concatenated = try loader.concatPlane(
            [("a.weight", rowsA), ("b.weight", rowsB)], k: cols,
            key: \QWQTensorEntry.hq4Off, extraZeroRows: extraZero)
        let rowBytes = format.rowBytes(k: cols)
        XCTAssertEqual(concatenated.bytes, geom.vocab * rowBytes)
        let basePtr = concatenated.region.buffer.contents()
            .advanced(by: concatenated.region.base + concatenated.offset)
        let dataPtr = UnsafeBufferPointer(
            start: basePtr.assumingMemoryBound(to: UInt8.self), count: dataRows * rowBytes)
        let tailPtr = UnsafeBufferPointer(
            start: basePtr.advanced(by: dataRows * rowBytes).assumingMemoryBound(to: UInt8.self),
            count: extraZero * rowBytes)
        XCTAssertFalse(dataPtr.allSatisfy { $0 == 0 })
        XCTAssertTrue(tailPtr.allSatisfy { $0 == 0 })
    }

    func testPlaneIOLoadSliceReadsExactRangeAndRejectsOutOfBounds() throws {
        let dir = try makeTempDir()
        let rowBytes = 2 * 20
        let totalE = 3, perExpert = 4 * rowBytes
        var payload = Data()
        for e in 0..<totalE {
            payload.append(Data(repeating: UInt8(0x30 + e), count: perExpert))
        }
        let packURL = dir.appendingPathComponent("slice.qwp")
        try writeQWP1(at: packURL, tensors: [
            "w": SyntheticTensor(shape: [totalE, 4, 64], payload: payload, experts: totalE, eHq4: perExpert)
        ])
        let pack = try QWQuant.Pack(url: packURL, descriptor: .qwen38_27b, skipIdentityCheck: true)
        let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: .qwen38_27b, env: ["QW_PLANE_BUFFERED": "1"]))

        var dst = [UInt8](repeating: 0, count: 2 * perExpert)
        try dst.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let ptr = try XCTUnwrap(raw.baseAddress)
            try io.loadSlice("w", layout: .hq4, byteOffset: perExpert, byteCount: perExpert, into: ptr)
        }
        XCTAssertTrue(dst[0..<perExpert].allSatisfy { $0 == 0x31 })
        XCTAssertTrue(dst[perExpert..<(2 * perExpert)].allSatisfy { $0 == 0 })

        XCTAssertThrowsError(try dst.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let ptr = try XCTUnwrap(raw.baseAddress)
            try io.loadSlice("w", layout: .hq4, byteOffset: totalE * perExpert, byteCount: 1, into: ptr)
        })
        XCTAssertThrowsError(try dst.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let ptr = try XCTUnwrap(raw.baseAddress)
            try io.loadSlice("w", layout: .hq4, byteOffset: -1, byteCount: 4, into: ptr)
        })
    }
}

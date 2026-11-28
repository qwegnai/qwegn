// QWQSACacheStateTests.swift — Layer-4 QSA cache state, partition isolation, and storage mode unit tests.
import XCTest
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWRope

private struct QWQSAMixerConfig: QWQSAMixerSpec, Sendable {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let indexHeads: Int
    let indexHeadDim: Int
    let rotaryDim: Int
    let compress: Int
    let blockTopK: Int
    let bankRows: Int
    let rope: QWRopeCPU
}

final class QWQSACacheStateTests: XCTestCase {
    private static func makeHAL() throws -> QWHal {
        try QWHal(geometry: .qwen38_27b)
    }

    private static func makeMixer(
        heads: Int = 16,
        kvHeads: Int = 4,
        headDim: Int = 256,
        indexHeads: Int = 4,
        indexHeadDim: Int = 128,
        rotaryDim: Int = 64,
        compress: Int = 4,
        blockTopK: Int = 512
    ) -> QWQSAMixerConfig {
        let bankRows = heads * 2 * headDim + 2 * kvHeads * headDim + indexHeads * indexHeadDim + indexHeadDim
        return QWQSAMixerConfig(
            heads: heads,
            kvHeads: kvHeads,
            headDim: headDim,
            indexHeads: indexHeads,
            indexHeadDim: indexHeadDim,
            rotaryDim: rotaryDim,
            compress: compress,
            blockTopK: blockTopK,
            bankRows: bankRows,
            rope: QWRopeCPU(rotaryDim: rotaryDim, theta: 10000.0)
        )
    }

    func testKVStorageModeResolveAndRejectInvalid() throws {
        XCTAssertEqual(try FNKVStorageMode.resolve(nil), .f16)
        XCTAssertEqual(try FNKVStorageMode.resolve(""), .f16)
        XCTAssertEqual(try FNKVStorageMode.resolve("auto"), .f16)
        XCTAssertEqual(try FNKVStorageMode.resolve("Q4"), .q4)
        XCTAssertEqual(try FNKVStorageMode.resolve("q8"), .q8)
        XCTAssertEqual(try FNKVStorageMode.resolve("f32"), .f32)
        XCTAssertEqual(QWQSAKVStorageMode.f16.perTokenBytes, 31_296)
        XCTAssertEqual(FNKVStorageMode.q4.prepKernelName, "qwg_attn_prep_q4")
        XCTAssertEqual(FNKVStorageMode.q8.sparseGatedNAXKernelName, "qwg_attn_sparse_gated_nax_q8")
        XCTAssertThrowsError(try FNKVStorageMode.resolve("int2")) { error in
            XCTAssertEqual(error as? QWQSAError, .kvModeNotImplemented("int2"))
        }
    }

    func testCachePartitionIsolationAndSharedVerifyState() throws {
        let hal = try Self.makeHAL()
        let cap = 1024
        let mixer = Self.makeMixer()
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 64 << 20, kv: 128 << 20, gdnStates: 0)
        let state: FNGPUAttnDecodeState = try QWQSACacheState(
            arena: arena, mixer: mixer, hidden: 2560, capacity: cap, kvMode: .q8
        )
        XCTAssertEqual(state.fullCapacity, cap)
        XCTAssertEqual(state.capacity, cap)

        var kOffsets: [Int] = []
        var idxOffsets: [Int] = []
        var poolOffsets: [Int] = []
        for slot in 0..<4 {
            state.selectPartition(slot: slot, totalSlots: 4, mixer: mixer)
            XCTAssertEqual(state.capacity, cap / 4)
            kOffsets.append(state.k.offset)
            idxOffsets.append(state.indexK.offset)
            poolOffsets.append(state.pooled.offset)
        }
        for slot in 1..<4 {
            XCTAssertGreaterThan(kOffsets[slot], kOffsets[slot - 1])
            XCTAssertGreaterThan(idxOffsets[slot], idxOffsets[slot - 1])
            XCTAssertGreaterThan(poolOffsets[slot], poolOffsets[slot - 1])
        }
        state.selectPartition(slot: 0, totalSlots: 1, mixer: mixer)
        XCTAssertEqual(state.capacity, cap)
        XCTAssertEqual(state.k.offset, kOffsets[0])

        let verifyState = try QWQSACacheState(sharingCachesOf: state, arena: arena, mixer: mixer, rows: 4)
        XCTAssertEqual(verifyState.rows, 4)
        XCTAssertEqual(verifyState.k.offset, state.k.offset)
        XCTAssertEqual(verifyState.pooled.offset, state.pooled.offset)

        let views = state.kvViews(pos: 8, count: 4, kvHeads: mixer.kvHeads, headDim: mixer.headDim)
        XCTAssertEqual(views.k.shape.elementCount, 4 * mixer.kvHeads * (mixer.headDim / 2))
        XCTAssertEqual(views.kMeta.shape.elementCount, 4 * mixer.kvHeads)
    }

    func testAllocateLSAClustersAndSelectionLimitError() throws {
        let hal = try Self.makeHAL()
        let mixer = Self.makeMixer()
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 64 << 20, kv: 32 << 20, gdnStates: 0)
        let state = try QWQSACacheState(arena: arena, mixer: mixer, hidden: 2560, capacity: 1024, kvMode: .q4)
        let emReg = try QWQSACacheState.allocateLSAClusters(
            hal: hal, kvCap: 8192, indexHeadDim: mixer.indexHeadDim, states: [(state, nil)]
        )
        XCTAssertGreaterThan(emReg.length, 0)
        XCTAssertEqual(state.lsaClusters?.isEmeans, true)

        let tinyScores = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        XCTAssertThrowsError(try QWQSAEncoders.encodeQSASelect(
            enc, hal: hal, idxQ: state.idxQ, pooled: state.pooled, list: state.list,
            rows: 1, blockTopK: 512, pos0: 16, scores: tinyScores
        )) { error in
            XCTAssertEqual(error as? QWQSAError, .selectionLimitExceeded(positions: 17, limit: 4))
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
    }
}

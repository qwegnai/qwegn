// QWResidencyPoolTests.swift — Multi-slot UMA admission, lease replacement, and ANE accounting tests (G5).

import Foundation
import QWCore
import QWEngine
import XCTest

private struct MockMemorySnapshotSource: QWDeviceMemorySnapshotSource {
    let snapshot: QWDeviceMemorySnapshot
    func capture() -> QWDeviceMemorySnapshot { snapshot }
}

final class QWResidencyPoolTests: XCTestCase {
    private let giB = 1 << 30

    func testMultiSlotAdmissionAndLeaseReplacement() async {
        let snap = QWDeviceMemorySnapshot(
            physicalBytes: 128 * giB,
            reclaimableBytes: 110 * giB,
            maxWorkingSetBytes: 116 * giB
        )
        let pool = QWResidencyPool(memorySnapshotSource: MockMemorySnapshotSource(snapshot: snap), osReserveGiB: 12.0)

        let primary = await pool.admit(
            slot: .primaryLLM,
            modelName: "qwen3-35b",
            packPath: "/models/qwen3-35b",
            estimatedBytes: 40 * giB,
            computeUnit: "metal",
            requestedWarmCacheBytes: 8 * giB,
            requestedSessions: 8
        )
        XCTAssertTrue(primary.admitted)
        XCTAssertEqual(primary.warmCache.admittedWarmBytes, 8 * giB)

        await pool.commitLease(QWResidencyLease(
            slot: .primaryLLM, modelName: "qwen3-35b", packPath: "/models/qwen3-35b",
            estimatedBytes: UInt64(40 * giB), computeUnit: "metal"
        ))

        let sidecar = await pool.admit(
            slot: .decisionSidecar,
            modelName: "d1-2b",
            packPath: "/models/d1-2b",
            estimatedBytes: 4 * giB,
            computeUnit: "metal",
            requestedWarmCacheBytes: 8 * giB,
            requestedSessions: 8
        )
        XCTAssertTrue(sidecar.admitted)
        await pool.commitLease(QWResidencyLease(
            slot: .decisionSidecar, modelName: "d1-2b", packPath: "/models/d1-2b",
            estimatedBytes: UInt64(4 * giB), computeUnit: "metal"
        ))

        let embed = await pool.admit(
            slot: .embeddingSidecar,
            modelName: "bge-m3",
            packPath: "/models/bge-m3",
            estimatedBytes: 2 * giB,
            computeUnit: "metal"
        )
        XCTAssertTrue(embed.admitted)
        await pool.commitLease(QWResidencyLease(
            slot: .embeddingSidecar, modelName: "bge-m3", packPath: "/models/bge-m3",
            estimatedBytes: UInt64(2 * giB), computeUnit: "metal"
        ))

        // Reloading same slot replaces rather than double-counting.
        let reloaded = await pool.admit(
            slot: .primaryLLM,
            modelName: "qwen3-80b",
            packPath: "/models/qwen3-80b",
            estimatedBytes: 80 * giB,
            computeUnit: "metal"
        )
        XCTAssertTrue(reloaded.admitted)
        XCTAssertEqual(reloaded.committedMetalBytes, UInt64(86 * giB))
        XCTAssertEqual(pool.currentLeases().count, 3)
    }

    func testUMARefusalWarmClampAndANEBypassAccounting() async {
        let snap = QWDeviceMemorySnapshot(
            physicalBytes: 64 * giB,
            reclaimableBytes: 40 * giB,
            maxWorkingSetBytes: 56 * giB
        )
        let pool = QWResidencyPool(memorySnapshotSource: MockMemorySnapshotSource(snapshot: snap), osReserveGiB: 8.0)

        await pool.commitLease(QWResidencyLease(
            slot: .primaryLLM, modelName: "model-a", packPath: "/a",
            estimatedBytes: UInt64(32 * giB), computeUnit: "metal"
        ))

        // Tight headroom clamps requested 16 GiB warm cache cleanly.
        let tight = await pool.admit(
            slot: .decisionSidecar,
            modelName: "d1-2b",
            packPath: "/d1",
            estimatedBytes: 2 * giB,
            computeUnit: "metal",
            requestedWarmCacheBytes: 16 * giB,
            requestedSessions: 8
        )
        XCTAssertTrue(tight.admitted)
        XCTAssertTrue(tight.warmCache.clamped)
        XCTAssertEqual(tight.warmCache.admittedWarmBytes, 0)

        // Exceeding 56 GiB available cap refuses cleanly before allocation.
        let overflow = await pool.admit(
            slot: .embeddingSidecar,
            modelName: "huge-embed",
            packPath: "/huge",
            estimatedBytes: 30 * giB,
            computeUnit: "metal"
        )
        XCTAssertFalse(overflow.admitted)
        XCTAssertNotNil(overflow.reason)

        // ANE lease bypasses GPU arbitrator accounting while tracking physical RAM.
        await pool.commitLease(QWResidencyLease(
            slot: .decisionSidecar, modelName: "d1-ane", packPath: "/d1.aimodel",
            estimatedBytes: UInt64(2 * giB), computeUnit: "ane", bypassesGPUArbitrator: true
        ))
        let summary = await pool.snapshot()
        XCTAssertEqual(summary.totalEstimatedMetalBytes, UInt64(32 * giB))
        XCTAssertEqual(summary.totalEstimatedANEBytes, UInt64(2 * giB))

        // Release frees capacity.
        let released = await pool.releaseLease(for: .primaryLLM)
        XCTAssertEqual(released?.modelName, "model-a")
        let afterRelease = await pool.admit(
            slot: .embeddingSidecar, modelName: "huge-embed", packPath: "/huge",
            estimatedBytes: 30 * giB, computeUnit: "metal"
        )
        XCTAssertTrue(afterRelease.admitted)
    }
}

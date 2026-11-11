// QWResidencyAdmissionTests.swift — Deterministic unit tests for unified 128 GB UMA residency admission.

import XCTest
@testable import QWCore

final class QWResidencyAdmissionTests: XCTestCase {
    private let GiB = 1 << 30
    private let MiB = 1 << 20

    func testFlashNextDeduplicatesPackDiskSizeAndGPUPLE() {
        // Reproduces F5: QWP pack tensor-table under-report (~3.5 GiB) vs real 67.71 GiB .qwp + 27.13 GiB VQ2 PLE.
        let synthetic = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 128 * GiB,
            reclaimableBytes: 108 * GiB,
            maxWorkingSetBytes: 118 * GiB
        )
        let qwpDiskBytes = 68 * GiB
        let pleBytes = 27 * GiB
        let scratchBytes = 2244 * MiB
        let admission = QWResidencyAdmission.planFlashNext(
            weightsEstimate: 3500 * MiB,
            packDiskBytes: qwpDiskBytes,
            gpuPLEBytes: pleBytes,
            ngramRowCacheBytes: 0,
            baseScratchBytes: scratchBytes,
            retainedPrefillBytes: 512 * MiB,
            lsaClusterBytes: 0,
            minKVBytes: 512 * MiB,
            requestedWarmCacheBytes: 8 * GiB,
            requestedSessions: 8,
            snapshot: synthetic.capture()
        )
        XCTAssertEqual(admission.weightsBytes, qwpDiskBytes)
        XCTAssertEqual(admission.pleResidentBytes, pleBytes)
        XCTAssertEqual(admission.fixedScratchBytes, (2244 + 512) * MiB)
        XCTAssertGreaterThan(admission.totalFixedBytes, 97 * GiB)
        // Post-wire reclaimable = 108 - ~98.2 = ~9.8 GiB; minus 6 GiB floor leaves ~3.8 GiB warm cache (< 8 GiB).
        XCTAssertTrue(admission.warmCache.clamped)
        XCTAssertLessThan(admission.warmCache.admittedWarmBytes, 5 * GiB)
        XCTAssertGreaterThan(admission.warmCache.admittedWarmBytes, 2 * GiB)
    }

    func testGLM53FlashPostWireClampsWarmCacheToZero() {
        // Reproduces P0.1: GLM-5.3-Flash SHVQ3/4 wires ~104 GiB on 128 GiB UMA, leaving ~4 GiB reclaimable.
        let postWireSnapshot = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 128 * GiB,
            reclaimableBytes: 4 * GiB,
            maxWorkingSetBytes: 118 * GiB
        )
        let res = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: 8 * GiB,
            requestedSessions: 8,
            wiredEngineBytes: 104 * GiB,
            measuredAfterWire: true,
            snapshot: postWireSnapshot.capture()
        )
        XCTAssertEqual(res.admittedWarmBytes, 0)
        XCTAssertEqual(res.maxConcurrentSessions, 1)
        XCTAssertTrue(res.clamped)
    }

    func testMediumModelPreservesRequestedWarmCache() {
        let postWireSnapshot = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 128 * GiB,
            reclaimableBytes: 86 * GiB,
            maxWorkingSetBytes: 118 * GiB
        )
        let res = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: 8 * GiB,
            requestedSessions: 8,
            wiredEngineBytes: 18 * GiB,
            measuredAfterWire: true,
            snapshot: postWireSnapshot.capture()
        )
        XCTAssertEqual(res.admittedWarmBytes, 8 * GiB)
        XCTAssertEqual(res.maxConcurrentSessions, 8)
        XCTAssertFalse(res.clamped)
    }

    func testSmallUnitFixtureUnderOneGiBIsNeverClamped() {
        let contendedSnapshot = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 16 * GiB,
            reclaimableBytes: 512 * MiB
        )
        let res = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: 256 * MiB,
            requestedSessions: 4,
            wiredEngineBytes: 128 * MiB,
            measuredAfterWire: true,
            snapshot: contendedSnapshot.capture()
        )
        XCTAssertEqual(res.admittedWarmBytes, 256 * MiB)
        XCTAssertEqual(res.maxConcurrentSessions, 4)
        XCTAssertFalse(res.clamped)
    }

    func testPLEAdmissibilityHonorsPhysicalAndReclaimableCaps() {
        let fit128 = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 128 * GiB,
            reclaimableBytes: 108 * GiB,
            maxWorkingSetBytes: 118 * GiB
        )
        XCTAssertTrue(QWResidencyAdmission.isPLEAdmissible(
            weightsEstimate: 4 * GiB,
            packDiskBytes: 68 * GiB,
            fixedScratchBytes: 2 * GiB,
            pleFileBytes: 27 * GiB,
            minKVBytes: 1 * GiB,
            snapshot: fit128.capture()
        ))

        let contended128 = QWSyntheticDeviceMemorySnapshot(
            physicalBytes: 128 * GiB,
            reclaimableBytes: 72 * GiB,
            maxWorkingSetBytes: 118 * GiB
        )
        XCTAssertFalse(QWResidencyAdmission.isPLEAdmissible(
            weightsEstimate: 4 * GiB,
            packDiskBytes: 68 * GiB,
            fixedScratchBytes: 2 * GiB,
            pleFileBytes: 27 * GiB,
            minKVBytes: 1 * GiB,
            snapshot: contended128.capture()
        ))
    }
}

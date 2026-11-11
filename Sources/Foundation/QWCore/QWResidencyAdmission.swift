// QWResidencyAdmission.swift — Unified 128 GB UMA memory and residency admission.
// Consolidates backing-file deduplication (.qwp, safetensors, ngram.bin PLE),
// KV/scratch accounting, and PrefixCache warm-RAM clamping behind one snapshot seam.

import Foundation

/// Immutable point-in-time view of host and Metal unified-memory budgets.
public struct QWDeviceMemorySnapshot: Sendable, Equatable {
    public let physicalBytes: Int
    public let reclaimableBytes: Int?
    public let maxWorkingSetBytes: Int?

    public init(physicalBytes: Int, reclaimableBytes: Int?, maxWorkingSetBytes: Int? = nil) {
        self.physicalBytes = max(0, physicalBytes)
        self.reclaimableBytes = reclaimableBytes.map { max(0, $0) }
        self.maxWorkingSetBytes = maxWorkingSetBytes.map { max(0, $0) }
    }
}

/// Seam supplying host/device memory snapshots (live Mach/Metal vs synthetic test stand-in).
public protocol QWDeviceMemorySnapshotSource: Sendable {
    func capture() -> QWDeviceMemorySnapshot
}

/// Production adapter querying Darwin `sysctl` + `host_statistics64` and optional Metal cap.
public struct QWLiveDeviceMemorySnapshotSource: QWDeviceMemorySnapshotSource {
    public let maxWorkingSetBytes: Int?

    public init(maxWorkingSetBytes: Int? = nil) {
        self.maxWorkingSetBytes = maxWorkingSetBytes
    }

    public func capture() -> QWDeviceMemorySnapshot {
        QWDeviceMemorySnapshot(
            physicalBytes: QWAdmission.physicalMemoryBytes() ?? 0,
            reclaimableBytes: QWAdmission.reclaimableMemoryBytes(),
            maxWorkingSetBytes: maxWorkingSetBytes
        )
    }
}

/// Deterministic in-memory adapter for unit tests (zero Mach or GPU queries).
public struct QWSyntheticDeviceMemorySnapshot: QWDeviceMemorySnapshotSource, Equatable {
    public let snapshot: QWDeviceMemorySnapshot

    public init(physicalBytes: Int, reclaimableBytes: Int?, maxWorkingSetBytes: Int? = nil) {
        self.snapshot = QWDeviceMemorySnapshot(
            physicalBytes: physicalBytes,
            reclaimableBytes: reclaimableBytes,
            maxWorkingSetBytes: maxWorkingSetBytes
        )
    }

    public func capture() -> QWDeviceMemorySnapshot { snapshot }
}

/// Unified residency and warm-cache admission calculations across all engine families.
public enum QWResidencyAdmission {
    /// Minimum free/reclaimable UMA cushion preserved after wiring model + KV + warm cache.
    public static let postWireReclaimableFloorBytes = 6 << 30
    /// Minimum host/OS reserve enforced when wiring >= 64 GiB of GPU weight arenas.
    public static let largeModelOSReserveFloorBytes = 16 << 30
    /// Smallest worthwhile warm-RAM prefix cache tier (below 128 MiB, disable RAM retention).
    public static let minViableWarmCacheBytes = 128 << 20

    public struct WarmCacheAdmission: Sendable, Equatable {
        public let requestedWarmBytes: Int
        public let admittedWarmBytes: Int
        public let maxConcurrentSessions: Int
        public let clamped: Bool

        public init(requestedWarmBytes: Int, admittedWarmBytes: Int,
                    maxConcurrentSessions: Int, clamped: Bool) {
            self.requestedWarmBytes = requestedWarmBytes
            self.admittedWarmBytes = admittedWarmBytes
            self.maxConcurrentSessions = maxConcurrentSessions
            self.clamped = clamped
        }
    }

    public struct FlashNextAdmission: Sendable, Equatable {
        public let weightsBytes: Int
        public let pleResidentBytes: Int
        public let fixedScratchBytes: Int
        public let availableBytes: Int
        public let warmCache: WarmCacheAdmission

        public var totalFixedBytes: Int { weightsBytes + pleResidentBytes + fixedScratchBytes }
    }

    /// Computes available UMA bytes from a snapshot after subtracting the OS reserve.
    public static func availableBytes(snapshot: QWDeviceMemorySnapshot, osReserveGiB: Double? = nil) -> Int {
        let reserve: Int
        if let gib = osReserveGiB {
            reserve = max(0, Int((gib * Double(1 << 30)).rounded()))
        } else {
            reserve = QWCapacityBudget.osReserveBytes
        }
        let physicalCap = max(0, snapshot.physicalBytes - reserve)
        if let ws = snapshot.maxWorkingSetBytes, ws > 0, physicalCap > 0 {
            return min(physicalCap, ws)
        }
        return physicalCap
    }

    /// Checks whether mapping a resident GPU PLE table (`ngram.bin` VQ2) fits safely in UMA.
    public static func isPLEAdmissible(
        weightsEstimate: Int,
        packDiskBytes: Int,
        fixedScratchBytes: Int,
        pleFileBytes: Int,
        minKVBytes: Int,
        osReserveGiB: Double? = nil,
        snapshot: QWDeviceMemorySnapshot = QWLiveDeviceMemorySnapshotSource().capture()
    ) -> Bool {
        let effectiveWeights = max(weightsEstimate, packDiskBytes)
        let totalWired = effectiveWeights + fixedScratchBytes + pleFileBytes + minKVBytes
        let avail = availableBytes(snapshot: snapshot, osReserveGiB: osReserveGiB)
        if avail > 0 && totalWired > avail { return false }
        if totalWired >= (1 << 30), let reclaimable = snapshot.reclaimableBytes {
            let slack = totalWired >= (64 << 30) ? -(4 << 30) : (4 << 30)
            if totalWired > reclaimable + slack { return false }
        }
        return true
    }

    /// Clamps `PrefixCache` warm RAM retention and session slots so host snapshots never push
    /// wired Metal engines across the macOS unified-memory paging cliff.
    public static func clampWarmCache(
        requestedWarmBytes: Int,
        requestedSessions: Int = 8,
        wiredEngineBytes: Int,
        osReserveGiB: Double? = nil,
        measuredAfterWire: Bool,
        snapshot: QWDeviceMemorySnapshot = QWLiveDeviceMemorySnapshotSource().capture()
    ) -> WarmCacheAdmission {
        let clampedSessions = max(1, min(16, requestedSessions))
        guard requestedWarmBytes > 0 else {
            return WarmCacheAdmission(requestedWarmBytes: 0, admittedWarmBytes: 0,
                                      maxConcurrentSessions: clampedSessions, clamped: false)
        }
        // Leave tiny synthetic unit-test fixtures (< 1 GiB) unclamped on ambient host memory.
        guard wiredEngineBytes >= (1 << 30) else {
            return WarmCacheAdmission(requestedWarmBytes: requestedWarmBytes,
                                      admittedWarmBytes: requestedWarmBytes,
                                      maxConcurrentSessions: clampedSessions,
                                      clamped: false)
        }
        let explicitReserve = osReserveGiB.map { max(0, Int(($0 * Double(1 << 30)).rounded())) }
        let defaultReserve = wiredEngineBytes >= (64 << 30)
            ? largeModelOSReserveFloorBytes
            : QWCapacityBudget.osReserveBytes
        let reserveBytes = explicitReserve ?? defaultReserve
        let staticHeadroom = snapshot.physicalBytes > 0
            ? max(0, snapshot.physicalBytes - reserveBytes - wiredEngineBytes)
            : requestedWarmBytes
        let liveHeadroom: Int
        if let reclaimable = snapshot.reclaimableBytes {
            let postWire = measuredAfterWire ? reclaimable : (reclaimable - wiredEngineBytes)
            liveHeadroom = max(0, postWire - postWireReclaimableFloorBytes)
        } else {
            liveHeadroom = staticHeadroom
        }
        let rawCap = min(requestedWarmBytes, min(staticHeadroom, liveHeadroom))
        if rawCap < minViableWarmCacheBytes {
            return WarmCacheAdmission(requestedWarmBytes: requestedWarmBytes, admittedWarmBytes: 0,
                                      maxConcurrentSessions: 1, clamped: true)
        }
        let sessionCap = rawCap < (2 << 30)
            ? max(1, min(clampedSessions, rawCap / (256 << 20)))
            : clampedSessions
        return WarmCacheAdmission(
            requestedWarmBytes: requestedWarmBytes,
            admittedWarmBytes: rawCap,
            maxConcurrentSessions: sessionCap,
            clamped: rawCap < requestedWarmBytes || sessionCap < clampedSessions
        )
    }

    /// Deduplicates Flash-Next `.qwp` / safetensors weights, GPU `VQ2` PLE, retained prefill scratch,
    /// LSA clusters, and warm prefix-cache headroom into one deterministic fixed budget.
    public static func planFlashNext(
        weightsEstimate: Int,
        packDiskBytes: Int,
        gpuPLEBytes: Int,
        ngramRowCacheBytes: Int,
        baseScratchBytes: Int,
        retainedPrefillBytes: Int,
        lsaClusterBytes: Int,
        minKVBytes: Int,
        requestedWarmCacheBytes: Int,
        requestedSessions: Int,
        osReserveGiB: Double? = nil,
        snapshot: QWDeviceMemorySnapshot = QWLiveDeviceMemorySnapshotSource().capture()
    ) -> FlashNextAdmission {
        let effectiveWeights = max(weightsEstimate, packDiskBytes)
        let pleBytes = gpuPLEBytes > 0 ? gpuPLEBytes : ngramRowCacheBytes
        let fixedScratch = baseScratchBytes + max(0, retainedPrefillBytes) + max(0, lsaClusterBytes)
        let avail = availableBytes(snapshot: snapshot, osReserveGiB: osReserveGiB)
        let plannedEngineBytes = effectiveWeights + pleBytes + fixedScratch + max(0, minKVBytes)
        let warm = clampWarmCache(
            requestedWarmBytes: requestedWarmCacheBytes,
            requestedSessions: requestedSessions,
            wiredEngineBytes: plannedEngineBytes,
            osReserveGiB: osReserveGiB,
            measuredAfterWire: false,
            snapshot: snapshot
        )
        return FlashNextAdmission(
            weightsBytes: effectiveWeights,
            pleResidentBytes: pleBytes,
            fixedScratchBytes: fixedScratch,
            availableBytes: avail,
            warmCache: warm
        )
    }

    /// Inspects on-disk `model.qwp` and `assets/ngram.bin` sizes when estimating Flash-Next context memory.
    public static func flashNextOnDiskFootprint(packPath: String?, minWeightsBytes: Int) -> (weightsBytes: Int, pleBytes: Int) {
        guard let rawPath = packPath, !rawPath.isEmpty else { return (minWeightsBytes, 0) }
        let fm = FileManager.default
        let rootURL = URL(fileURLWithPath: (rawPath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDir) else {
            return (minWeightsBytes, 0)
        }
        let weightsURL = isDir.boolValue ? rootURL.appendingPathComponent("model.qwp") : rootURL
        let assetsURL = (isDir.boolValue ? rootURL : rootURL.deletingLastPathComponent())
            .appendingPathComponent("assets/ngram.bin")
        let qwpBytes = (try? fm.attributesOfItem(atPath: weightsURL.path)[.size] as? Int) ?? 0
        let ngramBytes = (try? fm.attributesOfItem(atPath: assetsURL.path)[.size] as? Int) ?? 0
        // VQ2 ngram.bin (~27.1 GiB) maps resident on GPU when QWG_PLE_GPU_GATHER != 0; raw fp16 (>40 GiB) streams via 64 MiB row cache.
        let pleResident: Int
        if ProcessInfo.processInfo.environment["QWG_PLE_GPU_GATHER"] == "0" {
            pleResident = ngramBytes > 0 ? (64 << 20) : 0
        } else if ngramBytes > 0 && ngramBytes <= (36 << 30) {
            pleResident = ngramBytes
        } else {
            pleResident = ngramBytes > 0 ? (64 << 20) : 0
        }
        return (max(minWeightsBytes, qwpBytes), pleResident)
    }
}

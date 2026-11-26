// QWResidencyPool.swift — Multi-slot UMA residency orchestrator managing active model leases (Law 12 / G5).

import Foundation
import QWCore

/// Logical residency slot managed inside `QWResidencyPool`.
public enum QWResidencySlotKind: String, Codable, Sendable, CaseIterable {
    case primaryLLM = "primary_llm"
    case decisionSidecar = "decision_sidecar"
    case embeddingSidecar = "embedding_sidecar"
    case visionTower = "vision_tower"
    case audioEncoder = "audio_encoder"
}

/// Active UMA residency lease held by a loaded model or subsystem.
public struct QWResidencyLease: Codable, Sendable, Equatable {
    public let slot: QWResidencySlotKind
    public let modelName: String
    public let packPath: String
    public let estimatedBytes: UInt64
    public let computeUnit: String
    public let bypassesGPUArbitrator: Bool
    public let acquiredAt: Date

    public init(
        slot: QWResidencySlotKind,
        modelName: String,
        packPath: String,
        estimatedBytes: UInt64,
        computeUnit: String,
        bypassesGPUArbitrator: Bool = false,
        acquiredAt: Date = Date()
    ) {
        self.slot = slot
        self.modelName = modelName
        self.packPath = packPath
        self.estimatedBytes = estimatedBytes
        self.computeUnit = computeUnit
        self.bypassesGPUArbitrator = bypassesGPUArbitrator
        self.acquiredAt = acquiredAt
    }
}

/// Result of evaluating whether a candidate slot lease fits in the 128 GB UMA budget.
public struct QWResidencyAdmissionOutcome: Sendable, Equatable {
    public let admitted: Bool
    public let availableBytes: Int
    public let committedMetalBytes: UInt64
    public let warmCache: QWResidencyAdmission.WarmCacheAdmission
    public let reason: String?

    public init(
        admitted: Bool,
        availableBytes: Int,
        committedMetalBytes: UInt64,
        warmCache: QWResidencyAdmission.WarmCacheAdmission,
        reason: String?
    ) {
        self.admitted = admitted
        self.availableBytes = availableBytes
        self.committedMetalBytes = committedMetalBytes
        self.warmCache = warmCache
        self.reason = reason
    }
}

/// Point-in-time summary of all active UMA residency leases and remaining headroom.
public struct QWResidencyPoolSnapshot: Codable, Sendable, Equatable {
    public let leases: [QWResidencyLease]
    public let totalEstimatedMetalBytes: UInt64
    public let totalEstimatedANEBytes: UInt64
    public let availableUMABytes: UInt64
    public let headroomBytes: UInt64
    public let warmCacheAdmittedBytes: Int

    public init(
        leases: [QWResidencyLease],
        totalEstimatedMetalBytes: UInt64,
        totalEstimatedANEBytes: UInt64,
        availableUMABytes: UInt64,
        headroomBytes: UInt64,
        warmCacheAdmittedBytes: Int
    ) {
        self.leases = leases
        self.totalEstimatedMetalBytes = totalEstimatedMetalBytes
        self.totalEstimatedANEBytes = totalEstimatedANEBytes
        self.availableUMABytes = availableUMABytes
        self.headroomBytes = headroomBytes
        self.warmCacheAdmittedBytes = warmCacheAdmittedBytes
    }
}

/// Budgeted multi-slot UMA orchestrator across primary LLM, decision/embedding sidecars, and media leases.
public actor QWResidencyPool {
    private final class MirrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var leases: [QWResidencySlotKind: QWResidencyLease] = [:]

        func set(_ next: [QWResidencySlotKind: QWResidencyLease]) {
            lock.withLock { leases = next }
        }

        func allLeases() -> [QWResidencyLease] {
            lock.withLock {
                leases.values.sorted { $0.slot.rawValue < $1.slot.rawValue }
            }
        }
    }

    private let memorySnapshotSource: any QWDeviceMemorySnapshotSource
    private let osReserveGiB: Double?
    private var leases: [QWResidencySlotKind: QWResidencyLease] = [:]
    private let mirror = MirrorBox()

    public init(
        memorySnapshotSource: any QWDeviceMemorySnapshotSource = QWLiveDeviceMemorySnapshotSource(),
        osReserveGiB: Double? = nil
    ) {
        self.memorySnapshotSource = memorySnapshotSource
        self.osReserveGiB = osReserveGiB
    }

    /// Evaluates UMA admission for `slot` against all other active leases without mutating committed leases.
    public func admit(
        slot: QWResidencySlotKind,
        modelName: String,
        packPath: String,
        estimatedBytes: Int,
        computeUnit: String,
        measuredPrimaryWiredBytes: Int = 0,
        requestedWarmCacheBytes: Int = 0,
        requestedSessions: Int = 8
    ) -> QWResidencyAdmissionOutcome {
        _ = (modelName, packPath)
        let snapshot = memorySnapshotSource.capture()
        let avail = QWResidencyAdmission.availableBytes(snapshot: snapshot, osReserveGiB: osReserveGiB)
        let bypassesGPU = computeUnit.lowercased().hasPrefix("ane") || computeUnit.lowercased().hasPrefix("coreai")
        let candidateBytes = max(0, estimatedBytes)

        var otherMetalBytes = 0
        var otherTotalBytes = 0
        var hasPrimaryLease = false
        for (k, lease) in leases where k != slot {
            let leaseBytes = Int(min(UInt64(Int.max / 2), lease.estimatedBytes))
            let effectiveBytes = (k == .primaryLLM) ? max(leaseBytes, max(0, measuredPrimaryWiredBytes)) : leaseBytes
            if k == .primaryLLM { hasPrimaryLease = true }
            otherTotalBytes += effectiveBytes
            if !lease.bypassesGPUArbitrator {
                otherMetalBytes += effectiveBytes
            }
        }
        if slot != .primaryLLM && !hasPrimaryLease && measuredPrimaryWiredBytes > 0 {
            otherTotalBytes += measuredPrimaryWiredBytes
            otherMetalBytes += measuredPrimaryWiredBytes
        }

        let candidateMetal = bypassesGPU ? 0 : candidateBytes
        let admission = QWResidencyAdmission.admitDecisionSidecar(
            primaryWiredBytes: otherMetalBytes,
            sidecarEstimateBytes: candidateMetal,
            requestedWarmCacheBytes: requestedWarmCacheBytes,
            requestedSessions: requestedSessions,
            osReserveGiB: osReserveGiB,
            snapshot: snapshot
        )
        let totalCombined = otherTotalBytes + candidateBytes
        let enforceCap = (osReserveGiB != nil) || (totalCombined >= (1 << 30)) || (avail > 0)
        if !admission.admitted {
            return QWResidencyAdmissionOutcome(
                admitted: false,
                availableBytes: admission.availableBytes,
                committedMetalBytes: UInt64(otherMetalBytes),
                warmCache: admission.warmCache,
                reason: admission.reason
            )
        }
        if enforceCap && totalCombined > avail {
            let zeroWarm = QWResidencyAdmission.WarmCacheAdmission(
                requestedWarmBytes: max(0, requestedWarmCacheBytes),
                admittedWarmBytes: 0,
                maxConcurrentSessions: 1,
                clamped: true
            )
            return QWResidencyAdmissionOutcome(
                admitted: false,
                availableBytes: avail,
                committedMetalBytes: UInt64(otherMetalBytes),
                warmCache: zeroWarm,
                reason: "slot '\(slot.rawValue)' (\(candidateBytes) B) + resident (\(otherTotalBytes) B) exceeds available UMA (\(avail) B)"
            )
        }
        return QWResidencyAdmissionOutcome(
            admitted: true,
            availableBytes: admission.availableBytes,
            committedMetalBytes: UInt64(otherMetalBytes + candidateMetal),
            warmCache: admission.warmCache,
            reason: nil
        )
    }

    /// Commits or replaces the active lease for `lease.slot`.
    public func commitLease(_ lease: QWResidencyLease) {
        leases[lease.slot] = lease
        mirror.set(leases)
    }

    /// Releases any active lease for `slot` and returns the removed lease.
    @discardableResult
    public func releaseLease(for slot: QWResidencySlotKind) -> QWResidencyLease? {
        let removed = leases.removeValue(forKey: slot)
        mirror.set(leases)
        return removed
    }

    /// Returns the active lease for `slot`, if resident.
    public func lease(for slot: QWResidencySlotKind) -> QWResidencyLease? {
        leases[slot]
    }

    /// Computes a full residency pool snapshot including Metal vs ANE bytes and clamped warm cache.
    public func snapshot(
        requestedWarmCacheBytes: Int = 0,
        requestedSessions: Int = 8
    ) -> QWResidencyPoolSnapshot {
        let mem = memorySnapshotSource.capture()
        let avail = max(0, QWResidencyAdmission.availableBytes(snapshot: mem, osReserveGiB: osReserveGiB))
        var metalBytes: UInt64 = 0
        var aneBytes: UInt64 = 0
        for lease in leases.values {
            if lease.bypassesGPUArbitrator {
                aneBytes &+= lease.estimatedBytes
            } else {
                metalBytes &+= lease.estimatedBytes
            }
        }
        let total = metalBytes &+ aneBytes
        let totalClamped = Int(min(UInt64(Int.max / 2), total))
        let warm = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: requestedWarmCacheBytes,
            requestedSessions: requestedSessions,
            wiredEngineBytes: totalClamped,
            osReserveGiB: osReserveGiB,
            measuredAfterWire: true,
            snapshot: mem
        )
        let headroom = UInt64(avail) > total ? (UInt64(avail) - total) : 0
        return QWResidencyPoolSnapshot(
            leases: leases.values.sorted { $0.slot.rawValue < $1.slot.rawValue },
            totalEstimatedMetalBytes: metalBytes,
            totalEstimatedANEBytes: aneBytes,
            availableUMABytes: UInt64(avail),
            headroomBytes: headroom,
            warmCacheAdmittedBytes: warm.admittedWarmBytes
        )
    }

    /// Non-blocking snapshot of committed leases for telemetry and HTTP status routes.
    nonisolated public func currentLeases() -> [QWResidencyLease] {
        mirror.allLeases()
    }
}

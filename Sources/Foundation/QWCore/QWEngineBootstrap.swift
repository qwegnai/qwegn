// QWEngineBootstrap.swift — Shared engine plan construction and memory admission bootstrap.
// Unifies plan creation, capacity log formatting, single-instance lock acquisition,
// and live/physical memory validation across all GPU inference engines.

import Foundation

/// Shared bootstrap helper for constructing an engine's execution plan and validating memory admission.
public enum QWEngineBootstrap {
    /// Minimum weight footprint (4 GiB) that triggers the single-instance GPU engine lock.
    public static let singleInstanceLockThresholdBytes = 4 << 30

    /// Builds the frozen execution plan for a model pack and geometry.
    public static func makePlan(
        geometry: QWModelGeometry,
        packPath: String,
        prefillChunkRows: Int,
        kvStorageMode: String,
        kernelDispatch: QWKernelDispatchTable? = nil
    ) throws -> QWEnginePlan {
        try QWEnginePlan(
            geometry: geometry,
            modelId: QWEnginePlan.packIdentity(packPath: packPath),
            geometryDigest: geometry.descriptor.geometryDigestV2Hex,
            prefillChunkRows: prefillChunkRows,
            kvStorageMode: kvStorageMode,
            kernelDispatch: kernelDispatch
        )
    }

    /// Formats the startup capacity log line, acquires the single-instance lock for large models,
    /// and validates both physical and live reclaimable host RAM before arena allocation.
    public static func admit(
        weightsBytes: Int,
        fixedScratchBytes: Int,
        kvMode: String,
        perTokenBytes: Int,
        maxTokens: Int,
        totalPlannedBytes: Int,
        availableBytes: Int,
        osReserveGiB: Double?,
        logSuffix: String = ""
    ) throws -> String {
        let logMessage = QWCapacityPlan.formatLogLine(
            weightsBytes: weightsBytes,
            fixedScratchBytes: fixedScratchBytes,
            kvMode: kvMode,
            perTokenBytes: perTokenBytes,
            maxTokens: maxTokens,
            totalPlannedBytes: totalPlannedBytes,
            availableBytes: availableBytes
        ) + logSuffix
        if weightsBytes > singleInstanceLockThresholdBytes {
            try QWEngineInstanceLock.acquireEngineLock()
        }
        try QWCapacityBudget.validateMemoryForEngineLoad(
            requiredBytes: totalPlannedBytes,
            osReserveGiB: osReserveGiB
        )
        return logMessage
    }
}

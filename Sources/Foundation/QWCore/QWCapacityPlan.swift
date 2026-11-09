// QWCapacityPlan.swift — Derives usable context from the model config and runtime parameters.
// Pure: no GPU, no I/O. Caps come only from the native window in config.json and from memory.

public struct QWCapacityPlan: Equatable, Sendable {
    public enum Failure: Error, Equatable {
        case contextBelowMinimum(Int)
        case yarnNotAvailable(requested: Int, native: Int)
    }

    /// Tokens the engine allocates for.
    public let context: Int
    /// YaRN factor when `context` exceeds the native window; nil otherwise.
    public let yarnFactor: Double?
    /// Bytes needed for `context` (fixed + per-token × context).
    public let requiredBytes: Int
    /// Largest context that fits in the available bytes, never above the native window unless YaRN is allowed.
    public let maxFittingContext: Int

    /// True when `requiredBytes` fits in the available bytes.
    public let fits: Bool

    /// Caps come only from nativeWindow (config.json) and RAM. yarnAllowed gates requests above native.
    /// nil requested = native window.
    public static func plan(nativeWindow: Int, requested: Int?, perTokenBytes: Int,
                            fixedBytes: Int, availableBytes: Int,
                            yarnAllowed: Bool) throws -> QWCapacityPlan {
        let context = requested ?? nativeWindow
        guard context >= 1 else { throw Failure.contextBelowMinimum(context) }
        var yarn: Double? = nil
        if context > nativeWindow {
            guard yarnAllowed else {
                throw Failure.yarnNotAvailable(requested: context, native: nativeWindow)
            }
            yarn = Double(context) / Double(nativeWindow)
        }
        let perToken = max(1, perTokenBytes)
        let required = fixedBytes + perToken * context
        let freeForKV = max(0, availableBytes - fixedBytes)
        let maxFitting = min(freeForKV / perToken, yarnAllowed ? Int.max : nativeWindow)
        return QWCapacityPlan(context: context, yarnFactor: yarn, requiredBytes: required,
                              maxFittingContext: maxFitting, fits: required <= availableBytes)
    }

    /// K+V bytes per token across all attention layers.
    public static func kvBytesPerToken(attentionLayers: Int, kvHeads: Int, headDim: Int,
                                       bytesPerElement: Double, kvLoraRank: Int = 0) -> Int {
        if kvLoraRank > 0 {
            return Int((Double(attentionLayers) * Double(kvLoraRank) * bytesPerElement).rounded(.up))
        }
        return Int((Double(attentionLayers) * 2 * Double(kvHeads * headDim) * bytesPerElement).rounded(.up))
    }

    /// Stepped KV allocation quantum (8,192 tokens, matching oMLX QSAKVCache).
    public static let kvStepTokens = 8_192

    /// Aligns a requested token capacity to 32-token blocks for short contexts (<= 8,192) or 8,192-token steps above 8,192.
    public static func steppedCapacity(_ tokens: Int) -> Int {
        if tokens <= kvStepTokens {
            return ((max(32, tokens) + 31) / 32) * 32
        }
        return ((tokens + kvStepTokens - 1) / kvStepTokens) * kvStepTokens
    }

    /// Formats a standardized `[memory]` startup capacity log line.
    public static func formatLogLine(weightsBytes: Int, fixedScratchBytes: Int, kvMode: String,
                                     perTokenBytes: Int, maxTokens: Int, totalPlannedBytes: Int,
                                     availableBytes: Int) -> String {
        let gib = 1_073_741_824.0
        return String(
            format: "[memory] weights=%.2f GiB fixed_scratch=%.2f GiB kv_mode=%@ kv_per_tok=%d B max_tokens=%d total_planned=%.2f GiB available=%.2f GiB",
            Double(weightsBytes) / gib,
            Double(fixedScratchBytes) / gib,
            kvMode,
            perTokenBytes,
            maxTokens,
            Double(totalPlannedBytes) / gib,
            Double(availableBytes) / gib
        )
    }
}

/// Predicted GPU working-set footprint for a model descriptor and context configuration.
public struct QWContextMemoryEstimate: Sendable, Equatable {
    public let contextTokens: Int
    public let weightsBytes: Int
    public let kvBytes: Int
    public let scratchAndStateBytes: Int
    public let requiredBytes: Int
    public let availableBytes: Int
    public let maxFittingContext: Int
    public let fits: Bool

    public init(contextTokens: Int, weightsBytes: Int, kvBytes: Int, scratchAndStateBytes: Int,
                requiredBytes: Int, availableBytes: Int, maxFittingContext: Int, fits: Bool) {
        self.contextTokens = contextTokens
        self.weightsBytes = weightsBytes
        self.kvBytes = kvBytes
        self.scratchAndStateBytes = scratchAndStateBytes
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
        self.maxFittingContext = maxFittingContext
        self.fits = fits
    }
}

import Foundation

/// Memory the engine may plan against: physical RAM minus an OS reserve.
public enum QWCapacityBudget {
    public static let osReserveBytes = 8 << 30

    public enum MemoryValidationError: Error, LocalizedError, CustomStringConvertible, Equatable {
        case insufficientPhysicalMemory(requiredBytes: Int, budgetBytes: Int)
        case insufficientLiveMemory(requiredBytes: Int, reclaimableBytes: Int)

        public var description: String {
            let gib = 1_073_741_824.0
            switch self {
            case .insufficientPhysicalMemory(let req, let budget):
                return String(
                    format: "Insufficient system memory Budget: model + context requires %.2f GiB, but physical memory budget after OS reserve is %.2f GiB.",
                    Double(req) / gib, Double(budget) / gib
                )
            case .insufficientLiveMemory(let req, let avail):
                return String(
                    format: "Insufficient free/reclaimable memory to load engine safely: requires %.2f GiB, but only %.2f GiB is currently free/reclaimable. Close other memory-intensive processes or unload any running model first.",
                    Double(req) / gib, Double(avail) / gib
                )
            }
        }

        public var errorDescription: String? { description }
    }

    public static func availableBytes(osReserveGiB: Double? = nil) -> Int {
        let reserve: Int
        if let gib = osReserveGiB {
            reserve = max(0, Int((gib * Double(1 << 30)).rounded()))
        } else {
            reserve = osReserveBytes
        }
        return max(0, (QWAdmission.physicalMemoryBytes() ?? 0) - reserve)
    }

    /// Validates both the static physical budget and live reclaimable host RAM before allocating GPU model arenas.
    public static func validateMemoryForEngineLoad(requiredBytes: Int, osReserveGiB: Double? = nil) throws {
        // Skip live host check for tiny synthetic unit-test fixtures (< 1 GiB).
        guard requiredBytes >= (1 << 30) else { return }
        let budget = availableBytes(osReserveGiB: osReserveGiB)
        if budget > 0 && requiredBytes > budget {
            throw MemoryValidationError.insufficientPhysicalMemory(requiredBytes: requiredBytes, budgetBytes: budget)
        }
        if var reclaimable = QWAdmission.reclaimableMemoryBytes() {
            // For moderate arenas (< 64 GiB), allow up to 6 GiB of inactive anonymous compression slack.
            // For massive arenas (>= 64 GiB) that wire most of physical RAM, require strict live
            // reclaimable coverage with a 4 GiB safety floor so wired Metal pages never starve macOS.
            let slack = requiredBytes >= (64 << 30) ? -(4 << 30) : (6 << 30)
            if requiredBytes > reclaimable + slack {
                // If a previous model arena was just released, macOS IOKit/Metal reclaims wired
                // pages asynchronously; poll briefly before declaring an OOM failure.
                for _ in 0..<30 {
                    Thread.sleep(forTimeInterval: 0.1)
                    if let updated = QWAdmission.reclaimableMemoryBytes() {
                        reclaimable = max(reclaimable, updated)
                        if requiredBytes <= reclaimable + slack { break }
                    }
                }
            }
            if requiredBytes > reclaimable + slack {
                throw MemoryValidationError.insufficientLiveMemory(
                    requiredBytes: requiredBytes,
                    reclaimableBytes: reclaimable
                )
            }
        }
    }
}

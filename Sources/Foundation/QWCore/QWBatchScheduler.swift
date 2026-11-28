// QWBatchScheduler.swift — Shared L0 batch admission, slot lifecycle, partition sizing, and wave primitives.

import Darwin
import Dispatch
import Foundation

public enum QWBatchSchedulerError: Error, Equatable, CustomStringConvertible {
    case missingJobOutcome

    public var description: String {
        switch self {
        case .missingJobOutcome:
            return "batch job finished without an outcome"
        }
    }
}

/// Thread-safe admission preparation counter for micro-coalescing concurrent bursts.
public enum QWBatchAdmissionCounter {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0

    public static func beginPreparing() {
        lock.withLock { count += 1 }
    }

    public static func endPreparing() {
        lock.withLock { count = max(0, count - 1) }
    }

    public static var preparingCount: Int {
        lock.withLock { count }
    }

    /// Brief bounded wait when peer requests are actively tokenizing/preparing templates.
    public static func coalescePreparingBurst(
        maxSlots: Int,
        maxWaitUs: useconds_t = 10_000,
        pollIntervalUs: useconds_t = 250,
        queuedCount: () -> Int
    ) {
        guard preparingCount > 0, maxSlots > 1 else { return }
        var waitedUs: useconds_t = 0
        while waitedUs < maxWaitUs {
            if queuedCount() >= maxSlots { break }
            if preparingCount == 0 && waitedUs >= 1_000 { break }
            usleep(pollIntervalUs)
            waitedUs += pollIntervalUs
        }
    }
}

/// Single-shot thread-safe completion promise for a queued batch generation job.
public final class QWBatchJobPromise<Output: Sendable>: @unchecked Sendable {
    private let done = DispatchSemaphore(value: 0)
    private let outcomeLock = NSLock()
    private var outcome: Result<Output, any Error>?

    public init() {}

    public func complete(_ result: Result<Output, any Error>) {
        let shouldSignal = outcomeLock.withLock { () -> Bool in
            guard outcome == nil else { return false }
            outcome = result
            return true
        }
        if shouldSignal { done.signal() }
    }

    public func wait() throws -> Output {
        done.wait()
        guard let result = outcomeLock.withLock({ outcome }) else {
            throw QWBatchSchedulerError.missingJobOutcome
        }
        return try result.get()
    }
}

/// Pure partition capacity, slot admission, eligibility, and round-robin scheduling math.
public enum QWBatchPartitionSelection {
    public struct SlotAdmissionDemand: Sendable, Equatable {
        public let promptCount: Int
        public let maxNew: Int

        public init(promptCount: Int, maxNew: Int) {
            self.promptCount = promptCount
            self.maxNew = maxNew
        }
    }

    public struct SlotAdmissionAssignment: Sendable, Equatable {
        public let slot: Int
        public let queueIndex: Int
        public let inBatchMode: Bool

        public init(slot: Int, queueIndex: Int, inBatchMode: Bool) {
            self.slot = slot
            self.queueIndex = queueIndex
            self.inBatchMode = inBatchMode
        }
    }

    public static func slotCapacity(fullCapacity: Int, totalSlots: Int, alignment: Int = 1) -> Int {
        guard totalSlots > 1 else { return fullCapacity }
        let block = max(1, alignment)
        return max(block, ((fullCapacity / totalSlots) / block) * block)
    }

    public static func clampedMaxNew(requested: Int, promptCount: Int, activeCapacity: Int) -> Int {
        min(max(0, requested), max(0, activeCapacity - promptCount))
    }

    public static func fitsSlotCapacity(
        usedTokens: Int,
        remainingTokens: Int,
        margin: Int = 4,
        slotCapacity: Int
    ) -> Bool {
        usedTokens + remainingTokens + margin <= slotCapacity
    }

    public static func eligibleSlotIndices(
        count: Int,
        maxRows: Int,
        isEligible: (Int) -> Bool
    ) -> [Int] {
        guard count > 0, maxRows > 0 else { return [] }
        var out: [Int] = []
        out.reserveCapacity(min(maxRows, count))
        for idx in 0..<count where isEligible(idx) {
            out.append(idx)
            if out.count == maxRows { break }
        }
        return out
    }

    public static func nextRoundRobinIndex(
        current: Int,
        stepIndices: [Int],
        activeCount: Int,
        mixedShortAndLong: Bool
    ) -> Int {
        guard activeCount > 0 else { return 0 }
        if mixedShortAndLong {
            return ((stepIndices.last ?? current) + 1) % activeCount
        }
        return current >= activeCount ? 0 : current
    }

    public static func validateRequestBounds(
        prompt: [Int],
        maxNew: Int,
        stopIDs: Set<Int>,
        vocab: Int
    ) -> Bool {
        maxNew >= 0 && !prompt.isEmpty
            && prompt.allSatisfy({ (0..<vocab).contains($0) })
            && stopIDs.allSatisfy({ (0..<vocab).contains($0) })
    }

    public static func shouldRunBatchedStep(
        stepIndices: [Int],
        activeCount: Int,
        roundRobinIndex: Int,
        canRunBatchedDecode: Bool,
        hasSpeculative: Bool,
        allowsSpeculativeBatch: Bool
    ) -> Bool {
        guard stepIndices.count >= 2, canRunBatchedDecode else { return false }
        let mixed = stepIndices.count < activeCount
        if mixed && !stepIndices.contains(roundRobinIndex) { return false }
        return allowsSpeculativeBatch || !hasSpeculative
    }

    public static func serialQuantum(
        canRunBatchedDecode: Bool,
        activeCount: Int,
        canSpeculateCurrent: Bool
    ) -> Int {
        (!canRunBatchedDecode && activeCount > 1 && !canSpeculateCurrent) ? 4 : 1
    }

    public static func canEnableSpeculation(
        requested: Bool,
        mtpDrafts: Int,
        collectLogprobs: Bool,
        allowsSpeculation: Bool,
        hasExplicitSeed: Bool,
        isGreedy: Bool
    ) -> Bool {
        requested && mtpDrafts > 0 && !collectLogprobs && allowsSpeculation && !(hasExplicitSeed && !isGreedy)
    }

    public static func isSlotSpeculatable(
        speculativeEnabled: Bool,
        remaining: Int,
        hasLogprobsSampler: Bool,
        isUnconstrainedNow: Bool,
        hasConstraint: Bool,
        hasConstraintSampler: Bool,
        canSpeculateConstrained: Bool = false
    ) -> Bool {
        speculativeEnabled && remaining >= 2 && !hasLogprobsSampler
            && (isUnconstrainedNow || canSpeculateConstrained)
            && (!hasConstraint || hasConstraintSampler)
    }

    /// Plans which queued requests enter free partition slots in single-stream vs multi-slot mode.
    public static func planAdmissions(
        maxSlots: Int,
        slotCapacity: Int,
        slot0ActiveInBatchMode: Bool,
        slot0FitsInBatch: Bool,
        occupiedSlots: [Bool],
        pendingDemands: [SlotAdmissionDemand],
        margin: Int = 4
    ) -> [SlotAdmissionAssignment] {
        guard !pendingDemands.isEmpty, maxSlots > 0, occupiedSlots.count == maxSlots else { return [] }
        let currentActive = occupiedSlots.filter { $0 }.count
        let totalCandidate = currentActive + pendingDemands.count
        let first = pendingDemands[0]
        let firstFits = fitsSlotCapacity(
            usedTokens: first.promptCount, remainingTokens: first.maxNew,
            margin: margin, slotCapacity: slotCapacity
        )
        if currentActive == 0 && (totalCandidate == 1 || !firstFits) {
            return [SlotAdmissionAssignment(slot: 0, queueIndex: 0, inBatchMode: false)]
        }
        let slot0CanBatch = occupiedSlots[0] ? (slot0ActiveInBatchMode && slot0FitsInBatch) : true
        guard slot0CanBatch else { return [] }

        var assignments: [SlotAdmissionAssignment] = []
        var qIdx = 0
        for s in 0..<maxSlots where !occupiedSlots[s] && qIdx < pendingDemands.count {
            let cand = pendingDemands[qIdx]
            let fitsIn4 = fitsSlotCapacity(
                usedTokens: cand.promptCount, remainingTokens: cand.maxNew,
                margin: margin, slotCapacity: slotCapacity
            )
            let noActiveBeforeSlot = (currentActive == 0 && assignments.isEmpty)
            if fitsIn4 || (s == 0 && noActiveBeforeSlot) {
                let batch = fitsIn4 && (totalCandidate > 1 || s > 0)
                assignments.append(SlotAdmissionAssignment(slot: s, queueIndex: qIdx, inBatchMode: batch))
                qIdx += 1
                if !batch { break }
            } else {
                break
            }
        }
        return assignments
    }
}

/// Thread-safe slot coordinator managing queued jobs, active slot occupancy, and exclusive barriers.
public final class QWBatchSlotCoordinator<Job: AnyObject>: @unchecked Sendable {
    public let maxSlots: Int
    private let lock = NSLock()
    private let exclusiveCond = NSCondition()
    private var pendingQueue: [Job] = []
    private var activeSlots: [Job?]
    private var isDriving = false
    private var isExclusiveRunning = false

    public init(maxSlots: Int) {
        self.maxSlots = max(1, maxSlots)
        self.activeSlots = Array(repeating: nil, count: self.maxSlots)
    }

    public var queuedCount: Int {
        lock.withLock { pendingQueue.count }
    }

    public var hasPending: Bool {
        lock.withLock { !pendingQueue.isEmpty }
    }

    public var hasFreeSlot: Bool {
        activeSlots.contains { $0 == nil }
    }

    public func activeSlotIndices() -> [Int] {
        (0..<maxSlots).filter { activeSlots[$0] != nil }
    }

    public func job(at slot: Int) -> Job? {
        guard (0..<maxSlots).contains(slot) else { return nil }
        return activeSlots[slot]
    }

    public func setJob(_ job: Job?, at slot: Int) {
        guard (0..<maxSlots).contains(slot) else { return }
        activeSlots[slot] = job
    }

    /// Enqueues `job` after waiting out any exclusive barrier; returns `true` if caller became driver.
    public func submit(_ job: Job) -> Bool {
        exclusiveCond.withLock {
            while isExclusiveRunning {
                exclusiveCond.wait()
            }
            return lock.withLock {
                pendingQueue.append(job)
                guard !isDriving else { return false }
                isDriving = true
                return true
            }
        }
    }

    /// Executes `body` only when all batch driving has drained and no other exclusive call runs.
    public func runExclusive<T>(_ body: () throws -> T) rethrows -> T {
        exclusiveCond.withLock {
            while isExclusiveRunning || lock.withLock({ isDriving }) {
                exclusiveCond.wait()
            }
            isExclusiveRunning = true
        }
        defer {
            exclusiveCond.withLock {
                isExclusiveRunning = false
                exclusiveCond.broadcast()
            }
        }
        return try body()
    }

    public func withPendingAndSlots<T>(
        _ body: (_ pending: inout [Job], _ active: inout [Job?]) throws -> T
    ) rethrows -> T {
        try lock.withLock {
            try body(&pendingQueue, &activeSlots)
        }
    }

    public func pollPending(maxCount: Int) -> [Job] {
        guard maxCount > 0 else { return [] }
        return lock.withLock {
            let n = min(maxCount, pendingQueue.count)
            guard n > 0 else { return [] }
            let polled = Array(pendingQueue.prefix(n))
            pendingQueue.removeFirst(n)
            return polled
        }
    }

    /// Attempts to park the driver when `pendingQueue` is empty; runs `beforePark` outside locks first.
    public func tryParkDriverIfIdle(beforePark: () -> Void) -> Bool {
        guard lock.withLock({ pendingQueue.isEmpty }) else { return false }
        beforePark()
        return exclusiveCond.withLock {
            lock.withLock {
                guard pendingQueue.isEmpty else { return false }
                isDriving = false
                exclusiveCond.broadcast()
                return true
            }
        }
    }
}

/// Tracks KV-pressure deferral and round-robin progress for continuous batch waves.
public struct QWBatchWaveState: Sendable, Equatable {
    public var blockedFreeWatermark: Int?
    public var roundRobinIndex: Int

    public init(blockedFreeWatermark: Int? = nil, roundRobinIndex: Int = 0) {
        self.blockedFreeWatermark = blockedFreeWatermark
        self.roundRobinIndex = roundRobinIndex
    }

    public mutating func refreshBlock(currentFree: Int) {
        if let watermark = blockedFreeWatermark, currentFree > watermark {
            blockedFreeWatermark = nil
        }
    }

    public func canPollPending(openSlots: Int) -> Bool {
        openSlots > 0 && blockedFreeWatermark == nil
    }

    public mutating func clearBlock() {
        blockedFreeWatermark = nil
    }

    public mutating func markBlocked(freeWatermark: Int) {
        blockedFreeWatermark = freeWatermark
    }

    public mutating func advanceRoundRobin(
        stepIndices: [Int],
        activeCount: Int,
        mixedShortAndLong: Bool
    ) {
        roundRobinIndex = QWBatchPartitionSelection.nextRoundRobinIndex(
            current: roundRobinIndex,
            stepIndices: stepIndices,
            activeCount: activeCount,
            mixedShortAndLong: mixedShortAndLong
        )
    }

    public mutating func finishSerialSlotRemoved(activeCount: Int) {
        clearBlock()
        if roundRobinIndex >= activeCount { roundRobinIndex = 0 }
    }

    public mutating func advanceSerialSlot(activeCount: Int) {
        if activeCount > 0 { roundRobinIndex = (roundRobinIndex + 1) % activeCount }
    }
}

/// Shared per-slot token output, step timing, stop detection, and speculative round metrics.
public struct QWBatchSlotOutputState: Sendable {
    public let startedAt: Date
    public var prefillSeconds: Double
    public var firstTokenSeconds: Double
    public var prefillGPUSeconds: Double
    public var decodeStartedAt: Date?
    public var generated: [Int]
    public var stepMs: [Double]
    public var stepTokenCounts: [Int]
    public var speculativeRounds: Int
    public var speculativeRoundTokens: Int
    public var stoppedToken: Int?
    public var finished: Bool

    public init(
        startedAt: Date = Date(),
        prefillSeconds: Double = 0,
        firstTokenSeconds: Double = 0,
        prefillGPUSeconds: Double = 0
    ) {
        self.startedAt = startedAt
        self.prefillSeconds = prefillSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.prefillGPUSeconds = prefillGPUSeconds
        self.decodeStartedAt = nil
        self.generated = []
        self.stepMs = []
        self.stepTokenCounts = []
        self.speculativeRounds = 0
        self.speculativeRoundTokens = 0
        self.stoppedToken = nil
        self.finished = false
    }

    public mutating func markDecodeStartedIfNeeded() {
        if decodeStartedAt == nil { decodeStartedAt = Date() }
    }

    public mutating func recordStepTiming(_ stepMsValue: Double?, tokenCount: Int = 1) {
        if let stepMsValue {
            stepMs.append(stepMsValue)
            stepTokenCounts.append(tokenCount)
        }
    }

    @discardableResult
    public mutating func beginPredictedToken(
        _ token: Int,
        stopIDs: Set<Int>,
        stepMsValue: Double? = nil,
        onStopToken: ((Int) -> Void)? = nil
    ) -> Bool {
        recordStepTiming(stepMsValue, tokenCount: 1)
        if stopIDs.contains(token) {
            stoppedToken = token
            finished = true
            onStopToken?(token)
            return false
        }
        generated.append(token)
        return true
    }

    public mutating func finishPredictedToken(maxNew: Int, accepted: Bool) {
        if !accepted || generated.count >= maxNew { finished = true }
    }

    /// Records a single predicted token, checks stop IDs / budget / `onToken` cancellation,
    /// and returns `true` when the token was appended (`false` when a stop token terminated early).
    @discardableResult
    public mutating func recordPredictedToken(
        _ token: Int,
        stopIDs: Set<Int>,
        maxNew: Int,
        stepMsValue: Double? = nil,
        onStopToken: ((Int) -> Void)? = nil,
        onAccepted: ((Int) -> Bool)? = nil
    ) -> Bool {
        guard beginPredictedToken(token, stopIDs: stopIDs, stepMsValue: stepMsValue, onStopToken: onStopToken) else {
            return false
        }
        finishPredictedToken(maxNew: maxNew, accepted: onAccepted?(token) ?? true)
        return true
    }

    public mutating func recordSpeculativeRound(
        tokens: [Int],
        stepMsValue: Double,
        maxNew: Int,
        stopped: Bool,
        stoppedToken: Int?,
        onToken: ((Int) -> Bool)?
    ) {
        stepMs.append(stepMsValue)
        stepTokenCounts.append(tokens.count)
        speculativeRounds += 1
        speculativeRoundTokens += tokens.count
        generated.append(contentsOf: tokens)
        var cancelled = false
        for tok in tokens {
            if onToken?(tok) == false { cancelled = true }
        }
        if cancelled { finished = true }
        if stopped {
            self.stoppedToken = stoppedToken
            finished = true
        } else if generated.count >= maxNew {
            finished = true
        }
    }

    public func decodeDurationSeconds() -> Double? {
        decodeStartedAt.map { Date().timeIntervalSince($0) }
    }

    public func speculativeSummary(enabled: Bool, proposals: Int, accepted: Int) -> String? {
        if speculativeRounds > 0 {
            return String(format: "rounds=%d, mean=%.2f tok/rnd",
                          speculativeRounds,
                          Double(speculativeRoundTokens) / Double(speculativeRounds))
        }
        return enabled ? String(format: "mtp=%d/%d", accepted, proposals) : nil
    }
}

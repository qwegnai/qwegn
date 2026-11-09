// DriverRequests.swift — The driver's request-scoped doors.
// A module that drives the engine one step at a time needs the engine's
// admission permit for the whole request, not for one call.

import Foundation
import Metal
import QWCore

extension QWEngineDriver {
    /// Takes the engine's admission permit for a request that spans several
    /// calls. The caller releases it when the request is over.
    func acquireRequest() throws -> QWGenerationPermit {
        try requestGate.acquire(flight: flight)
    }

    /// Gives the admission permit back.
    func releaseRequest(_ permit: QWGenerationPermit) {
        requestGate.release(permit)
    }

    /// The GPU seconds the prefill spent since the ledger stood at `mark`.
    /// Zero when the ledger holds nothing for those chunks.
    func prefillGPUSecondsSince(mark: Int) -> Double {
        let samples = prefillTimingLedger.values()
        guard mark >= 0, mark < samples.count else { return 0 }
        let micros = samples[mark...].compactMap(\.gpuMicros).reduce(0, +)
        return micros / 1_000_000
    }
}

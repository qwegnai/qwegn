import Foundation

struct QWGenerationPermit: Sendable {
    fileprivate let identity: UUID
}

/// Admission and step validation use the same flight lock. The logical owner
/// survives GPU waits and callbacks, but no mutex is held across a callback.
final class QWGenerationRequestGate: @unchecked Sendable {
    private let lock = NSLock()
    private var active: UUID?

    func withRequest<T>(flight: NSLock, _ body: (QWGenerationPermit) throws -> T) throws -> T {
        let permit = try acquire(flight: flight)
        defer { release(permit) }
        return try body(permit)
    }

    /// Takes the gate for a request that spans several driver calls.
    /// The engine has one scratch and one KV cache, so two requests must
    /// never interleave. The caller releases the permit when it is over.
    func acquire(flight: NSLock) throws -> QWGenerationPermit {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        return try lock.withLock {
            guard active == nil else { throw QWEngineDriverError.busy }
            let permit = QWGenerationPermit(identity: UUID())
            active = permit.identity
            return permit
        }
    }

    /// Gives the gate back. Releasing a permit the gate does not hold is a
    /// programming error, never a silent no-op: a double release would let
    /// two requests believe they own the engine.
    func release(_ permit: QWGenerationPermit) {
        lock.withLock {
            precondition(active == permit.identity)
            active = nil
        }
    }

    /// The caller holds flight, so admission cannot race a raw step.
    func validate(_ permit: QWGenerationPermit? = nil) throws {
        try lock.withLock {
            guard active == permit?.identity else { throw QWEngineDriverError.busy }
        }
    }
}

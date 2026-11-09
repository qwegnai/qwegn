// QWEngineSessionPool.swift — Pre-carved sessions, recycled without allocating.
// The pool hands out module-owned state and takes it back. A steady-state
// generation must not allocate, so the states are carved once, up front,
// and the free list never grows past its reserved capacity.

import Foundation
import QWCore
import QWHead

/// A fixed pool of module-owned session states.
final class QWEngineSessionPool: @unchecked Sendable {
    /// One carved session and whether it is currently handed out.
    private struct Slot {
        let handle: QWEngineSessionHandle
        var live: Bool
    }

    private let lock = NSLock()
    /// The carved sessions, in slot order.
    private var slots: [Slot]
    /// The indices of the free slots, as a stack.
    private var free: [Int]

    /// Carves `maxSessions` states up front so the steady state allocates nothing.
    init(module: any QWInferenceModule, maxSessions: Int) throws {
        var slots: [Slot] = []
        slots.reserveCapacity(maxSessions)
        for id in 0..<maxSessions {
            let state = try module.makeSessionState()
            slots.append(Slot(handle: QWEngineSessionHandle(id: id, state: state), live: false))
        }
        var free: [Int] = []
        free.reserveCapacity(maxSessions)
        free.append(contentsOf: (0..<maxSessions).reversed())
        self.slots = slots
        self.free = free
    }

    /// Takes a clean session, or nil when every slot is live.
    /// A session that cannot be reset is an error, not an empty pool. The
    /// contract is handed to the state with the reset, not kept in the slot.
    func acquire(sampling: QWSamplerConfig,
                 speculative: QWSpeculativeParameters) throws -> QWEngineSessionHandle? {
        lock.lock()
        defer { lock.unlock() }
        guard let index = free.popLast() else { return nil }
        do {
            try slots[index].handle.state.reset(sampling: sampling, speculative: speculative)
        } catch {
            free.append(index)
            throw error
        }
        slots[index].live = true
        return slots[index].handle
    }

    /// Takes a session back. Releasing twice is a no-op, not a double free,
    /// and a handle this pool never carved is ignored — the caller cannot
    /// talk the pool into handing out a state it does not own.
    func release(_ handle: QWEngineSessionHandle) {
        lock.lock()
        defer { lock.unlock() }
        guard let index = slots.firstIndex(where: { $0.handle === handle }) else { return }
        guard slots[index].live else { return }
        slots[index].live = false
        free.append(index)
    }
}

// QWResponsesStore.swift — Thread-safe bounded LRU store for /v1/responses state,
// supporting previous_response_id chaining, background status updates, GET, and DELETE.

import Foundation

public final class QWResponsesStore: @unchecked Sendable {
    public struct StoredResponse: Sendable {
        public var resource: OpenAIResponseResource
        public var historyMessages: [ChatMessage]
        public var sessionNodeId: String?
        public var task: Task<Void, Never>?

        public init(
            resource: OpenAIResponseResource,
            historyMessages: [ChatMessage],
            sessionNodeId: String? = nil,
            task: Task<Void, Never>? = nil
        ) {
            self.resource = resource
            self.historyMessages = historyMessages
            self.sessionNodeId = sessionNodeId
            self.task = task
        }
    }

    public static let defaultMaxBytes = 256 * 1024 * 1024

    private let lock = NSLock()
    private let capacity: Int
    private let maxBytes: Int
    private var entries: [String: StoredResponse] = [:]
    private var entryBytes: [String: Int] = [:]
    private var lruOrder: [String] = []
    private var totalBytes: Int = 0

    public init(capacity: Int = 256, maxBytes: Int = QWResponsesStore.defaultMaxBytes) {
        self.capacity = max(1, capacity)
        self.maxBytes = max(1024, maxBytes)
    }

    public var count: Int {
        lock.withLock { entries.count }
    }

    public func contains(_ id: String) -> Bool {
        lock.withLock { entries[id] != nil }
    }

    @discardableResult
    public func putIfAbsent(_ entry: StoredResponse) -> Bool {
        let id = entry.resource.id
        let size = Self.estimateBytes(entry)
        var evictedTasks: [Task<Void, Never>] = []
        let inserted = lock.withLock { () -> Bool in
            guard entries[id] == nil else { return false }
            entries[id] = entry
            entryBytes[id] = size
            totalBytes += size
            lruOrder.append(id)
            evictExcessLocked(into: &evictedTasks)
            return true
        }
        for t in evictedTasks { t.cancel() }
        return inserted
    }

    public func put(_ entry: StoredResponse) {
        let id = entry.resource.id
        let size = Self.estimateBytes(entry)
        var cancelledTasks: [Task<Void, Never>] = []
        lock.withLock {
            if let old = entries.removeValue(forKey: id) {
                if let t = old.task { cancelledTasks.append(t) }
                totalBytes -= entryBytes.removeValue(forKey: id) ?? 0
            }
            entries[id] = entry
            entryBytes[id] = size
            totalBytes += size
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
            }
            lruOrder.append(id)
            evictExcessLocked(into: &cancelledTasks)
        }
        for t in cancelledTasks { t.cancel() }
    }

    public func get(_ id: String) -> StoredResponse? {
        lock.withLock {
            guard let existing = entries[id] else { return nil }
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
                lruOrder.append(id)
            }
            return existing
        }
    }

    public func update(
        id: String,
        resource: OpenAIResponseResource,
        historyMessages: [ChatMessage]? = nil,
        sessionNodeId: String? = nil
    ) {
        var evictedTasks: [Task<Void, Never>] = []
        lock.withLock {
            guard var existing = entries[id] else { return }
            existing.resource = resource
            if let historyMessages {
                existing.historyMessages = historyMessages
            }
            if let sessionNodeId {
                existing.sessionNodeId = sessionNodeId
            }
            existing.task = nil
            let newSize = Self.estimateBytes(existing)
            let oldSize = entryBytes[id] ?? 0
            entries[id] = existing
            entryBytes[id] = newSize
            totalBytes += (newSize - oldSize)
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
                lruOrder.append(id)
            }
            evictExcessLocked(into: &evictedTasks)
        }
        for t in evictedTasks { t.cancel() }
    }

    public func attachTask(id: String, task: Task<Void, Never>) {
        let shouldCancel: Bool = lock.withLock {
            guard var existing = entries[id] else { return true }
            let active = existing.resource.status == "queued" || existing.resource.status == "in_progress"
            guard active else { return false }
            existing.task = task
            entries[id] = existing
            return false
        }
        if shouldCancel {
            task.cancel()
        }
    }

    @discardableResult
    public func delete(_ id: String) -> Bool {
        let outcome: (removed: Bool, task: Task<Void, Never>?) = lock.withLock {
            guard let removed = entries.removeValue(forKey: id) else { return (false, nil) }
            totalBytes -= entryBytes.removeValue(forKey: id) ?? 0
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
            }
            return (true, removed.task)
        }
        outcome.task?.cancel()
        return outcome.removed
    }

    private func evictExcessLocked(into evictedTasks: inout [Task<Void, Never>]) {
        while (lruOrder.count > capacity || totalBytes > maxBytes) && lruOrder.count > 1 {
            let oldest = lruOrder.removeFirst()
            totalBytes -= entryBytes.removeValue(forKey: oldest) ?? 0
            if let removed = entries.removeValue(forKey: oldest), let t = removed.task {
                evictedTasks.append(t)
            }
        }
        if lruOrder.count > capacity, let oldest = lruOrder.first {
            lruOrder.removeFirst()
            totalBytes -= entryBytes.removeValue(forKey: oldest) ?? 0
            if let removed = entries.removeValue(forKey: oldest), let t = removed.task {
                evictedTasks.append(t)
            }
        }
    }

    private static func estimateBytes(_ entry: StoredResponse) -> Int {
        let resBytes = (try? JSONEncoder().encode(entry.resource).count) ?? 256
        let histBytes = (try? JSONEncoder().encode(entry.historyMessages).count) ?? 0
        return max(64, resBytes + histBytes)
    }
}

// QWTelemetrySeries.swift — A bounded series of readings for one metric.
// Pure: a gap in the data stays a gap. A window is never back-filled with
// the mean of the windows around it.

import Foundation

/// A bounded, ordered series of readings for one metric.
public struct QWTelemetrySeries: Sendable, Equatable {
    /// One reading. A nil value is an explicit gap, not a zero.
    public struct Sample: Sendable, Equatable {
        public let at: Date
        public let value: Double?

        public init(at: Date, value: Double?) {
            self.at = at
            self.value = value
        }
    }

    private var buffer: [Sample] = []
    private let capacity: Int

    public init(capacity: Int = 300) {
        self.capacity = max(2, capacity)
    }

    /// Appends a reading; a nil value records an explicit gap.
    public mutating func append(_ value: Double?, at: Date = Date()) {
        buffer.append(Sample(at: at, value: value))
        if buffer.count > capacity { buffer.removeFirst(buffer.count - capacity) }
    }

    /// The readings, oldest first.
    public var samples: [Sample] { buffer }

    /// Groups the readings into fixed windows of `seconds`, anchored on the
    /// oldest reading. A window with no reading, or with an explicit gap in
    /// it, is nil — never an average that hides the gap.
    public func bucket(_ seconds: Double) -> [Double?] {
        guard seconds > 0, let origin = buffer.first?.at else { return [] }
        var sums: [Double] = []
        var counts: [Int] = []
        var gapped: [Bool] = []
        for sample in buffer {
            let offset = sample.at.timeIntervalSince(origin)
            guard offset >= 0 else { continue }
            let index = Int(offset / seconds)
            while sums.count <= index {
                sums.append(0)
                counts.append(0)
                gapped.append(false)
            }
            if let value = sample.value {
                sums[index] += value
                counts[index] += 1
            } else {
                gapped[index] = true
            }
        }
        return (0..<sums.count).map { index in
            guard counts[index] > 0, !gapped[index] else { return nil }
            return sums[index] / Double(counts[index])
        }
    }
}

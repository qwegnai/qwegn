// WinnerMap.swift — the emitter.
//
// The bench REUSES QWKernelDispatchTable from QWCore: one schema, one
// encoder, so the producer and the consumer cannot drift apart. Each entry
// carries the runner-up and the margin over it, so a win that was inside
// the measurement's noise says so — and names the candidate it beat.

import Foundation
import QWCore

enum WinnerMap {
    /// The shape a measurement belongs to.
    private struct ShapeKey: Hashable {
        let nBucket: Int
        let op: String
        let bits: Int
        let group: Int
    }

    /// Picks the fastest candidate for each (bucket, op, bits, group).
    static func make(from points: [BenchPoint]) -> QWKernelDispatchTable {
        var ranked: [ShapeKey: [(ms: Double, name: String)]] = [:]
        for p in points {
            let key = ShapeKey(nBucket: p.nBucket, op: p.op, bits: p.bits, group: p.group)
            ranked[key, default: []].append((p.gpuMilliseconds, p.candidate))
        }
        let entries = ranked.map { key, list -> QWKernelDispatchEntry in
            // Sorted by name as well as time: two candidates can land on the
            // same millisecond, and the winner must not be arbitrary.
            let sorted = list.sorted { ($0.ms, $0.name) < ($1.ms, $1.name) }
            let best = sorted[0]
            let second = sorted.count > 1 ? sorted[1] : nil
            return QWKernelDispatchEntry(
                nBucket: key.nBucket, op: key.op, bits: key.bits, group: key.group,
                winner: best.name,
                runnerUp: second?.name,
                margin: second.map { ($0.ms - best.ms) / $0.ms })
        }
        return QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                     entries: entries.sorted {
                                         ($0.nBucket, $0.op, $0.bits, $0.group)
                                             < ($1.nBucket, $1.op, $1.bits, $1.group)
                                     })
    }
}

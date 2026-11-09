// BenchReport.swift — the Markdown report.
//
// The JSON is the map the engine reads; this is the same measurement written
// for a human. One section per operator, with the dispatch counts, the
// verification errors and the range of every timing, so a reader can see what
// was measured and how firmly it was measured.

import Foundation
import QWCore

enum BenchReport {
    /// Renders the whole report.
    static func markdown(map: QWKernelDispatchTable, points: [BenchPoint],
                         suites: [OperatorSuites.OperatorResult],
                         moeSuites: [OperatorSuites.OperatorResult],
                         prefill: [PrefillSuites.Race] = [],
                         cols: Int, samples: Int) -> String {
        let suiteOps = Set(suites.map(\.op)).union(moeSuites.map(\.op))
        var out = "# qwg-bench-kernels\n\n"
        out += "Schema `\(map.schema)` — \(cols) columns, \(samples) samples per candidate, "
        out += "\(map.entries.count) entries in the map.\n\n"

        out += "## The family sweep\n\n"
        out += "| rows | op | bits | group | winner | gpu ms | runner-up | margin |\n"
        out += "|---|---|---|---|---|---|---|---|\n"
        for entry in map.entries where !suiteOps.contains(entry.op) {
            out += "| \(entry.nBucket) | \(entry.op) | \(entry.bits) | \(entry.group) | "
            out += "`\(entry.winner)` | \(ms(points, entry)) | "
            out += entry.runnerUp.map { "`\($0)`" } ?? "-"
            out += " | \(pct(entry.margin)) |\n"
        }
        out += "\n"

        if !prefill.isEmpty {
            out += "## prefill_gemm\n\n"
            out += "\(PrefillSuites.tokens) tokens × \(PrefillSuites.rows) rows × \(cols) columns. "
            out += "The family's prefill twin against the incumbent's, over the same logical "
            out += "weights. Bench-only: the engine picks the prefill kernel by container.\n\n"
            out += "| pair | family ms | incumbent ms | ratio | family error | incumbent error |\n"
            out += "|---|---|---|---|---|---|\n"
            for race in prefill {
                let ratio = race.family.milliseconds / race.incumbent.milliseconds
                out += "| \(race.bits)-bit G\(race.group) | "
                out += String(format: "%.4f", race.family.milliseconds) + " | "
                out += String(format: "%.4f", race.incumbent.milliseconds) + " | "
                out += String(format: "%.4f", ratio) + " | "
                out += sci(race.family.error) + " | " + sci(race.incumbent.error) + " |\n"
            }
            out += "\n"
        }

        for suite in suites + moeSuites {
            out += "## \(suite.op)\n\n"
            out += "\(suite.rows) rows × \(suite.cols) columns, "
            out += "\(OperatorSuites.bits)-bit G\(OperatorSuites.group).\n\n"
            out += suite.note + "\n\n"
            out += "| candidate | publishable | dispatches | slots | worst error | median ms | range ms |\n"
            out += "|---|---|---|---|---|---|---|\n"
            for c in suite.candidates.sorted(by: { $0.median < $1.median }) {
                out += "| `\(c.name)` | \(c.publishable ? "yes" : "no") | \(c.dispatches) "
                out += "| \(c.slots.map(String.init) ?? "-") "
                out += "| \(sci(c.error)) | "
                out += String(format: "%.4f", c.median) + " | "
                out += String(format: "%.4f–%.4f", c.low, c.high) + " |\n"
            }
            out += "\nSeparation \(pct2(suite.separation)), noise floor \(pct2(suite.noise)). "
            if suite.separable, let winner = published(map: map, suite: suite) {
                out += "Published: `\(winner.winner)` over `\(winner.runnerUp ?? "-")`, "
                out += "margin \(pct2(winner.margin ?? 0)), noise floor \(pct2(suite.noise)).\n\n"
            } else {
                out += "Not separable, so no entry is published and the engine keeps the path "
                    + "it already runs.\n\n"
            }
        }
        return out
    }

    /// The map's entry for a suite: which candidate won, over which.
    private static func published(map: QWKernelDispatchTable,
                                  suite: OperatorSuites.OperatorResult)
        -> QWKernelDispatchEntry? {
        map.entries.first { $0.op == suite.op && $0.nBucket == suite.rows }
    }

    /// The winner's own timing, from the points it was measured at.
    private static func ms(_ points: [BenchPoint], _ entry: QWKernelDispatchEntry) -> String {
        let point = points.first {
            $0.nBucket == entry.nBucket && $0.op == entry.op && $0.bits == entry.bits
                && $0.group == entry.group && $0.candidate == entry.winner
        }
        guard let point else { return "-" }
        return String(format: "%.4f", point.gpuMilliseconds)
    }

    /// A fraction as a percentage.
    private static func pct(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.1f%%", value * 100)
    }

    /// A fraction as a percentage, with the precision a thin margin needs.
    private static func pct2(_ value: Double) -> String {
        String(format: "%.2f%%", value * 100)
    }

    /// An error in the notation a reader can read.
    private static func sci(_ value: Double) -> String {
        value == 0 ? "0" : String(format: "%.1e", value)
    }
}

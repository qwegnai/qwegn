// QWKernelDispatchTable.swift — the measured winner map.
//
// The bench (qwg-bench-kernels) writes it; the engine plan reads it back.
// It is DATA, not control flow: the engine asks for a kernel name by shape
// and never inspects the map's contents.

import Foundation

/// One shape's measured winner: the kernel the bench found fastest.
public struct QWKernelDispatchEntry: Sendable, Hashable, Codable {
    /// The row-count bucket the measurement was taken at. The bucket is the
    /// GEMV's **output-row count** — the N extent, not M.
    public let nBucket: Int
    /// The operation the kernel serves: "gemv", "mlx_nax_gemm", and so on.
    public let op: String
    /// The quantisation width the kernel reads.
    public let bits: Int
    /// The quantisation group the kernel reads.
    public let group: Int
    /// The kernel name, exactly as the metallib spells it.
    public let winner: String
    /// The candidate that came second, when there was one.
    public let runnerUp: String?
    /// How much faster the winner was than the runner-up, as a fraction.
    /// A small margin means the two were within the measurement's noise.
    public let margin: Double?

    public init(nBucket: Int, op: String, bits: Int, group: Int, winner: String,
                runnerUp: String? = nil, margin: Double? = nil) {
        self.nBucket = nBucket
        self.op = op
        self.bits = bits
        self.group = group
        self.winner = winner
        self.runnerUp = runnerUp
        self.margin = margin
    }

    /// The identity of a shape. One entry per shape: the map is a table, not
    /// a log of every measurement.
    public var shapeKey: String { "\(nBucket)|\(op)|\(bits)|\(group)" }

    /// A stable rendering of the entry, for the plan's digest. Every field is
    /// length-prefixed so the rendering cannot be ambiguous.
    public var canonical: String {
        var out = "\(shapeKey.count):\(shapeKey)"
        out += "|\(winner.count):\(winner)"
        out += "|\(runnerUp.map { "\($0.count):\($0)" } ?? "-")"
        out += "|\(margin.map { String(format: "%016llx", $0.bitPattern) } ?? "-")"
        return out
    }

    private enum CodingKeys: String, CodingKey {
        case nBucket = "n_bucket"
        case op, bits, group, winner
        case runnerUp = "runner_up"
        case margin
    }
}

/// The measured winner map, written by `qwg-bench-kernels` and read back by
/// the engine plan. A shape is `(bucket, op, bits, group)`; the bucket is
/// the GEMV's output-row count, the only extent a decode GEMV has.
public struct QWKernelDispatchTable: Sendable, Codable, Equatable {
    public static let currentSchema = "qwg-bench-kernels/2"

    /// The operations the bench measures. The engine may ask for any of them.
    /// The bench asserts every operation it emits is on the list, so the two
    /// cannot drift; reading a map refuses no unknown operation.
    public static let knownOps = [
        // The family sweep.
        "gemv", "mlx_nax_gemm",
        // The per-operator suites.
        "gdn_inproj_decode", "mlp_swiglu", "attn_qkv", "lm_head",
        // The routed-MoE suites.
        "moe_gateup_decode", "moe_gateup_prefill",
    ]

    public let schema: String
    public let entries: [QWKernelDispatchEntry]

    public init(schema: String, entries: [QWKernelDispatchEntry]) {
        self.schema = schema
        self.entries = entries
    }

    public enum Error: Swift.Error, Equatable {
        case unsupportedSchema(String)
        case duplicateShape(String)
    }

    /// Reads a winner map, refusing one this build does not understand.
    public static func decode(_ data: Data) throws -> QWKernelDispatchTable {
        try JSONDecoder().decode(QWKernelDispatchTable.self, from: data)
    }

    /// The refusal lives here, not in the caller: every path that decodes a
    /// map — the bench's file, the plan's own encoding — goes through it.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decode(String.self, forKey: .schema)
        entries = try container.decode([QWKernelDispatchEntry].self, forKey: .entries)
        guard schema == Self.currentSchema else {
            throw Error.unsupportedSchema(schema)
        }
        var seen = Set<String>()
        for entry in entries {
            guard seen.insert(entry.shapeKey).inserted else {
                throw Error.duplicateShape(entry.shapeKey)
            }
        }
    }

    private enum CodingKeys: String, CodingKey { case schema, entries }

    public static func load(from url: URL) throws -> QWKernelDispatchTable {
        try decode(Data(contentsOf: url))
    }

    /// The winner for a shape: the fastest candidate measured at the largest
    /// bucket at or below `nBucket`. `nil` when the shape was never measured
    /// — the map is an optimisation input, so that means no optimisation.
    public func winner(nBucket: Int, op: String, bits: Int, group: Int) -> String? {
        let atOrBelow = entries.filter {
            $0.op == op && $0.bits == bits && $0.group == group && $0.nBucket <= nBucket
        }
        guard let best = atOrBelow.map(\.nBucket).max() else { return nil }
        // Two entries for one shape are a malformed map. The tie is broken by
        // name so the answer never depends on the order they arrived in.
        return atOrBelow.filter { $0.nBucket == best }.map(\.winner).min()
    }

    /// The same bytes the bench writes.
    public func encoded() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    /// A stable rendering of the table's **content**. Two tables holding the
    /// same entries in a different order render the same; two tables that
    /// select different kernels do not.
    public var canonical: String {
        var out = "\(schema.count):\(schema)"
        for entry in entries.sorted(by: { $0.canonical < $1.canonical }) {
            out += "|\(entry.canonical)"
        }
        return out
    }
}

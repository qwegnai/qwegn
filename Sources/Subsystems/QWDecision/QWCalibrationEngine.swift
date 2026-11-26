// QWCalibrationEngine.swift — Multi-format temperature scaling, PAV knot interpolation, and 10-decimal roundPy renormalization.
import Foundation
import QWCore

public struct QWPlattParameters: Sendable, Hashable, Codable, Equatable {
    public let scale: Double
    public let bias: Double

    public init(scale: Double, bias: Double) {
        self.scale = scale
        self.bias = bias
    }

    public func apply(_ logit: Double) -> Double {
        1.0 / (1.0 + exp(-(scale * logit + bias)))
    }
}

public struct QWCalibrationProfile: Sendable, Hashable, Codable, Equatable {
    public var temperatures: [String: Double]
    public var defaultTemperature: Double
    public var tables: [String: [[Double]]]
    public var chainTemperatures: [String: [String: Double]]
    public var platt: [String: QWPlattParameters]

    public init(
        temperatures: [String: Double] = [:],
        defaultTemperature: Double = 1.0,
        tables: [String: [[Double]]] = [:],
        chainTemperatures: [String: [String: Double]] = [:],
        platt: [String: QWPlattParameters] = [:]
    ) {
        self.temperatures = temperatures
        self.defaultTemperature = defaultTemperature
        self.tables = tables
        self.chainTemperatures = chainTemperatures
        self.platt = platt
    }

    /// Loads `calibration.json`, inline `config.json` temperatures, or `rl_agent_config.json` per-chain tables.
    public static func load(fromData data: Data) throws -> QWCalibrationProfile {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QWDecisionError.calibrationFailed("expected JSON object")
        }
        var temps: [String: Double] = [:]
        if let tObj = obj["temperatures"] as? [String: Any] {
            for (k, v) in tObj {
                if let d = (v as? NSNumber)?.doubleValue { temps[k] = d }
            }
        }
        var chains: [String: [String: Double]] = [:]
        if let cObj = obj["chain_temperatures"] as? [String: Any] ?? obj["temperature_by_options"] as? [String: Any] {
            for (chain, rawVal) in cObj {
                if let sub = rawVal as? [String: Any] {
                    var parsed: [String: Double] = [:]
                    for (k, v) in sub { if let d = (v as? NSNumber)?.doubleValue { parsed[k] = d } }
                    chains[chain] = parsed
                } else if let d = (rawVal as? NSNumber)?.doubleValue {
                    temps[chain] = d
                }
            }
        }
        if let agents = obj["chains"] as? [String: Any] {
            for (chainName, rawAgent) in agents {
                guard let agentDict = rawAgent as? [String: Any],
                      let tbo = agentDict["temperature_by_options"] as? [String: Any] else { continue }
                var parsed: [String: Double] = [:]
                for (k, v) in tbo { if let d = (v as? NSNumber)?.doubleValue { parsed[k] = d } }
                chains[chainName] = parsed
            }
        }
        var tables: [String: [[Double]]] = [:]
        if let rawTables = obj["tables"] as? [String: Any] {
            for (k, v) in rawTables {
                guard let rows = v as? [[Any]] else { continue }
                tables[k] = rows.compactMap { row in
                    guard row.count >= 2,
                          let x = (row[0] as? NSNumber)?.doubleValue,
                          let y = (row[1] as? NSNumber)?.doubleValue else { return nil }
                    return [x, y]
                }
            }
        }
        var plattMap: [String: QWPlattParameters] = [:]
        if let rawPlatt = obj["platt"] as? [String: Any] {
            for (k, v) in rawPlatt {
                guard let d = v as? [String: Any],
                      let a = (d["scale"] as? NSNumber)?.doubleValue ?? (d["a"] as? NSNumber)?.doubleValue,
                      let b = (d["bias"] as? NSNumber)?.doubleValue ?? (d["b"] as? NSNumber)?.doubleValue else { continue }
                plattMap[k] = QWPlattParameters(scale: a, bias: b)
            }
        }
        let defTemp = (obj["default"] as? NSNumber)?.doubleValue
            ?? (obj["default_temperature"] as? NSNumber)?.doubleValue
            ?? 1.0
        return QWCalibrationProfile(
            temperatures: temps,
            defaultTemperature: defTemp,
            tables: tables,
            chainTemperatures: chains,
            platt: plattMap
        )
    }

    public func mergingInlineTemperatures(_ inline: [String: Double]) -> QWCalibrationProfile {
        guard !inline.isEmpty else { return self }
        var merged = temperatures
        for (k, v) in inline where merged[k] == nil { merged[k] = v }
        return QWCalibrationProfile(
            temperatures: merged,
            defaultTemperature: merged["default"] ?? defaultTemperature,
            tables: tables,
            chainTemperatures: chainTemperatures,
            platt: platt
        )
    }
}

public enum QWCalibrationEngine {
    public static let roundDecimals = 10
    public static let massSlack = 1e-12
    public static let residualMax = 1e-9

    /// Matches `calibration.py` + `d1-omni` + Qwegn bucket keys in exact fallback order.
    public static func candidateBucketKeys(kind: QWDecisionKind, cardinality k: Int) -> [String] {
        switch kind {
        case .noul:
            return ["noul:2", "noul"]
        case .score:
            var keys = ["score:\(k)"]
            if k <= 5 { keys.append("score:3-5") }
            else if k <= 10 { keys.append("score:6-10") }
            else { keys.append("score:11+") }
            keys.append("score")
            return keys
        case .choice, .multiChoice:
            var keys = ["choice:\(k)"]
            if k <= 2 { keys.append("choice:2") }
            if k <= 3 { keys.append("choice:small") }
            else if k <= 8 { keys.append("choice:mid") }
            else if k <= 26 { keys.append("choice:large") }
            else { keys.append("choice:inventory") }
            if k >= 3 && k <= 5 { keys.append("choice:3-5") }
            else if k >= 6 && k <= 10 { keys.append("choice:6-10") }
            else if k >= 11 { keys.append("choice:11+") }
            keys.append("choice")
            return keys
        case .read, .estimate, .box, .boxes:
            return [kind.rawValue]
        }
    }

    public static func temperature(
        profile: QWCalibrationProfile,
        kind: QWDecisionKind,
        cardinality: Int,
        chain: String = "default"
    ) -> Double {
        let keys = candidateBucketKeys(kind: kind, cardinality: cardinality)
        if let chainMap = profile.chainTemperatures[chain] {
            for key in keys {
                if let t = chainMap[key], t > 0, t.isFinite { return t }
            }
            if let d = chainMap["default"], d > 0, d.isFinite { return d }
        }
        for key in keys {
            if let t = profile.temperatures[key], t > 0, t.isFinite { return t }
        }
        return profile.defaultTemperature > 0 ? profile.defaultTemperature : 1.0
    }

    /// Piecewise-linear knot lookup with flat clamping past ends (`calibration.py` `through`).
    public static func through(_ value: Double, knots: [[Double]]?) -> Double {
        guard let knots, !knots.isEmpty else { return value }
        if value <= knots[0][0] { return knots[0][1] }
        let last = knots.count - 1
        if value >= knots[last][0] { return knots[last][1] }
        var lo = 0
        var hi = knots.count
        while lo < hi {
            let mid = (lo + hi) >> 1
            if knots[mid][0] < value { lo = mid + 1 } else { hi = mid }
        }
        if lo == 0 { return knots[0][1] }
        if lo >= knots.count { return knots[last][1] }
        let a = knots[lo - 1][0], pa = knots[lo - 1][1]
        let b = knots[lo][0], pb = knots[lo][1]
        if abs(b - a) <= 1e-18 { return 0.5 * (pa + pb) }
        return pa + (value - a) * (pb - pa) / (b - a)
    }

    /// Reverse lookup on `(x, y)` knots for `Estimate` target quantile levels (`calibration.py` `level`).
    public static func level(_ share: Double, knots: [[Double]]?) -> Double {
        guard let knots, !knots.isEmpty else { return share }
        let swapped = knots.map { [$0[1], $0[0]] }
        return through(share, knots: swapped)
    }

    public static func calibratedSoftmax(logits: [Double], temperature: Double) -> [Double] {
        guard !logits.isEmpty else { return [] }
        let t = (temperature > 0 && temperature.isFinite) ? temperature : 1.0
        let scaled = t == 1.0 ? logits : logits.map { $0 / t }
        guard let maxVal = scaled.max() else { return [] }
        var exps = scaled.map { exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        guard sum > 0, sum.isFinite else {
            return [Double](repeating: 1.0 / Double(logits.count), count: logits.count)
        }
        for i in exps.indices { exps[i] /= sum }
        return exps
    }

    /// Decimal half-to-even rounding matching CPython `round(v, nd)` via Darwin printf.
    public static func roundPy(_ v: Double, digits nd: Int = roundDecimals) -> Double {
        if !v.isFinite || abs(v) > 1e15 { return v }
        let s = String(format: "%.\(nd)f", v)
        let d = Double(s) ?? v
        if d == 0.0 && s.hasPrefix("-") { return -0.0 }
        return d
    }

    /// Insertion-order argmax matching Qwegn & d1 tie-break semantics.
    public static func argmaxIndex(_ values: [Double]) -> Int {
        guard !values.isEmpty else { return 0 }
        var best = 0
        for i in 1..<values.count where values[i] > values[best] { best = i }
        return best
    }

    /// Rounds probabilities to 10 decimals and adjusts residual onto argmax within 1e-12.
    public static func renormalize(
        questionName: String,
        keys: [String],
        rawProbs: [Double]
    ) throws -> [QWDecisionProbability] {
        guard keys.count == rawProbs.count, !keys.isEmpty else {
            throw QWDecisionError.calibrationFailed("question \(questionName): key/prob length mismatch")
        }
        var vals = rawProbs.map { roundPy(min(1.0, max(0.0, $0.isFinite ? $0 : 0.0))) }
        var mass = vals.reduce(0, +)
        guard mass > 0 else {
            throw QWDecisionError.calibrationFailed("question \(questionName): zero probability mass")
        }
        if abs(mass - 1.0) > massSlack {
            vals = vals.map { roundPy($0 / mass) }
        }
        var converged = false
        for _ in 0..<8 {
            mass = vals.reduce(0, +)
            let err = 1.0 - mass
            if abs(err) <= massSlack { converged = true; break }
            let movable = vals.indices.filter { err > 0 ? vals[$0] < 1.0 : vals[$0] > 0.0 }
            guard !movable.isEmpty else {
                throw QWDecisionError.calibrationFailed("question \(questionName): no movable probabilities")
            }
            let step = err / Double(movable.count)
            guard abs(step) <= residualMax else {
                throw QWDecisionError.calibrationFailed("question \(questionName): residual \(err) exceeds max")
            }
            for idx in movable { vals[idx] += step }
        }
        guard converged else {
            throw QWDecisionError.calibrationFailed("question \(questionName): mass did not converge")
        }
        let leftover = 1.0 - vals.reduce(0, +)
        guard abs(leftover) <= massSlack else {
            throw QWDecisionError.calibrationFailed("question \(questionName): leftover \(leftover) > \(massSlack)")
        }
        if leftover != 0.0 {
            vals[argmaxIndex(vals)] += leftover
        }
        return zip(keys, vals).map { QWDecisionProbability(key: $0, value: $1) }
    }
}

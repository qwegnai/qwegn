// QWJSONConstraint+Speculation.swift — Lossless speculative probes, response-local
// JSON formatting style learning, and canonical structural draft proposals.

import Foundation
import QWHead

/// Tracks whether the current response uses compact (`{"k":`) or pretty-printed (`{\n  "k": `) JSON,
/// along with any trailing whitespace bytes already consumed in the current structural transition.
struct QWJSONFormattingStyle {
    var observedFirstKey = false
    var observedFirstColon = false
    var usesNewlines = false
    var indentUnit: [UInt8] = []
    var colonSpace = false
    var currentWhitespaceRun: [UInt8] = []
    private var pendingIndent: [UInt8] = []
    private var collectingIndent = false
    private var collectingColonWS = false

    mutating func observe(byte b: UInt8, beforeState state: QWJState) {
        let isStringMode: Bool = {
            if case .string = state.mode { return true }
            return false
        }()
        if !isStringMode && QWJState.isWhitespace(b) {
            if currentWhitespaceRun.count < 32 { currentWhitespaceRun.append(b) }
        } else {
            currentWhitespaceRun.removeAll(keepingCapacity: true)
        }

        if !observedFirstKey {
            if state.frames == nil, case .value = state.mode, b == UInt8(ascii: "{") {
                collectingIndent = true
                pendingIndent.removeAll(keepingCapacity: true)
                return
            }
            if collectingIndent {
                if QWJState.isWhitespace(b) {
                    if pendingIndent.count < 16 { pendingIndent.append(b) }
                } else if b == UInt8(ascii: "\"") || b == UInt8(ascii: "}") {
                    collectingIndent = false
                    observedFirstKey = true
                    if let nlIdx = pendingIndent.lastIndex(of: 0x0A) {
                        usesNewlines = true
                        indentUnit = Array(pendingIndent[(nlIdx + 1)...])
                    } else {
                        usesNewlines = false
                        indentUnit = pendingIndent
                    }
                } else {
                    collectingIndent = false
                }
            }
        }
        if !observedFirstColon {
            if case .colon = state.mode, b == UInt8(ascii: ":") {
                collectingColonWS = true
            } else if collectingColonWS {
                if b == 0x20 {
                    colonSpace = true
                    collectingColonWS = false
                    observedFirstColon = true
                } else if !QWJState.isWhitespace(b) {
                    colonSpace = false
                    collectingColonWS = false
                    observedFirstColon = true
                }
            }
        }
    }

    func indent(depth: Int) -> [UInt8] {
        guard usesNewlines else { return indentUnit }
        var out: [UInt8] = [0x0A]
        if depth > 0 && !indentUnit.isEmpty {
            out.reserveCapacity(1 + depth * indentUnit.count)
            for _ in 0..<depth { out.append(contentsOf: indentUnit) }
        }
        return out
    }

    func remainingIndent(depth: Int) -> [UInt8]? {
        let full = indent(depth: depth)
        if currentWhitespaceRun.isEmpty { return full }
        guard full.count >= currentWhitespaceRun.count,
              full.starts(with: currentWhitespaceRun) else {
            return nil
        }
        return Array(full.dropFirst(currentWhitespaceRun.count))
    }

    var colonSuffix: [UInt8] {
        colonSpace ? [UInt8(ascii: ":"), 0x20] : [UInt8(ascii: ":")]
    }
}

extension QWJSLiteralTrie {
    /// Returns the deterministic remaining byte suffix when `node` has a unique valid completion.
    func uniqueSuffix(from node: Int32, allowedPayloadMask: UInt64? = nil) -> [UInt8]? {
        var cur = node
        var out: [UInt8] = []
        while true {
            let term = terminal(cur)
            let termValid: Bool
            if term >= 0 {
                if let mask = allowedPayloadMask {
                    termValid = term < 64 && (mask & (UInt64(1) << UInt64(term))) != 0
                } else {
                    termValid = true
                }
            } else {
                termValid = false
            }
            var validEdges: [(UInt8, Int32)] = []
            for (b, child) in next[Int(cur)] {
                if let mask = allowedPayloadMask {
                    if (reachable[Int(child)] & mask) != 0 {
                        validEdges.append((b, child))
                    }
                } else {
                    validEdges.append((b, child))
                }
            }
            if termValid && validEdges.isEmpty {
                return out
            }
            guard !termValid && validEdges.count == 1 else {
                return out.isEmpty ? nil : out
            }
            out.append(validEdges[0].0)
            cur = validEdges[0].1
        }
    }
}

extension QWJSONTokenTable {
    /// Greedily tokenizes a prefix of `targetBytes` via longest-match walks in `fullTrie`,
    /// accepting only non-stop tokens validated by `acceptToken`.
    func greedyTokenizePrefix(
        bytes targetBytes: [UInt8],
        maxTokens: Int,
        excludedTokens: Set<Int> = [],
        acceptToken: (Int) -> Bool
    ) -> (tokens: [Int], consumedBytes: Int) {
        guard maxTokens > 0, !targetBytes.isEmpty else { return ([], 0) }
        var out: [Int] = []
        out.reserveCapacity(min(maxTokens, 16))
        var offset = 0
        while offset < targetBytes.count && out.count < maxTokens {
            var node: Int32 = 0
            var bestToken = -1
            var bestLen = 0
            let limit = targetBytes.count - offset
            for step in 0..<limit {
                guard let child = childNode(in: fullTrie, node: node, byte: targetBytes[offset + step]) else {
                    break
                }
                let tc = Int(fullTrie.tokenCount[Int(child)])
                if tc > 0 {
                    let ts = Int(fullTrie.tokenStart[Int(child)])
                    for i in ts..<(ts + tc) {
                        let cand = Int(fullTrie.tokens[i])
                        if cand >= 0 && cand < vocabulary && !excludedTokens.contains(cand) {
                            bestToken = cand
                            bestLen = step + 1
                            break
                        }
                    }
                }
                node = child
            }
            guard bestToken >= 0, bestLen > 0, acceptToken(bestToken) else { break }
            out.append(bestToken)
            offset += bestLen
        }
        return (out, offset)
    }

    private func childNode(in trie: Trie, node: Int32, byte: UInt8) -> Int32? {
        var lo = Int(trie.childStart[Int(node)])
        var hi = lo + Int(trie.childCount[Int(node)]) - 1
        while lo <= hi {
            let mid = lo + ((hi - lo) >> 1)
            let b = trie.childByte[mid]
            if b == byte { return trie.childNode[mid] }
            if b < byte { lo = mid + 1 } else { hi = mid - 1 }
        }
        return nil
    }
}

extension QWJState {
    var frameDepth: Int {
        var d = 0
        var cur = frames
        while let f = cur {
            d += 1
            cur = f.parent
        }
        return d
    }

    /// Predicts canonical structural JSON continuation bytes matching the response's observed style.
    func canonicalStructuralBytes(
        style: QWJSONFormattingStyle,
        hasNewlineTokens: Bool,
        maxBytes: Int = 96
    ) -> [UInt8] {
        var sim = self
        var simStyle = style
        var out: [UInt8] = []
        out.reserveCapacity(min(maxBytes, 64))
        while out.count < maxBytes {
            guard let chunk = sim.nextCanonicalChunk(style: simStyle, hasNewlineTokens: hasNewlineTokens),
                  !chunk.isEmpty else { break }
            var advancedAll = true
            for b in chunk {
                guard out.count < maxBytes else {
                    advancedAll = false
                    break
                }
                simStyle.observe(byte: b, beforeState: sim)
                guard sim.step(b) else {
                    advancedAll = false
                    break
                }
                out.append(b)
            }
            if !advancedAll { break }
        }
        return out
    }

    private func nextCanonicalChunk(style: QWJSONFormattingStyle, hasNewlineTokens: Bool) -> [UInt8]? {
        let styleReady = style.observedFirstKey || !hasNewlineTokens
        let depth = frameDepth
        switch mode {
        case .done, .string, .number:
            return nil
        case .colon:
            guard style.observedFirstColon || !hasNewlineTokens else { return nil }
            guard style.currentWhitespaceRun.isEmpty else { return nil }
            return style.colonSuffix
        case .literal(let trie, let node, let isKey):
            if isKey {
                guard let top = frames, case .object(let object, let seen, _) = top.kind else { return nil }
                if let suffix = trie.uniqueSuffix(from: node, allowedPayloadMask: ~seen), !suffix.isEmpty {
                    if suffix.last == UInt8(ascii: "\""), style.observedFirstColon || !hasNewlineTokens {
                        return suffix + style.colonSuffix
                    }
                    return suffix
                }
                let unseenReq = object.requiredMask & ~seen
                if node == 0, object.additional == nil, unseenReq != 0 {
                    let idx = unseenReq.trailingZeroBitCount
                    if idx < object.names.count {
                        let col = (style.observedFirstColon || !hasNewlineTokens) ? style.colonSuffix : [UInt8(ascii: ":")]
                        return Array((object.names[idx] + "\"").utf8) + col
                    }
                }
                return nil
            }
            return trie.uniqueSuffix(from: node)
        case .keyOrEnd(let first):
            guard styleReady, let top = frames, case .object(let object, let seen, _) = top.kind else { return nil }
            let unseenAll = object.allMask & ~seen
            if object.additional == nil && unseenAll == 0 {
                if first && style.currentWhitespaceRun.isEmpty { return [UInt8(ascii: "}")] }
                guard let rem = style.remainingIndent(depth: depth - 1) else { return nil }
                return rem + [UInt8(ascii: "}")]
            }
            let unseenReq = object.requiredMask & ~seen
            if object.additional == nil, unseenReq != 0 {
                let idx = unseenReq.trailingZeroBitCount
                if idx < object.names.count, let rem = style.remainingIndent(depth: depth) {
                    return rem
                        + Array(("\"" + object.names[idx] + "\"").utf8)
                        + style.colonSuffix
                }
            }
            return nil
        case .arrayStart:
            guard styleReady, let top = frames, case .array(let array, _) = top.kind else { return nil }
            if let maxItems = array.maxItems, maxItems == 0, style.currentWhitespaceRun.isEmpty {
                return [UInt8(ascii: "]")]
            }
            guard array.minItems >= 1,
                  let rem = style.remainingIndent(depth: depth),
                  let itemPrefix = Self.valueOpenBytes(array.items) else { return nil }
            return rem + itemPrefix
        case .afterValue:
            guard styleReady, style.currentWhitespaceRun.isEmpty, let top = frames else { return nil }
            switch top.kind {
            case .object(let object, let seen, _):
                let unseenAll = object.allMask & ~seen
                if object.additional == nil && unseenAll == 0 {
                    return style.indent(depth: depth - 1) + [UInt8(ascii: "}")]
                }
                let unseenReq = object.requiredMask & ~seen
                if object.additional == nil && unseenReq != 0 {
                    let idx = unseenReq.trailingZeroBitCount
                    if idx < object.names.count {
                        return [UInt8(ascii: ",")]
                            + style.indent(depth: depth)
                            + Array(("\"" + object.names[idx] + "\"").utf8)
                            + style.colonSuffix
                    }
                }
                return nil
            case .array(let array, let count):
                if let maxItems = array.maxItems, count >= maxItems, count >= array.minItems {
                    return style.indent(depth: depth - 1) + [UInt8(ascii: "]")]
                }
                if count < array.minItems, let itemPrefix = Self.valueOpenBytes(array.items) {
                    return [UInt8(ascii: ",")] + style.indent(depth: depth) + itemPrefix
                }
                return nil
            }
        case .value(let node):
            if let lit = node.literals {
                return lit.uniqueSuffix(from: 0)
            }
            guard styleReady else { return nil }
            if style.colonSpace && style.currentWhitespaceRun.isEmpty && !style.observedFirstColon {
                return nil
            }
            return Self.valueOpenBytes(node)
        }
    }

    private static func valueOpenBytes(_ node: QWJSNode) -> [UInt8]? {
        if let lit = node.literals { return lit.uniqueSuffix(from: 0) }
        let t = node.types
        let typeCount = (t.object != nil ? 1 : 0) + (t.array != nil ? 1 : 0)
            + (t.string ? 1 : 0) + ((t.number || t.integer) ? 1 : 0)
            + (t.boolean ? 1 : 0) + (t.null ? 1 : 0)
        guard typeCount == 1 else { return nil }
        if t.object != nil { return [UInt8(ascii: "{")] }
        if t.array != nil { return [UInt8(ascii: "[")] }
        if t.string { return [UInt8(ascii: "\"")] }
        if t.null { return Array("null".utf8) }
        return nil
    }
}

extension QWStructuredOutputConstraint {
    public var allowsSpeculation: Bool {
        QWConstrainedSpeculationGate.isEnabled || phase == .thinking || phase == .free
    }

    public var isUnconstrainedNow: Bool {
        phase == .thinking || phase == .free
    }

    public var speculationBoundaryToken: Int? {
        switch phase {
        case .thinking: return endThinkToken
        case .choosing:
            return toolPrefixMatched < toolCallOpenTokens.count ? toolCallOpenTokens[toolPrefixMatched] : nil
        case .json, .free: return nil
        }
    }

    public func makeSpeculativeProbe() -> (any QWTokenConstraint)? {
        QWStructuredOutputConstraint(cloning: self)
    }

    public func proposeCanonicalDrafts(maxCount: Int) -> [Int] {
        guard QWConstrainedSpeculationGate.isEnabled, maxCount > 0 else { return [] }
        if phase == .choosing && toolPrefixMatched > 0 && toolPrefixMatched < toolCallOpenTokens.count {
            return Array(toolCallOpenTokens[toolPrefixMatched...].prefix(maxCount))
        }
        guard phase == .json else { return [] }
        let probe = QWStructuredOutputConstraint(cloning: self)
        let targetBytes = probe.state.canonicalStructuralBytes(
            style: probe.style,
            hasNewlineTokens: table.hasNewlineTokens,
            maxBytes: max(32, maxCount * 16)
        )
        guard !targetBytes.isEmpty else { return [] }
        let tokenized = table.greedyTokenizePrefix(
            bytes: targetBytes,
            maxTokens: maxCount,
            excludedTokens: stopTokenSet
        ) { tok in
            guard (try? probe.accept(tok)) != nil else { return false }
            return true
        }
        return tokenized.tokens
    }
}

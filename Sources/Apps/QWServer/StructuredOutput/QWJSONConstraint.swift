// QWJSONConstraint.swift — byte-level JSON parsing state, the tokenizer's
// byte trie, and the request-local structured-output token constraint.
//
// The parser state is a small value with an immutable frame stack, so the
// mask walk can fork it at every trie node cheaply. A token is allowed only
// if all of its bytes keep the parse alive; the parse never reaches a state
// with no completion, so the mask is never empty before the value ends.

import Foundation
import QWHead
import QWToken

/// One open container. Frames are immutable; a change makes a new frame.
final class QWJFrame: @unchecked Sendable {
    enum Kind {
        /// `seen` holds named keys used so far; `pending` is the value's key
        /// index (-1: an additional key).
        case object(QWJSObject, seen: UInt64, pending: Int)
        case array(QWJSArray, count: Int)
    }
    let kind: Kind
    let parent: QWJFrame?
    init(_ kind: Kind, parent: QWJFrame?) { self.kind = kind; self.parent = parent }
}

struct QWJString {
    var isKey: Bool
    /// Position in the key trie while the key still names a property (-1 otherwise).
    var keyNode: Int32
    var escape = false
    var hexLeft: UInt8 = 0
    var utf8Left: UInt8 = 0
    var nextLo: UInt8 = 0x80
    var nextHi: UInt8 = 0xBF
}

struct QWJState {
    enum Mode {
        case value(QWJSNode)
        case afterValue
        case keyOrEnd(first: Bool)
        case colon
        case arrayStart
        case string(QWJString)
        case literal(QWJSLiteralTrie, node: Int32, isKey: Bool)
        case number(state: UInt8, length: UInt8, integerOnly: Bool)
        case done
    }

    static let maxWhitespace: UInt8 = 64
    static let maxNumberLength: UInt8 = 32

    var frames: QWJFrame?
    var mode: Mode
    var whitespace: UInt8 = 0

    init(root: QWJSNode) { mode = .value(root) }

    var isDone: Bool { if case .done = mode { return true }; return false }

    /// Inside a free (non-key-matching) string with nothing pending: any
    /// byte run without quote, backslash, control or partial UTF-8 keeps it here.
    var isPlainString: Bool {
        if case .string(let s) = mode {
            return (!s.isKey || s.keyNode < 0) && !s.escape && s.hexLeft == 0 && s.utf8Left == 0
        }
        return false
    }

    /// Whether the document may end here (a root number or literal ends at EOS).
    var canEnd: Bool {
        var probe = self
        return probe.finish()
    }

    mutating func finish() -> Bool {
        switch mode {
        case .done: return true
        case .number(let state, _, _) where frames == nil:
            guard Self.numberAccepts(state) else { return false }
            mode = .done
            return true
        case .literal(let trie, let node, false) where frames == nil:
            guard trie.terminal(node) >= 0 else { return false }
            mode = .done
            return true
        default: return false
        }
    }

    @inline(__always) static func isWhitespace(_ b: UInt8) -> Bool {
        b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09
    }

    private mutating func takeWhitespace() -> Bool {
        guard whitespace < Self.maxWhitespace else { return false }
        whitespace += 1
        return true
    }

    private mutating func valueCompleted() {
        mode = frames == nil ? .done : .afterValue
    }

    private mutating func startValue(_ node: QWJSNode, _ b: UInt8) -> Bool {
        if let lit = node.literals, let child = lit.child(0, b) {
            mode = .literal(lit, node: child, isKey: false)
            if lit.isLeaf(child) && lit.terminal(child) >= 0 { valueCompleted() }
            return true
        }
        let t = node.types
        switch b {
        case UInt8(ascii: "{"):
            guard let object = t.object else { return false }
            frames = QWJFrame(.object(object, seen: 0, pending: -1), parent: frames)
            mode = .keyOrEnd(first: true)
        case UInt8(ascii: "["):
            guard let array = t.array else { return false }
            frames = QWJFrame(.array(array, count: 0), parent: frames)
            mode = .arrayStart
        case UInt8(ascii: "\""):
            guard t.string else { return false }
            mode = .string(QWJString(isKey: false, keyNode: -1))
        case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
            guard t.number || t.integer else { return false }
            mode = .number(state: 0, length: 0, integerOnly: !t.number)
            return step(b)
        case UInt8(ascii: "t"), UInt8(ascii: "f"):
            guard t.boolean else { return false }
            mode = .literal(b == UInt8(ascii: "t") ? QWJSONGrammar.trueTrie : QWJSONGrammar.falseTrie,
                            node: 0, isKey: false)
            return step(b)
        case UInt8(ascii: "n"):
            guard t.null else { return false }
            mode = .literal(QWJSONGrammar.nullTrie, node: 0, isKey: false)
            return step(b)
        default:
            return false
        }
        return true
    }

    /// Keys may still be added to the top object: a free key, or an unseen name.
    private static func canAddKey(_ object: QWJSObject, seen: UInt64) -> Bool {
        object.additional != nil || (object.allMask & ~seen) != 0
    }

    private mutating func keyFinished(index: Int) -> Bool {
        guard let top = frames, case .object(let object, let seen, _) = top.kind else { return false }
        var newSeen = seen
        if index >= 0 {
            let bit = UInt64(1) << UInt64(index)
            guard seen & bit == 0 else { return false }
            newSeen |= bit
        } else if object.additional == nil {
            return false
        }
        frames = QWJFrame(.object(object, seen: newSeen, pending: index), parent: top.parent)
        mode = .colon
        return true
    }

    private mutating func closeContainer(_ b: UInt8) -> Bool {
        guard let top = frames else { return false }
        switch top.kind {
        case .object(let object, let seen, _):
            guard b == UInt8(ascii: "}"), object.requiredMask & ~seen == 0 else { return false }
        case .array(let array, let count):
            guard b == UInt8(ascii: "]"), count >= array.minItems else { return false }
        }
        frames = top.parent
        valueCompleted()
        return true
    }

    static func numberAccepts(_ state: UInt8) -> Bool {
        state == 2 || state == 3 || state == 5 || state == 8
    }

    /// JSON number DFA. States: 0 start, 1 '-', 2 leading 0, 3 int digits,
    /// 4 '.', 5 fraction digits, 6 'e', 7 exponent sign, 8 exponent digits.
    static func numberNext(_ state: UInt8, _ b: UInt8, integerOnly: Bool) -> UInt8? {
        let digit = b >= UInt8(ascii: "0") && b <= UInt8(ascii: "9")
        switch state {
        case 0:
            if b == UInt8(ascii: "-") { return 1 }
            if b == UInt8(ascii: "0") { return 2 }
            return digit ? 3 : nil
        case 1:
            if b == UInt8(ascii: "0") { return 2 }
            return digit ? 3 : nil
        case 2, 3:
            if state == 3 && digit { return 3 }
            if integerOnly { return nil }
            if b == UInt8(ascii: ".") { return 4 }
            if b == UInt8(ascii: "e") || b == UInt8(ascii: "E") { return 6 }
            return nil
        case 4: return digit ? 5 : nil
        case 5:
            if digit { return 5 }
            if b == UInt8(ascii: "e") || b == UInt8(ascii: "E") { return 6 }
            return nil
        case 6:
            if b == UInt8(ascii: "+") || b == UInt8(ascii: "-") { return 7 }
            return digit ? 8 : nil
        case 7, 8: return digit ? 8 : nil
        default: return nil
        }
    }

    /// Advances one byte. False means the byte cannot continue any valid document.
    mutating func step(_ b: UInt8) -> Bool {
        while true {
            switch mode {
            case .done:
                return false

            case .value(let node):
                if Self.isWhitespace(b) { return takeWhitespace() }
                whitespace = 0
                return startValue(node, b)

            case .afterValue:
                if Self.isWhitespace(b) { return takeWhitespace() }
                whitespace = 0
                guard let top = frames else { return false }
                if b == UInt8(ascii: ",") {
                    switch top.kind {
                    case .object(let object, let seen, _):
                        guard Self.canAddKey(object, seen: seen) else { return false }
                        mode = .keyOrEnd(first: false)
                    case .array(let array, let count):
                        if let maxItems = array.maxItems, count >= maxItems { return false }
                        frames = QWJFrame(.array(array, count: count + 1), parent: top.parent)
                        mode = .value(array.items)
                    }
                    return true
                }
                return closeContainer(b)

            case .keyOrEnd(let first):
                if Self.isWhitespace(b) { return takeWhitespace() }
                whitespace = 0
                guard let top = frames, case .object(let object, let seen, _) = top.kind else { return false }
                if b == UInt8(ascii: "\"") {
                    guard Self.canAddKey(object, seen: seen) else { return false }
                    if object.additional == nil {
                        mode = .literal(object.keyTrie, node: 0, isKey: true)
                    } else {
                        mode = .string(QWJString(isKey: true, keyNode: object.names.isEmpty ? -1 : 0))
                    }
                    return true
                }
                guard first else { return false }
                return closeContainer(b)

            case .colon:
                if Self.isWhitespace(b) { return takeWhitespace() }
                whitespace = 0
                guard b == UInt8(ascii: ":"), let top = frames,
                      case .object(let object, _, let pending) = top.kind else { return false }
                guard let node = pending >= 0 ? object.nodes[pending] : object.additional else { return false }
                mode = .value(node)
                return true

            case .arrayStart:
                if Self.isWhitespace(b) { return takeWhitespace() }
                whitespace = 0
                guard let top = frames, case .array(let array, _) = top.kind else { return false }
                if b == UInt8(ascii: "]") { return closeContainer(b) }
                if let maxItems = array.maxItems, maxItems == 0 { return false }
                frames = QWJFrame(.array(array, count: 1), parent: top.parent)
                mode = .value(array.items)
                continue

            case .string(var s):
                if s.escape {
                    s.escape = false
                    switch b {
                    case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"), UInt8(ascii: "b"),
                         UInt8(ascii: "f"), UInt8(ascii: "n"), UInt8(ascii: "r"), UInt8(ascii: "t"):
                        break
                    case UInt8(ascii: "u"):
                        s.hexLeft = 4
                    default:
                        return false
                    }
                    mode = .string(s)
                    return true
                }
                if s.hexLeft > 0 {
                    let hex = (b >= 0x30 && b <= 0x39) || (b >= 0x41 && b <= 0x46) || (b >= 0x61 && b <= 0x66)
                    guard hex else { return false }
                    s.hexLeft -= 1
                    mode = .string(s)
                    return true
                }
                if s.utf8Left > 0 {
                    guard b >= s.nextLo && b <= s.nextHi else { return false }
                    s.utf8Left -= 1
                    s.nextLo = 0x80
                    s.nextHi = 0xBF
                    if s.isKey && s.keyNode >= 0 { s.keyNode = keyTrieChild(s.keyNode, b) }
                    mode = .string(s)
                    return true
                }
                if b == UInt8(ascii: "\"") {
                    if s.isKey {
                        var index = -1
                        if s.keyNode >= 0, let top = frames, case .object(let object, _, _) = top.kind,
                           let end = object.keyTrie.child(s.keyNode, b) {
                            index = Int(object.keyTrie.terminal(end))
                        }
                        return keyFinished(index: index)
                    }
                    valueCompleted()
                    return true
                }
                if b == UInt8(ascii: "\\") {
                    s.escape = true
                    if s.isKey { s.keyNode = -1 }
                    mode = .string(s)
                    return true
                }
                if b < 0x20 { return false }
                if b >= 0x80 {
                    switch b {
                    case 0xC2...0xDF: s.utf8Left = 1
                    case 0xE0: s.utf8Left = 2; s.nextLo = 0xA0
                    case 0xED: s.utf8Left = 2; s.nextHi = 0x9F
                    case 0xE1...0xEF: s.utf8Left = 2
                    case 0xF0: s.utf8Left = 3; s.nextLo = 0x90
                    case 0xF4: s.utf8Left = 3; s.nextHi = 0x8F
                    case 0xF1...0xF3: s.utf8Left = 3
                    default: return false
                    }
                }
                if s.isKey && s.keyNode >= 0 { s.keyNode = keyTrieChild(s.keyNode, b) }
                mode = .string(s)
                return true

            case .literal(let trie, let node, let isKey):
                if let child = trie.child(node, b) {
                    if isKey {
                        guard let top = frames, case .object(_, let seen, _) = top.kind,
                              trie.reachable[Int(child)] & ~seen != 0 else { return false }
                        if trie.isLeaf(child) { return keyFinished(index: Int(trie.terminal(child))) }
                    }
                    mode = .literal(trie, node: child, isKey: isKey)
                    if !isKey && trie.isLeaf(child) && trie.terminal(child) >= 0 { valueCompleted() }
                    return true
                }
                guard !isKey, trie.terminal(node) >= 0 else { return false }
                valueCompleted()
                continue

            case .number(let state, let length, let integerOnly):
                if let next = Self.numberNext(state, b, integerOnly: integerOnly) {
                    guard length < Self.maxNumberLength else { return false }
                    mode = .number(state: next, length: length + 1, integerOnly: integerOnly)
                    return true
                }
                guard Self.numberAccepts(state) else { return false }
                valueCompleted()
                continue
            }
        }
    }

    private func keyTrieChild(_ node: Int32, _ b: UInt8) -> Int32 {
        guard let top = frames, case .object(let object, _, _) = top.kind else { return -1 }
        return object.keyTrie.child(node, b) ?? -1
    }
}

/// The tokenizer's vocabulary as a byte trie (CSR), built once per tokenizer.
public final class QWJSONTokenTable: @unchecked Sendable {
    struct Trie {
        var childStart: [Int32] = []
        var childCount: [UInt16] = []
        var tokenStart: [Int32] = []
        var tokenCount: [UInt16] = []
        var childByte: [UInt8] = []
        var childNode: [Int32] = []
        var tokens: [Int32] = []
    }

    let bytes: [[UInt8]?]
    let vocabulary: Int
    /// Every token that keeps a free string in its plain state, ascending.
    let plainStringTokens: [Int]
    let fullTrie: Trie
    /// Tokens not in `plainStringTokens` (walked from a plain string state).
    let stringEdgeTrie: Trie

    public convenience init(tokenizer: QWTokenizer, vocabulary: Int) {
        self.init(tokenBytes: tokenizer.tokenByteTable(), vocabulary: vocabulary)
    }

    public init(tokenBytes: [[UInt8]?], vocabulary: Int) {
        var table = Array(tokenBytes.prefix(vocabulary))
        if table.count < vocabulary { table += [[UInt8]?](repeating: nil, count: vocabulary - table.count) }
        bytes = table
        self.vocabulary = vocabulary
        var plain: [Int] = []
        var edge: [Int] = []
        var all: [Int] = []
        for (id, entry) in table.enumerated() {
            guard let b = entry, !b.isEmpty else { continue }
            all.append(id)
            if Self.isPlainStringToken(b) { plain.append(id) } else { edge.append(id) }
        }
        plainStringTokens = plain
        fullTrie = Self.buildTrie(all, bytes: table)
        stringEdgeTrie = Self.buildTrie(edge, bytes: table)
    }

    private static func isPlainStringToken(_ b: [UInt8]) -> Bool {
        var state = QWJState(root: QWJSNode())
        state.mode = .string(QWJString(isKey: false, keyNode: -1))
        state.frames = QWJFrame(.array(QWJSArray(items: QWJSNode(), minItems: 0, maxItems: nil), count: 1), parent: nil)
        for byte in b {
            if byte == UInt8(ascii: "\"") || byte == UInt8(ascii: "\\") { return false }
            guard state.step(byte) else { return false }
        }
        return state.isPlainString
    }

    private static func buildTrie(_ ids: [Int], bytes: [[UInt8]?]) -> Trie {
        let sorted = ids.sorted { bytes[$0]!.lexicographicallyPrecedes(bytes[$1]!) }
        var trie = Trie()
        func build(_ lo: Int, _ hi: Int, _ depth: Int) -> Int32 {
            let node = Int32(trie.childStart.count)
            trie.childStart.append(0)
            trie.childCount.append(0)
            trie.tokenStart.append(Int32(trie.tokens.count))
            trie.tokenCount.append(0)
            var i = lo
            while i < hi && bytes[sorted[i]]!.count == depth {
                trie.tokens.append(Int32(sorted[i]))
                i += 1
            }
            trie.tokenCount[Int(node)] = UInt16(i - lo)
            var groups: [(UInt8, Int, Int)] = []
            while i < hi {
                let b = bytes[sorted[i]]![depth]
                var j = i + 1
                while j < hi && bytes[sorted[j]]![depth] == b { j += 1 }
                groups.append((b, i, j))
                i = j
            }
            let start = trie.childByte.count
            trie.childStart[Int(node)] = Int32(start)
            trie.childCount[Int(node)] = UInt16(groups.count)
            trie.childByte.append(contentsOf: groups.map(\.0))
            trie.childNode.append(contentsOf: [Int32](repeating: 0, count: groups.count))
            for (k, g) in groups.enumerated() {
                trie.childNode[start + k] = build(g.1, g.2, depth + 1)
            }
            return node
        }
        _ = build(0, sorted.count, 0)
        return trie
    }

    /// Appends every token whose bytes keep `state` alive.
    func collect(_ trie: Trie, from state: QWJState, into out: inout [Int]) {
        func walk(_ node: Int32, _ state: QWJState) {
            let start = Int(trie.childStart[Int(node)])
            let end = start + Int(trie.childCount[Int(node)])
            for k in start..<end {
                var next = state
                guard next.step(trie.childByte[k]) else { continue }
                let child = Int(trie.childNode[k])
                let ts = Int(trie.tokenStart[child])
                for t in ts..<(ts + Int(trie.tokenCount[child])) { out.append(Int(trie.tokens[t])) }
                walk(Int32(child), next)
            }
        }
        walk(0, state)
    }

    func allowed(from state: QWJState, stopTokens: [Int]) -> [Int] {
        var extra: [Int] = []
        if state.isPlainString {
            collect(stringEdgeTrie, from: state, into: &extra)
        } else if !state.isDone {
            collect(fullTrie, from: state, into: &extra)
        }
        if state.canEnd { extra.append(contentsOf: stopTokens) }
        extra.sort()
        guard state.isPlainString else { return extra }
        // Both lists are ascending and disjoint (stop ids are special tokens).
        var out: [Int] = []
        out.reserveCapacity(plainStringTokens.count + extra.count)
        var j = 0
        for id in plainStringTokens {
            while j < extra.count && extra[j] < id { out.append(extra[j]); j += 1 }
            out.append(id)
        }
        out.append(contentsOf: extra[j...])
        return out
    }
}

/// The structured-output constraint for one request: optional thinking phase, optional
/// tool-call choice, then the constrained JSON document followed by a stop token.
public final class QWStructuredOutputConstraint: QWTokenConstraint, @unchecked Sendable {
    enum Phase { case thinking, choosing, json, free }

    private let table: QWJSONTokenTable
    private let stopTokens: [Int]
    private let endThinkToken: Int?
    private let toolCallToken: Int?
    private var phase: Phase
    private(set) var state: QWJState
    /// Within one plain string the mask cannot change; it is keyed by the open frame.
    private var plainCache: (frame: QWJFrame?, isKey: Bool, allowed: [Int])?

    public init(grammar: QWJSONGrammar, table: QWJSONTokenTable, stopTokens: Set<Int>,
                startsInThinking: Bool = false, endThinkToken: Int? = nil,
                toolCallToken: Int? = nil) {
        self.table = table
        self.stopTokens = stopTokens.filter { $0 >= 0 && $0 < table.vocabulary }.sorted()
        self.endThinkToken = endThinkToken
        self.toolCallToken = toolCallToken
        self.state = QWJState(root: grammar.root)
        if startsInThinking && endThinkToken != nil {
            phase = .thinking
        } else {
            phase = toolCallToken != nil ? .choosing : .json
        }
    }

    /// Feeds text the model is continuing from (an assistant prefill).
    public func consumePrefix(_ text: String) throws {
        guard phase == .json || phase == .choosing else { return }
        for b in text.utf8 {
            guard state.step(b) else {
                throw QWJSONSchemaError("the assistant prefill is not a valid prefix of the requested JSON")
            }
        }
        if !isAtRootStart { phase = .json }
    }

    /// Whether the JSON document is complete.
    public var isComplete: Bool { state.isDone }

    private var isAtRootStart: Bool {
        if state.frames == nil, case .value = state.mode { return true }
        return false
    }

    public func allowedTokens() throws -> [Int]? {
        switch phase {
        case .thinking, .free:
            return nil
        case .json, .choosing:
            if state.isPlainString, case .string(let s) = state.mode, let cache = plainCache,
               cache.frame === state.frames, cache.isKey == s.isKey {
                return cache.allowed
            }
            var allowed = table.allowed(from: state, stopTokens: stopTokens)
            if state.isPlainString, case .string(let s) = state.mode {
                plainCache = (state.frames, s.isKey, allowed)
            }
            if phase == .choosing, let tool = toolCallToken {
                let at = allowed.firstIndex(where: { $0 >= tool }) ?? allowed.count
                if at == allowed.count || allowed[at] != tool { allowed.insert(tool, at: at) }
            }
            guard !allowed.isEmpty else { throw QWTokenConstraintError.emptySupport }
            return allowed
        }
    }

    public func accept(_ token: Int) throws {
        switch phase {
        case .free:
            return
        case .thinking:
            if token == endThinkToken { phase = toolCallToken != nil ? .choosing : .json }
            return
        case .choosing, .json:
            if phase == .choosing && token == toolCallToken {
                phase = .free
                return
            }
            if stopTokens.contains(token) {
                guard state.canEnd else { throw QWTokenConstraintError.rejectedToken(token) }
                _ = state.finish()
                return
            }
            guard token >= 0 && token < table.vocabulary, let bytes = table.bytes[token], !bytes.isEmpty else {
                throw QWTokenConstraintError.rejectedToken(token)
            }
            for b in bytes {
                guard state.step(b) else { throw QWTokenConstraintError.rejectedToken(token) }
            }
            if phase == .choosing && !isAtRootStart { phase = .json }
        }
    }
}

// QWToolCallConstraint.swift — request-local token constraint that restricts
// tool-call function and parameter names inside tool blocks across Qwen XML,
// GCML (<｜GCML｜tool_calls>), GLM XML, and JSON formats to declared schemas.

import Foundation
import QWCore
import QWHead
import QWToken

/// Constrains tool and parameter names inside `<tool_call>` / `<｜GCML｜tool_calls>` blocks
/// to the request's declared tool names and parameter schemas.
public final class QWToolCallConstraint: QWTokenConstraint, @unchecked Sendable {
    private enum TagKind {
        case header
        case function
        case parameter
        case argKey
        case jsonFunction
    }

    private static let endThinkUTF8 = Array("</think>".utf8)
    private static let gcmlInvokePrefix = "<｜GCML｜invoke name=\""
    private static let gcmlInvokeOpenUTF8 = Array("<｜GCML｜invoke".utf8)
    private static let gcmlInvokeCloseUTF8 = Array("</｜GCML｜invoke>".utf8)
    private static let gcmlParamOpenUTF8 = Array("<｜GCML｜parameter".utf8)

    private let table: QWJSONTokenTable
    private let toolCallOpenTokens: [Int]
    private let toolCallEndToken: Int?
    private let requireToolCall: Bool
    private let endThinkToken: Int?
    private let toolCloseBytesList: [[UInt8]]
    private let toolClosePrefixes: [String]
    private let endThinkBytesList: [[UInt8]]
    private let recentCap: Int
    private let recentThinkCap: Int
    private let headerTargets: [[UInt8]]
    private let toolTargets: [[UInt8]]
    private let xmlInvokeTargets: [[UInt8]]
    private let jsonToolTargets: [[UInt8]]
    private let parameterTargetsByTool: [String: [[UInt8]]]
    private let gcmlParameterTargetsByTool: [String: [[UInt8]]]
    private let argKeyTargetsByTool: [String: [[UInt8]]]

    private var inThinking: Bool
    private var hasStartedToolCall = false
    private var openPrefixMatched = 0
    private var recentThinkBytes: [UInt8] = []
    private var inToolCall = false
    private var currentToolName: String?
    private var recentBytes: [UInt8] = []
    private var activeTag: (kind: TagKind, targets: [[UInt8]], prefix: [UInt8])?

    private var mustEmitToolCallNow: Bool {
        requireToolCall && !hasStartedToolCall && !inThinking && !toolCallOpenTokens.isEmpty
    }

    public var allowsSpeculation: Bool {
        QWConstrainedSpeculationGate.isEnabled ? true : !mustEmitToolCallNow
    }
    public var isUnconstrainedNow: Bool { !inToolCall && activeTag == nil && !mustEmitToolCallNow }
    public var speculationBoundaryToken: Int? {
        if requireToolCall && !hasStartedToolCall && inThinking {
            return endThinkToken
        }
        return openPrefixMatched < toolCallOpenTokens.count ? toolCallOpenTokens[openPrefixMatched] : toolCallOpenTokens.first
    }

    private init(cloning other: QWToolCallConstraint) {
        self.table = other.table
        self.toolCallOpenTokens = other.toolCallOpenTokens
        self.toolCallEndToken = other.toolCallEndToken
        self.requireToolCall = other.requireToolCall
        self.endThinkToken = other.endThinkToken
        self.toolCloseBytesList = other.toolCloseBytesList
        self.toolClosePrefixes = other.toolClosePrefixes
        self.endThinkBytesList = other.endThinkBytesList
        self.recentCap = other.recentCap
        self.recentThinkCap = other.recentThinkCap
        self.headerTargets = other.headerTargets
        self.toolTargets = other.toolTargets
        self.xmlInvokeTargets = other.xmlInvokeTargets
        self.jsonToolTargets = other.jsonToolTargets
        self.parameterTargetsByTool = other.parameterTargetsByTool
        self.gcmlParameterTargetsByTool = other.gcmlParameterTargetsByTool
        self.argKeyTargetsByTool = other.argKeyTargetsByTool
        self.inThinking = other.inThinking
        self.hasStartedToolCall = other.hasStartedToolCall
        self.openPrefixMatched = other.openPrefixMatched
        self.recentThinkBytes = other.recentThinkBytes
        self.inToolCall = other.inToolCall
        self.currentToolName = other.currentToolName
        self.recentBytes = other.recentBytes
        self.activeTag = other.activeTag
    }

    public func makeSpeculativeProbe() -> (any QWTokenConstraint)? {
        QWToolCallConstraint(cloning: self)
    }

    public func proposeCanonicalDrafts(maxCount: Int) -> [Int] {
        guard QWConstrainedSpeculationGate.isEnabled, maxCount > 0 else { return [] }
        let probe = QWToolCallConstraint(cloning: self)
        var out: [Int] = []
        while out.count < maxCount && probe.mustEmitToolCallNow,
              probe.openPrefixMatched < probe.toolCallOpenTokens.count {
            let startTok = probe.toolCallOpenTokens[probe.openPrefixMatched]
            guard (try? probe.accept(startTok)) != nil else { return out }
            out.append(startTok)
        }
        while out.count < maxCount, probe.inToolCall, let tag = probe.activeTag {
            let allMatching = tag.targets.filter { $0.starts(with: tag.prefix) }
            let preferred: [[UInt8]]
            if tag.prefix.isEmpty && (tag.kind == .parameter || tag.kind == .function) {
                let eqForms = allMatching.filter { $0.first == UInt8(ascii: "=") }
                preferred = eqForms.isEmpty ? allMatching : eqForms
            } else {
                preferred = allMatching
            }
            guard let first = preferred.first else { break }
            var lcpEnd = first.count
            for other in preferred.dropFirst() {
                let bound = min(lcpEnd, other.count)
                var idx = tag.prefix.count
                while idx < bound && first[idx] == other[idx] { idx += 1 }
                lcpEnd = idx
                if lcpEnd <= tag.prefix.count { break }
            }
            guard lcpEnd > tag.prefix.count else { break }
            let suffix = Array(first[tag.prefix.count..<lcpEnd])
            var excluded: Set<Int> = []
            if let endTok = probe.toolCallEndToken { excluded.insert(endTok) }
            let res = table.greedyTokenizePrefix(
                bytes: suffix,
                maxTokens: maxCount - out.count,
                excludedTokens: excluded
            ) { tok in
                guard (try? probe.accept(tok)) != nil else { return false }
                return true
            }
            guard !res.tokens.isEmpty else { break }
            out.append(contentsOf: res.tokens)
            if res.consumedBytes < suffix.count { break }
        }
        return out
    }

    public init?(tools: [QWJson], forcedToolName: String?,
                 table: QWJSONTokenTable, toolCallToken: Int? = nil,
                 toolCallOpenTokens: [Int]? = nil,
                 toolCallEndToken: Int?,
                 toolCallCloseTag: String? = nil,
                 requireToolCall: Bool = false, startsInThinking: Bool = false,
                 endThinkToken: Int? = nil, endThinkCloseTag: String? = nil) {
        var names: [String] = []
        var paramsByTool: [String: [[UInt8]]] = [:]
        var gcmlParamsByTool: [String: [[UInt8]]] = [:]
        var argKeysByTool: [String: [[UInt8]]] = [:]
        for item in tools {
            let fn = item.get("function").isUndefined ? item : item.get("function")
            guard case .string(let rawName) = fn.get("name") else { continue }
            let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { continue }
            if let forced = forcedToolName, !forced.isEmpty, name != forced { continue }
            if !names.contains(name) { names.append(name) }
            let params = fn.get("parameters").isUndefined ? fn.get("input_schema") : fn.get("parameters")
            let allowExtra: Bool
            if case .bool(true) = params.get("additionalProperties") {
                allowExtra = true
            } else {
                allowExtra = false
            }
            if !allowExtra, case .object(let props) = params.get("properties"), !props.isEmpty {
                var pList = props.map { Array(("=" + $0.0 + ">").utf8) }
                pList.append(contentsOf: props.map { Array((" name=\"" + $0.0 + "\">").utf8) })
                pList.append(contentsOf: props.map { Array((" name=\"" + $0.0 + "\" string=\"true\">").utf8) })
                pList.append(contentsOf: props.map { Array((" name=\"" + $0.0 + "\" string=\"false\">").utf8) })
                paramsByTool[name] = pList
                var gcmlList = props.map { Array((" name=\"" + $0.0 + "\" string=\"true\">").utf8) }
                gcmlList.append(contentsOf: props.map { Array((" name=\"" + $0.0 + "\" string=\"false\">").utf8) })
                gcmlList.append(contentsOf: props.map { Array((" name=\"" + $0.0 + "\">").utf8) })
                gcmlParamsByTool[name] = gcmlList
                argKeysByTool[name] = props.map { Array(($0.0 + "</arg_key>").utf8) }
            }
        }
        guard !names.isEmpty else { return nil }
        let resolvedOpen = toolCallOpenTokens ?? (toolCallToken.map { [$0] } ?? [])
        let firstOpenId = resolvedOpen.count == 1 ? resolvedOpen[0] : nil
        let derivedCloseTag: String = {
            if let explicit = toolCallCloseTag, !explicit.isEmpty { return explicit }
            if let endId = toolCallEndToken, endId >= 0, endId < table.bytes.count, let eb = table.bytes[endId],
               let s = String(bytes: eb, encoding: .utf8), !s.isEmpty {
                return s
            }
            if let openId = firstOpenId, openId >= 0, openId < table.bytes.count, let ob = table.bytes[openId],
               let s = String(bytes: ob, encoding: .utf8),
               let env = QWToolCallEnvelope.fromOpenTag(s) {
                return env.closeTag
            }
            return QWToolCallEnvelope.standard.closeTag
        }()
        var closeTags: [String] = [QWToolCallEnvelope.standard.closeTag]
        if !closeTags.contains(derivedCloseTag) { closeTags.append(derivedCloseTag) }
        var thinkTags: [[UInt8]] = [Self.endThinkUTF8]
        if let explicitThink = endThinkCloseTag, !explicitThink.isEmpty {
            let b = Array(explicitThink.utf8)
            if !thinkTags.contains(b) { thinkTags.append(b) }
        } else if let et = endThinkToken, et >= 0, et < table.bytes.count, let tb = table.bytes[et],
                  let s = String(bytes: tb, encoding: .utf8), s.hasPrefix("</"), s.hasSuffix(">"),
                  !thinkTags.contains(tb) {
            thinkTags.append(tb)
        }

        self.table = table
        self.toolCallOpenTokens = resolvedOpen
        self.toolCallEndToken = toolCallEndToken
        self.requireToolCall = requireToolCall
        self.endThinkToken = endThinkToken
        self.toolCloseBytesList = closeTags.map { Array($0.utf8) }
        self.toolClosePrefixes = closeTags.map { $0.hasSuffix(">") ? String($0.dropLast()) : $0 }
        self.endThinkBytesList = thinkTags
        self.recentCap = max(48, closeTags.map(\.utf8.count).max() ?? 12)
        self.recentThinkCap = max(24, thinkTags.map(\.count).max() ?? 8)
        self.inThinking = startsInThinking && requireToolCall
        self.toolTargets = names.map { Array(("=" + $0 + ">").utf8) }
        self.xmlInvokeTargets = names.map { Array((" name=\"" + $0 + "\">").utf8) }
        self.jsonToolTargets = names.map { Array(($0 + "\"").utf8) }
        var hdr: [[UInt8]] = []
        for name in names {
            hdr.append(Array(("<function=" + name + ">").utf8))
            hdr.append(Array(("<invoke name=\"" + name + "\">").utf8))
            hdr.append(Array((Self.gcmlInvokePrefix + name + "\">").utf8))
            hdr.append(Array((name + "<arg_key>").utf8))
            for closeTag in closeTags {
                hdr.append(Array((name + closeTag).utf8))
                hdr.append(Array((name + "\n" + closeTag).utf8))
            }
        }
        self.headerTargets = hdr
        self.parameterTargetsByTool = paramsByTool
        self.gcmlParameterTargetsByTool = gcmlParamsByTool
        self.argKeyTargetsByTool = argKeysByTool
    }

    @inline(__always)
    private static func isWhitespace(_ b: UInt8) -> Bool {
        b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09
    }

    private func enterToolCall() {
        hasStartedToolCall = true
        inThinking = false
        openPrefixMatched = 0
        recentThinkBytes.removeAll(keepingCapacity: false)
        inToolCall = true
        currentToolName = nil
        recentBytes.removeAll(keepingCapacity: true)
        activeTag = (.header, headerTargets, [])
    }

    public func allowedTokens() throws -> [Int]? {
        if mustEmitToolCallNow, openPrefixMatched < toolCallOpenTokens.count {
            return [toolCallOpenTokens[openPrefixMatched]]
        }
        guard inToolCall, let tag = activeTag else { return nil }
        let trie = table.fullTrie
        let targets = tag.targets
        let prefix = tag.prefix
        let kind = tag.kind
        let allIndices = Array(targets.indices)
        var out: [Int] = []

        let initialActive = targets.indices.filter { targets[$0].starts(with: prefix) }
        var stack: [(node: Int, depth: Int, active: [Int], inFreeJson: Bool)] = [(0, 0, initialActive, false)]
        while let (node, depth, active, inFreeJson) = stack.popLast() {
            let start = Int(trie.childStart[node])
            let end = start + Int(trie.childCount[node])
            let pos = prefix.count + depth
            for k in start..<end {
                let b = trie.childByte[k]
                let child = Int(trie.childNode[k])
                if inFreeJson {
                    let ts = Int(trie.tokenStart[child])
                    for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                        out.append(Int(trie.tokens[ti]))
                    }
                    stack.append((child, depth + 1, [], true))
                    continue
                }
                if kind == .header && pos == 0 {
                    if Self.isWhitespace(b) {
                        let ts = Int(trie.tokenStart[child])
                        for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                            out.append(Int(trie.tokens[ti]))
                        }
                        stack.append((child, 0, allIndices, false))
                        continue
                    }
                    if b == UInt8(ascii: "{") {
                        let ts = Int(trie.tokenStart[child])
                        for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                            out.append(Int(trie.tokens[ti]))
                        }
                        stack.append((child, depth + 1, [], true))
                        continue
                    }
                }
                var nextActive: [Int] = []
                for idx in active {
                    let t = targets[idx]
                    if pos < t.count {
                        if t[pos] == b { nextActive.append(idx) }
                    } else if kind == .jsonFunction {
                        if b == 0x2C || b == 0x7D || Self.isWhitespace(b) {
                            nextActive.append(idx)
                        }
                    } else if kind == .parameter || b == 0x0A || b == 0x0D || b == UInt8(ascii: "<") {
                        nextActive.append(idx)
                    }
                }
                guard !nextActive.isEmpty else { continue }
                let ts = Int(trie.tokenStart[child])
                for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                    out.append(Int(trie.tokens[ti]))
                }
                stack.append((child, depth + 1, nextActive, false))
            }
        }
        if kind == .header, let endTok = toolCallEndToken,
           targets.contains(where: { t in toolCloseBytesList.contains(where: { t == prefix + $0 }) }) {
            out.append(endTok)
        }
        let vocab = table.vocabulary
        out.removeAll { $0 < 0 || $0 >= vocab }
        guard !out.isEmpty else { return nil }
        out.sort()
        var write = 1
        for idx in 1..<out.count where out[idx] != out[write - 1] {
            out[write] = out[idx]
            write += 1
        }
        if write < out.count {
            out.removeSubrange(write..<out.count)
        }
        return out
    }

    public func accept(_ token: Int) throws {
        if !inToolCall && !(requireToolCall && !hasStartedToolCall && inThinking) && !toolCallOpenTokens.isEmpty {
            if mustEmitToolCallNow {
                guard token == toolCallOpenTokens[openPrefixMatched] else {
                    throw QWTokenConstraintError.rejectedToken(token)
                }
                openPrefixMatched += 1
                if openPrefixMatched >= toolCallOpenTokens.count { enterToolCall() }
                return
            }
            if token == toolCallOpenTokens[openPrefixMatched] {
                openPrefixMatched += 1
                if openPrefixMatched >= toolCallOpenTokens.count { enterToolCall() }
                return
            }
            openPrefixMatched = (token == toolCallOpenTokens[0]) ? 1 : 0
            return
        }
        if let endTok = toolCallEndToken, token == endTok {
            inToolCall = false
            openPrefixMatched = 0
            currentToolName = nil
            recentBytes.removeAll(keepingCapacity: true)
            activeTag = nil
            return
        }
        if requireToolCall && !hasStartedToolCall && inThinking {
            if let et = endThinkToken, token == et {
                inThinking = false
                openPrefixMatched = 0
                recentThinkBytes.removeAll(keepingCapacity: false)
                return
            }
            if token >= 0, token < table.vocabulary, let bytes = table.bytes[token], !bytes.isEmpty {
                recentThinkBytes.append(contentsOf: bytes)
                if recentThinkBytes.count > recentThinkCap {
                    recentThinkBytes.removeFirst(recentThinkBytes.count - recentThinkCap)
                }
                if endThinkBytesList.contains(where: { recentThinkBytes.suffix($0.count).elementsEqual($0) }) {
                    inThinking = false
                    openPrefixMatched = 0
                    recentThinkBytes.removeAll(keepingCapacity: false)
                }
            }
            return
        }
        if inToolCall && toolCallOpenTokens.count == 1 && token == toolCallOpenTokens[0] {
            enterToolCall()
            return
        }
        guard inToolCall, token >= 0, token < table.vocabulary,
              let bytes = table.bytes[token] else { return }

        for b in bytes {
            if var tag = activeTag {
                if tag.kind == .header && tag.prefix.isEmpty {
                    if Self.isWhitespace(b) { continue }
                    if b == UInt8(ascii: "{") {
                        activeTag = nil
                        recentBytes = [b]
                        continue
                    }
                }
                if tag.kind == .jsonFunction {
                    if b == UInt8(ascii: "\"") {
                        currentToolName = String(decoding: tag.prefix, as: UTF8.self)
                        activeTag = nil
                        recentBytes.removeAll(keepingCapacity: true)
                    } else {
                        tag.prefix.append(b)
                        activeTag = tag.targets.contains(where: { $0.starts(with: tag.prefix) }) ? tag : nil
                    }
                    continue
                }
                if b == UInt8(ascii: ">") {
                    switch tag.kind {
                    case .function:
                        if tag.prefix.first == UInt8(ascii: "=") {
                            currentToolName = String(decoding: tag.prefix.dropFirst(), as: UTF8.self)
                        } else if let str = String(bytes: tag.prefix, encoding: .utf8),
                                  str.hasPrefix(" name=\""), str.hasSuffix("\"") {
                            currentToolName = String(str.dropFirst(7).dropLast(1))
                        }
                        activeTag = nil
                    case .header:
                        let str = String(decoding: tag.prefix, as: UTF8.self)
                        if str.hasPrefix("<function=") {
                            currentToolName = String(str.dropFirst(10))
                            activeTag = nil
                        } else if str.hasPrefix("<invoke name=\""), str.hasSuffix("\"") {
                            currentToolName = String(str.dropFirst(14).dropLast(1))
                            activeTag = nil
                        } else if str.hasPrefix(Self.gcmlInvokePrefix), str.hasSuffix("\"") {
                            currentToolName = String(str.dropFirst(Self.gcmlInvokePrefix.count).dropLast(1))
                            activeTag = nil
                        } else if str.hasSuffix("<arg_key") {
                            let tool = str.dropLast(8).trimmingCharacters(in: .whitespacesAndNewlines)
                            currentToolName = tool
                            if let pTargets = argKeyTargetsByTool[tool], !pTargets.isEmpty {
                                activeTag = (.argKey, pTargets, [])
                            } else {
                                activeTag = nil
                            }
                        } else if toolClosePrefixes.contains(where: { str.hasSuffix($0) }) {
                            inToolCall = false
                            currentToolName = nil
                            activeTag = nil
                        } else {
                            activeTag = nil
                        }
                    case .parameter, .argKey, .jsonFunction:
                        activeTag = nil
                    }
                    recentBytes.removeAll(keepingCapacity: true)
                } else {
                    tag.prefix.append(b)
                    if tag.targets.contains(where: { $0.starts(with: tag.prefix) }) {
                        activeTag = tag
                    } else {
                        activeTag = nil
                    }
                }
            } else {
                recentBytes.append(b)
                if recentBytes.count > recentCap {
                    recentBytes.removeFirst(recentBytes.count - recentCap)
                }
                if recentBytes.suffix(9).elementsEqual("<function".utf8) {
                    activeTag = (.function, toolTargets, [])
                } else if recentBytes.suffix(7).elementsEqual("<invoke".utf8)
                    || recentBytes.suffix(Self.gcmlInvokeOpenUTF8.count).elementsEqual(Self.gcmlInvokeOpenUTF8) {
                    activeTag = (.function, xmlInvokeTargets, [])
                } else if recentBytes.suffix(9).elementsEqual("</invoke>".utf8)
                    || recentBytes.suffix(Self.gcmlInvokeCloseUTF8.count).elementsEqual(Self.gcmlInvokeCloseUTF8) {
                    currentToolName = nil
                } else if recentBytes.suffix(Self.gcmlParamOpenUTF8.count).elementsEqual(Self.gcmlParamOpenUTF8),
                          let tool = currentToolName,
                          let pTargets = gcmlParameterTargetsByTool[tool],
                          !pTargets.isEmpty {
                    activeTag = (.parameter, pTargets, [])
                } else if recentBytes.suffix(10).elementsEqual("<parameter".utf8),
                          let tool = currentToolName,
                          let pTargets = parameterTargetsByTool[tool],
                          !pTargets.isEmpty {
                    activeTag = (.parameter, pTargets, [])
                } else if recentBytes.suffix(9).elementsEqual("<arg_key>".utf8),
                          let tool = currentToolName,
                          let pTargets = argKeyTargetsByTool[tool],
                          !pTargets.isEmpty {
                    activeTag = (.argKey, pTargets, [])
                } else if currentToolName == nil,
                          (recentBytes.suffix(8).elementsEqual("\"name\":\"".utf8)
                           || recentBytes.suffix(9).elementsEqual("\"name\": \"".utf8)
                           || recentBytes.suffix(12).elementsEqual("\"function\":\"".utf8)
                           || recentBytes.suffix(13).elementsEqual("\"function\": \"".utf8)) {
                    activeTag = (.jsonFunction, jsonToolTargets, [])
                } else if toolCloseBytesList.contains(where: { recentBytes.suffix($0.count).elementsEqual($0) }) {
                    inToolCall = false
                    currentToolName = nil
                    recentBytes.removeAll(keepingCapacity: true)
                }
            }
        }
    }
}

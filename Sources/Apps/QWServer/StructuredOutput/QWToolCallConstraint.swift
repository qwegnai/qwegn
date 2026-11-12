// QWToolCallConstraint.swift — request-local token constraint that restricts
// tool-call function and parameter names inside <tool_call> blocks across
// Qwen XML (<function=NAME> / <parameter=NAME>), GLM XML (NAME<arg_key>NAME</arg_key>),
// and JSON (<tool_call>{"name": "NAME", ...}) formats to the request's tool schemas.

import Foundation
import QWCore
import QWHead
import QWToken

/// Constrains tool and parameter names inside `<tool_call>` blocks to the
/// request's declared tool names and parameter schemas.
public final class QWToolCallConstraint: QWTokenConstraint, @unchecked Sendable {
    private enum TagKind {
        case header
        case function
        case parameter
        case argKey
        case jsonFunction
    }

    private let table: QWJSONTokenTable
    private let toolCallToken: Int?
    private let toolCallEndToken: Int?
    private let headerTargets: [[UInt8]]
    private let toolTargets: [[UInt8]]
    private let jsonToolTargets: [[UInt8]]
    private let parameterTargetsByTool: [String: [[UInt8]]]
    private let argKeyTargetsByTool: [String: [[UInt8]]]

    private var inToolCall = false
    private var currentToolName: String?
    private var recentBytes: [UInt8] = []
    private var activeTag: (kind: TagKind, targets: [[UInt8]], prefix: [UInt8])?

    public var allowsSpeculation: Bool { true }
    public var isUnconstrainedNow: Bool { !inToolCall && activeTag == nil }
    public var speculationBoundaryToken: Int? { toolCallToken }

    public init?(tools: [QWJson], forcedToolName: String?,
                 table: QWJSONTokenTable, toolCallToken: Int?, toolCallEndToken: Int?) {
        var names: [String] = []
        var paramsByTool: [String: [[UInt8]]] = [:]
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
                paramsByTool[name] = props.map { Array(("=" + $0.0 + ">").utf8) }
                argKeysByTool[name] = props.map { Array(($0.0 + "</arg_key>").utf8) }
            }
        }
        guard !names.isEmpty else { return nil }
        self.table = table
        self.toolCallToken = toolCallToken
        self.toolCallEndToken = toolCallEndToken
        self.toolTargets = names.map { Array(("=" + $0 + ">").utf8) }
        self.jsonToolTargets = names.map { Array(($0 + "\"").utf8) }
        var hdr: [[UInt8]] = []
        for name in names {
            hdr.append(Array(("<function=" + name + ">").utf8))
            hdr.append(Array((name + "<arg_key>").utf8))
            hdr.append(Array((name + "</tool_call>").utf8))
            hdr.append(Array((name + "\n<arg_key>").utf8))
            hdr.append(Array((name + "\n</tool_call>").utf8))
        }
        self.headerTargets = hdr
        self.parameterTargetsByTool = paramsByTool
        self.argKeyTargetsByTool = argKeysByTool
    }

    @inline(__always)
    private static func isWhitespace(_ b: UInt8) -> Bool {
        b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09
    }

    public func allowedTokens() throws -> [Int]? {
        guard inToolCall, let tag = activeTag else { return nil }
        let trie = table.fullTrie
        let targets = tag.targets
        let prefix = tag.prefix
        let kind = tag.kind
        let allIndices = Array(targets.indices)
        var out: [Int] = []

        func walk(_ node: Int32, _ depth: Int, _ active: [Int], _ inFreeJson: Bool) {
            let start = Int(trie.childStart[Int(node)])
            let end = start + Int(trie.childCount[Int(node)])
            let pos = prefix.count + depth
            for k in start..<end {
                let b = trie.childByte[k]
                let child = Int(trie.childNode[k])
                if inFreeJson {
                    let ts = Int(trie.tokenStart[child])
                    for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                        out.append(Int(trie.tokens[ti]))
                    }
                    walk(Int32(child), depth + 1, [], true)
                    continue
                }
                if kind == .header && pos == 0 {
                    if Self.isWhitespace(b) {
                        let ts = Int(trie.tokenStart[child])
                        for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                            out.append(Int(trie.tokens[ti]))
                        }
                        walk(Int32(child), 0, allIndices, false)
                        continue
                    }
                    if b == UInt8(ascii: "{") {
                        let ts = Int(trie.tokenStart[child])
                        for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                            out.append(Int(trie.tokens[ti]))
                        }
                        walk(Int32(child), depth + 1, [], true)
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
                    } else if b == 0x0A || b == 0x0D || b == UInt8(ascii: "<") {
                        nextActive.append(idx)
                    }
                }
                guard !nextActive.isEmpty else { continue }
                let ts = Int(trie.tokenStart[child])
                for ti in ts..<(ts + Int(trie.tokenCount[child])) {
                    out.append(Int(trie.tokens[ti]))
                }
                walk(Int32(child), depth + 1, nextActive, false)
            }
        }

        let initialActive = targets.indices.filter { targets[$0].starts(with: prefix) }
        walk(0, 0, initialActive, false)
        if kind == .header, let endTok = toolCallEndToken,
           targets.contains(where: { $0 == prefix + Array("</tool_call>".utf8) }) {
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
        if let startTok = toolCallToken, token == startTok {
            inToolCall = true
            currentToolName = nil
            recentBytes.removeAll(keepingCapacity: true)
            activeTag = (.header, headerTargets, [])
            return
        }
        if let endTok = toolCallEndToken, token == endTok {
            inToolCall = false
            currentToolName = nil
            recentBytes.removeAll(keepingCapacity: true)
            activeTag = nil
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
                        }
                        activeTag = nil
                    case .header:
                        let str = String(decoding: tag.prefix, as: UTF8.self)
                        if str.hasPrefix("<function=") {
                            currentToolName = String(str.dropFirst(10))
                            activeTag = nil
                        } else if str.hasSuffix("<arg_key") {
                            let tool = str.dropLast(8).trimmingCharacters(in: .whitespacesAndNewlines)
                            currentToolName = tool
                            if let pTargets = argKeyTargetsByTool[tool], !pTargets.isEmpty {
                                activeTag = (.argKey, pTargets, [])
                            } else {
                                activeTag = nil
                            }
                        } else if str.hasSuffix("</tool_call") {
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
                if recentBytes.count > 20 {
                    recentBytes.removeFirst(recentBytes.count - 20)
                }
                if recentBytes.suffix(9).elementsEqual("<function".utf8) {
                    activeTag = (.function, toolTargets, [])
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
                } else if recentBytes.suffix(12).elementsEqual("</tool_call>".utf8) {
                    inToolCall = false
                    currentToolName = nil
                    recentBytes.removeAll(keepingCapacity: true)
                }
            }
        }
    }
}

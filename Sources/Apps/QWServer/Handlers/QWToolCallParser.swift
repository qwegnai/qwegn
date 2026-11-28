import Foundation
import QWCore
import QWToken

public struct ToolCallFunction: Codable, Sendable, Equatable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCall: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let type: String
    public let function: ToolCallFunction

    public init(id: String = QWToolCallParser.generateCallId(), type: String = "function", function: ToolCallFunction) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ToolCallChunk: Codable, Sendable {
    public struct FunctionChunk: Codable, Sendable {
        public let name: String?
        public let arguments: String?

        public init(name: String? = nil, arguments: String? = nil) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let index: Int
    public let id: String?
    public let type: String?
    public let function: FunctionChunk?

    public init(index: Int, id: String? = nil, type: String? = "function", function: FunctionChunk? = nil) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public enum QWToolCallParser {
    public struct ParseResult: Sendable {
        public let content: String?
        public let toolCalls: [ToolCall]?
        public let reasoningContent: String?
        public let finishReason: String

        public init(content: String?, toolCalls: [ToolCall]?, reasoningContent: String?, finishReason: String) {
            self.content = content
            self.toolCalls = toolCalls
            self.reasoningContent = reasoningContent
            self.finishReason = finishReason
        }
    }

    /// Generates standard OpenAI-compatible tool call ID (e.g. "call_a1b2c3d4e5f6")
    public static func generateCallId() -> String {
        let hex = UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased()
        return "call_\(hex)"
    }

/// Parses complete generated text into reasoning, content and tool calls;
/// supports both Qwen XML-tag and JSON tool-call formats. Unterminated
/// calls on a truncated generation are dropped; finish reason becomes length.
    public static func parse(text: String,
                             tools: [QWJson]? = nil,
                             startsInThinking: Bool = false,
                             truncated: Bool = false,
                             allowToolCalls: Bool = true,
                             allowReasoningTags: Bool = true,
                             allowInk2Markers: Bool = false,
                             toolEnvelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes,
                             thinkTags: [(open: String, close: String)] = QWChatStreamParser.defaultThinkTags) -> ParseResult {
        let parser = QWChatStreamParser(
            startsInReasoning: startsInThinking && allowReasoningTags,
            allowToolCalls: allowToolCalls,
            allowReasoningTags: allowReasoningTags,
            allowInk2Markers: allowInk2Markers,
            toolEnvelopes: toolEnvelopes,
            thinkTags: thinkTags
        )
        _ = parser.feed(text)
        _ = parser.finish()
        return parser.result(tools: allowToolCalls ? tools : nil, truncated: truncated)
    }

    /// Splits accumulated tool-envelope segments into calls and flags unterminated segments.
    static func parseToolSegments(
        _ text: String,
        tools: [QWJson]?,
        dropIncomplete: Bool,
        envelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> (calls: [ToolCall], incomplete: Bool) {
        let schemas = ToolSchemas(tools)
        var calls: [ToolCall] = []
        var incomplete = false
        var rest = Substring(text)
        while let hit = earliestToolSegmentOpen(in: rest, envelopes: envelopes) {
            let afterOpen = rest[hit.range.upperBound...]
            let inner: Substring
            let closed: Bool
            if let close = matchingToolCallClose(in: afterOpen, envelope: hit.envelope) {
                inner = afterOpen[..<close.lowerBound]
                rest = afterOpen[close.upperBound...]
                closed = true
            } else {
                inner = afterOpen
                rest = Substring("")
                closed = false
            }
            if !closed {
                incomplete = true
                if dropIncomplete { continue }
            }
            let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
            calls.append(contentsOf: parseToolSegmentCalls(inner: trimmed, schemas: schemas))
        }
        return (calls, incomplete)
    }

    static func parseSingleToolCall(inner: String, schemas: ToolSchemas) -> ToolCall? {
        // Format A: JSON format inside <tool_call>
        if inner.hasPrefix("{") && inner.hasSuffix("}") {
            if let data = inner.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let name = (json["name"] as? String) ?? (json["function"] as? String) ?? ""
                if !name.isEmpty {
                    let argsString: String
                    if let rawArgs = json["arguments"] ?? json["parameters"] {
                        if let dict = rawArgs as? [String: Any] {
                            let coerced = schemas.coerceDictionary(dict, function: name)
                            if let d = try? JSONSerialization.data(withJSONObject: coerced, options: [.sortedKeys, .withoutEscapingSlashes]),
                               let str = String(data: d, encoding: .utf8) {
                                argsString = str
                            } else {
                                argsString = "{}"
                            }
                        } else if let s = rawArgs as? String {
                            if let sData = s.data(using: .utf8),
                               let dict = (try? JSONSerialization.jsonObject(with: sData)) as? [String: Any],
                               let d = try? JSONSerialization.data(
                                   withJSONObject: schemas.coerceDictionary(dict, function: name),
                                   options: [.sortedKeys, .withoutEscapingSlashes]
                               ),
                               let str = String(data: d, encoding: .utf8) {
                                argsString = str
                            } else {
                                argsString = s
                            }
                        } else if let d = try? JSONSerialization.data(withJSONObject: rawArgs, options: [.sortedKeys, .withoutEscapingSlashes]),
                                  let str = String(data: d, encoding: .utf8) {
                            argsString = str
                        } else {
                            argsString = "{}"
                        }
                    } else {
                        argsString = "{}"
                    }
                    return ToolCall(
                        id: generateCallId(),
                        type: "function",
                        function: ToolCallFunction(name: name, arguments: argsString)
                    )
                }
            }
        }

        // Format B: XML tag format (function/parameter open-close pairs);
        // Format A above is the JSON-object variant.
        if let funcOpen = inner.range(of: "<function="),
           let gtIdx = inner[funcOpen.upperBound...].firstIndex(of: ">") {
            let name = String(inner[funcOpen.upperBound..<gtIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty, !name.contains("\n") {
                let bodyStart = inner.index(after: gtIdx)
                let afterOpen = inner[bodyStart...]
                let body: String
                if let funcClose = afterOpen.range(of: "</function>", options: .backwards),
                   afterOpen[funcClose.upperBound...].allSatisfy(\.isWhitespace) {
                    body = String(afterOpen[..<funcClose.lowerBound])
                } else {
                    body = String(afterOpen)
                }

                // Arguments are assembled by hand so the parameter order the model produced is
                // preserved: the chat template re-renders history from this JSON, and a reordered
                // call would no longer match what the model actually generated.
                var members: [String] = []
                for (pName, rawVal) in Self.extractXMLParameters(in: body) {
                    var pVal = rawVal
                    if pVal.hasPrefix("\n") { pVal.removeFirst() }
                    if pVal.hasSuffix("\n") { pVal.removeLast() }
                    members.append("\(jsonLiteral(pName)):\(jsonValue(pVal, type: schemas.type(function: name, parameter: pName)))")
                }

                return ToolCall(
                    id: generateCallId(),
                    type: "function",
                    function: ToolCallFunction(name: name, arguments: "{" + members.joined(separator: ",") + "}")
                )
            }
        }

        // Format C: <tool_call>{function-name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
        if let firstArgRange = inner.range(of: "<arg_key>") {
            let name = String(inner[..<firstArgRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            let body = String(inner[firstArgRange.lowerBound...])
            var members: [String] = []
            for (pName, rawVal) in Self.extractGLMArgPairs(in: body) {
                var pVal = rawVal
                if pVal.hasPrefix("\n") { pVal.removeFirst() }
                if pVal.hasSuffix("\n") { pVal.removeLast() }
                members.append("\(jsonLiteral(pName)):\(jsonValue(pVal, type: schemas.type(function: name, parameter: pName)))")
            }
            return ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: name, arguments: "{" + members.joined(separator: ",") + "}")
            )
        }

        let bareName = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        if !bareName.isEmpty,
           bareName.range(of: #"^[A-Za-z_][A-Za-z0-9_.-]*$"#, options: .regularExpression) != nil {
            return ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: bareName, arguments: "{}")
            )
        }

        return nil
    }

    private static func extractXMLParameters(in body: String) -> [(String, String)] {
        var out: [(String, String)] = []
        var cursor = body.startIndex
        while let open = body[cursor...].range(of: "<parameter="),
              let gt = body[open.upperBound...].firstIndex(of: ">") {
            let rawName = String(body[open.upperBound..<gt])
            if rawName.contains("\n") {
                cursor = body.index(after: open.lowerBound)
                continue
            }
            let pName = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            let valStart = body.index(after: gt)
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: valStart, closeTag: "</parameter>", nextOpenTag: "<parameter="
            ) {
                out.append((pName, String(body[valStart..<valEnd])))
                cursor = nextCursor
            } else {
                out.append((pName, String(body[valStart...])))
                break
            }
        }
        return out
    }

    private static func extractGLMArgPairs(in body: String) -> [(String, String)] {
        var out: [(String, String)] = []
        var cursor = body.startIndex
        while let kOpen = body[cursor...].range(of: "<arg_key>"),
              let kClose = body[kOpen.upperBound...].range(of: "</arg_key>"),
              let vOpen = body[kClose.upperBound...].range(of: "<arg_value>") {
            let between = body[kClose.upperBound..<vOpen.lowerBound]
            guard between.allSatisfy(\.isWhitespace) else {
                cursor = kClose.upperBound
                continue
            }
            let pName = String(body[kOpen.upperBound..<kClose.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let valStart = vOpen.upperBound
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: valStart, closeTag: "</arg_value>", nextOpenTag: "<arg_key>"
            ) {
                out.append((pName, String(body[valStart..<valEnd])))
                cursor = nextCursor
            } else {
                out.append((pName, String(body[valStart...])))
                break
            }
        }
        return out
    }

    static func findBoundaryCloseTag(
        in body: String,
        from valStart: String.Index,
        closeTag: String,
        nextOpenTag: String
    ) -> (valEnd: String.Index, nextCursor: String.Index)? {
        var search = valStart
        var fallbackNextOpen: String.Index? = nil
        while let close = body[search...].range(of: closeTag) {
            let after = body[close.upperBound...].drop(while: { $0.isWhitespace })
            if after.isEmpty || after.hasPrefix(nextOpenTag) {
                return (close.lowerBound, close.upperBound)
            }
            search = close.upperBound
        }
        if fallbackNextOpen == nil, let nextOpen = body[valStart...].range(of: nextOpenTag) {
            fallbackNextOpen = nextOpen.lowerBound
        }
        if let end = fallbackNextOpen {
            return (end, end)
        }
        return nil
    }

    static func jsonLiteral(_ string: String) -> String {
        if let d = try? JSONSerialization.data(withJSONObject: string, options: [.fragmentsAllowed, .withoutEscapingSlashes]),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        return "\"\""
    }

    /// Converts a raw parameter body into a JSON value. When the tool schema declares the
    /// parameter as a string the body is NEVER interpreted: file contents that happen to be
    /// valid JSON (package.json, a bare number, ...) must reach the tool unchanged.
    static func jsonValue(_ raw: String, type: ToolSchemas.ParamType) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case .string:
            return jsonLiteral(raw)
        case .boolean:
            let low = trimmed.lowercased()
            if low == "true" || low == "\"true\"" || low == "1" { return "true" }
            if low == "false" || low == "\"false\"" || low == "0" { return "false" }
            return jsonLiteral(raw)
        case .integer:
            let unquoted = (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"") && trimmed.count >= 2)
                ? String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
                : trimmed
            if let iv = Int64(unquoted) { return String(iv) }
            if let dv = Double(unquoted), dv.isFinite, let iv = Int64(exactly: dv) { return String(iv) }
            return jsonLiteral(raw)
        case .number:
            let unquoted = (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"") && trimmed.count >= 2)
                ? String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
                : trimmed
            if let iv = Int64(unquoted) { return String(iv) }
            if let dv = Double(unquoted), dv.isFinite {
                if let d = try? JSONSerialization.data(withJSONObject: dv, options: [.fragmentsAllowed]),
                   let s = String(data: d, encoding: .utf8) {
                    return s
                }
            }
            return jsonLiteral(raw)
        case .structured, .unknown:
            let low = trimmed.lowercased()
            if low == "true" { return "true" }
            if low == "false" { return "false" }
            if low == "null" || low == "none" { return "null" }
            if let data = trimmed.data(using: .utf8),
               let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
               let out = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes]),
               let str = String(data: out, encoding: .utf8) {
                return str
            }
            return jsonLiteral(raw)
        }
    }

    /// Per-function parameter type table extracted from the request's OpenAI tool list.
    struct ToolSchemas {
        enum ParamType { case string, boolean, integer, number, structured, unknown }
        private var table: [String: [String: ParamType]] = [:]

        init(_ tools: [QWJson]?) {
            for tool in tools ?? [] {
                let fn = tool.get("function").isUndefined ? tool : tool.get("function")
                guard case .string(let name) = fn.get("name") else { continue }
                var params: [String: ParamType] = [:]
                let paramRoot = fn.get("parameters").isUndefined ? fn.get("input_schema") : fn.get("parameters")
                if case .object(let props) = paramRoot.get("properties") {
                    for (key, schema) in props { params[key] = Self.classify(schema) }
                }
                table[name] = params
            }
        }

        func type(function: String, parameter: String) -> ParamType {
            table[function]?[parameter] ?? .unknown
        }

        func coerceDictionary(_ dict: [String: Any], function: String) -> [String: Any] {
            guard let params = table[function] else { return dict }
            var out = dict
            for (k, v) in dict {
                guard let pType = params[k] else { continue }
                if let strVal = v as? String {
                    let trimmed = strVal.trimmingCharacters(in: .whitespacesAndNewlines)
                    switch pType {
                    case .boolean:
                        let low = trimmed.lowercased()
                        if low == "true" || low == "1" { out[k] = true }
                        else if low == "false" || low == "0" { out[k] = false }
                    case .integer:
                        if let iv = Int64(trimmed) { out[k] = iv }
                        else if let dv = Double(trimmed), dv.isFinite, let iv = Int64(exactly: dv) { out[k] = iv }
                    case .number:
                        if let iv = Int64(trimmed) { out[k] = iv }
                        else if let dv = Double(trimmed), dv.isFinite { out[k] = dv }
                    default:
                        break
                    }
                }
            }
            return out
        }

        private static func classify(_ schema: QWJson) -> ParamType {
            switch schema.get("type") {
            case .string(let t):
                switch t.lowercased() {
                case "string": return .string
                case "boolean", "bool": return .boolean
                case "integer", "int": return .integer
                case "number", "float", "double": return .number
                default: return .structured
                }
            case .array(let types):
                if types.contains(where: { $0.isString("string") }) { return .string }
                if types.contains(where: { $0.isString("boolean") }) { return .boolean }
                if types.contains(where: { $0.isString("integer") }) { return .integer }
                if types.contains(where: { $0.isString("number") }) { return .number }
                return .structured
            default:
                break
            }
            for key in ["anyOf", "oneOf"] {
                if case .array(let options) = schema.get(key) {
                    if options.contains(where: { $0.get("type").isString("string") }) { return .string }
                    if options.contains(where: { $0.get("type").isString("boolean") }) { return .boolean }
                    if options.contains(where: { $0.get("type").isString("integer") }) { return .integer }
                    if options.contains(where: { $0.get("type").isString("number") }) { return .number }
                }
            }
            return .unknown
        }
    }
}

/// Incremental splitter for model output. Starts in reasoning mode when the
/// generation prompt ended inside the thinking block; switches on close/call
/// tags. Tags split across token pieces are held back until classifiable.
public final class QWChatStreamParser: @unchecked Sendable {
    public enum Delta: Equatable, Sendable {
        case reasoning(String)
        case content(String)
    }

    private enum Mode { case reasoning, content, toolCall }

    public static let defaultThinkTags: [(open: String, close: String)] = [
        ("<think>", "</think>"),
        ("<ernie:think>", "</ernie:think>"),
    ]

    private var mode: Mode
    let allowToolCalls: Bool
    private var allowReasoningTags: Bool
    private var allowInk2Markers: Bool
    private var toolEnvelopes: [QWToolCallEnvelope]
    private var toolOpenTags: [String]
    private let thinkOpenTags: [String]
    private let thinkCloseTags: [String]
    var buffer = ""
    private var leadingPending = true
    private var contentEmitted = false
    var ink2Mode = false
    var ink2State: Ink2ChannelState = .awaitingChannel
    var ink2ToolPayloads: [String] = []
    var ink2IncompleteToolPayload: String? = nil

    public private(set) var reasoning = ""
    public private(set) var content = ""
    public private(set) var toolText = ""

    public init(startsInReasoning: Bool,
                allowToolCalls: Bool = true,
                allowReasoningTags: Bool = true,
                allowInk2Markers: Bool = false,
                toolEnvelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes,
                thinkTags: [(open: String, close: String)] = QWChatStreamParser.defaultThinkTags) {
        self.mode = (startsInReasoning && allowReasoningTags) ? .reasoning : .content
        self.allowToolCalls = allowToolCalls
        self.allowReasoningTags = allowReasoningTags
        self.allowInk2Markers = allowInk2Markers
        self.toolEnvelopes = toolEnvelopes
        self.toolOpenTags = toolEnvelopes.map(\.openTag)
        self.thinkOpenTags = thinkTags.map(\.open)
        self.thinkCloseTags = thinkTags.map(\.close)
    }

    public func setStartsInReasoning(
        _ startsInReasoning: Bool,
        allowReasoningTags: Bool? = nil,
        allowInk2Markers: Bool? = nil,
        toolEnvelopes: [QWToolCallEnvelope]? = nil
    ) {
        guard buffer.isEmpty && reasoning.isEmpty && content.isEmpty && toolText.isEmpty else { return }
        if let allowReasoningTags { self.allowReasoningTags = allowReasoningTags }
        if let allowInk2Markers { self.allowInk2Markers = allowInk2Markers }
        if let toolEnvelopes, !toolEnvelopes.isEmpty {
            self.toolEnvelopes = toolEnvelopes
            self.toolOpenTags = toolEnvelopes.map(\.openTag)
        }
        mode = (startsInReasoning && self.allowReasoningTags) ? .reasoning : .content
    }

    public func feed(_ piece: String) -> [Delta] {
        buffer += piece
        return drain(final: false)
    }

    public func finish() -> [Delta] {
        drain(final: true)
    }

    /// Longest suffix of `text` that is a proper prefix of one of `tags`.
    func heldMarkerSuffixCount(_ text: String, tags: [String]) -> Int {
        Self.heldSuffix(text, tags: tags)
    }

    private static func heldSuffix(_ text: String, tags: [String]) -> Int {
        var best = 0
        for tag in tags {
            let maxLen = min(tag.count - 1, text.count)
            guard maxLen > best else { continue }
            for len in stride(from: maxLen, to: best, by: -1) where text.hasSuffix(String(tag.prefix(len))) {
                best = len
                break
            }
        }
        return best
    }

    func emitSubstring(_ text: Substring, as targetMode: DeltaTarget, into out: inout [Delta]) {
        emit(text, as: targetMode == .reasoning ? .reasoning : .content, into: &out)
    }

    enum DeltaTarget { case reasoning, content }

    private func emit(_ text: Substring, as mode: Mode, into out: inout [Delta]) {
        guard !text.isEmpty else { return }
        let s = String(text)
        if mode == .reasoning {
            reasoning += s
            out.append(.reasoning(s))
        } else {
            content += s
            contentEmitted = true
            out.append(.content(s))
        }
    }

    private static func earliestTagRange(in text: String, tags: [String]) -> Range<String.Index>? {
        var earliest: Range<String.Index>? = nil
        for tag in tags {
            if let r = text.range(of: tag), earliest.map({ r.lowerBound < $0.lowerBound }) ?? true {
                earliest = r
            }
        }
        return earliest
    }

    private func drain(final: Bool) -> [Delta] {
        var out: [Delta] = []
        while true {
            if ink2Mode {
                _ = drainInk2(final: final, out: &out)
                return out
            }
            let inkTags = allowInk2Markers ? QWToolCallParser.ink2AllMarkers : []
            if allowInk2Markers, let inkMatch = Self.earliestTagMatch(in: buffer, tags: QWToolCallParser.ink2AllMarkers) {
                if inkMatch.range.lowerBound > buffer.startIndex {
                    let target: Mode = (mode == .reasoning) ? .reasoning : .content
                    emit(buffer[..<inkMatch.range.lowerBound], as: target, into: &out)
                }
                buffer.removeSubrange(..<inkMatch.range.lowerBound)
                ink2Mode = true
                ink2State = .awaitingChannel
                continue
            }
            switch mode {
            case .toolCall:
                if let env = toolEnvelopes.first(where: { buffer.hasPrefix($0.openTag) }) {
                    let afterOpen = buffer.dropFirst(env.openTag.count)
                    if let close = QWToolCallParser.matchingToolCallClose(in: afterOpen, envelope: env) {
                        let inner = afterOpen[..<close.lowerBound]
                        let balanced = QWToolCallParser.isBalancedToolPayload(inner)
                        let unclosed = QWToolCallParser.hasUnclosedLiteralContainer(inner)
                        let hasTail = !afterOpen[close.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        if balanced || (!unclosed && (final || hasTail)) {
                            toolText += String(buffer[..<close.upperBound])
                            buffer = String(afterOpen[close.upperBound...])
                            mode = .content
                            leadingPending = false
                            continue
                        }
                    }
                }
                if final {
                    toolText += buffer
                    buffer = ""
                }
                return out

            case .reasoning:
                if leadingPending {
                    buffer = String(buffer.drop(while: { $0.isWhitespace }))
                    if !final && allowInk2Markers && Self.heldSuffix(buffer, tags: inkTags) >= 2 { return out }
                    if let openTag = thinkOpenTags.first(where: { buffer.hasPrefix($0) }) {
                        buffer.removeFirst(openTag.count)
                        continue
                    }
                    if buffer.isEmpty { return out }
                    if !final && thinkOpenTags.contains(where: { $0.hasPrefix(buffer) }) { return out }
                    leadingPending = false
                }
                let close = Self.earliestTagRange(in: buffer, tags: thinkCloseTags)
                let tool = allowToolCalls ? Self.earliestTagRange(in: buffer, tags: toolOpenTags) : nil
                if let close, tool.map({ close.lowerBound < $0.lowerBound }) ?? true {
                    emit(buffer[..<close.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[close.upperBound...])
                    mode = .content
                    leadingPending = false
                    continue
                }
                if let tool {
                    emit(buffer[..<tool.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = (allowToolCalls ? (thinkCloseTags + toolOpenTags) : thinkCloseTags) + inkTags
                let hold = final ? 0 : Self.heldSuffix(buffer, tags: activeTags)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emit(buffer[..<cut], as: .reasoning, into: &out)
                buffer = String(buffer[cut...])
                return out

            case .content:
                if leadingPending {
                    if allowReasoningTags && !contentEmitted {
                        if let nonWS = buffer.firstIndex(where: { !$0.isWhitespace }) {
                            let afterWS = buffer[nonWS...]
                            if let openTag = thinkOpenTags.first(where: { afterWS.hasPrefix($0) }) {
                                buffer = String(afterWS.dropFirst(openTag.count))
                                mode = .reasoning
                                leadingPending = true
                                continue
                            }
                            if !final && (thinkOpenTags.contains(where: { $0.hasPrefix(afterWS) })
                                || (allowInk2Markers && Self.heldSuffix(String(afterWS), tags: inkTags) >= 2)) {
                                return out
                            }
                            leadingPending = false
                        } else if !final && buffer.count < 16 {
                            return out
                        } else {
                            leadingPending = false
                        }
                    } else {
                        leadingPending = false
                    }
                }
                if allowToolCalls, let tool = Self.earliestTagRange(in: buffer, tags: toolOpenTags) {
                    emit(buffer[..<tool.lowerBound], as: .content, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = (allowToolCalls ? toolOpenTags : []) + inkTags
                let hold = final ? 0 : Self.heldSuffix(buffer, tags: activeTags)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emit(buffer[..<cut], as: .content, into: &out)
                buffer = String(buffer[cut...])
                return out
            }
        }
    }

    /// Final structured result. Call after `finish()`.
    public func result(tools: [QWJson]?, truncated: Bool) -> QWToolCallParser.ParseResult {
        var (calls, _) = allowToolCalls
            ? QWToolCallParser.parseToolSegments(toolText, tools: tools, dropIncomplete: truncated, envelopes: toolEnvelopes)
            : ([], false)
        if allowToolCalls {
            let inkCalls = QWToolCallParser.parseInk2ToolPayloads(
                ink2ToolPayloads, trailingIncomplete: ink2IncompleteToolPayload,
                tools: tools, dropIncomplete: truncated
            )
            calls.append(contentsOf: inkCalls)
        }
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let finish: String
        if !calls.isEmpty && !truncated {
            finish = "tool_calls"
        } else if truncated {
            finish = "length"
        } else {
            finish = "stop"
        }
        return QWToolCallParser.ParseResult(
            content: trimmedContent.isEmpty ? nil : trimmedContent,
            toolCalls: calls.isEmpty ? nil : calls,
            reasoningContent: trimmedReasoning.isEmpty ? nil : trimmedReasoning,
            finishReason: finish
        )
    }
}


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
                             allowReasoningTags: Bool = true) -> ParseResult {
        let parser = QWChatStreamParser(
            startsInReasoning: startsInThinking && allowReasoningTags,
            allowToolCalls: allowToolCalls,
            allowReasoningTags: allowReasoningTags
        )
        _ = parser.feed(text)
        _ = parser.finish()
        return parser.result(tools: allowToolCalls ? tools : nil, truncated: truncated)
    }

    /// Splits accumulated `<tool_call>` segments into calls. A segment without its closing
    /// tag is reported through `incomplete` and only returned as a call when `dropIncomplete`
    /// is false.
    static func parseToolSegments(_ text: String, tools: [QWJson]?, dropIncomplete: Bool)
        -> (calls: [ToolCall], incomplete: Bool) {
        let schemas = ToolSchemas(tools)
        var calls: [ToolCall] = []
        var incomplete = false
        var rest = Substring(text)
        while let open = rest.range(of: "<tool_call>") {
            let afterOpen = rest[open.upperBound...]
            let inner: Substring
            let closed: Bool
            if let close = matchingToolCallClose(in: afterOpen) {
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
            if let call = parseSingleToolCall(inner: trimmed, schemas: schemas) {
                calls.append(call)
            }
        }
        return (calls, incomplete)
    }

    static func firstCompleteToolCallEnd(in text: String) -> String.Index? {
        guard let open = text.range(of: "<tool_call>") else { return nil }
        return matchingToolCallClose(in: text[open.upperBound...])?.upperBound
    }

    private static func isBalancedToolPayload(_ raw: Substring) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") {
            var depth = 0
            var inString = false
            var escaped = false
            let lastIdx = trimmed.index(before: trimmed.endIndex)
            for idx in trimmed.indices {
                let ch = trimmed[idx]
                if inString {
                    if escaped {
                        escaped = false
                    } else if ch == "\\" {
                        escaped = true
                    } else if ch == "\"" {
                        inString = false
                    }
                    continue
                }
                if ch == "\"" {
                    inString = true
                } else if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth < 0 { return false }
                    if depth == 0 && idx < lastIdx { return false }
                }
            }
            return !inString && depth == 0
        }
        if trimmed.hasPrefix("<function=") && trimmed.hasSuffix("</function>") {
            return trimmed.components(separatedBy: "<function=").count == 2
                && trimmed.components(separatedBy: "</function>").count == 2
                && trimmed.components(separatedBy: "<parameter=").count
                    == trimmed.components(separatedBy: "</parameter>").count
        }
        return false
    }

    private static func matchingToolCallClose(in afterOpen: Substring) -> Range<String.Index>? {
        var searchFrom = afterOpen.startIndex
        while let close = afterOpen[searchFrom...].range(of: "</tool_call>") {
            if isBalancedToolPayload(afterOpen[..<close.lowerBound]) {
                return close
            }
            let tail = afterOpen[close.upperBound...].drop(while: { $0.isWhitespace })
            if tail.isEmpty || tail.hasPrefix("<tool_call>") || tail.range(of: "</tool_call>") == nil {
                return close
            }
            searchFrom = close.upperBound
        }
        return nil
    }

    private static func parseSingleToolCall(inner: String, schemas: ToolSchemas) -> ToolCall? {
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
        let funcRegex = try? NSRegularExpression(pattern: "<function=([^>\\n]+)>([\\s\\S]*?)(?:</function>|$)", options: [])
        let nsInner = inner as NSString
        let funcMatches = funcRegex?.matches(in: inner, options: [], range: NSRange(location: 0, length: nsInner.length)) ?? []

        if let firstMatch = funcMatches.first, firstMatch.numberOfRanges > 2 {
            let name = nsInner.substring(with: firstMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let body = nsInner.substring(with: firstMatch.range(at: 2))

            // Match parameters: <parameter=([^>\n]+)>([\s\S]*?)(?:</parameter>|(?=<parameter=)|$)
            let paramRegex = try? NSRegularExpression(pattern: "<parameter=([^>\\n]+)>([\\s\\S]*?)(?:</parameter>|(?=<parameter=)|$)", options: [])
            let nsBody = body as NSString
            let paramMatches = paramRegex?.matches(in: body, options: [], range: NSRange(location: 0, length: nsBody.length)) ?? []

            // Arguments are assembled by hand so the parameter order the model produced is
            // preserved: the chat template re-renders history from this JSON, and a reordered
            // call would no longer match what the model actually generated.
            var members: [String] = []
            for pMatch in paramMatches {
                guard pMatch.numberOfRanges > 2 else { continue }
                let pName = nsBody.substring(with: pMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                var pVal = nsBody.substring(with: pMatch.range(at: 2))

                // Strip single leading/trailing newlines added by XML formatting
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

        // Format C: <tool_call>{function-name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
        if let firstArgRange = inner.range(of: "<arg_key>") {
            let name = String(inner[..<firstArgRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            let body = String(inner[firstArgRange.lowerBound...])
            let argRegex = try? NSRegularExpression(
                pattern: "<arg_key>([\\s\\S]*?)</arg_key>\\s*<arg_value>([\\s\\S]*?)(?:</arg_value>|(?=<arg_key>)|$)",
                options: []
            )
            let nsBody = body as NSString
            let argMatches = argRegex?.matches(in: body, options: [], range: NSRange(location: 0, length: nsBody.length)) ?? []
            var members: [String] = []
            for aMatch in argMatches {
                guard aMatch.numberOfRanges > 2 else { continue }
                let pName = nsBody.substring(with: aMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                var pVal = nsBody.substring(with: aMatch.range(at: 2))
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

    private static func jsonLiteral(_ string: String) -> String {
        if let d = try? JSONSerialization.data(withJSONObject: string, options: [.fragmentsAllowed, .withoutEscapingSlashes]),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        return "\"\""
    }

    /// Converts a raw parameter body into a JSON value. When the tool schema declares the
    /// parameter as a string the body is NEVER interpreted: file contents that happen to be
    /// valid JSON (package.json, a bare number, ...) must reach the tool unchanged.
    private static func jsonValue(_ raw: String, type: ToolSchemas.ParamType) -> String {
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
            if let dv = Double(unquoted), dv.isFinite, dv == Double(Int64(dv)) { return String(Int64(dv)) }
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
                        else if let dv = Double(trimmed), dv.isFinite, dv == Double(Int64(dv)) { out[k] = Int64(dv) }
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

    private static let thinkOpen = "<think>"
    private static let thinkClose = "</think>"
    private static let toolOpen = "<tool_call>"

    private var mode: Mode
    private let allowToolCalls: Bool
    private let allowReasoningTags: Bool
    private var buffer = ""
    private var leadingPending = true
    private var contentEmitted = false

    public private(set) var reasoning = ""
    public private(set) var content = ""
    public private(set) var toolText = ""

    public var isInToolCall: Bool { mode == .toolCall }
    public var isInReasoning: Bool { mode == .reasoning }

    public init(startsInReasoning: Bool, allowToolCalls: Bool = true, allowReasoningTags: Bool = true) {
        self.mode = (startsInReasoning && allowReasoningTags) ? .reasoning : .content
        self.allowToolCalls = allowToolCalls
        self.allowReasoningTags = allowReasoningTags
    }

    public func feed(_ piece: String) -> [Delta] {
        buffer += piece
        return drain(final: false)
    }

    public func finish() -> [Delta] {
        drain(final: true)
    }

    /// Longest suffix of `text` that is a proper prefix of one of `tags`.
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

    private func drain(final: Bool) -> [Delta] {
        var out: [Delta] = []
        while true {
            switch mode {
            case .toolCall:
                toolText += buffer
                buffer = ""
                return out

            case .reasoning:
                if leadingPending {
                    // Drop leading whitespace and a redundant literal opening tag.
                    buffer = String(buffer.drop(while: { $0.isWhitespace }))
                    if buffer.hasPrefix(Self.thinkOpen) {
                        buffer.removeFirst(Self.thinkOpen.count)
                        continue
                    }
                    if buffer.isEmpty { return out }
                    if !final && Self.thinkOpen.hasPrefix(buffer) { return out }
                    leadingPending = false
                }
                let close = buffer.range(of: Self.thinkClose)
                let tool = allowToolCalls ? buffer.range(of: Self.toolOpen) : nil
                if let close, tool == nil || close.lowerBound < tool!.lowerBound {
                    emit(buffer[..<close.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[close.upperBound...])
                    mode = .content
                    leadingPending = true
                    continue
                }
                if let tool {
                    emit(buffer[..<tool.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = allowToolCalls ? [Self.thinkClose, Self.toolOpen] : [Self.thinkClose]
                let hold = final ? 0 : Self.heldSuffix(buffer, tags: activeTags)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emit(buffer[..<cut], as: .reasoning, into: &out)
                buffer = String(buffer[cut...])
                return out

            case .content:
                if leadingPending {
                    buffer = String(buffer.drop(while: { $0.isWhitespace }))
                    if buffer.isEmpty { return out }
                    if allowReasoningTags && !contentEmitted {
                        if buffer.hasPrefix(Self.thinkOpen) {
                            buffer.removeFirst(Self.thinkOpen.count)
                            mode = .reasoning
                            leadingPending = true
                            continue
                        }
                        if !final && Self.thinkOpen.hasPrefix(buffer) { return out }
                    }
                    leadingPending = false
                }
                if allowToolCalls, let tool = buffer.range(of: Self.toolOpen) {
                    emit(buffer[..<tool.lowerBound], as: .content, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = allowToolCalls ? [Self.toolOpen] : []
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
        let (calls, _) = allowToolCalls
            ? QWToolCallParser.parseToolSegments(toolText, tools: tools, dropIncomplete: truncated)
            : ([], false)
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


import Foundation
import Jinja
import QWCore
import QWToken

public struct QWChatTemplateOptions: Sendable {
    public var addGenerationPrompt = true
    public var tools: QWJson = .undefined
    public var reasoningEffort: QWJson = .undefined
    public var enableThinking: QWJson = .undefined
    public var reasoning: QWJson = .undefined
    public var preserveThinking: QWJson = .undefined
    public var addVisionId = false

    public init() {}
}

public enum QWChatTemplateError: Error, CustomStringConvertible, Sendable {
    case emptyConversation

    public var description: String {
        switch self {
        case .emptyConversation:
            return "Cannot apply chat template to an empty conversation"
        }
    }
}

public struct QWChatTemplate: Sendable {
    private let template: Template
    private let tokenizerContext: [String: QWJson]
    public let rawSource: String
    public let isInkling2Template: Bool
    private let usesGlobalSettingProtocol: Bool
    public let toolCallEnvelope: QWToolCallEnvelope
    public let thinkOpenTag: String?
    public let thinkCloseTag: String?
    public let supportsThinkingTags: Bool
    public var usesStandardToolCallHint: Bool { !isInkling2Template && toolCallEnvelope == .standard }

    public init(source: String, tokenizerContext: [String: QWJson] = [:]) throws {
        let normalizedSource = Self.normalizePythonJinjaSyntax(source)
        self.template = try Template(normalizedSource)
        self.tokenizerContext = tokenizerContext
        self.rawSource = source
        self.isInkling2Template = source.contains("<|content_thinking|>") && source.contains("<|message_model|>")
        self.usesGlobalSettingProtocol = source.contains("<|ernie:im_start|>") && source.contains("<ernie:global_setting>")
        self.toolCallEnvelope = QWToolCallEnvelope.detect(in: normalizedSource)
        let detectedThink = Self.detectThinkTags(in: normalizedSource)
        self.thinkOpenTag = detectedThink?.open
        self.thinkCloseTag = detectedThink?.close
        self.supportsThinkingTags = Self.supportsThinkingTags(inNormalizedSource: normalizedSource, detectedThink: detectedThink)
    }

    public static func supportsThinkingTags(inSource source: String) -> Bool {
        let normalized = normalizePythonJinjaSyntax(source)
        return supportsThinkingTags(inNormalizedSource: normalized, detectedThink: detectThinkTags(in: normalized))
    }

    private static func supportsThinkingTags(
        inNormalizedSource normalized: String,
        detectedThink: (open: String, close: String)?
    ) -> Bool {
        let isInkling2 = normalized.contains("<|content_thinking|>") && normalized.contains("<|message_model|>")
        return detectedThink != nil || isInkling2
    }

    /// Probes `chat_template.jinja` or `tokenizer_config.json` in `packPath` without compiling Jinja or loading weights.
    public static func detectSupportsThinkingTags(atPackPath packPath: String) -> Bool? {
        let expanded = packPath.trimmingCharacters(in: .whitespacesAndNewlines).expandingTildePath
        guard !expanded.isEmpty else { return nil }
        let fm = FileManager.default
        guard fm.fileExists(atPath: expanded) else { return nil }
        let assets = QWModelRoot(url: URL(fileURLWithPath: expanded)).assets
        let jinjaURL = assets.appendingPathComponent("chat_template.jinja")
        if fm.fileExists(atPath: jinjaURL.path),
           let source = try? String(contentsOf: jinjaURL, encoding: .utf8) {
            return supportsThinkingTags(inSource: source)
        }
        let tokCfgURL = assets.appendingPathComponent("tokenizer_config.json")
        guard fm.fileExists(atPath: tokCfgURL.path),
              let data = try? Data(contentsOf: tokCfgURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawTemplate = obj["chat_template"] else {
            return nil
        }
        if let str = rawTemplate as? String {
            return supportsThinkingTags(inSource: str)
        }
        if let entries = rawTemplate as? [[String: Any]], !entries.isEmpty {
            let selected = entries.first(where: { ($0["name"] as? String) == "default" })
                ?? (entries.count == 1 ? entries[0] : nil)
            if let str = selected?["template"] as? String {
                return supportsThinkingTags(inSource: str)
            }
        }
        return nil
    }

    private static func detectThinkTags(in source: String) -> (open: String, close: String)? {
        guard let regex = try? NSRegularExpression(pattern: #"</?((?:[A-Za-z0-9_]+:)?think)>"#),
              let match = regex.firstMatch(in: source, range: NSRange(source.startIndex..<source.endIndex, in: source)),
              let nameRange = Range(match.range(at: 1), in: source) else {
            return nil
        }
        let tagName = String(source[nameRange])
        return ("<\(tagName)>", "</\(tagName)>")
    }

    public func tokenizerTokenString(_ key: String) -> String? {
        guard case .string(let token) = tokenizerContext[key], !token.isEmpty else { return nil }
        return token
    }

    /// Normalizes Python Jinja2 syntax (`m.content.0`, `call` variables, and `namespace(items=...)`
    /// attributes shadowed by `swift-jinja` dictionary methods) outside string literals.
    private static func normalizePythonJinjaSyntax(_ source: String) -> String {
        let shadowedNSPairs = Self.detectShadowedNamespaceAttributes(in: source)
        guard let intAttrRegex = try? NSRegularExpression(pattern: #"(\b[A-Za-z_][A-Za-z0-9_]*|\])\.([0-9]+)\b"#),
              let callIdentRegex = try? NSRegularExpression(pattern: #"\bcall\b"#),
              let dictCallRegex = try? NSRegularExpression(pattern: #"(?<![A-Za-z0-9_.])dict(\s*\()"#),
              let nsKwargRegex = try? NSRegularExpression(
                  pattern: #"(?<![!<>=\w.])(items|keys|values|get)(\s*=(?!=))"#
              ) else {
            return source
        }
        let hasCallBlock = source.contains("endcall")
        let nsMemberRegexes: [(NSRegularExpression, String)] = shadowedNSPairs.compactMap { nsVar, attr in
            let escVar = NSRegularExpression.escapedPattern(for: nsVar)
            let escAttr = NSRegularExpression.escapedPattern(for: attr)
            guard let re = try? NSRegularExpression(pattern: "\\b\(escVar)\\.\(escAttr)\\b(?!\\s*\\()") else {
                return nil
            }
            return (re, "\(nsVar).__qw_ns_\(attr)")
        }
        var inNamespaceSetTag = false
        func rewriteCodeSegment(_ segment: Substring, isStatementHead: Bool) -> String {
            var current = String(segment)
            if isStatementHead && !shadowedNSPairs.isEmpty,
               current.range(of: #"^[-+]?\s*set\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*namespace\s*\("#, options: .regularExpression) != nil {
                inNamespaceSetTag = true
            }
            for _ in 0..<8 {
                let range = NSRange(current.startIndex..<current.endIndex, in: current)
                let next = intAttrRegex.stringByReplacingMatches(in: current, range: range, withTemplate: "$1[$2]")
                if next == current { break }
                current = next
            }
            if !shadowedNSPairs.isEmpty {
                if inNamespaceSetTag {
                    let range = NSRange(current.startIndex..<current.endIndex, in: current)
                    current = nsKwargRegex.stringByReplacingMatches(
                        in: current, range: range, withTemplate: "__qw_ns_$1$2"
                    )
                }
                for (re, replacement) in nsMemberRegexes {
                    let range = NSRange(current.startIndex..<current.endIndex, in: current)
                    current = re.stringByReplacingMatches(in: current, range: range, withTemplate: replacement)
                }
            }
            if !hasCallBlock {
                let range = NSRange(current.startIndex..<current.endIndex, in: current)
                current = callIdentRegex.stringByReplacingMatches(in: current, range: range, withTemplate: "__qw_call")
            } else if !isStatementHead || current.range(of: #"^[-+]?\s*call\b"#, options: .regularExpression) == nil {
                let range = NSRange(current.startIndex..<current.endIndex, in: current)
                current = callIdentRegex.stringByReplacingMatches(in: current, range: range, withTemplate: "__qw_call")
            }
            let dictRange = NSRange(current.startIndex..<current.endIndex, in: current)
            current = dictCallRegex.stringByReplacingMatches(in: current, range: dictRange, withTemplate: "__qw_dict$1")
            return current
        }

        var out = ""
        out.reserveCapacity(source.count)
        var i = source.startIndex
        var inTag = false
        var isStatementHead = false
        var tagClose = ""
        var codeStart = source.startIndex

        while i < source.endIndex {
            if !inTag {
                let rest = source[i...]
                if rest.hasPrefix("{{") || rest.hasPrefix("{%") {
                    out += source[codeStart..<i]
                    tagClose = rest.hasPrefix("{{") ? "}}" : "%}"
                    isStatementHead = (tagClose == "%}")
                    let tagOpenEnd = source.index(i, offsetBy: 2)
                    out += source[i..<tagOpenEnd]
                    i = tagOpenEnd
                    codeStart = i
                    inTag = true
                    continue
                }
                if rest.hasPrefix("{#") {
                    out += source[codeStart..<i]
                    if let commentEnd = rest.range(of: "#}")?.upperBound {
                        out += source[i..<commentEnd]
                        i = commentEnd
                        codeStart = i
                        continue
                    }
                    out += source[i...]
                    return out
                }
                i = source.index(after: i)
                continue
            }

            let ch = source[i]
            if ch == "\"" || ch == "'" {
                out += rewriteCodeSegment(source[codeStart..<i], isStatementHead: isStatementHead)
                isStatementHead = false
                let quoteStart = i
                var escaped = false
                i = source.index(after: i)
                while i < source.endIndex {
                    let c = source[i]
                    if escaped {
                        escaped = false
                    } else if c == "\\" {
                        escaped = true
                    } else if c == ch {
                        i = source.index(after: i)
                        break
                    }
                    i = source.index(after: i)
                }
                out += source[quoteStart..<i]
                codeStart = i
                continue
            }

            if source[i...].hasPrefix(tagClose) {
                out += rewriteCodeSegment(source[codeStart..<i], isStatementHead: isStatementHead)
                isStatementHead = false
                inNamespaceSetTag = false
                let tagEnd = source.index(i, offsetBy: 2)
                out += source[i..<tagEnd]
                i = tagEnd
                codeStart = i
                inTag = false
                continue
            }
            i = source.index(after: i)
        }
        if codeStart < source.endIndex {
            out += inTag
                ? rewriteCodeSegment(source[codeStart..<source.endIndex], isStatementHead: isStatementHead)
                : String(source[codeStart..<source.endIndex])
        }
        return out
    }

    private static func detectShadowedNamespaceAttributes(in source: String) -> [(String, String)] {
        guard source.contains("namespace(") || source.contains("namespace ("),
              let headRegex = try? NSRegularExpression(
                  pattern: #"\bset\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*namespace\s*\("#
              ),
              let kwRegex = try? NSRegularExpression(pattern: #"\b(items|keys|values|get)\s*="#) else {
            return []
        }
        var pairs: [(String, String)] = []
        let fullRange = NSRange(source.startIndex..<source.endIndex, in: source)
        for m in headRegex.matches(in: source, range: fullRange) {
            guard let varRange = Range(m.range(at: 1), in: source),
                  let callEnd = Range(m.range, in: source)?.upperBound else { continue }
            let nsVar = String(source[varRange])
            var idx = callEnd
            var depth = 1
            while idx < source.endIndex && depth > 0 {
                let ch = source[idx]
                if ch == "(" { depth += 1 }
                else if ch == ")" { depth -= 1 }
                idx = source.index(after: idx)
            }
            let argSlice = String(source[callEnd..<idx])
            let argRange = NSRange(argSlice.startIndex..<argSlice.endIndex, in: argSlice)
            for km in kwRegex.matches(in: argSlice, range: argRange) {
                guard let attrRange = Range(km.range(at: 1), in: argSlice) else { continue }
                let attr = String(argSlice[attrRange])
                if !pairs.contains(where: { $0.0 == nsVar && $0.1 == attr }) {
                    pairs.append((nsVar, attr))
                }
            }
        }
        return pairs
    }

    public func render(messages: [QWJson],
                       options: QWChatTemplateOptions = QWChatTemplateOptions()) throws -> String {
        guard !messages.isEmpty else {
            throw QWChatTemplateError.emptyConversation
        }
        if isInkling2Template {
            return try renderInkling2(messages: messages, options: options)
        }
        if usesGlobalSettingProtocol {
            return try renderErnie(messages: messages, options: options)
        }
        var context = tokenizerContext.mapValues(Self.value(from:))
        context["__qw_dict"] = .function(Self.pythonCompatibleDict)
        context["messages"] = .array(messages.map(Self.value(from:)))
        context["add_generation_prompt"] = .boolean(options.addGenerationPrompt)
        context["tools"] = Self.value(from: options.tools)
        let effectiveThinking = (options.enableThinking == .undefined) ? options.reasoning : options.enableThinking
        let effectiveReasoning = (options.reasoning == .undefined) ? options.enableThinking : options.reasoning
        context["reasoning_effort"] = Self.value(from: options.reasoningEffort)
        context["enable_thinking"] = .init(Self.value(from: effectiveThinking))
        context["reasoning"] = Self.value(from: effectiveReasoning)
        context["preserve_thinking"] = Self.value(from: options.preserveThinking)
        context["add_vision_id"] = .boolean(options.addVisionId)
        var rendered = try template.render(context)
        if options.addGenerationPrompt,
           case .bool(false) = effectiveThinking,
           let openThink = thinkOpenTag,
           let closeThink = thinkCloseTag,
           rendered.hasSuffix(openThink) {
            rendered += closeThink
        }
        return rendered
    }

    private static let pythonCompatibleDict: @Sendable ([Value], [String: Value], Environment) throws -> Value = {
        args, kwargs, _ in
        var ordered = OrderedDictionary<ObjectKey, Value>()
        if let first = args.first {
            switch first {
            case .object(let existing):
                for (k, v) in existing { ordered[k] = v }
            case .array(let items):
                for item in items {
                    guard case .array(let pair) = item, pair.count == 2 else { continue }
                    switch pair[0] {
                    case .string(let s): ordered[.string(s)] = pair[1]
                    case .int(let n): ordered[.int(n)] = pair[1]
                    default: break
                    }
                }
            default:
                break
            }
        }
        for (key, value) in kwargs.sorted(by: { $0.key < $1.key }) {
            ordered[.string(key)] = value
        }
        return .object(ordered)
    }

    private static func value(from json: QWJson) -> Value {
        switch json {
        case .undefined: return .undefined
        case .null: return .null
        case .bool(let value): return .boolean(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map(Self.value(from:)))
        case .object(let pairs):
            var object = OrderedDictionary<String, Value>()
            for (key, value) in pairs {
                object[key] = Self.value(from: value)
            }
            return .object(object)
        }
    }
}

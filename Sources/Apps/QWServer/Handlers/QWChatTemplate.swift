import Foundation
import Jinja
import QWCore
import QWToken

public struct QWChatTemplateOptions: Sendable {
    public var addGenerationPrompt = true
    public var tools: QWJson = .undefined
    public var reasoningEffort: QWJson = .undefined
    public var enableThinking: QWJson = .undefined
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

    /// Normalizes Python Jinja2 integer attribute accesses (e.g. `m.content.0.type`)
    /// into subscript indexing (`m.content[0].type`) outside quoted string literals.
    private static func normalizePythonJinjaSyntax(_ source: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"(\b[A-Za-z_][A-Za-z0-9_]*|\])\.([0-9]+)\b"#) else {
            return source
        }
        func rewriteCodeSegment(_ segment: Substring) -> String {
            var current = String(segment)
            for _ in 0..<8 {
                let range = NSRange(current.startIndex..<current.endIndex, in: current)
                let next = regex.stringByReplacingMatches(in: current, range: range, withTemplate: "$1[$2]")
                if next == current { break }
                current = next
            }
            return current
        }

        var out = ""
        out.reserveCapacity(source.count)
        var i = source.startIndex
        var inTag = false
        var tagClose = ""
        var codeStart = source.startIndex

        while i < source.endIndex {
            if !inTag {
                let rest = source[i...]
                if rest.hasPrefix("{{") || rest.hasPrefix("{%") {
                    out += source[codeStart..<i]
                    tagClose = rest.hasPrefix("{{") ? "}}" : "%}"
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
                out += rewriteCodeSegment(source[codeStart..<i])
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
                out += rewriteCodeSegment(source[codeStart..<i])
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
            out += inTag ? rewriteCodeSegment(source[codeStart..<source.endIndex]) : String(source[codeStart..<source.endIndex])
        }
        return out
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
        context["messages"] = .array(messages.map(Self.value(from:)))
        context["add_generation_prompt"] = .boolean(options.addGenerationPrompt)
        context["tools"] = Self.value(from: options.tools)
        context["reasoning_effort"] = Self.value(from: options.reasoningEffort)
        context["enable_thinking"] = .init(Self.value(from: options.enableThinking))
        context["preserve_thinking"] = Self.value(from: options.preserveThinking)
        context["add_vision_id"] = .boolean(options.addVisionId)
        var rendered = try template.render(context)
        if options.addGenerationPrompt,
           case .bool(false) = options.enableThinking,
           let openThink = thinkOpenTag,
           let closeThink = thinkCloseTag,
           rendered.hasSuffix(openThink) {
            rendered += closeThink
        }
        return rendered
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

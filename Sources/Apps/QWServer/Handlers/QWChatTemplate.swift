import Foundation
import Jinja
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

    public init(source: String, tokenizerContext: [String: QWJson] = [:]) throws {
        let normalizedSource = Self.normalizePythonJinjaSyntax(source)
        self.template = try Template(normalizedSource)
        self.tokenizerContext = tokenizerContext
    }

    /// Normalizes Python Jinja2 integer attribute accesses (e.g. `m.content.0.type`)
    /// into subscript indexing (`m.content[0].type`) required by Swift Jinja.
    private static func normalizePythonJinjaSyntax(_ source: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"(\b[A-Za-z_][A-Za-z0-9_]*|\])\.([0-9]+)\b"#) else {
            return source
        }
        var current = source
        for _ in 0..<4 {
            let range = NSRange(current.startIndex..<current.endIndex, in: current)
            let next = regex.stringByReplacingMatches(in: current, range: range, withTemplate: "$1[$2]")
            if next == current { break }
            current = next
        }
        return current
    }

    public func render(messages: [QWJson],
                       options: QWChatTemplateOptions = QWChatTemplateOptions()) throws -> String {
        guard !messages.isEmpty else {
            throw QWChatTemplateError.emptyConversation
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
           rendered.hasSuffix("<think>") {
            rendered += "</think>"
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

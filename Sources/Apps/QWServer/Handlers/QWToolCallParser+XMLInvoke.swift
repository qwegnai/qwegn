// QWToolCallParser+XMLInvoke.swift — Dialect-agnostic tool envelopes and XML/GCML <invoke name="..."><parameter name="..."> parsing.
import Foundation
import QWCore
import QWToken

/// Open/close XML wrapper pair enclosing structured tool calls on the wire.
public struct QWToolCallEnvelope: Equatable, Hashable, Sendable {
    public let openTag: String
    public let closeTag: String
    public let closeTagPrefix: String

    public static let standard = QWToolCallEnvelope(openTag: "<tool_call>", closeTag: "</tool_call>")
    public static let ernie = QWToolCallEnvelope(openTag: "<ernie:tool_call>", closeTag: "</ernie:tool_call>")
    public static let gcml = QWToolCallEnvelope(openTag: "<｜GCML｜tool_calls>", closeTag: "</｜GCML｜tool_calls>")
    public static let knownEnvelopes: [QWToolCallEnvelope] = [
        .standard,
        .ernie,
        .gcml,
    ]

    public init(openTag: String, closeTag: String) {
        self.openTag = openTag
        self.closeTag = closeTag
        self.closeTagPrefix = closeTag.hasSuffix(">") ? String(closeTag.dropLast()) : closeTag
    }

    public static func envelopes(prioritizing active: QWToolCallEnvelope) -> [QWToolCallEnvelope] {
        var list: [QWToolCallEnvelope] = [active]
        for env in knownEnvelopes where !list.contains(env) {
            list.append(env)
        }
        return list
    }

    /// Builds an open/close pair from `<...>` (e.g. `<acme:tool_call>` -> `</acme:tool_call>`).
    public static func fromOpenTag(_ openTag: String) -> QWToolCallEnvelope? {
        guard openTag.hasPrefix("<"), openTag.hasSuffix(">"), !openTag.hasPrefix("</"), openTag.count > 2 else {
            return nil
        }
        return QWToolCallEnvelope(openTag: openTag, closeTag: "</" + String(openTag.dropFirst()))
    }

    /// Detects the active tool-call envelope declared by a chat template source.
    public static func detect(in templateSource: String) -> QWToolCallEnvelope {
        if templateSource.contains(gcml.openTag) {
            return .gcml
        }
        if let regex = try? NSRegularExpression(pattern: #"<｜[A-Za-z0-9_]+｜tool_calls?>"#),
           let m = regex.firstMatch(
               in: templateSource,
               range: NSRange(templateSource.startIndex..<templateSource.endIndex, in: templateSource)
           ),
           let r = Range(m.range, in: templateSource),
           let env = fromOpenTag(String(templateSource[r])) {
            return env
        }
        if let regex = try? NSRegularExpression(pattern: #"<([A-Za-z0-9_]+):tool_call>"#),
           let m = regex.firstMatch(
               in: templateSource,
               range: NSRange(templateSource.startIndex..<templateSource.endIndex, in: templateSource)
           ),
           let r = Range(m.range, in: templateSource),
           let env = fromOpenTag(String(templateSource[r])) {
            return env
        }
        if templateSource.contains("<|ernie:im_start|>") && templateSource.contains("<ernie:global_setting>") {
            return .ernie
        }
        return .standard
    }
}

extension QWToolCallParser {
    private static let invokeTagPairs: [(open: String, close: String)] = [
        ("<｜GCML｜invoke", "</｜GCML｜invoke>"),
        ("<invoke", "</invoke>"),
    ]
    private static let paramTagPairs: [(open: String, close: String)] = [
        ("<｜GCML｜parameter", "</｜GCML｜parameter>"),
        ("<parameter", "</parameter>"),
    ]

    static func earliestToolSegmentOpen(
        in text: Substring,
        envelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> (range: Range<String.Index>, envelope: QWToolCallEnvelope)? {
        var best: (range: Range<String.Index>, envelope: QWToolCallEnvelope)?
        for env in envelopes {
            guard let r = text.range(of: env.openTag) else { continue }
            if best.map({ r.lowerBound < $0.range.lowerBound }) ?? true {
                best = (r, env)
            }
        }
        return best
    }

    static func firstCompleteToolCallEnd(
        in text: String,
        envelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> String.Index? {
        var earliest: String.Index?
        for env in envelopes {
            guard let open = text.range(of: env.openTag),
                  let close = matchingToolCallClose(in: text[open.upperBound...], envelope: env) else {
                continue
            }
            if earliest.map({ close.upperBound < $0 }) ?? true {
                earliest = close.upperBound
            }
        }
        return earliest
    }

    static func parseToolSegmentCalls(inner: String, schemas: ToolSchemas) -> [ToolCall] {
        if inner.hasPrefix("{") && inner.hasSuffix("}"),
           let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        if inner.contains("<function=") || inner.contains("<arg_key>"),
           let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        if inner.contains("<invoke") || inner.contains("<｜GCML｜invoke") {
            let xmlCalls = parseXMLInvokeToolCalls(inner, schemas: schemas)
            if !xmlCalls.isEmpty { return xmlCalls }
        }
        if let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        return parseXMLInvokeToolCalls(inner, schemas: schemas)
    }

    static func isBalancedToolPayload(_ raw: Substring) -> Bool {
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
        return isBalancedXMLInvokePayload(raw)
    }

    static func hasUnclosedLiteralContainer(_ raw: Substring) -> Bool {
        let trimmed = raw.drop(while: { $0.isWhitespace })
        if trimmed.hasPrefix("{") {
            var depth = 0, inString = false, escaped = false
            for ch in trimmed {
                if inString {
                    if escaped { escaped = false } else if ch == "\\" { escaped = true } else if ch == "\"" { inString = false }
                    continue
                }
                if ch == "\"" { inString = true } else if ch == "{" { depth += 1 } else if ch == "}" { depth -= 1 }
            }
            return inString || depth > 0
        }
        if trimmed.hasPrefix("<function=") {
            return trimmed.components(separatedBy: "<parameter=").count > trimmed.components(separatedBy: "</parameter>").count
        }
        return hasUnclosedXMLInvoke(trimmed)
    }

    static func matchingToolCallClose(
        in afterOpen: Substring,
        envelope: QWToolCallEnvelope = .standard
    ) -> Range<String.Index>? {
        var searchFrom = afterOpen.startIndex
        while let close = afterOpen[searchFrom...].range(of: envelope.closeTag) {
            let prefix = afterOpen[..<close.lowerBound]
            if isBalancedToolPayload(prefix) {
                return close
            }
            let tail = afterOpen[close.upperBound...].drop(while: { $0.isWhitespace })
            guard let nextClose = tail.range(of: envelope.closeTag) else { return close }
            if !hasUnclosedLiteralContainer(prefix) {
                if tail.hasPrefix(envelope.openTag) { return close }
                if let nextOpen = tail.range(of: envelope.openTag), nextOpen.lowerBound < nextClose.lowerBound {
                    return close
                }
            }
            searchFrom = close.upperBound
        }
        return nil
    }

    static func isBalancedXMLInvokePayload(_ raw: Substring) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for (iIdx, inv) in invokeTagPairs.enumerated() where trimmed.hasSuffix(inv.close) {
            let prm = paramTagPairs[iIdx]
            let invokeOpens = trimmed.components(separatedBy: inv.open).count - 1
            let invokeCloses = trimmed.components(separatedBy: inv.close).count - 1
            let paramOpens = trimmed.components(separatedBy: prm.open).count - 1
            let paramCloses = trimmed.components(separatedBy: prm.close).count - 1
            if invokeOpens >= 1 && invokeOpens == invokeCloses && paramOpens == paramCloses {
                return true
            }
        }
        return false
    }

    static func hasUnclosedXMLInvoke(_ raw: Substring) -> Bool {
        let trimmed = raw.drop(while: { $0.isWhitespace })
        for (iIdx, inv) in invokeTagPairs.enumerated() where trimmed.contains(inv.open) {
            let prm = paramTagPairs[iIdx]
            let invokeOpens = trimmed.components(separatedBy: inv.open).count - 1
            let invokeCloses = trimmed.components(separatedBy: inv.close).count - 1
            let paramOpens = trimmed.components(separatedBy: prm.open).count - 1
            let paramCloses = trimmed.components(separatedBy: prm.close).count - 1
            if invokeOpens > invokeCloses || paramOpens > paramCloses { return true }
        }
        return false
    }

    static func parseXMLInvokeToolCalls(_ inner: String, schemas: ToolSchemas) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = inner.startIndex
        while let match = nextNamedTag(in: inner, from: cursor, tagPairs: invokeTagPairs) {
            let body: String
            let nextCursor: String.Index
            if let (end, after) = findBoundaryCloseTag(
                in: inner, from: match.bodyStart, closeTag: match.closeTag, nextOpenTag: match.openTag
            ) {
                body = String(inner[match.bodyStart..<end])
                nextCursor = after
            } else {
                body = String(inner[match.bodyStart...])
                nextCursor = inner.endIndex
            }
            var members: [String] = []
            for param in extractXMLNamedParameters(in: body) {
                let encodedVal = formatXMLParameterValue(
                    param.value,
                    stringAttr: param.stringAttr,
                    schemaType: schemas.type(function: match.name, parameter: param.name)
                )
                members.append("\(jsonLiteral(param.name)):\(encodedVal)")
            }
            calls.append(ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: match.name, arguments: "{" + members.joined(separator: ",") + "}")
            ))
            cursor = nextCursor
        }
        return calls
    }

    private static func formatXMLParameterValue(
        _ raw: String,
        stringAttr: Bool?,
        schemaType: ToolSchemas.ParamType
    ) -> String {
        if stringAttr == true {
            return jsonLiteral(raw)
        }
        if stringAttr == false {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let low = trimmed.lowercased()
            if low == "true" { return "true" }
            if low == "false" { return "false" }
            if low == "null" { return "null" }
            if let iv = Int64(trimmed) { return String(iv) }
            if let data = trimmed.data(using: .utf8),
               let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
               let out = try? JSONSerialization.data(
                   withJSONObject: value,
                   options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]
               ),
               let str = String(data: out, encoding: .utf8) {
                return str
            }
            if schemaType != .string {
                return jsonValue(raw, type: schemaType)
            }
            return jsonLiteral(raw)
        }
        return jsonValue(raw, type: schemaType)
    }

    private struct NamedParameter {
        let name: String
        let value: String
        let stringAttr: Bool?
    }

    private struct NamedTagMatch {
        let name: String
        let stringAttr: Bool?
        let openTag: String
        let closeTag: String
        let bodyStart: String.Index
    }

    private static func extractXMLNamedParameters(in body: String) -> [NamedParameter] {
        var out: [NamedParameter] = []
        var cursor = body.startIndex
        while let match = nextNamedTag(in: body, from: cursor, tagPairs: paramTagPairs) {
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: match.bodyStart, closeTag: match.closeTag, nextOpenTag: match.openTag
            ) {
                out.append(NamedParameter(
                    name: match.name,
                    value: String(body[match.bodyStart..<valEnd]),
                    stringAttr: match.stringAttr
                ))
                cursor = nextCursor
            } else {
                out.append(NamedParameter(
                    name: match.name,
                    value: String(body[match.bodyStart...]),
                    stringAttr: match.stringAttr
                ))
                break
            }
        }
        return out
    }

    private static func nextNamedTag(
        in text: String,
        from start: String.Index,
        tagPairs: [(open: String, close: String)]
    ) -> NamedTagMatch? {
        var search = start
        while search < text.endIndex {
            var earliest: (range: Range<String.Index>, open: String, close: String)?
            for pair in tagPairs {
                if let r = text[search...].range(of: pair.open),
                   earliest.map({ r.lowerBound < $0.range.lowerBound }) ?? true {
                    earliest = (r, pair.open, pair.close)
                }
            }
            guard let hit = earliest else { return nil }
            var idx = hit.range.upperBound
            guard idx < text.endIndex, text[idx].isWhitespace else {
                search = hit.range.upperBound
                continue
            }
            var parsedName: String?
            var parsedStringAttr: Bool?
            var validAttrs = true
            while idx < text.endIndex {
                while idx < text.endIndex, text[idx].isWhitespace { idx = text.index(after: idx) }
                guard idx < text.endIndex else { validAttrs = false; break }
                if text[idx] == ">" { break }
                guard let eqIdx = text[idx...].firstIndex(of: "=") else { validAttrs = false; break }
                let attrKey = text[idx..<eqIdx].trimmingCharacters(in: .whitespaces)
                let afterEq = text.index(after: eqIdx)
                guard afterEq < text.endIndex, (text[afterEq] == "\"" || text[afterEq] == "'") else {
                    validAttrs = false
                    break
                }
                let quote = text[afterEq]
                let valStart = text.index(after: afterEq)
                guard let quoteEnd = text[valStart...].firstIndex(of: quote) else { return nil }
                let attrVal = String(text[valStart..<quoteEnd])
                if attrVal.contains("\n") { validAttrs = false; break }
                if attrKey == "name" {
                    parsedName = attrVal
                } else if attrKey == "string" {
                    let low = attrVal.lowercased()
                    if low == "true" { parsedStringAttr = true }
                    else if low == "false" { parsedStringAttr = false }
                }
                idx = text.index(after: quoteEnd)
            }
            if validAttrs, idx < text.endIndex, text[idx] == ">",
               let name = parsedName, !name.isEmpty {
                return NamedTagMatch(
                    name: name,
                    stringAttr: parsedStringAttr,
                    openTag: hit.open,
                    closeTag: hit.close,
                    bodyStart: text.index(after: idx)
                )
            }
            search = hit.range.upperBound
        }
        return nil
    }
}

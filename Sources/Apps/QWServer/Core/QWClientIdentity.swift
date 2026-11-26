// QWClientIdentity.swift — Privacy-safe client fingerprinting and trace context.

import Foundation
import Network

/// Per-request trace metadata carried across HTTP handlers, engine logs, and request records.
public struct QWRequestTraceContext: Sendable, Equatable {
    public let method: String
    public let route: String
    public let client: String
    public let requestId: String?
    public let sessionId: String?
    public let peerHost: String?

    public init(
        method: String,
        route: String,
        client: String,
        requestId: String? = nil,
        sessionId: String? = nil,
        peerHost: String? = nil
    ) {
        self.method = method
        self.route = route
        self.client = client
        self.requestId = requestId
        self.sessionId = sessionId
        self.peerHost = peerHost
    }

    public var logTag: String {
        var parts = ["client=\(client)"]
        if let requestId, !requestId.isEmpty { parts.append("req=\(requestId)") }
        if let sessionId, !sessionId.isEmpty { parts.append("sess=\(sessionId)") }
        if let peerHost, !peerHost.isEmpty { parts.append("peer=\(peerHost)") }
        return parts.joined(separator: " ")
    }

    public func withRequestId(_ id: String?) -> QWRequestTraceContext {
        let sanitized = QWClientIdentity.sanitizeIdentifier(id) ?? requestId
        return QWRequestTraceContext(
            method: method,
            route: route,
            client: client,
            requestId: sanitized,
            sessionId: sessionId,
            peerHost: peerHost
        )
    }

    public func withClient(_ newClient: String) -> QWRequestTraceContext {
        let sanitized = QWClientIdentity.sanitizeLabel(newClient) ?? client
        return QWRequestTraceContext(
            method: method,
            route: route,
            client: sanitized,
            requestId: requestId,
            sessionId: sessionId,
            peerHost: peerHost
        )
    }
}

/// Resolves sanitized client labels and allowlisted debug header summaries from HTTP metadata.
public enum QWClientIdentity {
    private static let explicitClientHeaders = ["x-qwegn-client", "x-client-name", "x-app-name"]

    private static let debugHeaderAllowlist = [
        "user-agent",
        "x-qwegn-client",
        "x-client-name",
        "x-app-name",
        "x-title",
        "originator",
        "origin",
        "referer",
        "http-referer",
        "x-stainless-lang",
        "x-stainless-runtime",
        "x-stainless-package-version",
        "anthropic-version",
        "x-request-id",
        "x-session-id",
    ]

    private static let knownAgentMatchers: [(needles: [String], label: String)] = [
        (["claude-cli", "claude-code", "claude code"], "claude-code"),
        (["codex_cli_rs", "openai/codex", "codex-cli"], "codex"),
        (["opencode"], "opencode"),
        (["hermes"], "hermes"),
        (["llmprobe"], "llmprobe"),
        (["qwg-cli", "qwegine", "qwg/"], "qwg-cli"),
        (["cursor"], "cursor"),
        (["zed/"], "zed"),
        (["continue"], "continue"),
        (["aider"], "aider"),
        (["cline"], "cline"),
        (["roo-code", "roocode"], "roo-code"),
        (["goose"], "goose"),
        (["cherrystudio", "cherry-studio"], "cherry-studio"),
        (["open-webui", "openwebui"], "open-webui"),
        (["ollama"], "ollama"),
    ]

    private static let genericRuntimePrefixes = [
        "undici", "node", "python-httpx", "aiohttp", "python-requests",
        "openai/", "asyncopenai/", "anthropic/",
    ]

    public static func makeTraceContext(
        method: String,
        route: String,
        headers: [String: String],
        peerEndpoint: NWEndpoint? = nil,
        mcpClientName: String? = nil
    ) -> QWRequestTraceContext {
        let resolved = resolve(headers: headers, peerEndpoint: peerEndpoint, mcpClientName: mcpClientName)
        let reqId = sanitizeIdentifier(header("x-request-id", in: headers))
        let sessId = sanitizeIdentifier(header("x-session-id", in: headers))
        return QWRequestTraceContext(
            method: method.uppercased(),
            route: route,
            client: resolved.client,
            requestId: reqId,
            sessionId: sessId,
            peerHost: resolved.peerHost
        )
    }

    public static func resolve(
        headers: [String: String],
        peerEndpoint: NWEndpoint? = nil,
        mcpClientName: String? = nil
    ) -> (client: String, peerHost: String?) {
        let peerHost = nonLoopbackPeerHost(from: peerEndpoint)

        for key in explicitClientHeaders {
            if let val = sanitizeLabel(header(key, in: headers)) {
                return (val, peerHost)
            }
        }
        if let title = sanitizeLabel(header("x-title", in: headers)) {
            return (title, peerHost)
        }
        if let originator = header("originator", in: headers) {
            if originator.lowercased().contains("codex") {
                return ("codex", peerHost)
            }
            if let label = sanitizeLabel(originator) {
                return (label, peerHost)
            }
        }
        if let mcp = sanitizeLabel(mcpClientName) {
            return (mcp, peerHost)
        }

        let referer = header("http-referer", in: headers) ?? header("referer", in: headers)
        let stainlessLang = header("x-stainless-lang", in: headers)
        let anthropicVersion = header("anthropic-version", in: headers)

        if let ua = header("user-agent", in: headers) {
            let client = fingerprintUserAgent(
                ua,
                referer: referer,
                stainlessLang: stainlessLang,
                anthropicVersion: anthropicVersion
            )
            if let client {
                return (client, peerHost)
            }
        }

        if let referer {
            if isWebUIReferer(referer) {
                return ("webui", peerHost)
            }
            if let host = URL(string: referer)?.host, let label = sanitizeLabel(host) {
                return (label, peerHost)
            }
        }

        if anthropicVersion != nil {
            return ("anthropic-sdk", peerHost)
        }
        if let lang = normalizedStainlessLang(stainlessLang) {
            return ("stainless-\(lang)", peerHost)
        }
        return ("unknown", peerHost)
    }

    public static func isGenericClient(_ client: String) -> Bool {
        let lower = client.lowercased()
        if lower == "unknown" || lower == "curl" || lower.hasPrefix("curl/") {
            return true
        }
        return lower == "node" || lower == "undici"
            || lower.hasPrefix("undici/") || lower.hasPrefix("node/")
            || lower.hasPrefix("python-httpx") || lower.hasPrefix("aiohttp")
            || lower.hasPrefix("python-requests")
    }

    public static func nonLoopbackPeerHost(from endpoint: NWEndpoint?) -> String? {
        guard case .hostPort(let host, _) = endpoint else { return nil }
        let raw = "\(host)".trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = raw.split(separator: "%", maxSplits: 1).first.map(String.init) ?? raw
        guard !cleaned.isEmpty else { return nil }
        let lower = cleaned.lowercased()
        if QWHttpSecurity.isLoopbackHost(cleaned) || lower.hasPrefix("::ffff:127.") {
            return nil
        }
        return sanitizeIdentifier(cleaned, maxLength: 48)
    }

    public static func sanitizeLabel(_ raw: String?, maxLength: Int = 32) -> String? {
        guard let raw else { return nil }
        let collapsed = raw
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
        guard !collapsed.isEmpty else { return nil }
        let allowed = String(collapsed.unicodeScalars.filter { s in
            (s.value >= 0x30 && s.value <= 0x39)
                || (s.value >= 0x41 && s.value <= 0x5A)
                || (s.value >= 0x61 && s.value <= 0x7A)
                || s == "." || s == "_" || s == ":" || s == "/" || s == "-"
        })
        let trimSet = CharacterSet(charactersIn: ".-_:/")
        let trimmed = String(allowed.trimmingCharacters(in: trimSet).prefix(maxLength))
        return trimmed.isEmpty ? nil : trimmed
    }

    public static func sanitizeIdentifier(_ raw: String?, maxLength: Int = 48) -> String? {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else { return nil }
        let filtered = String(trimmed.unicodeScalars.filter { s in
            (s.value >= 0x30 && s.value <= 0x39)
                || (s.value >= 0x41 && s.value <= 0x5A)
                || (s.value >= 0x61 && s.value <= 0x7A)
                || s == "." || s == "_" || s == ":" || s == "-"
        }.prefix(maxLength))
        return filtered.isEmpty ? nil : filtered
    }

    public static func debugHeaderSummary(headers: [String: String]) -> String {
        var parts: [String] = []
        for key in debugHeaderAllowlist {
            guard let val = header(key, in: headers) else { continue }
            let clean = String(val.unicodeScalars.filter { s in
                s.value >= 0x20 && s.value != 0x7F
            }.prefix(96)).trimmingCharacters(in: .whitespaces)
            if !clean.isEmpty {
                parts.append("\(key)=\(clean)")
            }
        }
        return parts.joined(separator: " ")
    }

    public static func header(_ name: String, in headers: [String: String]) -> String? {
        if let exact = headers[name] {
            let trimmed = exact.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        for (k, v) in headers where k.caseInsensitiveCompare(name) == .orderedSame {
            let trimmed = v.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        return nil
    }

    private static func fingerprintUserAgent(
        _ ua: String,
        referer: String?,
        stainlessLang: String?,
        anthropicVersion: String?
    ) -> String? {
        let lower = ua.lowercased()
        for matcher in knownAgentMatchers where matcher.needles.contains(where: { lower.contains($0) }) {
            return matcher.label
        }

        if lower.contains("mozilla/5.0") {
            if let referer, isWebUIReferer(referer) {
                return "webui"
            }
            if lower.contains("edg/") { return "browser/edge" }
            if lower.contains("firefox/") { return "browser/firefox" }
            if lower.contains("chrome/") || lower.contains("chromium/") { return "browser/chrome" }
            if lower.contains("safari/") { return "browser/safari" }
            return "browser"
        }

        let isSDKOrRuntime = genericRuntimePrefixes.contains(where: { lower.hasPrefix($0) || lower.contains($0) })
        if isSDKOrRuntime, let lang = normalizedStainlessLang(stainlessLang) {
            let prefix = (anthropicVersion != nil || lower.contains("anthropic"))
                ? "anthropic-"
                : (lower.contains("openai") ? "openai-" : "stainless-")
            return sanitizeLabel("\(prefix)\(lang)")
        }

        let firstToken = ua.split(whereSeparator: { $0.isWhitespace }).first.map(String.init)
        return sanitizeLabel(firstToken)
    }

    private static func isWebUIReferer(_ referer: String) -> Bool {
        guard let url = URL(string: referer) else {
            return referer.lowercased().contains("/ui")
        }
        return url.path == "/ui" || url.path.hasPrefix("/ui/")
    }

    private static func normalizedStainlessLang(_ raw: String?) -> String? {
        guard let lower = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !lower.isEmpty else { return nil }
        switch lower {
        case "python": return "py"
        case "js", "javascript", "typescript", "ts": return "js"
        case "go", "golang": return "go"
        default: return sanitizeLabel(lower, maxLength: 12)
        }
    }
}

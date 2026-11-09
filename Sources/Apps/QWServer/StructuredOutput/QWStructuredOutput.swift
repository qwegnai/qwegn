// QWStructuredOutput.swift — maps a chat request's response_format onto a
// compiled grammar, and caches the per-tokenizer token trie.

import Foundation
import QWToken

enum QWStructuredOutput {
    /// The grammar a request asks for; nil when it has no JSON response_format.
    static func grammar(for request: ChatCompletionRequest) throws -> QWJSONGrammar? {
        switch request.responseFormatType {
        case "json_object":
            return QWJSONGrammar.anyObject()
        case "json_schema":
            guard let format = request.responseFormat else { return nil }
            let nested = format.get("json_schema").get("schema")
            let schema = nested.isUndefined ? format.get("schema") : nested
            guard !schema.isUndefined else { return QWJSONGrammar.anyObject() }
            return try QWJSONGrammar.compile(schema: schema)
        default:
            return nil
        }
    }

    /// Whether the request demands enforcement (`json_schema.strict: true`).
    static func isStrict(_ request: ChatCompletionRequest) -> Bool {
        guard request.responseFormatType == "json_schema", let format = request.responseFormat else { return false }
        if case .bool(true) = format.get("json_schema").get("strict") { return true }
        if case .bool(true) = format.get("strict") { return true }
        return false
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var tables: [ObjectIdentifier: (QWTokenizer, QWJSONTokenTable)] = [:]

    /// The token trie for a tokenizer, built on first use.
    static func tokenTable(for tokenizer: QWTokenizer, vocabulary: Int) -> QWJSONTokenTable {
        let key = ObjectIdentifier(tokenizer)
        lock.lock()
        defer { lock.unlock() }
        if let (_, table) = tables[key], table.vocabulary == vocabulary { return table }
        let table = QWJSONTokenTable(tokenizer: tokenizer, vocabulary: vocabulary)
        // Holding the tokenizer keeps its identifier from being reused.
        tables = [key: (tokenizer, table)]
        return table
    }
}

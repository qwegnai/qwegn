// QWJSONSchemaGrammar.swift — supported JSON Schema subset compiled to a byte-level grammar.
// Supported: type, properties, required, additionalProperties, items, minItems, maxItems,
// enum, const, nullable, local $ref, and first-byte-disjoint anyOf/oneOf.
// Annotation keywords are ignored; any other keyword is rejected loudly.

import Foundation
import QWToken

public struct QWJSONSchemaError: Error, CustomStringConvertible, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { "unsupported response_format schema: \(message)" }
}

/// A literal set (enum/const values or object key names) as a byte trie.
/// Each terminal carries a payload (the key index for names, 0 for values).
final class QWJSLiteralTrie: @unchecked Sendable {
    private(set) var next: [[UInt8: Int32]] = [[:]]
    private(set) var payload: [Int32] = [-1]
    /// Bitmask of key payloads reachable from each node (key tries hold <= 64 keys).
    private(set) var reachable: [UInt64] = [0]

    func insert(_ bytes: [UInt8], payload value: Int32) {
        let bit: UInt64 = (value >= 0 && value < 64) ? (UInt64(1) << UInt64(value)) : 0
        var node: Int32 = 0
        reachable[0] |= bit
        for b in bytes {
            if let child = next[Int(node)][b] {
                node = child
            } else {
                let child = Int32(next.count)
                next.append([:])
                payload.append(-1)
                reachable.append(0)
                next[Int(node)][b] = child
                node = child
            }
            reachable[Int(node)] |= bit
        }
        payload[Int(node)] = value
    }

    @inline(__always) func child(_ node: Int32, _ b: UInt8) -> Int32? { next[Int(node)][b] }
    @inline(__always) func terminal(_ node: Int32) -> Int32 { payload[Int(node)] }
    @inline(__always) func isLeaf(_ node: Int32) -> Bool { next[Int(node)].isEmpty }
    var firstBytes: Set<UInt8> { Set(next[0].keys) }

    func words() -> [[UInt8]] {
        var out: [[UInt8]] = []
        func walk(_ node: Int32, _ prefix: [UInt8]) {
            if terminal(node) >= 0 { out.append(prefix) }
            for (b, c) in next[Int(node)] { walk(c, prefix + [b]) }
        }
        walk(0, [])
        return out
    }
}

final class QWJSObject: @unchecked Sendable {
    var names: [String] = []
    var nodes: [QWJSNode] = []
    var requiredMask: UInt64 = 0
    /// Schema for keys outside `names`; nil forbids them.
    var additional: QWJSNode?
    let keyTrie = QWJSLiteralTrie()
    var allMask: UInt64 {
        names.count >= 64 ? ~0 : (UInt64(1) << UInt64(names.count)) - 1
    }
}

final class QWJSArray: @unchecked Sendable {
    let items: QWJSNode
    let minItems: Int
    let maxItems: Int?
    init(items: QWJSNode, minItems: Int, maxItems: Int?) {
        self.items = items; self.minItems = minItems; self.maxItems = maxItems
    }
}

struct QWJSTypes {
    var object: QWJSObject?
    var array: QWJSArray?
    var string = false
    var number = false
    var integer = false
    var boolean = false
    var null = false

    var firstBytes: Set<UInt8> {
        var s = Set<UInt8>()
        if object != nil { s.insert(UInt8(ascii: "{")) }
        if array != nil { s.insert(UInt8(ascii: "[")) }
        if string { s.insert(UInt8(ascii: "\"")) }
        if number || integer { s.formUnion(Array("-0123456789".utf8)) }
        if boolean { s.formUnion([UInt8(ascii: "t"), UInt8(ascii: "f")]) }
        if null { s.insert(UInt8(ascii: "n")) }
        return s
    }
}

/// A schema node: the JSON types it admits plus an optional literal set.
/// The two never share a first byte, so a value's first byte picks one.
final class QWJSNode: @unchecked Sendable {
    var types = QWJSTypes()
    var literals: QWJSLiteralTrie?
}

/// A compiled response format: the root node of the byte grammar.
public final class QWJSONGrammar: @unchecked Sendable {
    let root: QWJSNode
    static let trueTrie = literalTrie("true")
    static let falseTrie = literalTrie("false")
    static let nullTrie = literalTrie("null")

    init(root: QWJSNode) { self.root = root }

    private static func literalTrie(_ word: String) -> QWJSLiteralTrie {
        let t = QWJSLiteralTrie()
        t.insert(Array(word.utf8), payload: 0)
        return t
    }

    /// Any JSON object (response_format json_object).
    public static func anyObject() -> QWJSONGrammar {
        let node = QWJSNode()
        node.types.object = anyNode().types.object
        return QWJSONGrammar(root: node)
    }

    static func anyNode() -> QWJSNode {
        let node = QWJSNode()
        let object = QWJSObject()
        object.additional = node
        node.types = QWJSTypes(object: object, array: QWJSArray(items: node, minItems: 0, maxItems: nil),
                               string: true, number: true, integer: false, boolean: true, null: true)
        return node
    }

    /// Compiles a JSON Schema (the supported subset) into a grammar.
    public static func compile(schema: QWJson) throws -> QWJSONGrammar {
        let compiler = Compiler(root: schema)
        return QWJSONGrammar(root: try compiler.node(for: schema))
    }

    private final class Compiler {
        let root: QWJson
        var refs: [String: QWJSNode] = [:]
        init(root: QWJson) { self.root = root }

        static let annotations: Set<String> = [
            "title", "description", "default", "examples", "$schema", "$id", "$comment",
            "deprecated", "readOnly", "writeOnly", "format", "contentMediaType",
            "contentEncoding", "$defs", "definitions",
        ]
        static let handled: Set<String> = [
            "type", "properties", "required", "additionalProperties", "items", "minItems",
            "maxItems", "enum", "const", "nullable", "$ref", "anyOf", "oneOf", "allOf",
        ]

        func node(for schema: QWJson) throws -> QWJSNode {
            switch schema {
            case .bool(true): return QWJSONGrammar.anyNode()
            case .bool(false): throw QWJSONSchemaError("a false schema admits no value")
            case .object(let pairs): return try objectSchema(pairs)
            default: throw QWJSONSchemaError("a schema must be an object or a boolean")
            }
        }

        private func resolve(_ ref: String) throws -> QWJSNode {
            if let cached = refs[ref] { return cached }
            let target: QWJson
            if ref == "#" {
                target = root
            } else if ref.hasPrefix("#/$defs/") || ref.hasPrefix("#/definitions/") {
                let parts = ref.dropFirst(2).split(separator: "/", maxSplits: 1).map(String.init)
                let name = parts[1].replacingOccurrences(of: "~1", with: "/")
                    .replacingOccurrences(of: "~0", with: "~")
                let found = root.get(parts[0]).get(name)
                guard !found.isUndefined else { throw QWJSONSchemaError("$ref \(ref) does not resolve") }
                target = found
            } else {
                throw QWJSONSchemaError("only local $ref (#, #/$defs/..., #/definitions/...) is supported, got \(ref)")
            }
            // A recursive reference sees this node before it is filled in.
            let placeholder = QWJSNode()
            refs[ref] = placeholder
            let built = try node(for: target)
            placeholder.types = built.types
            placeholder.literals = built.literals
            return placeholder
        }

        private func onlyAnnotationsBesides(_ key: String, _ dict: [String: QWJson]) -> Bool {
            dict.keys.allSatisfy { $0 == key || Self.annotations.contains($0) }
        }

        private func objectSchema(_ pairs: [(String, QWJson)]) throws -> QWJSNode {
            var dict: [String: QWJson] = [:]
            for (k, v) in pairs { dict[k] = v }
            for key in dict.keys.sorted() where !Self.handled.contains(key) && !Self.annotations.contains(key) {
                throw QWJSONSchemaError("keyword '\(key)' is not supported")
            }
            if let ref = dict["$ref"] {
                guard case .string(let r) = ref else { throw QWJSONSchemaError("$ref must be a string") }
                guard onlyAnnotationsBesides("$ref", dict) else {
                    throw QWJSONSchemaError("$ref with sibling constraints is not supported")
                }
                return try resolve(r)
            }
            if let all = dict["allOf"] {
                guard case .array(let items) = all, items.count == 1, onlyAnnotationsBesides("allOf", dict) else {
                    throw QWJSONSchemaError("allOf is supported only with a single subschema and no siblings")
                }
                return try node(for: items[0])
            }
            for key in ["anyOf", "oneOf"] {
                guard let branches = dict[key] else { continue }
                guard case .array(let items) = branches, !items.isEmpty else {
                    throw QWJSONSchemaError("\(key) must be a non-empty array")
                }
                guard onlyAnnotationsBesides(key, dict) else {
                    throw QWJSONSchemaError("\(key) with sibling constraints is not supported")
                }
                var merged = try node(for: items[0])
                for item in items.dropFirst() { merged = try union(merged, try node(for: item)) }
                return merged
            }
            let nullable: Bool
            if case .bool(true)? = dict["nullable"] { nullable = true } else { nullable = false }
            let result = QWJSNode()
            if let values = dict["enum"] ?? dict["const"].map({ QWJson.array([$0]) }) {
                guard case .array(let list) = values, !list.isEmpty else {
                    throw QWJSONSchemaError("enum must be a non-empty array")
                }
                let trie = QWJSLiteralTrie()
                for v in list { trie.insert(Array(QWJSONGrammar.canonical(v).utf8), payload: 0) }
                if nullable { trie.insert(Array("null".utf8), payload: 0) }
                result.literals = trie
                return result
            }
            var typeNames: [String]
            switch dict["type"] {
            case .string(let t)?: typeNames = [t]
            case .array(let list)?:
                typeNames = try list.map {
                    guard case .string(let t) = $0 else { throw QWJSONSchemaError("type entries must be strings") }
                    return t
                }
            case nil:
                if dict["properties"] != nil || dict["required"] != nil || dict["additionalProperties"] != nil {
                    typeNames = ["object"]
                } else if dict["items"] != nil || dict["minItems"] != nil || dict["maxItems"] != nil {
                    typeNames = ["array"]
                } else {
                    return QWJSONGrammar.anyNode()
                }
            default: throw QWJSONSchemaError("type must be a string or an array of strings")
            }
            if nullable { typeNames.append("null") }
            for t in typeNames {
                switch t {
                case "object": result.types.object = try objectType(dict)
                case "array": result.types.array = try arrayType(dict)
                case "string": result.types.string = true
                case "number": result.types.number = true
                case "integer": result.types.integer = true
                case "boolean": result.types.boolean = true
                case "null": result.types.null = true
                default: throw QWJSONSchemaError("unknown type '\(t)'")
                }
            }
            return result
        }

        private func objectType(_ dict: [String: QWJson]) throws -> QWJSObject {
            let object = QWJSObject()
            if let props = dict["properties"] {
                guard case .object(let pairs) = props else { throw QWJSONSchemaError("properties must be an object") }
                for (name, sub) in pairs {
                    guard !object.names.contains(name) else { throw QWJSONSchemaError("duplicate property '\(name)'") }
                    object.names.append(name)
                    object.nodes.append(try node(for: sub))
                }
            }
            switch dict["additionalProperties"] {
            case nil, .bool(true)?: object.additional = QWJSONGrammar.anyNode()
            case .bool(false)?: object.additional = nil
            case let sub?: object.additional = try node(for: sub)
            }
            var required: [String] = []
            if let req = dict["required"] {
                guard case .array(let list) = req else { throw QWJSONSchemaError("required must be an array") }
                for entry in list {
                    guard case .string(let name) = entry else { throw QWJSONSchemaError("required entries must be strings") }
                    required.append(name)
                }
            }
            for name in required where !object.names.contains(name) {
                guard let extra = object.additional else {
                    throw QWJSONSchemaError("required property '\(name)' is not allowed by properties")
                }
                object.names.append(name)
                object.nodes.append(extra)
            }
            guard object.names.count <= 64 else {
                throw QWJSONSchemaError("objects with more than 64 named properties are not supported")
            }
            for name in required {
                object.requiredMask |= UInt64(1) << UInt64(object.names.firstIndex(of: name)!)
            }
            for (index, name) in object.names.enumerated() {
                object.keyTrie.insert(Array(QWJSONGrammar.escapedString(name).utf8) + [UInt8(ascii: "\"")],
                                      payload: Int32(index))
            }
            return object
        }

        private func arrayType(_ dict: [String: QWJson]) throws -> QWJSArray {
            let items: QWJSNode
            switch dict["items"] {
            case nil: items = QWJSONGrammar.anyNode()
            case .array?: throw QWJSONSchemaError("tuple-form items is not supported")
            case let sub?: items = try node(for: sub)
            }
            func count(_ key: String) throws -> Int? {
                switch dict[key] {
                case nil: return nil
                case .int(let n)? where n >= 0: return n
                case .double(let d)? where d >= 0 && d == d.rounded() && d < 1e9: return Int(d)
                default: throw QWJSONSchemaError("\(key) must be a non-negative integer")
                }
            }
            let minItems = try count("minItems") ?? 0
            let maxItems = try count("maxItems")
            if let maxItems, maxItems < minItems { throw QWJSONSchemaError("maxItems is below minItems") }
            return QWJSArray(items: items, minItems: minItems, maxItems: maxItems)
        }

        private func union(_ a: QWJSNode, _ b: QWJSNode) throws -> QWJSNode {
            let out = QWJSNode()
            var types = a.types
            if let object = b.types.object {
                guard types.object == nil else { throw QWJSONSchemaError("anyOf with two object branches is not supported") }
                types.object = object
            }
            if let array = b.types.array {
                guard types.array == nil else { throw QWJSONSchemaError("anyOf with two array branches is not supported") }
                types.array = array
            }
            types.string = types.string || b.types.string
            types.number = types.number || b.types.number
            types.integer = types.integer || b.types.integer
            types.boolean = types.boolean || b.types.boolean
            types.null = types.null || b.types.null
            out.types = types
            switch (a.literals, b.literals) {
            case (nil, nil): break
            case (let l?, nil), (nil, let l?): out.literals = l
            case (let l?, let r?):
                let merged = QWJSLiteralTrie()
                for word in l.words() + r.words() { merged.insert(word, payload: 0) }
                out.literals = merged
            }
            if let lit = out.literals, !lit.firstBytes.isDisjoint(with: out.types.firstBytes) {
                throw QWJSONSchemaError("anyOf branches must be distinguishable by their first character")
            }
            return out
        }
    }

    /// The single canonical compact serialization the grammar accepts for a literal.
    static func canonical(_ value: QWJson) -> String {
        switch value {
        case .undefined, .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d):
            if d == d.rounded() && abs(d) < 1e15 { return String(Int64(d)) }
            return String(d)
        case .string(let s): return "\"" + escapedString(s) + "\""
        case .array(let items): return "[" + items.map(canonical).joined(separator: ",") + "]"
        case .object(let pairs):
            return "{" + pairs.map { "\"" + escapedString($0.0) + "\":" + canonical($0.1) }.joined(separator: ",") + "}"
        }
    }

    /// Minimal JSON string escaping: quote, backslash, and control characters only.
    static func escapedString(_ s: String) -> String {
        var out = ""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }
}

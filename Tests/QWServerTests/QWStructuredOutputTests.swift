import XCTest
@testable import QWServer
import QWHead
import QWToken

final class QWStructuredOutputTests: XCTestCase {
    private func grammar(_ schema: String) throws -> QWJSONGrammar {
        try QWJSONGrammar.compile(schema: try QWJson.parse(schema))
    }

    /// Whether `text` is a complete document of `grammar`.
    private func accepts(_ grammar: QWJSONGrammar, _ text: String) -> Bool {
        var state = QWJState(root: grammar.root)
        for b in text.utf8 where !state.step(b) { return false }
        return state.canEnd
    }

    /// Whether `text` is a prefix some completion could extend.
    private func prefixAlive(_ grammar: QWJSONGrammar, _ text: String) -> Bool {
        var state = QWJState(root: grammar.root)
        for b in text.utf8 where !state.step(b) { return false }
        return true
    }

    func testNestedObjectWithRequiredAndClosedProperties() throws {
        let g = try grammar("""
        {"type":"object","properties":{"name":{"type":"string"},
         "address":{"type":"object","properties":{"city":{"type":"string"},"zip":{"type":"integer"}},
                    "required":["city","zip"],"additionalProperties":false},
         "tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":2}},
         "required":["name","address","tags"],"additionalProperties":false}
        """)
        XCTAssertTrue(accepts(g, #"{"name":"Ann","address":{"city":"Oslo","zip":123},"tags":["a"]}"#))
        XCTAssertTrue(accepts(g, "{ \"tags\" : [\"a\", \"b\"],\n \"address\": {\"zip\": 1, \"city\": \"x\"}, \"name\": \"\" }"))
        XCTAssertFalse(accepts(g, #"{"name":"Ann","address":{"city":"Oslo"},"tags":["a"]}"#), "missing zip")
        XCTAssertFalse(accepts(g, #"{"name":"Ann","address":{"city":"Oslo","zip":1.5},"tags":["a"]}"#), "zip is integer")
        XCTAssertFalse(accepts(g, #"{"name":"Ann","address":{"city":"O","zip":1},"tags":[]}"#), "minItems")
        XCTAssertFalse(prefixAlive(g, #"{"name":"Ann","address":{"city":"O","zip":1},"tags":["a","b","#), "maxItems")
        XCTAssertFalse(prefixAlive(g, #"{"extra""#), "additionalProperties false")
        XCTAssertFalse(prefixAlive(g, #"{"name":"a","name""#), "duplicate key")
        XCTAssertFalse(prefixAlive(g, "```"), "no fences")
        XCTAssertFalse(prefixAlive(g, #"{"name":"a"}"#), "required keys before close")
    }

    func testEnumWithMarkerLikeStrings() throws {
        let g = try grammar(#"{"type":"object","properties":{"v":{"enum":["<think>","<tool_call>","to=self<|message|>"]}},"required":["v"]}"#)
        XCTAssertTrue(accepts(g, #"{"v":"<think>"}"#))
        XCTAssertTrue(accepts(g, #"{"v":"to=self<|message|>"}"#))
        XCTAssertFalse(prefixAlive(g, #"{"v":"to=self<message"#))
        XCTAssertFalse(accepts(g, #"{"v":"<think"}"#))
    }

    func testNumbersLiteralsAndRootScalars() throws {
        let number = try grammar(#"{"type":"number"}"#)
        for ok in ["0", "-0.5", "12e3", "1.25E-7"] { XCTAssertTrue(accepts(number, ok), ok) }
        for bad in ["01", "1.", "-", ".5", "1e"] { XCTAssertFalse(accepts(number, bad), bad) }
        let bools = try grammar(#"{"type":["boolean","null"]}"#)
        XCTAssertTrue(accepts(bools, "true"))
        XCTAssertTrue(accepts(bools, "null"))
        XCTAssertFalse(accepts(bools, "nul"))
        XCTAssertFalse(prefixAlive(bools, "1"))
    }

    func testStringEscapesAndUTF8() throws {
        let g = try grammar(#"{"type":"string"}"#)
        XCTAssertTrue(accepts(g, #""a\"b\\c\u00e9\n""#))
        XCTAssertTrue(accepts(g, "\"héllo 世界 🙂\""))
        XCTAssertFalse(prefixAlive(g, #""\x"#))
        XCTAssertFalse(prefixAlive(g, "\"a\nb"), "raw control byte")
        var state = QWJState(root: g.root)
        XCTAssertTrue(state.step(UInt8(ascii: "\"")))
        XCTAssertFalse(state.step(0xC0), "overlong lead byte")
        var surrogate = QWJState(root: g.root)
        _ = surrogate.step(UInt8(ascii: "\""))
        XCTAssertTrue(surrogate.step(0xED))
        XCTAssertFalse(surrogate.step(0xA0), "UTF-16 surrogate in UTF-8")
    }

    func testRefAnyOfAndAdditionalSchema() throws {
        let g = try grammar("""
        {"$defs":{"node":{"type":"object","properties":{"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},
          "additionalProperties":false}},
         "type":"object","properties":{"root":{"$ref":"#/$defs/node"},"v":{"anyOf":[{"type":"string"},{"type":"null"}]}},
         "additionalProperties":{"type":"integer"}}
        """)
        XCTAssertTrue(accepts(g, #"{"root":{"kids":[{"kids":[]},{}]},"v":null,"n":3}"#))
        XCTAssertTrue(accepts(g, #"{"v":"s"}"#))
        XCTAssertFalse(prefixAlive(g, #"{"n":"x""#), "additional keys are integers")
        XCTAssertFalse(prefixAlive(g, #"{"v":1"#))
    }

    func testUnsupportedKeywordsAreRejected() {
        for schema in [#"{"type":"string","pattern":"^a$"}"#, #"{"type":"string","minLength":2}"#,
                       #"{"$ref":"https://x/y"}"#, #"{"anyOf":[{"type":"object"},{"type":"object"}]}"#,
                       #"{"type":"array","items":[{"type":"string"}]}"#, "false"] {
            XCTAssertThrowsError(try grammar(schema), schema)
        }
    }

    func testJsonObjectAndWhitespaceCap() throws {
        let g = QWJSONGrammar.anyObject()
        XCTAssertTrue(accepts(g, #"{"a":[1,{"b":null}],"c":"d"}"#))
        XCTAssertFalse(prefixAlive(g, "["))
        XCTAssertFalse(prefixAlive(g, String(repeating: " ", count: 65)), "whitespace is bounded")
    }

    /// A tiny byte-level vocabulary: masks must be exactly the tokens whose bytes stay valid.
    private func table() -> QWJSONTokenTable {
        let pieces: [String?] = ["{", "}", "\"", "a", "b", "\":", "\"a\"", ":", ",", " ", "1", "x\"}", "true", nil, "ab"]
        return QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: 16)
    }

    func testTokenMaskMatchesBytewiseFilter() throws {
        let g = try grammar(#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#)
        let table = table()
        let constraint = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [13])
        XCTAssertEqual(try constraint.allowedTokens(), [0, 9])
        try constraint.accept(0)
        XCTAssertEqual(try constraint.allowedTokens(), [2, 6, 9], "only the key a")
        try constraint.accept(6)
        XCTAssertEqual(try constraint.allowedTokens(), [7, 9])
        try constraint.accept(7)
        try constraint.accept(2)
        let inString = try XCTUnwrap(try constraint.allowedTokens())
        XCTAssertEqual(inString, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14], "plain content, a closing quote, or a valid close")
        try constraint.accept(11)
        XCTAssertTrue(constraint.isComplete)
        XCTAssertEqual(try constraint.allowedTokens(), [13])
        XCTAssertThrowsError(try constraint.accept(3))
    }

    func testPhasesThinkingAndToolChoice() throws {
        let g = QWJSONGrammar.anyObject()
        let thinking = QWStructuredOutputConstraint(grammar: g, table: table(), stopTokens: [13],
                                                    startsInThinking: true, endThinkToken: 15)
        XCTAssertNil(try thinking.allowedTokens())
        try thinking.accept(3)
        try thinking.accept(15)
        XCTAssertEqual(try thinking.allowedTokens(), [0, 9])

        let tools = QWStructuredOutputConstraint(grammar: g, table: table(), stopTokens: [13], toolCallToken: 15)
        XCTAssertEqual(try tools.allowedTokens(), [0, 9, 15])
        try tools.accept(9)
        XCTAssertEqual(try tools.allowedTokens(), [0, 9, 15], "whitespace keeps the choice open")
        try tools.accept(15)
        XCTAssertNil(try tools.allowedTokens(), "a tool call frees the rest of the turn")

        let answer = QWStructuredOutputConstraint(grammar: g, table: table(), stopTokens: [13], toolCallToken: 15)
        try answer.accept(0)
        XCTAssertFalse(try XCTUnwrap(try answer.allowedTokens()).contains(15))
    }

    func testPrefillMustBeAValidPrefix() throws {
        let g = QWJSONGrammar.anyObject()
        let ok = QWStructuredOutputConstraint(grammar: g, table: table(), stopTokens: [13])
        XCTAssertNoThrow(try ok.consumePrefix("{\"a\":"))
        let bad = QWStructuredOutputConstraint(grammar: g, table: table(), stopTokens: [13])
        XCTAssertThrowsError(try bad.consumePrefix("Sure: {"))
    }

    func testPartialUTF8TokensAreNotPlainAndDeadEndsThrow() throws {
        let pieces: [[UInt8]?] = [Array("{\"".utf8), Array("é".utf8), [0xC3], [0xA9], nil]
        let table = QWJSONTokenTable(tokenBytes: pieces, vocabulary: 5)
        XCTAssertEqual(table.plainStringTokens, [1])
        let g = try grammar(#"{"type":"string"}"#)
        let c = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [4])
        XCTAssertThrowsError(try c.allowedTokens(), "no token can start a string here")
    }

    func testToolCallConstraintRestrictsFunctionAndParameterNames() throws {
        let pieces: [String?] = [
            "<tool_call>", "</tool_call>", "\n", "<function", "=read", "_file>", "=grep>",
            "<parameter", "=path>", "=pattern>", ">", "src/main.rs"
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
        """)
        let c = try XCTUnwrap(QWToolCallConstraint(tools: [toolJson], forcedToolName: nil,
                                                   table: table, toolCallToken: 0, toolCallEndToken: 1))
        XCTAssertNil(try c.allowedTokens())
        try c.accept(0) // <tool_call>
        try c.accept(2) // \n
        try c.accept(3) // <function
        XCTAssertEqual(try c.allowedTokens(), [4], "must pick =read, not =grep>")
        try c.accept(4) // =read
        XCTAssertEqual(try c.allowedTokens(), [5], "must finish with _file>")
        try c.accept(5) // _file>
        XCTAssertNil(try c.allowedTokens(), "free outside tag")
        try c.accept(2) // \n
        try c.accept(7) // <parameter
        XCTAssertEqual(try c.allowedTokens(), [8], "must pick =path>, not =pattern>")
        try c.accept(8) // =path>
        XCTAssertNil(try c.allowedTokens(), "free inside parameter value")
        try c.accept(11) // src/main.rs
        try c.accept(1) // </tool_call>
        XCTAssertNil(try c.allowedTokens())
        XCTAssertTrue(c.isUnconstrainedNow)
    }

    func testToolCallConstraintSupportsGLMArgKeyAndJSONFormats() throws {
        let pieces: [String?] = [
            "<tool_call>", "</tool_call>", "read_file", "grep", "<arg_key>", "</arg_key>",
            "path", "pattern", "<arg_value>src/main.rs</arg_value>", "{\"name\":\"", "\""
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
        """)
        let c = try XCTUnwrap(QWToolCallConstraint(tools: [toolJson], forcedToolName: nil,
                                                   table: table, toolCallToken: 0, toolCallEndToken: 1))
        // Format C (GLM): <tool_call>read_file<arg_key>path</arg_key><arg_value>...</arg_value></tool_call>
        try c.accept(0) // <tool_call>
        XCTAssertFalse(c.isUnconstrainedNow)
        let hdrAllowed = try XCTUnwrap(try c.allowedTokens())
        XCTAssertTrue(hdrAllowed.contains(2), "must allow read_file")
        XCTAssertFalse(hdrAllowed.contains(3), "must block grep")
        try c.accept(2) // read_file
        try c.accept(4) // <arg_key>
        XCTAssertEqual(try c.allowedTokens(), [6], "must allow path, not pattern")
        try c.accept(6) // path
        XCTAssertEqual(try c.allowedTokens(), [5], "must finish with </arg_key>")
        try c.accept(5) // </arg_key>
        XCTAssertNil(try c.allowedTokens(), "free inside <arg_value>")
        try c.accept(8) // <arg_value>src/main.rs</arg_value>
        try c.accept(1) // </tool_call>
        XCTAssertTrue(c.isUnconstrainedNow)

        // Format A (JSON): <tool_call>{"name":"read_file"...}</tool_call>
        try c.accept(0) // <tool_call>
        try c.accept(9) // {"name":"
        XCTAssertEqual(try c.allowedTokens(), [2], "must allow read_file, not grep")
        try c.accept(2) // read_file
        try c.accept(10) // "
        XCTAssertNil(try c.allowedTokens())
        try c.accept(1) // </tool_call>
        XCTAssertTrue(c.isUnconstrainedNow)
    }
}

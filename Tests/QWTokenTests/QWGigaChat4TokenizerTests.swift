// QWGigaChat4TokenizerTests.swift — Tests Split(Removed, invert:true), conditional BOS, and GigaChat4 special tokens.
import Foundation
import XCTest
import QWCore
@testable import QWToken

final class QWGigaChat4TokenizerTests: XCTestCase {
    private static func makeSyntheticGigaChat4TokenizerJSON() throws -> Data {
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
        ]
        var nextId = 10
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        vocab["ab"] = nextId
        let obj: [String: Any] = [
            "version": "1.0",
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": ["Regex": #"[a-zA-Z]+|\d| ?[^\s\w]+"#],
                        "behavior": "Removed",
                        "invert": true,
                    ] as [String: Any],
                    [
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": false,
                        "use_regex": false,
                    ] as [String: Any],
                ],
            ] as [String: Any],
            "post_processor": [
                "type": "TemplateProcessing",
                "single": [
                    ["SpecialToken": ["id": "<s>", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                ],
                "pair": [
                    ["SpecialToken": ["id": "<s>", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                    ["Sequence": ["id": "B", "type_id": 1]],
                ],
                "special_tokens": [
                    "<s>": ["id": "<s>", "ids": [1], "tokens": ["<s>"]],
                ],
            ] as [String: Any],
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true,
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "vocab": vocab,
                "merges": [["a", "b"]],
                "byte_fallback": false,
            ] as [String: Any],
            "added_tokens": [
                ["id": 1, "content": "<s>", "special": true],
                ["id": 2, "content": "</s>", "special": true],
                ["id": 128000, "content": "<|role_sep|>\n", "special": true],
                ["id": 128001, "content": "<|message_sep|>\n\n", "special": true],
                ["id": 128014, "content": "<think>", "special": true],
                ["id": 128015, "content": "</think>", "special": true],
            ],
        ]
        return try JSONSerialization.data(withJSONObject: obj)
    }

    func testSplitRemovedInvertedDropsUnmatchedSpansAndHonorsAddSpecialTokens() throws {
        let tok = try QWTokenizer(json: Self.makeSyntheticGigaChat4TokenizerJSON())
        let abId = try XCTUnwrap(tok.vocab["ab"])
        let oneId = try XCTUnwrap(tok.vocab["1"])
        let twoId = try XCTUnwrap(tok.vocab["2"])

        // Spaces between "ab" and "12" are unmatched by the inverted Split regex and must be removed,
        // while digits "1" and "2" match individually via `\d`.
        let rawIds = try tok.encode("ab   12")
        XCTAssertEqual(rawIds, [abId, oneId, twoId], "default addSpecialTokens=false must not prepend <s>")

        let bosIds = try tok.encode("ab   12", addSpecialTokens: true)
        XCTAssertEqual(bosIds, [1, abId, oneId, twoId], "addSpecialTokens=true must prepend TemplateProcessing <s>=1")

        let spans = try tok.encodeWithOffsets("ab12", addSpecialTokens: true)
        XCTAssertEqual(spans.ids, [1, abId, oneId, twoId])
        XCTAssertEqual(spans.spans.count, 4)
        XCTAssertEqual(spans.spans[0].tokenId, 1)
        XCTAssertEqual(spans.spans[0].utf8Range, 0..<0)
        XCTAssertEqual(spans.substring(forTokenRange: 1..<2), "ab")
        XCTAssertEqual(spans.substring(forTokenRange: 2..<4), "12")

        let defaults = try XCTUnwrap(QWModelFamilyRegistry.shared.spec(forModelType: "gigachat4")?.defaultSpecialTokens)
        let resolved = tok.resolveSpecialTokens(fallback: defaults)
        XCTAssertEqual(resolved.eosTokenId, 2)
        XCTAssertEqual(resolved.imStartTokenId, 128000)
        XCTAssertEqual(resolved.imEndTokenId, 128001)
        XCTAssertEqual(resolved.thinkTokenId, 128014)
        XCTAssertEqual(resolved.endThinkTokenId, 128015)
        XCTAssertTrue(resolved.extraStopTokenIds.contains(2))
        XCTAssertTrue(resolved.extraStopTokenIds.contains(128001))
    }

    func testRealGigaChat4TokenizerParityAndGCMLTokensIfPresent() throws {
        let tokPath = NSHomeDirectory() + "/.cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B/"
            + "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc/tokenizer.json"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tokPath), "local GigaChat4 tokenizer.json absent")

        let tok = try QWTokenizer(json: Data(contentsOf: URL(fileURLWithPath: tokPath)))
        XCTAssertEqual(try tok.encode("Hello, world! 12345"), [16656, 14, 3621, 3, 223, 19, 20, 21, 22, 23])
        XCTAssertEqual(
            try tok.encode("Hello, world! 12345", addSpecialTokens: true),
            [1, 16656, 14, 3621, 3, 223, 19, 20, 21, 22, 23]
        )
        XCTAssertEqual(try tok.encode("<｜GCML｜tool_calls>"), [30, 128013, 27081, 74920, 32])
        XCTAssertEqual(tok.tokenId(for: "<|role_sep|>\n"), 128000)
        XCTAssertEqual(tok.tokenId(for: "<|message_sep|>\n\n"), 128001)

        let defaults = try XCTUnwrap(QWModelFamilyRegistry.shared.spec(forModelType: "gigachat4")?.defaultSpecialTokens)
        let resolved = tok.resolveSpecialTokens(fallback: defaults)
        XCTAssertEqual(resolved.imStartTokenId, 128000)
        XCTAssertEqual(resolved.imEndTokenId, 128001)
        XCTAssertTrue(resolved.extraStopTokenIds.contains(128001))

        let roundtripSample = "<s>system<|role_sep|>\nПривет, GigaChat 4! x < y && a == 42<|message_sep|>\n\n"
            + "assistant<|role_sep|>\n<think>Reasoning</think><｜GCML｜tool_calls><｜GCML｜invoke name=\"calc\">"
            + "</｜GCML｜invoke></｜GCML｜tool_calls><|message_sep|>\n\n"
        let encoded = try tok.encode(roundtripSample)
        XCTAssertEqual(try tok.decode(encoded, skipSpecialTokens: false), roundtripSample)
    }
}

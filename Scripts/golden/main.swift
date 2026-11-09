// golden_pre_p3 — P0-4 golden capture (p1_p3_plan.md §2 P0-4).
//
// Deterministic run: fixed prompt (32 ids through the REAL tokenizer, the
// Full note: docs/STYLE_NOTES.md#main.swift.

import Foundation
import QWCore
import QWEngine
import QWToken
import QWQuant

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    FileHandle.standardError.write("usage: golden_pre_p3 <model> [assets] [--check] [--fixture PATH] [--weight-arena planned|legacy-fixed]\n".data(using: .utf8)!)
    exit(2)
}
let packPath = argv[1]
let assetsOverride = argv.count >= 3 && !argv[2].hasPrefix("--")
    ? URL(fileURLWithPath: argv[2]) : nil
let assets = try QWModelLocation(url: URL(fileURLWithPath: packPath),
    assetsOverride: assetsOverride).assets.path
let check = argv.contains("--check")
let weightArenaSizing: QWWeightArenaSizing
if let i = argv.firstIndex(of: "--weight-arena") {
    guard i + 1 < argv.count, let sizing = QWWeightArenaSizing(rawValue: argv[i + 1]) else {
        FileHandle.standardError.write(Data("--weight-arena requires planned|legacy-fixed\n".utf8))
        exit(2)
    }
    weightArenaSizing = sizing
} else {
    weightArenaSizing = .planned
}

let defaultFixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Tests/QWEngineTests/Fixtures/golden_pre_p3.bin")
let fixture: URL
if let i = argv.firstIndex(of: "--fixture") {
    guard i + 1 < argv.count, !argv[i + 1].hasPrefix("--") else {
        FileHandle.standardError.write(Data("--fixture requires a path\n".utf8))
        exit(2)
    }
    fixture = URL(fileURLWithPath: argv[i + 1])
} else {
    fixture = defaultFixture
}

// Fixed prompt: 32 ids through the REAL tokenizer (deterministic ids).
let tokJSON = try Data(contentsOf: URL(fileURLWithPath: assets + "/tokenizer.json"))
let tok = try QWTokenizer(json: tokJSON)
let para = try tok.encode("The capital of France is Paris. The largest ocean on Earth is the Pacific Ocean. Water boils at 100 degrees Celsius at sea level. The planet known as the red planet is Mars. The chemical symbol for gold is Au. ")
var ids: [Int] = []
while ids.count < 32 { ids += para }          // m1 corpus law: repeat, then
let prompt = Array(ids.prefix(32))            // truncate — works at any ¶ size

let t0 = Date()
let engine = try QWEngineDriver(packPath: packPath, maxTokens: 1024,
                               weightArenaSizing: weightArenaSizing)
FileHandle.standardError.write("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s\n".data(using: .utf8)!)

var tokens: [Int32] = prompt.map(Int32.init)
let session = try engine.makeSession(promptTokens: prompt.count + 64)
defer { engine.finishSession(session) }
var next = try engine.prefill(session, tokens: prompt)     // chunked default, greedy
tokens.append(Int32(next))
for _ in 1..<64 {                                          // 64 decode tokens total
    next = try engine.decodeStep(session, token: next)
    tokens.append(Int32(next))
}
// Final-step logits, f32 bytes exactly as written (greedy = no sampler).
let stats = engine.logitsStats()
precondition(stats.nan == 0, "final logits contain NaN — engine broken, fixture refused")
let finalLogits = engine.logitVector()

func loadFloats(_ d: Data, _ base: Int, _ n: Int) -> [Float] {
    d.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: base).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}
func loadInt32s(_ d: Data, _ base: Int, _ n: Int) -> [Int32] {
    d.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: base).assumingMemoryBound(to: Int32.self)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}

// The same prompt through the unified core loop: the dense family on the
// module seam must produce the very same tokens, byte for byte.
if argv.contains("--module") {
    let module = try DenseModule.make(plan: engine.plan, engine: engine)
    // The dense module serves no speculative round yet, so the request asks
    // for none — a request that asked for drafts would be refused.
    let noDrafts = QWSpeculativeParameters(enabled: false, mtpDrafts: 0)
    let core = try QWEngineCore(configuration: QWEngineCoreConfiguration(
        plan: engine.plan, module: module, sampling: .greedy, maxSessions: 1,
        speculative: noDrafts))
    var moduleTokens: [Int32] = prompt.map(Int32.init)
    _ = try core.generate(prompt: prompt, maxNew: 64, stopIDs: [], sampling: .greedy,
                          speculative: noDrafts) { token in
        moduleTokens.append(Int32(token))
        return true
    }
    if moduleTokens == tokens {
        print("ok module tokens (\(moduleTokens.count) exact)")
    } else {
        var i = 0
        while i < min(moduleTokens.count, tokens.count), moduleTokens[i] == tokens[i] { i += 1 }
        print("FAIL module tokens: first diff at \(i) "
              + "(\(i < moduleTokens.count ? moduleTokens[i] : -1) vs \(i < tokens.count ? tokens[i] : -1))")
        exit(1)
    }
}

if check {
    let data = try Data(contentsOf: fixture)
    let magic = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
    guard magic == 0x4B473350 else { print("FAIL: bad magic"); exit(1) }
    let nTok = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
    let a = loadInt32s(data, 8, nTok)
    var ok = true
    if a != tokens {
        // Swift 6: Zip2Sequence has no firstIndex(where:) (not a
        // BidirectionalCollection) — plain loop, first diff only.
        var i = 0
        while i < min(a.count, tokens.count), a[i] == tokens[i] { i += 1 }
        print("FAIL tokens: first diff at \(i) (\(i < a.count ? a[i] : -1) vs \(i < tokens.count ? tokens[i] : -1))")
        ok = false
    } else {
        print("ok tokens (\(nTok) exact)")
    }
    let lgBase = 8 + nTok * 4
    let vocab = engine.geometry.vocab
    let b = loadFloats(data, lgBase, vocab)
    var li = 0
    while li < vocab, b[li].bitPattern == finalLogits[li].bitPattern { li += 1 }
    if li < vocab {
        print("FAIL logits: first bit-diff at \(li) (\(b[li]) vs \(finalLogits[li]))")
        ok = false
    } else {
        print("ok logits (\(vocab) bit-exact)")
    }
    exit(ok ? 0 : 1)
}

// Capture mode: build and write the fixture.
var out = Data()
@MainActor func put(_ v: UInt32) { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
put(0x4B473350)
put(UInt32(tokens.count))
out.append(contentsOf: tokens.withUnsafeBytes { Array($0) })
out.append(contentsOf: finalLogits.withUnsafeBytes { Array($0) })
try FileManager.default.createDirectory(at: fixture.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
try out.write(to: fixture, options: .atomic)
print("wrote \(fixture.path) (\(out.count) bytes): \(tokens.count) tokens + \(engine.geometry.vocab) logits")
print("first10 tokens: \(tokens.prefix(10).map(Int.init))")
print("argmax final: \(stats.argmax) max \(stats.max)")

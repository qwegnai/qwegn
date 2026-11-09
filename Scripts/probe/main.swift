// c23_probe — dump raw generated ids (no decode) to find what the engine emits.
import Foundation
import QWEngine
import QWToken
import QWQuant
import QWServer

let args = CommandLine.arguments
guard args.count >= 2 else { FileHandle.standardError.write("usage: c23_probe <pack> [model-dir]\n".data(using: .utf8)!); exit(2) }
let assetsOverride = args.count >= 3 ? URL(fileURLWithPath: args[2]) : nil
let location = try QWModelLocation(url: URL(fileURLWithPath: args[1]),
    assetsOverride: assetsOverride)
let runtimeAssets = try QWModelRuntimeAssets.load(packPath: args[1],
    assetsOverride: assetsOverride)
let tokJSON = try Data(contentsOf: location.assets.appendingPathComponent("tokenizer.json"))
let tok = try QWTokenizer(json: tokJSON)
let msgs: [QWJson] = [.object([("role", .string("user")), ("content", .string("What is 2+2? Answer with just the number."))])]
let wire = try runtimeAssets.chatTemplate.render(messages: msgs)
let ids = try tok.encode(wire)
FileHandle.standardError.write("prompt ids: \(ids.count), first: \(ids.prefix(8))\n".data(using: .utf8)!)
let eng = try QWEngineDriver(packPath: args[1], maxTokens: 1024)
let s = try eng.makeSession(promptTokens: ids.count)
if CommandLine.arguments.count >= 4 && CommandLine.arguments[3] == "--onestep" {
    // single step on the FIRST prompt token: argmax + logits stats (bisect probe)
    let t0 = Date()
    let first = try eng.decodeStep(s, token: ids[0])
    let ms = Date().timeIntervalSince(t0) * 1000
    let st = eng.logitsStats()
    print("onestep argmax: \(first)  (\(Int(ms)) ms)  logits nan=\(st.nan) max=\(st.max) min=\(st.min)")
    let top = eng.topLogits(5)
    for t in top { print("  top \(t.id) (\(String(format: "%.3f", t.logit))) = \(try tok.decode([t.id], skipSpecialTokens: false))") }
    exit(0)
}
let gen = try eng.generate(session: s, prompt: ids, maxNew: 8, stopIds: [])
print("gen ids: \(gen)")
print("decoded: \(try tok.decode(gen, skipSpecialTokens: false))")

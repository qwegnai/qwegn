import Testing
import Foundation
@testable import QWCli
import QWQuant

// C22 — the CLI logic library. The rules this suite binds:
//  1. npy bytes: writeNpy ≡ numpy.save BYTE-IDENTICAL (goldens in fixtures/,
//     produced by the live oracle venv — rule 2; the alignment law: header
//     padded with \n to a 64-byte multiple).
//  2. downloadPlan: total coverage, disjoint, ascending, ≤1 chunk / 4 MiB.
//  3. inspectReport: the geometry-guard verdict + provenance round-trip on
//     the shipped selfpack.qwp fixture (the packer's own header bytes).
//  4. quantizePlan: thin-wrapper law — argv names Scripts/quantize/pack.py,
//     the oracle python, and NEVER a Swift reimplementation.

private func fixtures() -> URL {
    var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<6 {
        let c = d.appendingPathComponent("Tests/QWCliTests/fixtures")
        if FileManager.default.fileExists(atPath: c.path) { return c }
        d = d.deletingLastPathComponent()
    }
    fatalError("Tests/QWCliTests/fixtures missing")
}

@Test("npy writer is byte-identical to numpy.save (all shape classes)")
func npyWriterByteParity() throws {
    let fx = fixtures()
    // the Swift side writes the SAME logical tensors numpy wrote for goldens
    // (values recomputed here must match numpy's inputs EXACTLY — they are
    //  trivially reproducible integer/scalar data, no RNG stream involved).
    struct Case { let golden: String; let values: [Float]; let shape: [Int] }
    let cases: [Case] = [
        Case(golden: "npy_ref_2x4.npy",
             values: (0..<8).map { Float($0) * 0.5 }, shape: [2, 4]),
        Case(golden: "npy_ref_1d.npy",
             values: [1.5, -2.25, 3, 4, 5, 6, 7, 8], shape: [8]),
        Case(golden: "npy_ref_2x3x4.npy",
             values: (0..<24).map { Float($0) }, shape: [2, 3, 4]),
    ]
    for c in cases {
        let out = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swift_\(c.golden)")
        try QWCli.writeNpy(c.values, shape: c.shape, to: out)
        let got = try Data(contentsOf: out)
        let want = try Data(contentsOf: fx.appendingPathComponent(c.golden))
        #expect(got == want, "npy bytes differ for \(c.golden) (len \(got.count) vs \(want.count))")
    }
}

@Test("downloadPlan covers the file exactly, disjoint ascending, part cap")
func downloadPlanAlgebra() {
    // exact cover law over a spread of sizes/parts
    for size in [1, 127, 4 << 20, 40 << 20, 1_073_741_824] {
        for parts in [1, 2, 3, 8, 16] {
            let plan = QWCli.downloadPlan(size: size, parts: parts)
            #expect(!plan.isEmpty, "size \(size) parts \(parts)")
            var expect = 0
            for r in plan {
                #expect(r.end >= r.start)
                #expect(r.start == expect, "gap/overlap at \(r.start)")
                expect = r.end + 1
            }
            #expect(expect == size, "tail not covered for \(size)/\(parts)")
            #expect(plan.count <= max(1, size / (4 << 20)), "part cap \(size)/\(parts)")
        }
    }
    // degenerate inputs produce an empty plan, never a crash or a bogus range
    #expect(QWCli.downloadPlan(size: 0, parts: 4).isEmpty)
    #expect(QWCli.downloadPlan(size: 100, parts: 0).isEmpty)
    // concrete spot-check: 10 MiB / 2 parts → two exact 5-MiB ranges
    // (labelled tuples carry no synthesized ==; compare elementwise)
    let two = QWCli.downloadPlan(size: 10 << 20, parts: 2)
    #expect(two.count == 2)
    #expect(two[0].start == 0 && two[0].end == 5_242_879)
    #expect(two[1].start == 5_242_880 && two[1].end == 10_485_759)
    #expect(QWCli.shouldDownloadRepoFile("config.json"))
    #expect(QWCli.shouldDownloadRepoFile("model.safetensors"))
    #expect(QWCli.shouldDownloadRepoFile("weights.00.safetensors"))
    #expect(!QWCli.shouldDownloadRepoFile(".gitattributes"))
    #expect(!QWCli.shouldDownloadRepoFile("README.md"))
    // Native Qwegn pack layout: weights, LSA sidecar, checksum and the assets/ tree.
    #expect(QWCli.shouldDownloadRepoFile("model.qwp"))
    #expect(QWCli.shouldDownloadRepoFile("model.qlsa"))
    #expect(QWCli.shouldDownloadRepoFile("model.qwp.sha256"))
    #expect(QWCli.shouldDownloadRepoFile("assets/tokenizer.json"))
    #expect(QWCli.shouldDownloadRepoFile("assets/README.md"))
    #expect(QWCli.shouldDownloadRepoFile("assets/LICENSE"))
    #expect(!QWCli.shouldDownloadRepoFile("LICENSE"))
    #expect(!QWCli.shouldDownloadRepoFile("media/qwegn_lsa_scaling.png"))
}

@Test("inspect report: provenance + geometry guard on the shipped fixture")
func inspectReportOnFixture() throws {
    var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    var packURL: URL?
    for _ in 0..<6 {
        let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
        if FileManager.default.fileExists(atPath: c.path) { packURL = c; break }
        d = d.deletingLastPathComponent()
    }
    guard let packURL else { Issue.record("selfpack.qwp missing"); return }
    let json = try QWCli.inspectReport(packURL: packURL)
    let obj = try #require(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    #expect(obj["tensors"] as? Int == 2)   // w_test (q4g only) + norm_test (verbatim bf16)
    #expect(obj["sourceModel"] as? String == "selfpack")
    // the fixture was packed with THIS tree's geometry → guard must be ok
    // (DIGEST LAW: the pack stores sha256(geometry-string); compare with
    //  QWQuant.geometryHash, the hashed form — never the raw string).
    #expect(obj["geometryGuard"] as? String == "ok")
    #expect(obj["geometryHash"] as? String == QWQuant.geometryHash)
    let table = try #require(obj["table"] as? [[String: Any]])
    // THE CLASSIFICATION LAW in the report: quantized rows carry rows/
    // cols/groups, raw rows carry kind=bf16 and NO group fields.
    let byName = Dictionary(uniqueKeysWithValues: table.map { ($0["name"] as? String ?? "?", $0) })
    let w = try #require(byName["w_test"])
    #expect(w["shape"] as? [Int] == [256, 1024])
    #expect(w["groups"] as? Int == 32)     // 1024 / 32 (HQ4_G32_M5)
    #expect(w["kind"] as? String == "q4g")   // q8-removal law: q4_g is the only quant plane
    let n = try #require(byName["norm_test"])
    #expect(n["kind"] as? String == "bf16")
    #expect(n["shape"] as? [Int] == [256])
    #expect(n["groups"] == nil)            // raw planes carry no group algebra
}

@Test("quantizePlan is the thin wrapper: python packer argv, no reimplementation")
func quantizePlanLaw() {
    let repo = URL(fileURLWithPath: #filePath)   // any file inside the repo
    var root = repo.deletingLastPathComponent()
    for _ in 0..<6 {
        if FileManager.default.fileExists(atPath: root.appendingPathComponent("Scripts/quantize/pack.py").path) { break }
        root = root.deletingLastPathComponent()
    }
    let scripts = root.appendingPathComponent("Scripts")
    let out = URL(fileURLWithPath: "/tmp/qw-model")
    let p = QWCli.quantizePlan(repo: "some/model", outPack: out, quant: "q4g", scriptsDir: scripts)
    // the python is the oracle venv python INSIDE the Scripts dir's parent
    #expect(p.command[0].hasSuffix(".venv-oracle/bin/python"))
    #expect(p.command[1].hasSuffix("Scripts/quantize/pack.py"))
    #expect(p.command.contains("--model") && p.command.contains("some/model"))
    #expect(p.command.elementsEqual([p.command[0], p.command[1], "--model", "some/model",
                                     "--out", "/tmp/qw-model", "--quant", "q4g"]))
    #expect(p.provenanceLine.contains("source=some/model"))
    // THE DRIVER LAW: the referenced pack.py accepts this exact argv —
    // verified against its argparse contract (--model/--out/--quant exist,
    // quant = q4g, --out is a directory); the packer owns the quantizer.
}

@Test("dequantToNpy: pack tensor → .npy readable and numerically identical to the packer sidecar")
func dequantRoundTrip() throws {
    var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    var packURL: URL?
    for _ in 0..<6 {
        let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
        if FileManager.default.fileExists(atPath: c.path) { packURL = c; break }
        d = d.deletingLastPathComponent()
    }
    guard let packURL else { Issue.record("selfpack.qwp missing"); return }
    let out = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("dq_roundtrip.npy")
    try QWCli.dequantToNpy(packURL: packURL, tensor: "w_test", out: out)
    let raw = try Data(contentsOf: out)
    // header ≡ 0 mod 64 (alignment law), body = 256×1024 f32
    #expect(raw.count % 64 == 0)
    #expect(raw.count == 128 + 256 * 1024 * 4)
    // numerically: identical to the packer's dq4 numpy sidecar — q4_g is the
    // only quant plane a new pack carries (q8-removal law 2026-10-04), so
    // dequantToNpy dispatches there. Scale/min f16→f32 widening is exact and
    let sidecar = URL(fileURLWithPath: #filePath)
    var sd = sidecar.deletingLastPathComponent()
    for _ in 0..<6 {
        let c = sd.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp.dq4.npy")
        if FileManager.default.fileExists(atPath: c.path) { break }
        sd = sd.deletingLastPathComponent()
    }
    let golden = try Data(contentsOf: sd.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp.dq4.npy"))
    #expect(Data(raw.suffix(256 * 1024 * 4)) == Data(golden.suffix(256 * 1024 * 4)),
            "dequant bytes diverged from the numpy q4g sidecar")
}

@Test("QWCli.Color terminal styling: ANSI codes emitted when enabled, empty strings when disabled")
func colorStylingParity() {
    let disabled = QWCli.Color(enabled: false)
    #expect(disabled.thinking.isEmpty)
    #expect(disabled.reset.isEmpty)
    #expect(disabled.prompt.isEmpty)
    #expect(disabled.systemNotice.isEmpty)
    #expect(disabled.dim.isEmpty)
    #expect(disabled.bold.isEmpty)

    let enabled = QWCli.Color(enabled: true)
    #expect(enabled.thinking == "\u{001B}[2;90m")
    #expect(enabled.reset == "\u{001B}[0m")
    #expect(enabled.prompt == "\u{001B}[1;36m")
    #expect(enabled.systemNotice == "\u{001B}[33m")
    #expect(enabled.dim == "\u{001B}[2m")
    #expect(enabled.bold == "\u{001B}[1m")

    let forcedOn = QWCli.Color.auto(force: true)
    #expect(forcedOn.enabled == true)
    let forcedOff = QWCli.Color.auto(force: false)
    #expect(forcedOff.enabled == false)
}

@Test("QWCli.resolveKVMode: f16, q8, q4 case-insensitive parsing and error on invalid")
func resolveKVModeValidation() throws {
    #expect(try QWCli.resolveKVMode("f16") == .f16)
    #expect(try QWCli.resolveKVMode("F16") == .f16)
    #expect(try QWCli.resolveKVMode("  q8  ") == .q8)
    #expect(try QWCli.resolveKVMode("Q8") == .q8)
    #expect(try QWCli.resolveKVMode("q4") == .q4)
    #expect(try QWCli.resolveKVMode("Q4") == .q4)

    #expect(throws: QWCli.CLIError.invalidKVMode("fp32")) {
        try QWCli.resolveKVMode("fp32")
    }
    #expect(throws: QWCli.CLIError.invalidKVMode("q2")) {
        try QWCli.resolveKVMode("q2")
    }
}

@Test("QWCli.validateContextBudget: accepts up to 1M tokens, rejects <=0 or >1M")
func validateContextBudgetBounds() throws {
    // Valid context budgets (powers of two and up to 1,048,576 tokens)
    try QWCli.validateContextBudget(tokens: 1)
    try QWCli.validateContextBudget(tokens: 4096)
    try QWCli.validateContextBudget(tokens: 32768)
    try QWCli.validateContextBudget(tokens: 262144)
    try QWCli.validateContextBudget(tokens: 1048576)

    // Invalid budgets: zero, negative, and over ceiling
    #expect(throws: QWCli.CLIError.invalidContextBudget(0, max: 1048576)) {
        try QWCli.validateContextBudget(tokens: 0)
    }
    #expect(throws: QWCli.CLIError.invalidContextBudget(-100, max: 1048576)) {
        try QWCli.validateContextBudget(tokens: -100)
    }
    #expect(throws: QWCli.CLIError.invalidContextBudget(1048577, max: 1048576)) {
        try QWCli.validateContextBudget(tokens: 1048577)
    }
}

@Test("QWCli.MultimodalInputOptions: validates existing and missing input paths")
func multimodalInputOptionsValidation() throws {
    let empty = QWCli.MultimodalInputOptions()
    #expect(!empty.hasInputs)
    try empty.validate()

    let nonExistent = QWCli.MultimodalInputOptions(imagePath: "/nonexistent/test_image.jpg")
    #expect(nonExistent.hasInputs)
    #expect(throws: QWCli.CLIError.inputFileNotFound("/nonexistent/test_image.jpg")) {
        try nonExistent.validate()
    }

    // Existing temp file passes validation
    let tmpFile = NSTemporaryDirectory() + "test_qwg_media.png"
    FileManager.default.createFile(atPath: tmpFile, contents: Data([0x00, 0x01]))
    defer { try? FileManager.default.removeItem(atPath: tmpFile) }

    let validImg = QWCli.MultimodalInputOptions(imagePath: tmpFile)
    #expect(validImg.hasInputs)
    try validImg.validate()
}




// swift-tools-version: 6.2
// qwegine — native Swift+Metal inference engine for Qwen3.8-27B (M5 Max).
// Strict layered DAG: a target may depend only on targets in lower layers.
//   L0 QWCore | L1 QWHAL QWQuant QWRope QWToken | L2 QWAlloc | L3 QWGEMM QWKV
//   L4 QWNorm QWAttn QWGDN QWMLP | L5 QWHead QWState | L6 QWMTP QWMM
//   L7 QWEngine | L8 qwg
import PackageDescription

let targetPaths: [String: String] = [
    // Foundation (L0–L2)
    "QWCore": "Sources/Foundation/QWCore",
    "QWHAL": "Sources/Foundation/QWHAL",
    "QWMetalCounterSupport": "Sources/Foundation/QWMetalCounterSupport",
    "QWAlloc": "Sources/Foundation/QWAlloc",
    "QWQuant": "Sources/Foundation/QWQuant",
    "QWLoad": "Sources/Foundation/QWLoad",
    // Operators (L1, L3–L5)
    "QWGEMM": "Sources/Operators/QWGEMM",
    "QWNorm": "Sources/Operators/QWNorm",
    "QWRope": "Sources/Operators/QWRope",
    "QWKV": "Sources/Operators/QWKV",
    "QWAttn": "Sources/Operators/QWAttn",
    "QWGDN": "Sources/Operators/QWGDN",
    "QWMLP": "Sources/Operators/QWMLP",
    "QWHead": "Sources/Operators/QWHead",
    // Subsystems (L4–L6)
    "QWFlashNext": "Sources/Subsystems/QWFlashNext",
    "QWMTP": "Sources/Subsystems/QWMTP",
    "QWToken": "Sources/Subsystems/QWToken",
    "QWVision": "Sources/Subsystems/QWVision",
    "QWMM": "Sources/Subsystems/QWMM",
    "QWState": "Sources/Subsystems/QWState",
    // Runtime (L7)
    "QWEngine": "Sources/Runtime/QWEngine",
    // Apps (L7.5–L8.5)
    "QWCli": "Sources/Apps/QWCli",
    "QWServer": "Sources/Apps/QWServer",
    "qwg": "Sources/Apps/qwg",
    "qwg-menu": "Sources/Apps/QWMenuBar",
    "QWBenchKernels": "Sources/Apps/QWBenchKernels",
]

func T(_ name: String, _ deps: [String] = []) -> Target {
    .target(name: name, dependencies: deps.map { Target.Dependency.byName(name: $0) },
            path: targetPaths[name],
            swiftSettings: [.swiftLanguageMode(.v6)])
}

// (module, test-target-name)
let components: [(String, String)] = [
    ("QWCore", "QWCoreTests"), ("QWHAL", "QWHALTests"), ("QWAlloc", "QWAllocTests"),
    ("QWQuant", "QWQuantTests"), ("QWGEMM", "QWGEMMTests"), ("QWNorm", "QWNormTests"),
    ("QWRope", "QWRopeTests"), ("QWKV", "QWKVTests"), ("QWGDN", "QWGDNTests"),
    ("QWAttn", "QWAttnTests"),
    ("QWMLP", "QWMLPTests"), ("QWHead", "QWHeadTests"), ("QWMTP", "QWMTPTests"),
    ("QWLoad", "QWLoadTests"),
    ("QWState", "QWStateTests"), ("QWEngine", "QWEngineTests"),
    ("QWToken", "QWTokenTests"), ("QWMM", "QWMMTests"),
    // C22: CLI logic as a library (the executable is a thin parser shell).
    ("QWCli", "QWCliTests"),
    ("QWServer", "QWServerTests"),
    // FlashNext (qwen4_exp_text) CPU reference + goldens; depends on QWCore only.
    ("QWFlashNext", "QWFlashNextTests"),
]

var targets: [Target] = [
    // Layer 0 — host-only, zero Metal.
    T("QWCore"),
    // Layer 1 (QWRope: L1→L2 via QWHAL/QWAlloc for its encoder; still acyclic)
    T("QWHAL", ["QWCore", "QWMetalCounterSupport"]),
    T("QWQuant", ["QWCore"]),
    // QWRope tables are pure host math, but the apply-encoder needs Metal →
    // it sits above HAL/Alloc (spec 07: tables + ONE apply kernel for C08/C09/C16).
    T("QWRope", ["QWCore", "QWHAL", "QWAlloc"]),
    T("QWToken", ["QWCore"]),
    // Layer 2
    T("QWAlloc", ["QWCore", "QWHAL"]),
    // LOAD-PATH DRY (2026-10-17): the one pack→memory mover both engines use.
    T("QWLoad", ["QWCore", "QWHAL", "QWQuant", "QWAlloc", "QWGEMM"]),
    // Layer 3
    T("QWGEMM", ["QWCore", "QWHAL", "QWAlloc", "QWQuant"]),
    T("QWKV", ["QWCore", "QWHAL", "QWAlloc"]),
    // Layer 4
    T("QWNorm", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM"]),
    T("QWAttn", ["QWCore", "QWHAL", "QWAlloc", "QWQuant", "QWGEMM", "QWKV", "QWRope"]),
    T("QWGDN", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM"]),
    T("QWMLP", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM"]),
    T("QWVision", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM", "QWQuant"]),
    // Layer 5
    T("QWHead", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM"]),
    // C20 replays its golden trace with the shipped C17 JSON parser
    // (reuse law — no second parser in the tree).
    T("QWState", ["QWCore", "QWHAL", "QWAlloc", "QWKV", "QWGDN", "QWToken"]),
    // Layer 6
    // C16 draft chain: norms (enorm/hnorm/lns), the QWGEMM fc+seam, rope,
    // attn decode on its OWN kv slice, MLP, and the shared head. Grows with
    // the chain legs — each rung REUSES a shipped door (never a private copy).
    T("QWMTP", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM", "QWNorm", "QWRope", "QWKV", "QWAttn", "QWMLP", "QWHead"]),
    T("QWMM", ["QWCore", "QWHAL", "QWAlloc", "QWToken", "QWVision"]),
    // Layer 7
    // The module registry maps a family to the module that serves it, so the
    // runtime sees Flash-Next's adapter. Acyclic: QWFlashNext depends on
    // QWLoad, which depends only on QWCore/QWHAL/QWQuant.
    T("QWEngine", ["QWCore", "QWHAL", "QWAlloc", "QWGEMM", "QWNorm", "QWRope",
                   "QWKV", "QWAttn", "QWGDN", "QWMLP", "QWHead", "QWMTP",
                   "QWState", "QWQuant", "QWVision", "QWMM", "QWMetalCounterSupport",
                   "QWLoad", "QWFlashNext"]),
    // Layer 7.5 — C22 CLI logic (library; needs pack format, KV types, geometry).
    T("QWCli", ["QWCore", "QWQuant", "QWKV"]),
    // Layer 7.8 — Server logic (library; HTTP 1.1 + SSE + OpenAI/Ollama schemas + lifecycle actor).
    .target(name: "QWServer",
            dependencies: ["QWCore", "QWHAL", "QWQuant", "QWKV", "QWToken", "QWVision", "QWMM",
                           "QWEngine", "QWCli", "QWHead",
                           .product(name: "Jinja", package: "swift-jinja")],
            path: targetPaths["QWServer"],
            resources: [.process("Resources")],
            swiftSettings: [.swiftLanguageMode(.v6)]),
    // Layer 8 — CLI.
    .executableTarget(name: "qwg",
                      dependencies: ["QWCore", "QWEngine", "QWToken", "QWCli", "QWKV", "QWServer",
                                     .product(name: "ArgumentParser", package: "swift-argument-parser")],
                      path: targetPaths["qwg"],
                      swiftSettings: [.swiftLanguageMode(.v6)]),
    // Layer 8.5 — Native macOS Menu Bar Controller (SwiftUI/AppKit Cupertino Telemetry Glass)
    .executableTarget(name: "qwg-menu",
                      dependencies: ["QWCore", "QWHAL", "QWQuant", "QWKV", "QWToken", "QWServer"],
                      path: targetPaths["qwg-menu"],
                      swiftSettings: [.swiftLanguageMode(.v6)]),
    // The standalone kernel shootout: the families' decode GEMVs against the
    // incumbent's, and MLX's tile GEMM as the reference point. It writes the
    // winner map the engine plan reads.
    .executableTarget(name: "qwg-bench-kernels",
                      dependencies: ["QWCore", "QWHAL", "QWQuant"],
                      path: targetPaths["QWBenchKernels"],
                      swiftSettings: [.swiftLanguageMode(.v6)]),
    // P0-4 golden capture/checker (p1_p3_plan.md): fixed 32+64-token greedy
    // run → Tests/QWEngineTests/Fixtures/golden_pre_p3.bin; every P3/P1
    // step must reproduce it byte-exact.
    .executableTarget(name: "golden_pre_p3", dependencies: ["QWCore", "QWEngine", "QWToken", "QWQuant"],
                      path: "Scripts/golden", swiftSettings: [.swiftLanguageMode(.v6)]),
    .executableTarget(name: "c23_probe", dependencies: ["QWCore", "QWEngine", "QWToken", "QWQuant", "QWServer"],
                      path: "Scripts/probe", swiftSettings: [.swiftLanguageMode(.v6)]),
    // M1 gate harness (spec-21 "Gate"; spec 40 §4 T6): the operator engine
    // soak and the CPU-only reference soak. One schema, one producer.
    .executableTarget(name: "m1", dependencies: ["QWCore", "QWEngine", "QWToken", "QWQuant", "QWCli"],
                      path: "Scripts/m1", swiftSettings: [.swiftLanguageMode(.v6)]),
    T("QWFlashNext", ["QWCore", "QWQuant", "QWHAL", "QWAlloc", "QWGEMM", "QWGDN", "QWHead", "QWLoad"]),
    .target(name: "QWMetalCounterSupport", path: targetPaths["QWMetalCounterSupport"],
            publicHeadersPath: "include"),
]
for (mod, test) in components {
    var deps: [Target.Dependency] = [.byName(name: mod)]
    // QWRopeTests reads f16 via the C04 converter; QWRope's own closure has no
    // QWQuant (rope math is quant-free by design) → test-only edge.
    if mod == "QWRope" || mod == "QWAttn" || mod == "QWGDN" { deps.append(.byName(name: "QWQuant")) }
    // QWKVTests drives arenas/HAL directly (page accounting, table mirrors).
    // QWMMTests parses golden JSON via the C17 value model.
    if mod == "QWMM" { deps.append(contentsOf: [.byName(name: "QWToken"), .byName(name: "QWVision")]) }
    // QWEngineTests binds planning laws that name C20 slots + C19 plans.
    if mod == "QWEngine" { deps.append(contentsOf: [.byName(name: "QWState"), .byName(name: "QWMM"), .byName(name: "QWToken"), .byName(name: "QWGEMM"), .byName(name: "QWCli"), .byName(name: "QWFlashNext"), .byName(name: "QWVision")]) }
    // QWVisionTests parses vis_index.json via the C17 value model.
    if mod == "QWVision" { deps.append(contentsOf: [.byName(name: "QWCore"), .byName(name: "QWHAL"), .byName(name: "QWAlloc"), .byName(name: "QWToken"), .byName(name: "QWState")]) }
    if mod == "QWKV" { deps.append(contentsOf: [.byName(name: "QWCore"), .byName(name: "QWHAL"), .byName(name: "QWAlloc")]) }
    // QWLoadTests race the mover against the frozen legacy FN loader (parity law).
    if mod == "QWLoad" { deps.append(contentsOf: [.byName(name: "QWQuant"), .byName(name: "QWHAL"), .byName(name: "QWEngine")]) }
    if mod == "QWCore" { deps.append(contentsOf: [.byName(name: "QWHAL"), .byName(name: "QWQuant")]) }
    // QWMLPTests cross-checks the batched FFN against the C05 ADR-001 GEMV
    // pipelines (the raced winners) and reads f16 via the C04 converter.
    if mod == "QWMLP" { deps.append(contentsOf: [.byName(name: "QWHAL"), .byName(name: "QWAlloc"), .byName(name: "QWGEMM"), .byName(name: "QWQuant")]) }
    // The CLI bindings live in the `qwg` executable target, and the
    // suite that holds them against the registry imports them from there.
    // The soak suite also runs the `m1` harness — the soak's producer —
    // in its own process, because the allocation law is only
    // attributable in a process that does nothing else.
    if mod == "QWCli" {
        deps.append(.byName(name: "qwg"))
        deps.append(.byName(name: "m1"))
        deps.append(.byName(name: "QWEngine"))
        deps.append(.byName(name: "QWServer"))
        deps.append(.product(name: "ArgumentParser", package: "swift-argument-parser"))
    }
    if mod == "QWToken" { deps.append(.byName(name: "QWServer")) }
    if mod == "QWServer" { deps.append(contentsOf: [.byName(name: "QWEngine"), .byName(name: "QWToken"), .byName(name: "QWMM"), .byName(name: "QWVision"), .byName(name: "QWCli")]) }
    // Fixture dirs are read from the SOURCE tree via #filePath — declaring
    // them as resources would copy (and duplicate) big golden blobs, so
    // exclude them explicitly (silences the unhandled-file warnings).
    let excl: [String] = (mod == "QWQuant" || mod == "QWCli" || mod == "QWFlashNext") ? ["fixtures"]
        : (mod == "QWEngine" || mod == "QWCore" ? ["Fixtures"] : [])
    targets.append(.testTarget(name: test, dependencies: deps, exclude: excl))
}

// FlashNext Metal (GPU) goldens: the dense HQ4 kernels driven on FlashNext expert weights.
targets.append(.testTarget(
    name: "QWFlashNextMetalTests",
    dependencies: [.byName(name: "QWFlashNext"), .byName(name: "QWCore"), .byName(name: "QWQuant"),
                   .byName(name: "QWHAL"), .byName(name: "QWAlloc"), .byName(name: "QWGEMM"),
                   .byName(name: "QWGDN")],
    path: "Tests/QWFlashNextMetalTests"))

let package = Package(
    name: "qwegine",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.1"),
        // C22: Apple's argument parser — audited 2026-10-02 in a throwaway
        // clone: the ArgumentParser target SwiftPM links is pure string /
        // reflection code; the network + Process hits live only in
        // TestHelpers/Tools, which qwg never links.
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
    ],
    targets: targets
)

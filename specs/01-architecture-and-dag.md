# 01 — Architecture & Layered Target DAG

> Normative target dependency architecture for Qwegn (`Package.swift`, enforced by `Scripts/lint_package_dag.py` and `Scripts/ci.sh`).

## 1. Design Law
Dependencies flow strictly **upward** along numeric layers (`LAYERS[dep] < LAYERS[target]` in `Scripts/lint_package_dag.py`). No target may depend on a sibling at the same layer or on any target in a higher layer. `QWState` is banned as a production target (the C20 reference harness resides solely in `Tests/QWStateTests/`).

## 2. Target Hierarchy (`29` Classified Targets)

| Layer | Targets | Role & Boundary |
| :--- | :--- | :--- |
| **L0.0** | `QWCore`, `QWMetalCounterSupport` | Model descriptor (`QWModelDescriptor`, `QWDecisionModelConfig`), geometry/capacity planning, `QWGPUTurnArbitrator`, checked POSIX I/O (`QWPOSIX`), workspace paths (`QWPaths`), `QWWeightView` / `QWSlotGeometry`, Objective-C Metal counter shim. |
| **L1.0** | `QWHAL`, `QWQuant`, `QWToken` | Metal 4 HAL (`QWHal`, lock-free PSO snapshot, doors), quantization codecs (`HQ*`, `AQ*`, `SHVQ*`, `QNGRAM`), BPE/Chat tokenizer + bijective character/UTF-8 offsets. |
| **L1.5** | `QWDecision` | Deep pure-host decision subsystem (zero Metal deps): sequence framing (`turnVerbalizer`, `optionMarkers`, `sequencePair` + cross-encoder fan-out), multi-source calibration + PAV isotonic/NLL fitter, pointer/grid/4D-Halton box decoders, `QWDecisionExecutionBackend` seam (`specs/07`). |
| **L2.0** | `QWAlloc` | Buffer/arena allocator (`QWArena`), `QWWeightView` (`L0` contract) over GPU storage. |
| **L2.5** | `QWLoad`, `QWRope` | Memory-mapped weight/slab loader (`PlaneIO`, `PlaneLayout`, `QWCheckpointTensorLoader` — no `QWGEMM` dependency), RoPE kernels. |
| **L3.0** | `QWGEMM`, `QWKV` | Matrix multiply / dequant GEMM & GEMV dispatch (`QWDeviceEmbedTable.gather`), KV cache arenas. |
| **L4.0** | `QWNorm`, `QWAttn`, `QWGDN`, `QWMLP`, `QWHyperPLE`, `QWVision` | Transformer normalization, dense/paged/sparse/MLA/QSA attention (`QWQSACacheState`, `QWQSAEncoders`), Gated DeltaNet recurrent/chunked operators, MLP/MoE FFN, HyperPLE operators (CPU references in `Tests/QWFlashNextTests/Reference/`), vision encoder (`ViT`). |
| **L5.0** | `QWHead` | LM-head projection & GPU sampling (`qwhead*`). |
| **L6.0** | `QWMTP`, `QWMM` | Multi-token prediction (`MTP` / `DFlash` / `DSpine`) and multimodal vision-text fusion. |
| **L7.0** | `QWEngine` | Module registry (`QWModuleRegistry`), unified `DenseModule`, compiled layer graphs (`QWLayerKernelRegistry` with `ple: true` via `QWQSALayerKernels`), single `QWEngineDriver` with `QWTrunkExecutor` routing `QWHyperStreamRuntime` + `QWInkling2Runtime` behind `QWEngineDriver`, `QWDecisionPackEngine` as the sole isolated stateless decision runtime, `QWBatchScheduler` (`QWBatchAdmissionTracker`), `ColdExpertStreamer`, telemetry registry (retired `L6.5 QWFlashNext`, `FlashNextModule`, `StandardModule`, `QWEngineCore`). |
| **L7.5** | `QWCli` | ArgumentParser CLI library (`pull`, `quantize`, `inspect`, `bench`, soak harnesses). |
| **L7.8** | `QWServer` | `NWListener`-based HTTP/1.1 API server, lifecycle actor, `/ui` WebUI & dashboard view-model builder, security/SSRF/framing policy. |
| **L8.0** | `qwg`, `qwg-bench-kernels`, `golden_pre_p3`, `m1` | Thin executable entry points. |
| **L8.5** | `QWMenuBarUI` | Reusable SwiftUI/`@Observable` library for `Qwegn.app` (`TelemetryObservable`, `CupertinoTelemetryView`, `SettingsView`, `LogViewerWindow`). |
| **L9.0** | `qwg-menu` (`Sources/Apps/QWMenuBarEntry`) | Minimal `@main` AppKit entry point wrapping `QWMenuBarUI`. |

## 3. Test Isolation Caps & Code Hygiene
- **Layer-capped test targets (`TEST_MAX_LAYER`):**
  - `QWCoreTests` `<= L0.0`
  - `QWTokenTests` `<= L1.0`
  - `QWDecisionTests` `<= L1.5`
  - `QWLoadTests` `<= L2.5`
  - `QWEngineTests` `<= L7.0`
- **Compiler safety:** Every Swift target sets `.swiftLanguageMode(.v6)` and `-warnings-as-errors` in `Package.swift`.
- **Style & size gates (`Scripts/lint_style.py`, `make lint`):**
  - All production Swift files remain compact and single-responsibility (hard lint ceiling `<= 800` LOC across `Sources/`).
  - File headers `<= 7` lines; block/doc comment runs `<= 3` lines.
  - Zero `try!` / `as!` in production engine/server targets (`Sources/`, excluding `QWBenchKernels` harness).
- **Kernel reachability (`Scripts/lint_kernels.py`, `Scripts/test_kernel_inventory.py`):** Every Metal kernel compiled into `kernel.metallib` / `kernel_std.metallib` is bidirectionally checked against production dispatch sites in `Sources/` (rejecting unreferenced or `Tests/`-only `.metal` kernels) and manifest-backed prewarm inventories.

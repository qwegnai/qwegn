# AGENTS.md — Working on the Qwegn Engine

Read this first in every session, from every harness (Hermes, Antigravity,
MiniMax Code, Copilot CLI, …). These laws are distilled from 247 recorded agent
sessions and ~138k tool calls of engine work; each exists because violating it
once cost a measured failure. The long-form analysis with evidence quotes is
the owner's private `docs/agent-retrospective-*.md`.

## Session protocol

1. **Open from artifacts, not vibes**: read `specs/` for the component you
   touch, the worktree's `task_plan.md` / `findings.md` / `progress.md` if
   present, then `git status` + `git branch --show-current`. Name the worktree
   and branch in your first message; assume the next agent has no chat memory.
2. **Worktree per task.** Never implement in the primary checkout. Lifecycle:
   create → implement → prove → merge → delete the worktree only after the
   merge is confirmed.
3. **Ground before writing**: a bounded, read-only audit (explicit file list,
   fixed questions) precedes every implementation thread. Do not mutate during
   an audit.
4. **End every turn with the resume tuple**: (a) next step, (b) the single
   command to run, (c) current measured numbers. Resumption must be zero-cost.
5. **Write like the context is already lost**: one ledger/DECISIONS row and a
   commit after each measured success. Never start work that cannot be built,
   gated, and committed inside the current session — land the green subset.

## Measurement law

6. **Every perf claim is a paired, interleaved A/B** (`Scripts/paired_perf.py`
   conventions): base vs candidate, ≥2 alternating rounds, min-of-N, raw
   artifacts kept, dispersion reported.
7. **Numbers belong to the exact fresh release binary.** A debug build or a
   sweep predating the last kernel edit is a plumbing smoke, not a result.
8. **Quiet GPU or no result.** Benchmarks run only with co-runners parked and
   the idle/idle probes in the log. If you cannot quiet the machine yourself,
   hand the operator one copy-paste command and stop.
9. **Publish only when separation beats the noise floor.** A delta inside
   run-to-run noise means "same speed". Non-regression is a gate: an
   optimization that does not gain is rolled back, not shipped with caveats.
   Throughput never excuses degraded tokens: every new lossy quant or pruning
   scheme must beat or match existing formats at equal bpw in an offline
   quality oracle before entering `Sources/`; sub-2-bit PTQ without QAT is
   restricted to exact-verified paths (speculative drafters with bit-exact
   greedy target parity) or cold banks passing `llmprobe` and needle gates.
   Cross-cutting runtime, kernel, or quant changes must compare fresh `master`
   release vs candidate across **every** supported flagship pack in a concise
   4-category table: throughput, UMA/GPU memory (including residency toggles),
   `llmprobe` task evaluation, and determinism / needle fidelity.
10. **Never compare across checkpoint formats as if same-model** (native HQ
    pack vs MLX pack). Match weights, KV mode, prompt, context, sampling,
    speculation settings, and state the target up front.

## Correctness law

11. **Parity is binary, not statistical.** Golden gates are bit-exact
    (bytes/sha256/token-stream identity). Loosening a tolerance requires an
    explicit written ruling, never a silent edit of `isAlmostEqual`.
12. **Memory discipline.** 128 GB unified is the budget; loading a pack next
    to a resident co-runner OOMs. Pre-flight free memory, load strictly
    sequentially, and make refusing guards exit nonzero — a refusal that
    prevents a contaminated run is a success state.
13. **One SwiftPM in flight per checkout** (shared `.build` lock); run tests
    serially (GPU bundles are not parallel-safe); wake on completion events,
    never sleep-poll loops.
14. **Test economy.** No unit test loads a real model pack; keep unit runs
    under ~4 minutes; perf gates live in `make perf`, not the test suite;
    full `make ci` at milestones, targeted suites in between.
15. **Config-sourced truth & capability-wide generalization.** Geometry,
    context caps, and quant metadata come from the model's own config/assets
    under `~/.qwg/models` — never hardcoded, never capped below native
    capacity. On spec/code conflict, the code is the fact; fix the spec. Every
    new weight or table container (`HQ*`, `AQ*`, `SHVQ*`, `VQ*`, `QNGRAM`) must
    honor runtime residency toggles (`--load-vision` / `--no-load-vision`,
    `--stream-ngrams` / `--no-stream-ngrams`) and `--no-profile` hermetic
    benchmarks. Generalize every perf or correctness fix by architectural
    capability, container family, attention complexity, or UMA ratio across
    both `DenseModule` and `HybridMoEModule`, CLI and `QWServer`, and all KV
    modes (`f16`/`q8`/`q4`) — never gate fast paths on a single incumbent
    format or model name. `Sources/Operators/`, `Sources/Subsystems/`,
    `Sources/Runtime/`, and `Sources/Apps/` must dispatch strictly on loaded
    capability structs (`KDAConfig`, `MLAConfig`, `MoERoutingConfig`,
    `SwiGLUClampConfig`, `VisionTowerConfig`, `MTPConfig`, `HCAConfig`,
    `BlockSparseGQAConfig`, `MRoPELayout`) enforced by
    `Scripts/lint_capability_seams.py` — never on `descriptor.family` or
    `model_type` strings outside `QWCore` config decoding and `QWLoad`
    tensor-name resolution. When onboarding or decoding a new HF checkpoint,
    inspect `configuration_*.py` `__init__` defaults alongside `config.json`,
    since checkpoints omit keys matching Python class defaults.

## Code law

16. **Gate before done**: `make lint && make test` green (then `make ci` at
    milestones). Machine gates are the only accepted done-signal — a prose
    "it works" is not. Never commit with lint violations.
17. **Sweep before you delete.** Grep the whole call graph (CLI dispatch,
    probes, plan tables, geometric fallbacks) before removing a flag, route,
    or kernel; a kernel that looks dead may be a live fallback.
18. **Clean trunk.** Rejected experiments leave the repository. No half-built
    doors, no disabled-attempt litter, no unrelated drive-by refactors, and no
    dead or test-only code in `Sources/` — every production symbol and kernel
    must serve a live product path (`qwg`, `QWServer`, `QWMenuBar`).
19. **Keep judge and builder separate.** Review tasks report
    `VERDICT: PASS|FAIL|PARTIAL` with evidence and never fix what they judge;
    fixes get re-reviewed. Numeric surfaces (telemetry, banners) are
    validated by running the binary against live sources, not by reading code.
20. **Style gates are hard limits**: files ≤ 800 lines, file headers ≤ 7
    lines, comment blocks ≤ 3 lines, zero compiler warnings, kernel
    reachability enforced by the linter.

## Agent-hygiene law

21. **Two failures of the same experiment = stop and change strategy** or
    escalate with a diagnosis. A third identical retry is prohibited.
22. **Keep shell output out of context**: pipe command results to
    `out/*.log`, then grep/summarize; only the verdict re-enters context.
23. **One language per thread; ask with letters.** Decide by offering
    "A (Recommended) / B" with evidence, and record rulings (ADRs).
24. **Never ingest raw chat-template/Jinja blobs into your own loop** while
    your harness is served by models sharing that template syntax — route
    such content through a stripping script.
25. **Cross-version compile sanity & zero-warning discipline**:
    All code must compile cleanly without warnings under `-Xswiftc -warnings-as-errors`.
    Avoid ambiguous standard library closures (such as `Data.withUnsafeBytes` without explicit buffer signatures or `DispatchData.copyBytes`); use explicit types or direct initializers (e.g. `Data(dispatchData)` or `replaceSubrange`) to prevent overload ambiguities across Swift minor versions.
    In Swift Testing (`import Testing`), use `#expect(Bool(true))` instead of `#expect(true)`, and never bind a result to a local variable with the same base name as a helper function called inside `#require(...)` (e.g. `let argument = try #require(argument(for: ...))` shadows `argument(for:)` inside the macro expansion; bind the optional first).
    Use scoped `lock.withLock { ... }` (rather than separate `lock.lock()` / `lock.unlock()` calls) inside `async` functions and `@Sendable` closures so Swift 6 concurrency checks pass on all Apple SDKs.
    Always hoist `MTLBuffer.contents()` and `region.pointer` outside throwing or rethrowing `withRawPointer`, `withBytes`, and `withUnsafeBytes` closures under `withExtendedLifetime(buffer) { ... }` using `dst.copyMemory(from:byteCount:)` (never `memcpy(buf.contents(), ...)` inside a throwing closure), preventing Xcode 26.6 AArch64 `[UNSAVED_X21_TAIL]` `swifterror` ABI clobbers on `macos-26` release runners even when newer local toolchains compile cleanly.
26. **CI transparency, pragmatic GitHub Actions gating & diagnostic preservation**:
    CI scripts (`ci.sh`) must never swallow compiler or test failure output through `tail` or silent pipes on nonzero exit. If a gate step exits nonzero, the full captured build/test log must be printed to standard output before terminating. Distinguish CI runner infrastructure timeouts (queue acquisition) from test/build failures before retrying code changes: check `jobs[].steps` and `gh api repos/<owner>/<repo>/check-runs/<job-id>/annotations` first (`steps: []` with `"not acquired by Runner of type hosted"` is pure runner-pool starvation); when simultaneous `main` + tag pushes starve `macos-26` runners, rerun `release` solo first and `ci` sequentially after it completes.
    Never alter or weaken working production code or kernels to work around GitHub Actions virtualized runner limitations (`Apple Paravirtual device`, 16 GiB RAM, no `~/.qwg/models`). Full hardware, Metal kernel, golden-oracle (`Scripts/spike/out`), and performance tests run only on real Apple Silicon machines (`GITHUB_ACTIONS != true`).
    When `GITHUB_ACTIONS=true`, omit `.venv-oracle` / `make goldens` in `.github/workflows/ci.yml`, skip performance harness suites (`test_paired*perf.py`) and `qwg-bench-kernels` execution in `Scripts/ci.sh`, and run only the non-Metal, non-golden host/app unit test targets (`QWCoreTests`, `QWQuantTests`, `QWTokenTests`, `QWCliTests`, `QWServerTests`, `QWMenuBarTests`, with any `Scripts/spike/out` fixture loaders gated via `XCTSkipIf`). In Python scripts that call `subprocess.run(..., timeout=...)`, isolate module `time` via `SimpleNamespace` so `unittest.mock.patch("<mod>.time.sleep")` never intercepts stdlib `subprocess.Popen._wait` backoff sleeps.
    Always verify `FileManager.default.fileExists(atPath:isDirectory:)` before constructing `QWModelRoot` or calling `fm.enumerator(at: root.directory)` (because `QWModelRoot` falls back to `url.deletingLastPathComponent()` on missing paths and would crawl `/tmp` or `~/.qwg/models/`), and keep `< 1 GiB` synthetic test fixtures exempt from `avail == 0` physical-reserve rejection on `<= 8 GiB` CI runners while strictly enforcing `>= 1 GiB` footprints and explicit `osReserveGiB` budgets.
27. **AGX 32 KiB constant-cache ceiling & mixed-plane capacity sizing**:
    Apple Silicon AGX spills `constant` address space to device DRAM when a single PSO references `> 32 KiB` of constant arrays; keep per-PSO constant LUTs `<= 32 KiB` by isolating new codebook families into separate `.metalh` headers and PSOs rather than unioning all tables into shared kernels.
    When sizing mixed raw/quant weight buffers (such as `DFlash` / `DSpine` drafters), always resolve per-entry byte counts via `PlaneLayout.resolve(e)` (`rawOff != nil` checked before quant plane tags, `16 KiB`-aligned).
28. **Multi-turn & multi-knob coherence gate**:
    Single-turn greedy checks never certify a runtime, sampling, cache,
    speculation, or server change. Validate live `QWServer` multi-turn chats
    with warm prefix cache hits, stochastic sampling (`temperature > 0`,
    `repetition_penalty > 1.0`), thinking off and across `reasoning.effort`
    budgets, uncached streaming, structured/tool calls, multi-protocol
    endpoints, and concurrent `B=4` batches across flagship architectures
    (follow `.agents/skills/qwegine-inference-qa-and-sampling/SKILL.md`).
29. **Metal 4 NAX deployment-target floor (`-mmacosx-version-min=26.2`) & GitHub-artifact parity**:
    Every `metal -std=metal4.0` compilation MUST pass `-mmacosx-version-min=26.2`
    (`__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ >= 260200`). Below `26.2`, Apple's
    `MetalPerformancePrimitives` SDK (`MPPTensorOpsAvailability.h`) silently falls
    back to broken pre-26.2 cooperative-tensor builtins without `operand_index`,
    corrupting all NAX `matmul2d` prefill kernels with zero compiler warnings;
    unpinned local `macOS 27.x` builds also emit `27.x` metallibs that fail to
    load on `macOS 26.2+`. Every standalone NAX Metal header/TU using
    `mpp::tensor_ops::matmul2d` must guard `< 260200` with `#error` and pass
    `Scripts/check_metallib_target.py`. Never certify a public release from local
    uploads alone when `.github/workflows/release.yml` overwrites assets with
    `--clobber` — validate the final GitHub-compiled tarball/`.pkg` after
    `release.yml` finishes.
30. **Graceful `NWConnection` close & deterministic server test synchronization**:
    Every terminal HTTP `Connection: close` send (`sendRawData`, `OPTIONS`,
    `503` over-capacity rejects) MUST pass
    `contentContext: .finalMessage, isComplete: true` and release
    `cancellationHubs` state synchronously in `.contentProcessed` before
    `connection.cancel()`. Early reject paths that respond before reading the
    peer request (`rejectOverCapacity`) MUST drain peer bytes up to a bounded
    cap (`64 KiB`) until `isComplete || error != nil` (with a short fallback
    deadline, e.g. `2.0s`) rather than cancelling immediately, preventing Darwin
    TCP `RST` (`POSIXErrorCode 54`) on virtualized CI runners. Raw
    `NWConnection` test clients must start `receive` concurrently with `send`
    on `.ready` and poll `server.activeConnectionCount` deterministically rather
    than relying on fixed sleeps.
31. **Headless-first, zero-cockpit-bloat & compact MCP surface (`ADR-012`, `ADR-022`)**:
    New engine and MCP capabilities ship headless (CLI/HTTP/MCP) by default.
    Add `0` new `/ui` tabs, `0` new `Qwegn.app` `440pt` cards, and `0` new
    `QWKnobRegistry` knobs unless a human operates that control weekly
    (requires an explicit ADR). Keep Qwegn's `/mcp` surface compact (`<= 16`
    cohesive tools) to avoid client tool-schema token bloat. Never clone
    external coding harnesses (`qwg agent`), reimplement
    `Scripts/quantize/pack.py` in Swift/Metal, or display unprivileged
    fabricated hardware gauges (`ANE %`/`AMX %`).
32. **Spike-first gate & independent scope critic (`ADR-019`, Law 19, Law 21)**:
    Any speculative compute or transport subsystem (ANE offload, multi-node
    cluster, new quant/attention paradigm) must pass a bounded `Scripts/spike/`
    paired interleaved A/B beating the incumbent above noise floor with
    `100%` SHA-256 parity (`<= 2` attempts or auto-drop) before creating
    `Sources/` targets. Multi-file feature proposals must pass a read-only
    critic (`VERDICT: PASS|FAIL|PARTIAL`) checking YAGNI, `Package.swift` DAG,
    and existing `master` capabilities before implementation.
33. **Skill hygiene, Code Mode & papercut flywheel**:
    Every skill under `.agents/skills/` and `skills/` must pass
    `python3 Scripts/lint_skills.py` (enforced by `make lint` and
    `Scripts/ci.sh`): `<= 250` lines, numbered checkable steps, a structured
    `## Error Recovery` table, live path references, `0` `.mcp.json` files,
    and deep references offloaded to `references/`. Put deterministic
    multi-step checks in `Scripts/` (Code Mode: `./Scripts/preflight.sh`,
    `Scripts/log_papercut.py`). When hitting a broken flag, stale skill step,
    wrong doc, or silent error, record one structured entry via
    `python3 Scripts/log_papercut.py log --area ... --target ... --tried ... --expected ... --actual ... --fix ...`
    and graduate recurring friction via `/qwegine-skill-workshop`.
34. **Qwegn-only naming across APIs, MCP tools, skills & specs**:
    Every API endpoint, MCP tool, Swift symbol, WebUI dialect, skill, spec,
    plan, scenario, and validation script in this repository must use strictly
    Qwegn-native naming (`/v1/qwegn`, `/v1/decisions`, `/v1/cognitive/*`,
    `qwg_ask`, `qwg_guard`, `qwg_status`, `qwegn-*` skills) with zero
    legacy third-party decision-daemon prefixes in skills, routes, or docs
    (enforced on skills by `Scripts/lint_skills.py` `S8`).
35. **`QWP` model quantization naming convention**:
    Quantized `QWP1` model packs (Hugging Face repo names under `qwegnai/` and
    local `~/.qwg/models/` directories) use the `QWP` naming convention rather
    than encoding internal container/group tags in the name:
    - Uniform single-quantization packs: `<Model>-QWP_{4,8,3,2,15,1}` (e.g.,
      `Qwen3.8-27B-QWP_4`, `GigaChat4-37B-A4B-QWP_4`).
    - Mixed-quantization packs: `<Model>-QWP-Mixed` (e.g.,
      `Edge0-145B-QWP-Mixed`).
    - Multiple variants for the same model and quant tier: disambiguate by
      tested profile (e.g., `-Fast` for the faster tested variant, base
      `-QWP-Mixed` for the higher-precision variant).
    - Unquantized packs retain their precision suffix (e.g., `d1-2B-bf16`).
    - Exact quantization details (plane formats, group sizes, head/PLE quant)
      MUST be documented in the model card (`README.md`), never encoded in the
      repo or directory name.

Verification footer: when you finish and merge to `master`, do **not** bump the
version and do **not** touch release artifacts (`QWVersion.swift`, `README.md`
badge/BibTeX, `CITATION.cff`, `assets/qwegn_engine_bento.svg`). Record the
change under `## [Unreleased]` in `CHANGELOG.md`, sync affected
`~/.qwg/models/*/README.md` + `media/*.{svg,png}`, preserve
`out/llmprobe/*.json` when cleaning worktrees/artifacts,
and state which gates you ran — including any you did not.

Version bumps are release-only: change the semantic version in
`Sources/Foundation/QWCore/QWVersion.swift`, rename `[Unreleased]` to
`[X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`, and update `README.md`,
`CITATION.cff`, `assets/qwegn_engine_bento.svg`, and the component ledger row
in `docs/tuning.md` only in one dedicated release-preparation commit, and only
when the owner explicitly requests publishing a release. Choose the level by
the significance of everything accumulated since the last tag
(fix/cleanup = patch, feature = minor, breaking/major milestone = major).

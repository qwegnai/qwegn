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
15. **Config-sourced truth.** Geometry, context caps, and quant metadata come
    from the model's own config/assets under `~/.qwg/models` — never
    hardcoded, never capped below native capacity. On spec/code conflict, the
    code is the fact; fix the spec. Every new weight or table container
    (`HQ*`, `AQ*`, `SHVQ*`, `VQ*`, `QNGRAM`) must honor runtime residency
    toggles (`--load-vision` / `--no-load-vision`, `--stream-ngrams` /
    `--no-stream-ngrams`) and `--no-profile` hermetic benchmarks.

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
26. **CI transparency & diagnostic preservation**:
    CI scripts (`ci.sh`) must never swallow compiler or test failure output through `tail` or silent pipes on nonzero exit. If a gate step exits nonzero, the full captured build/test log must be printed to standard output before terminating. Distinguish CI runner infrastructure timeouts (queue acquisition) from test/build failures before retrying code changes.

Verification footer: when you finish and merge to `master`, bump the semantic
version in `Sources/Foundation/QWCore/QWVersion.swift`, `CHANGELOG.md`,
`README.md`, `CITATION.cff`, and `assets/qwegn_engine_bento.svg` according to significance
(fix/cleanup = patch, feature = minor, breaking/major milestone = major),
sync affected `~/.qwg/models/*/README.md` + `media/*.{svg,png}`, preserve
`out/llmprobe/*.json` when cleaning worktrees/artifacts,
update the component ledger row, and state which gates you ran — including any
you did not.

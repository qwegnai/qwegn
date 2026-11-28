---
name: qwegine-sdlc-and-architecture
description: >-
  Use when starting a Qwegn worktree task, designing a new feature/subsystem,
  running a pre-coding YAGNI/architecture critic, delegating read-only
  subagents, or logging SDLC/tool friction.
---

# Qwegn SDLC, Architecture Gate & Papercut Runbook

Follow these five steps on every Qwegn engineering task to prevent scope creep, dirty-GPU contamination, and wasted retries (`AGENTS.md` Laws 1–5, 16–22, 31–33).

## Procedure

1. **Preflight & Isolated Worktree (`Law 2`, Code Mode):**
   - Run `./Scripts/preflight.sh` (pass `--require-quiet-gpu` before GPU tests/benchmarks).
   - Create an isolated worktree and branch from `master` before editing any file:
     ```bash
     git worktree add -b feat/<task> .worktrees/<task> master
     ```
2. **Pre-Coding Architecture & YAGNI Scope Gate (`ADR-012`, `ADR-019`, `ADR-022`):**
   - Read relevant `specs/` files and verify layer boundaries (`python3 Scripts/lint_package_dag.py` and `python3 Scripts/lint_capability_seams.py`).
   - Enforce headless-first scope: `0` new `/ui` tabs, `0` new `Qwegn.app` `440pt` cards, `0` new `QWKnobRegistry` knobs unless human-operated weekly, and `/mcp` kept `<= 16` cohesive tools.
   - Check [disqualified-patterns.md](./references/disqualified-patterns.md) before proposing UI chrome, native agents, Swift quantizers, or unprivileged hardware gauges.
   - Gate any speculative compute/transport subsystem (ANE offload, multi-Mac cluster, new quant/attention paradigm) behind a bounded `Scripts/spike/` paired A/B (`>= +10%` or noise-floor separation + `100%` SHA-256 parity, `<= 2` attempts) before touching `Sources/`.
3. **Read-Only Grounding & Separate Critic (`Law 3`, `Law 19`):**
   - Delegate bounded read-only codebase audits and pre-implementation / post-fix critics to isolated subagents returning `VERDICT: PASS|FAIL|PARTIAL` with evidence. Keep judge and builder separate; never poll background runs in a sleep loop.
4. **Log Friction Immediately (`Law 21`, `Law 33`):**
   - After 2 identical failures, stop and change strategy. Whenever hitting a broken flag, stale skill step, wrong doc, or silent error, record it once via `Scripts/log_papercut.py`:
     ```bash
     python3 Scripts/log_papercut.py log --area script --target "<name>" --tried "<cmd>" --expected "<want>" --actual "<err>" --fix "<fix>"
     ```
5. **Gate, Merge & Clean Only Your Own Worktree:**
   - Run `make lint && ./Scripts/ci.sh`, record the change under `## [Unreleased]` in `CHANGELOG.md` (zero bare `@` tokens, no version bump), merge to `master`, remove temporary `.venv-oracle`/`Scripts/spike/out` symlinks, and delete **only** your own `.worktrees/<task>` and feature branch.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| Editing in primary checkout (`[primary]` warning from `./Scripts/preflight.sh`) | Stash or reset primary checkout, create `.worktrees/<task>`, and `cd .worktrees/<task>` before editing. |
| Active GPU co-runner reported (`gpu_idle: NO`) | Park `Qwegn.app` / `qwg serve` or hand the user one copy-paste command and stop; never benchmark on a dirty GPU. |
| Parallel `swift test` `.build` lock contention or GPU failure | Run `swift test --no-parallel` strictly one SwiftPM invocation per checkout (`Law 13`). |
| Same experiment/build fix failed twice (`Law 21`) | Stop, run `python3 Scripts/log_papercut.py log ...`, and escalate or switch strategy — never attempt a 3rd identical retry. |
| Leftover `.venv-oracle` or `Scripts/spike/out` symlink blocks commit | Run `rm -f .venv-oracle Scripts/spike/out out/vision_decision_oracle` before `git commit` or `git worktree remove`. |
| `QWChatTemplateGoldenTests` / `QWAttn*` golden tests fail in worktree (`Scripts/spike/out` missing) | Untracked golden fixtures live only in the primary checkout; run golden-fixture suites on `master` post-merge (or symlink `Scripts/spike/out` temporarily and clean up before commit). |

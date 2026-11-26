---
name: qwegine-skill-workshop
description: "Triage agent papercuts, lint skills, run paired ABBA trajectory evals, and ablate/retire stale skills."
disable-model-invocation: true
---

# Qwegn Skill Workshop (`/qwegine-skill-workshop`)

Maintain high skill signal-to-noise with zero bloat: turn recorded session friction into deterministic scripts or crisp recovery steps, verify trajectories, and trim words or entire skills when models no longer need them.

## Procedure

1. **Triage Papercuts & Run Static Skill Linter:**
   - Inspect recurring tool/skill/doc failures and validate all `SKILL.md` hygiene rules (`S1–S7`):
     ```bash
     python3 Scripts/log_papercut.py triage && python3 Scripts/lint_skills.py && python3 Scripts/test_sdlc_scripts.py
     ```
2. **Graduate Friction by Code-Mode Precedence (Cheapest Token Layer First):**
   - Fix root causes in priority order:
     1. **Deterministic lint/check in `Scripts/`** (e.g., `Scripts/lint_skills.py`, `Scripts/preflight.sh`) — `0` LLM context tokens.
     2. **Deterministic CLI/script wrapper in `Scripts/`** for multi-step boilerplate.
     3. **`## Error Recovery` row** in the relevant `SKILL.md` (concrete symptom -> concrete fix; never vague retries).
3. **Evaluate Outcome + Trajectory (`pass@1`, `pass^3`, Tool Economy):**
   - Re-run affected tasks in detached worktrees with paired interleaved A/B (`with skill` vs `without skill`) verifying **both** end-state gates (`make lint`, tests exit `0`) and **trajectory discipline** (e.g., `qwg_vault_query` called before `grep` sweeps, `qwg_guard` called before mutating shell commands, error recovery step followed on failure).
   - Seed every real failure trajectory from `.agents/papercuts.jsonl` or `docs/agent-retrospective-*.md` into future regression prompts (aim for non-saturated 70–90% difficulty tasks, never toy 100%-pass prompts).
4. **Quarterly / Model-Upgrade Ablation ("Less Is Sometimes More"):**
   - When upgrading the primary serving model or harness, replay 5–10 representative tasks comparing `full SKILL.md` vs `50%-trimmed` vs `skill removed`.
   - Trim redundant instructions or retire (`git rm -r`) any skill whose removal yields identical `pass@1`, tool calls, and error count.
5. **Gate & Commit:**
   - Verify all modified skills and scripts pass:
     ```bash
     python3 Scripts/lint_skills.py && python3 Scripts/test_sdlc_scripts.py && make lint
     ```

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `S2` violation (`SKILL.md > 250` lines) | Move deep reference tables/flags into `references/*.md` under the skill directory and link via relative Markdown path. |
| Agent ignores a model-invoked skill at the right boundary | Front-load the trigger branch in `description` (`<= 500` chars) matching the exact prompt vocabulary (`writing-for-agents`). |
| Skill cites an unbuilt MCP tool or future roadmap flag | Remove the unbuilt tool name or split into a deferred draft under `roadmap/` until the feature lands on `master`. |
| Eval shows `100%` pass rate across both arms (`A == B`) | Replace toy prompts with real multi-repo/noisy failure cases from `.agents/papercuts.jsonl` or ablate/retire the skill. |

---
name: qwegn-orchestrate
description: "Supervise background subagents or async jobs with qwg_supervise, shortlist large skill catalogs with qwg_skill_select, route model tiers with qwg_route_turn, or score closed-set rubrics with qwg_decide."
---

# Calibrated Multi-Agent Orchestration (`14 ms` System-1)

Offload binary gates, closed-set picks, worker supervision, skill shortlisting, and model routing to Qwegn's calibrated System-1 sidecar instead of burning generative turns.

## Procedure

1. **Supervise Background Subagents & Jobs (`qwg_supervise`):**
   - Pass `{ "task_id": "...", "exit_code": int?, "seconds_since_output": num, "tail_output": "<last ~1500 chars>", "consecutive_alerts": int }` to `qwg_supervise`.
   - Follow the fact-overridden verdict (`180s` silence and nonzero `exit_code` override probabilities; `alert_after = 2` suppresses single-poll jitter): wait on `healthy`; intervene or re-scope on `needs_input` / `blocked` / `failed` when `should_alert == true`.
2. **Shortlist Large Skill Catalogs (`qwg_skill_select`):**
   - When `>= 30` skills exist and relevance is uncertain, call `qwg_skill_select` with `user_turn`, `catalog: [{"name", "description", "path"}]`, and `max_selected: 3`, loading `SKILL.md` only for returned `selected` finalists.
3. **Route Model Tiers (`qwg_route_turn`) & Score Rubrics (`qwg_decide`):**
   - Call `qwg_route_turn` (`prompt`, `context_tokens`, `current_tier`, `has_tools`) before subagent dispatch while honoring the `>= 32K` sticky-context guard.
   - Call `qwg_decide` (`noul`, `choice`, `score`, `read`, or `boxes`) for calibrated `< 15 ms` judgments.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `qwg_route_turn` returns low-confidence routing (`< 0.70`) | Keep the active heavy tier (fail open) so warm KV prefix caches are preserved. |
| `qwg_supervise` alerts on a single transient stall | Increment `consecutive_alerts` and only interrupt the worker when `should_alert == true`. |
| `qwg_status` reports `sidecar_loaded: false` | Run `/qwegn-mcp-setup` or start `qwg serve` with `--decision-model` before invoking orchestration tools. |

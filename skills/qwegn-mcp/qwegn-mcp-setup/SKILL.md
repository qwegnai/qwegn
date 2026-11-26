---
name: qwegn-mcp-setup
description: "Wire Qwegn MCP and System-One endpoints into hermes, omp, Claude Code, or OpenCode, index .qwcache, and verify sidecar health."
disable-model-invocation: true
---

# Qwegn MCP & Harness Setup (`/qwegn-mcp-setup`)

Run this checklist when connecting an agent harness to a local `qwg serve` instance or diagnosing missing sidecar/Vault capabilities.

## Procedure

1. **Verify `qwg serve` & Co-Resident Sidecar Health:**
   - Call `qwg_status` over MCP or `curl -sf http://127.0.0.1:8080/v1/models` and confirm `primary_loaded == true` and `sidecar_loaded == true`.
   - Call `qwg_vault_status` for the workspace; if unindexed, build `.qwcache` once:
     ```bash
     qwg index .
     ```
2. **Register Qwegn MCP in Your Target Harness:**
   - **Hermes:** `hermes mcp add qwegn --url http://127.0.0.1:8080/mcp && hermes mcp test qwegn`
   - **OMP (`oh-my-pi`):** Add `{"mcpServers": {"qwegn": {"url": "http://127.0.0.1:8080/mcp", "headers": {"X-Qwegn-Client": "omp"}}}}` to `.omp/mcp.json` and export `TYPESAFE_BASE_URL="http://127.0.0.1:8080/v1/systemone"`.
   - **Stdio Clients (`Claude Code` / `Codex` / `Zed`):** Configure command `qwg` with args `["mcp", "--server", "http://127.0.0.1:8080"]`.
3. **Smoke-Check Tools End-to-End:**
   - Verify `qwg_status` (`sidecar_loaded: true`), `qwg_guard` (`"echo ok"` -> `"allow"`), and `qwg_vault_status` (readable manifest).

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `sidecar_loaded: false` in `qwg_status` | Restart `qwg serve` with `--decision-model LiquidAI/d1-2B-vis-hq8` (or `d1-2B-hq8`). |
| Connection refused on `127.0.0.1:8080` | Launch `qwg serve <model-pack>` or check active port via `lsof -nP -iTCP -sTCP:LISTEN | rg qwg`. |
| `qwg_vault_status` fails on workspace path | Pass an absolute directory path and run `qwg index <abs-path>` once. |

---
name: qwegn-mcp-setup
description: "Wire Qwegn MCP and System-One endpoints into hermes, omp, Claude Code, or OpenCode, index Vault v2 code/PDF packs, and verify sidecar health."
disable-model-invocation: true
---

# Qwegn MCP & Harness Setup (`/qwegn-mcp-setup`)

Run this checklist when connecting an agent harness to a local `qwg serve` instance or mounting multi-repo/PDF Vault v2 packs.

## Procedure

1. **Verify `qwg serve`, Sidecar & Multi-Pack Vault v2:**
   - Call `qwg_status` over MCP or `curl -sf http://127.0.0.1:8080/v1/models` and confirm `primary_loaded == true` and `sidecar_loaded == true`.
   - Build or mount codebase and PDF/Markdown spec packs:
     ```bash
     qwg index . --docs ./docs
     qwg serve <model-pack> --pin-model --decision-model d1-2B-bf16 --repo . --docs ./docs
     ```
2. **Register Qwegn MCP in Your Target Harness:**
   - **OMP (`oh-my-pi`):** Add `{"mcpServers": {"qwegn": {"type": "http", "url": "http://127.0.0.1:8080/mcp", "headers": {"X-Qwegn-Client": "omp"}}}}` to `.mcp.json` (or `.omp/mcp.json`); `--pin-model` scopes `GET /v1/models` so `omp` binds all roles to the active pack.
   - **Hermes:** Add `qwegn` under `mcp_servers:` in `~/.hermes/config.yaml` via Streamable HTTP (`url: "http://127.0.0.1:8080/mcp"`, `headers: {"X-Qwegn-Client": "hermes"}`) or stdio (`command: qwg`, `args: ["mcp", "--server", "http://127.0.0.1:8080"]`).
   - **Stdio Clients (`Claude Code` / `Codex` / `Zed`):** Configure command `qwg` with args `["mcp", "--server", "http://127.0.0.1:8080"]`.
3. **Smoke-Check Tools End-to-End:**
   - Verify `qwg_status` (`sidecar_loaded: true`), `qwg_guard` (`"echo ok"` -> `"allow"`), and `qwg_vault_status` (`mounted_packs` non-empty).

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `sidecar_loaded: false` in `qwg_status` | Restart `qwg serve` with `--decision-model d1-2B-bf16` (or `LiquidAI/d1-2B-vis-hq8`). |
| Connection refused on `127.0.0.1:8080` | Launch `qwg serve <model-pack>` or check active port via `lsof -nP -iTCP -sTCP:LISTEN | rg qwg`. |
| `qwg_vault_status` reports empty `mounted_packs` | Call `qwg_vault_mount` with absolute `paths` / `docs` or pass `--repo <abs-path>` to `qwg serve`. |

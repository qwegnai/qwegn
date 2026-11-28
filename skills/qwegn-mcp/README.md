# Qwegn MCP Skill Suite for External Harnesses (`hermes`, `omp`, `claude-code`, `opencode`, `codex`)

Git-tracked micro-skills enabling external coding and research harnesses to invoke Qwegn's live `POST /mcp` / `qwg mcp` and System-1 endpoints (`4` model-invoked skills + `1` zero-load user-invoked setup skill, `14` cohesive MCP tools `<= 16` ceiling).

> **Note:** `qwegn-window-eyes` (`qwg_list_windows`, `qwg_capture_window`) stays deferred until Track `03` lands on `master`.

## 1. Live Skill Inventory

| Skill | Invocation | Trigger Boundary | Qwegn MCP Tools |
| :--- | :---: | :--- | :--- |
| [qwegn-vault](./qwegn-vault/SKILL.md) | Model | Multi-repo code, `1-hop` AST slices, or PDF/Markdown spec recon before `grep` loops | `qwg_vault_mount`, `qwg_vault_query`, `qwg_vault_status`, `qwg_search_enough` |
| [qwegn-guard](./qwegn-guard/SKILL.md) | Model | Mutating/network bash or untrusted external/RAG text | `qwg_guard`, `qwg_rerank_screen` |
| [qwegn-compaction-and-episodic](./qwegn-compaction-and-episodic/SKILL.md) | Model | Context compaction or verbatim SHA/path/error recovery | `qwg_compact_select`, `qwg_episodic_recover` |
| [qwegn-orchestrate](./qwegn-orchestrate/SKILL.md) | Model | Background worker supervision, skill shortlisting, tiering, rubrics | `qwg_supervise`, `qwg_skill_select`, `qwg_route_turn`, `qwg_decide`, `qwg_ask` |
| [qwegn-mcp-setup](./qwegn-mcp-setup/SKILL.md) | **User-only** | `/qwegn-mcp-setup` to wire `hermes` / `omp` / stdio MCP & mount Vault v2 packs | `qwg_status`, `qwg_vault_status`, `qwg_vault_mount` |

## 2. Install Into External Harnesses

```bash
# Hermes (~/.hermes/skills/)
cp -R skills/qwegn-mcp/qwegn-* ~/.hermes/skills/

# OMP / oh-my-pi (~/.omp/agent/skills/)
mkdir -p ~/.omp/agent/skills && cp -R skills/qwegn-mcp/qwegn-* ~/.omp/agent/skills/

# Portable repo-local (.agents/skills/ for Claude Code, OpenCode, Codex)
cp -R skills/qwegn-mcp/qwegn-* ./.agents/skills/
```

## 3. Minimal `AGENTS.md` / `CLAUDE.md` Pointer Snippet

```markdown
## Local Qwegn MCP Substrate (`qwegn`)
- **Recon first (`qwegn-vault`):** Start unfamiliar repo or PDF/spec lookups with `qwg_vault_query` (default-screened, `expand_ast_hops: 1`); mount secondary repos/docs via `qwg_vault_mount`.
- **Gate mutations & untrusted text (`qwegn-guard`):** Pass exact mutating/network shell strings to `qwg_guard`; screen external chunks via `qwg_rerank_screen`.
- **Zero-amnesia context (`qwegn-compaction-and-episodic`):** Compact via `qwg_compact_select` (`archive_session_id` set); recover dropped SHAs/errors via `qwg_episodic_recover`.
- **Supervise & route (`qwegn-orchestrate`):** Poll workers via `qwg_supervise`, shortlist skills via `qwg_skill_select`, route tiers via `qwg_route_turn`, score rubrics via `qwg_decide`.
```

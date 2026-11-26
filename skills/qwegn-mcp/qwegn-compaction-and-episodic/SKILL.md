---
name: qwegn-compaction-and-episodic
description: "Compact long transcripts losslessly via qwg_compact_select and recover dropped commit SHAs, file paths, or error logs verbatim via qwg_episodic_recover."
---

# Lossless Compaction & Verbatim Episodic Recovery

Pair Qwegn's calibrated turn selector with its `0o600` BM25 episodic archive so shrinking long contexts never loses exact SHAs, line numbers, ports, or diagnostics.

## Procedure

1. **Recover Missing Facts Before Re-Running Work (`qwg_episodic_recover`):**
   - Whenever an earlier commit SHA, command flag, stack trace, or user constraint was compacted out, call `qwg_episodic_recover` with `{"session_id": "<current-session-id>", "query": "<exact identifier or keyword>", "limit": 5}` before re-running builds or asking the user.
2. **Shrink Long Contexts Without Amnesia (`qwg_compact_select`):**
   - Batch turns (`<= 40` turns / `40,000` chars) to `qwg_compact_select` with `turns: [{"index": i, "role": "...", "text": "...", "pinned": bool}]`, `keep_tail_count: 6`, `drop_confidence_rail: 0.70`, and mandatory `archive_session_id: "<current-session-id>"`.
   - Apply verdicts strictly: `keep` -> retain verbatim; `summarize` -> 1-line factual pointer citing `session_id` + turn `index` for `qwg_episodic_recover`; `drop` -> remove only when calibrated `confidence >= 0.70`.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `qwg_episodic_recover` returns `0` matches | Check that `archive_session_id` was passed during `qwg_compact_select` and broaden `query` to a single token (SHA prefix or symbol). |
| `qwg_compact_select` rejects oversized batch | Split turns into chunks of `<= 40` turns and `<= 40,000` characters per request. |
| Critical constraint marked `summarize` or `drop` | Re-submit with `pinned: true` on system/decision turns or recover verbatim via `qwg_episodic_recover`. |

#!/usr/bin/env python3
"""Record and triage local agent papercuts (.agents/papercuts.jsonl)."""
import argparse
import datetime as dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

AREAS = ("skill", "script", "cli", "test", "spec", "harness")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_log_path() -> Path:
    return repo_root() / ".agents" / "papercuts.jsonl"


def git_info(cwd: Path) -> tuple[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip() or "detached"
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip() or "unknown"
        return branch, head
    except Exception:
        return "unknown", "unknown"


def cmd_log(args: argparse.Namespace) -> int:
    log_path = Path(args.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    branch, head = git_info(repo_root())
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": branch,
        "head": head,
        "area": args.area,
        "target": args.target,
        "tried": args.tried,
        "expected": args.expected,
        "actual": args.actual,
        "fix": args.fix,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"papercut logged: [{args.area}] {args.target} -> {log_path}")
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    log_path = Path(args.file)
    if not log_path.exists():
        print("papercuts: 0 recorded")
        return 0
    rows: list[dict[str, str]] = []
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        print("papercuts: 0 recorded")
        return 0
    counts = Counter((r.get("area", "?"), r.get("target", "?")) for r in rows)
    print(f"papercuts: {len(rows)} recorded across {len(counts)} target(s)")
    for (area, target), n in counts.most_common():
        print(f"  {n:3d}x  [{area}] {target}")
    print(f"latest {min(args.limit, len(rows))}:")
    for r in rows[-args.limit :]:
        print(
            f"  - {r.get('timestamp', '?')} [{r.get('area')}] {r.get('target')}: "
            f"actual={r.get('actual')} | fix={r.get('fix')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log or triage agent friction papercuts.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_log = sub.add_parser("log", help="Append one structured papercut entry")
    p_log.add_argument("--area", required=True, choices=AREAS)
    p_log.add_argument("--target", required=True, help="Skill, script, command, or spec name")
    p_log.add_argument("--tried", required=True, help="Command or action attempted")
    p_log.add_argument("--expected", required=True, help="Expected outcome")
    p_log.add_argument("--actual", required=True, help="Observed error or friction")
    p_log.add_argument("--fix", required=True, help="Workaround or proposed permanent fix")
    p_log.add_argument("--file", default=str(default_log_path()))
    p_log.set_defaults(func=cmd_log)

    p_tri = sub.add_parser("triage", help="Summarize and group recorded papercuts")
    p_tri.add_argument("--file", default=str(default_log_path()))
    p_tri.add_argument("--limit", type=int, default=10)
    p_tri.set_defaults(func=cmd_triage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

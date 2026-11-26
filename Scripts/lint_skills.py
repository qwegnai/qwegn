#!/usr/bin/env python3
"""Skill hygiene linter for .agents/skills and skills/ (S1-S7)."""
import argparse
import re
import sys
from pathlib import Path

MAX_LINES = 250
MAX_DESC_CHARS = 500
DEFAULT_ROOTS = (".agents/skills", "skills")
GITIGNORED_LOCAL_PATHS = {
    "Scripts/publish_release.sh",
    "Scripts/spike/out",
    ".agents/papercuts.jsonl",
}
COMMAND_RE = re.compile(
    r"(?:^|[\s`(/])(?:\./)?(?:make|swift|python3?|./Scripts/|Scripts/|qwg|git|rg|curl|gh|hf|"
    r"lsof|pgrep|shasum|codesign|xcrun|pkgutil|hermes|omp|screencapture|qwg_[a-z0-9_]+)\b"
)
FORBIDDEN_BRAND_RE = re.compile(
    r"(?:\b\x6c\x61\x79\x61\b|\b\x6c\x61\x79\x61_[a-z0-9_]+|\b\x6c\x61\x79\x61-[a-z0-9_-]+|/v1/\x6c\x61\x79\x61\b)",
    re.IGNORECASE,
)
WEAK_RECOVERY_RE = re.compile(r"\b(try again|just retry|retry again|re-run it)\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
REPO_PREFIXES = ("./Scripts/", "Scripts/", "specs/", ".agents/", "skills/")


def parse_frontmatter(lines: list[str]) -> tuple[dict[str, str], int, list[str]]:
    if not lines or lines[0].strip() != "---":
        return {}, 0, ["S1 missing opening YAML frontmatter '---'"]
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}, 0, ["S1 unterminated YAML frontmatter '---'"]
    meta: dict[str, str] = {}
    i = 1
    while i < end:
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", raw)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in (">-", ">", "|", "|-"):
            block: list[str] = []
            i += 1
            while i < end and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i].strip())
                i += 1
            joiner = "\n" if val.startswith("|") else " "
            meta[key] = joiner.join(block).strip()
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        meta[key] = val
        i += 1
    return meta, end + 1, []


def strip_fenced_code(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def check_error_recovery(body_lines: list[str]) -> list[str]:
    errs: list[str] = []
    heading_idx = None
    for idx, line in enumerate(body_lines):
        if re.match(r"^##\s+Error Recovery\b", line.strip(), re.IGNORECASE):
            heading_idx = idx
            break
    if heading_idx is None:
        return ["S4 missing '## Error Recovery' section with failure-to-fix table"]
    section: list[str] = []
    for line in body_lines[heading_idx + 1 :]:
        if re.match(r"^##\s+", line.strip()):
            break
        section.append(line)
    table_rows = [ln.strip() for ln in section if ln.strip().startswith("|")]
    sep_idx = next(
        (i for i, row in enumerate(table_rows) if re.match(r"^\|[\s:|-]+\|$", row)),
        None,
    )
    if sep_idx is None or sep_idx == 0 or sep_idx + 1 >= len(table_rows):
        errs.append("S4 '## Error Recovery' must include a Markdown table with >= 1 data row")
        return errs
    for row in table_rows[sep_idx + 1 :]:
        if WEAK_RECOVERY_RE.search(row):
            errs.append(f"S4 weak recovery advice not allowed: '{row}'")
    return errs


def check_paths(skill_path: Path, body_text: str, repo_root: Path) -> list[str]:
    errs: list[str] = []
    skill_dir = skill_path.parent
    for target in LINK_RE.findall(body_text):
        if target.startswith(("http://", "https://", "mailto:", "#", "file://")):
            continue
        clean = target.split("#", 1)[0].strip()
        if not clean or any(c in clean for c in "<>*{}$"):
            continue
        if not (skill_dir / clean).exists():
            errs.append(f"S5 broken relative link '{target}'")
    for span in BACKTICK_RE.findall(body_text):
        for raw_tok in span.split():
            tok = raw_tok.strip(",;.)]:'\"")
            if not tok.startswith(REPO_PREFIXES):
                continue
            rel = tok[2:] if tok.startswith("./") else tok
            rel = rel.split("#", 1)[0].rstrip("/")
            if not rel or any(c in rel for c in "<>*{}$") or "..." in rel:
                continue
            if rel in GITIGNORED_LOCAL_PATHS:
                continue
            if not (repo_root / rel).exists():
                errs.append(f"S5 missing repo path reference '{tok}'")
    return errs


def lint_skill_file(skill_path: Path, repo_root: Path) -> list[str]:
    text = skill_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    errs: list[str] = []
    meta, body_start, fm_errs = parse_frontmatter(lines)
    errs.extend(fm_errs)
    if not fm_errs:
        name = meta.get("name", "").strip()
        desc = meta.get("description", "").strip()
        dmi = meta.get("disable-model-invocation", "").strip()
        if not name:
            errs.append("S1 frontmatter missing 'name'")
        elif name != skill_path.parent.name:
            errs.append(f"S1 name '{name}' != directory '{skill_path.parent.name}'")
        if not desc:
            errs.append("S1 frontmatter missing 'description'")
        elif len(desc) > MAX_DESC_CHARS:
            errs.append(f"S1 description is {len(desc)} chars (max {MAX_DESC_CHARS})")
        if dmi and dmi not in ("true", "false"):
            errs.append(f"S1 disable-model-invocation must be true/false, got '{dmi}'")
    if len(lines) > MAX_LINES:
        errs.append(f"S2 file is {len(lines)} lines (max {MAX_LINES})")
    body_lines = lines[body_start:]
    body_text = "\n".join(body_lines)
    non_fence = strip_fenced_code(body_lines)
    if not any(re.match(r"^\s*1\.\s+\S", ln) for ln in non_fence):
        errs.append("S3 missing numbered procedural steps ('1. ...')")
    has_fence = any(ln.strip().startswith("```") for ln in body_lines)
    has_cmd = has_fence or any(COMMAND_RE.search(span) for span in BACKTICK_RE.findall(body_text))
    if not has_cmd:
        errs.append("S3 missing concrete verification command or code block")
    errs.extend(check_error_recovery(body_lines))
    errs.extend(check_paths(skill_path, body_text, repo_root))
    if "{%" in text or "{{" in text:
        errs.append("S7 raw Jinja tag ('{%' or '{{') forbidden in SKILL.md (Law 24)")
    if FORBIDDEN_BRAND_RE.search(text):
        errs.append("S8 non-Qwegn external naming forbidden in SKILL.md (Law 34: use qwg_*/qwegn-*)")
    return [f"{skill_path}: {e}" for e in errs]


def discover_skills(roots: list[Path]) -> tuple[list[Path], list[str]]:
    skills: list[Path] = []
    errs: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == "SKILL.md":
            skills.append(root)
            continue
        for mcp_file in sorted(root.rglob(".mcp.json")):
            errs.append(f"{mcp_file}: S6 .mcp.json forbidden inside skill trees (use Code Mode)")
        skills.extend(sorted(root.rglob("SKILL.md")))
    return skills, errs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint Qwegn SKILL.md files (S1-S7).")
    parser.add_argument("paths", nargs="*", help="Skill root directories or SKILL.md paths")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root for S5 path checks")
    args = parser.parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    roots = [Path(p) if Path(p).is_absolute() else (repo_root / p) for p in (args.paths or DEFAULT_ROOTS)]
    skills, errs = discover_skills(roots)
    if not skills:
        print("skills: 0 checked, 1 violation(s)")
        print("no SKILL.md files discovered", file=sys.stderr)
        return 1
    for skill in skills:
        errs.extend(lint_skill_file(skill, repo_root))
    for err in errs:
        print(err)
    print(f"skills: {len(skills)} checked, {len(errs)} violation(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

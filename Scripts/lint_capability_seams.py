#!/usr/bin/env python3
"""Enforce L0 capability-seam isolation across Operators, Subsystems, Runtime, and Apps."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = [
    ROOT / "Sources/Operators",
    ROOT / "Sources/Subsystems",
    ROOT / "Sources/Runtime",
    ROOT / "Sources/Apps",
]
ALLOWLIST = {
    (ROOT / "Sources/Runtime/QWEngine/Module/QWModuleRegistry.swift").resolve(),
}

GEMM_BUCKET_FAMILY = re.compile(r"\b(?:plan|shape)\s*\.\s*family\b")
PATTERNS = [
    ("model family comparison", re.compile(r"(?:\b\w+\s*\.\s*)?\.?family\s*(?:==|!=)\s*\.(?!f[1-9]\w*)")),
    ("family case equality", re.compile(r"\.(?:qwen38\w*|glm5Next|ernie55MoE|maxwellMoE|senseNovaLoop|inkling2Loop|bailingMoeV3|gigachat4|qwen3Dense|qwen3MoE|qwen2Dense|llamaDense|mistralDense|lfm2|lfm3|lfm2Encoder|modernBert)\s*(?:==|!=)")),
    ("switch on model family", re.compile(r"\bswitch\s+[^\{\n]*\.family\b")),
    ("contains model family", re.compile(r"\.contains\s*\(\s*[^)]*\.family\b")),
    ("modelType comparison", re.compile(r"\.modelType\s*(?:==|!=)")),
    ("descriptor.family.rawValue", re.compile(r"\bdescriptor\s*\.\s*family\s*\.\s*rawValue\b")),
    ("modelType string check", re.compile(r"\bmodelType\s*\.\s*(?:contains|hasPrefix)\s*\(")),
]
COMMENT_STRIPPED_PATTERNS = [
    ('json["model_type"]', re.compile(r"""\bjson\s*\[\s*"model_type"\s*\]""")),
]
# Temporary allowlist for pre-existing unowned legacy hits (flagged to @coordinator).
LEGACY_EXACT_HITS = {
    (
        "Sources/Subsystems/QWMTP/DFlash/QWDFlashConfig.swift",
        'json["model_type"]',
        'if let mt = (json["model_type"] as? String)?.lowercased(),',
    ),
    (
        "Sources/Runtime/QWEngine/Core/QWUnifiedEngine.swift",
        "descriptor.family.rawValue",
        "packPath: descriptor.family.rawValue,",
    ),
    (
        "Sources/Apps/qwg/QwgReplHelpers.swift",
        "descriptor.family.rawValue",
        "modelName: runtimeAssets.descriptor.family.rawValue,",
    ),
}


def _strip(src: str, strip_strings: bool) -> list[str]:
    out = []
    i, n = 0, len(src)
    in_block = False
    while i < n:
        if in_block:
            if src[i:i + 2] == "*/":
                out.append("  ")
                i += 2
                in_block = False
            else:
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
        elif src[i:i + 2] == "/*":
            out.append("  ")
            i += 2
            in_block = True
        elif src[i:i + 2] == "//":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
        elif src[i] == '"':
            out.append('"')
            i += 1
            while i < n and src[i] != '"':
                if src[i] == "\\" and i + 1 < n:
                    out.append("  " if strip_strings else src[i:i + 2])
                    i += 2
                elif src[i] == "\n":
                    break
                else:
                    out.append(" " if strip_strings else src[i])
                    i += 1
            if i < n and src[i] == '"':
                out.append('"')
                i += 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out).splitlines()


def strip_comments_and_strings(src: str) -> list[str]:
    return _strip(src, strip_strings=True)


def scan() -> list[str]:
    violations: list[str] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for path in sorted(scan_dir.rglob("*.swift")):
            if path.resolve() in ALLOWLIST:
                continue
            src = path.read_text(encoding="utf-8")
            lines = strip_comments_and_strings(src)
            comment_lines = _strip(src, strip_strings=False)
            rel = str(path.relative_to(ROOT))
            for lineno, line in enumerate(lines, start=1):
                sanitized = GEMM_BUCKET_FAMILY.sub("gemm_bucket", line)
                matched = False
                for label, pat in PATTERNS:
                    if pat.search(sanitized):
                        if (rel, label, line.strip()) not in LEGACY_EXACT_HITS:
                            violations.append(f"{rel}:{lineno}: banned {label}: {line.strip()}")
                        matched = True
                        break
                if matched:
                    continue
                cline = comment_lines[lineno - 1]
                for label, pat in COMMENT_STRIPPED_PATTERNS:
                    if pat.search(cline):
                        if (rel, label, cline.strip()) not in LEGACY_EXACT_HITS:
                            violations.append(f"{rel}:{lineno}: banned {label}: {cline.strip()}")
                        break
    return violations


def main() -> int:
    violations = scan()
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        print(f"capability_seams: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("capability_seams: 0 violation(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

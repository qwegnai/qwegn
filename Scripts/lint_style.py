#!/usr/bin/env python3
"""Style gate for Swift, Metal and Python: R1 file<=800 lines, R2 header<=7
lines, R3 pre-declaration comment blocks<=3 lines. Long design notes belong
in specs/*.md. Usage: lint_style.py [dir ...] (default: Sources Tests Scripts)
"""
import os, re, sys

MAX_FILE, MAX_HEADER, MAX_BLOCK = 800, 7, 3
SKIP_DIRS = {'.build', '.venv-oracle', '__pycache__', 'results', 'node_modules', '.git'}

DECL_SWIFT = re.compile(r'^(?:@\w+.*|public |internal |private |fileprivate |final |open |'
                        r'indirect |static |class |convenience |required |override |const )*'
                        r'(?:class|struct|enum|actor|protocol|func|init|var|let|typealias|'
                        r'extension|kernel|typedef|constant)\b')
DECL_PY = re.compile(r'^(?:@[\w.]+|async def |def |class |if __name__)\b')
MARKS = ('// MARK', '// region', '// region:', '# MARK')


def comment_block(lines, i, prefix):
    """Size (excluding MARK/region banners) and end index of the block at i."""
    j, size = i, 0
    while j < len(lines) and lines[j].strip().startswith(prefix):
        size = 0 if lines[j].strip().startswith(MARKS) else size + 1
        j += 1
    return size, j


def check_c_like(path, ext):
    errs = []
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    if len(lines) > MAX_FILE:
        errs.append(f"{path}:1: R1 file is {len(lines)} lines (max {MAX_FILE})")
    hdr, i = 0, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('//'):
            hdr, _ = comment_block(lines, i, '//')
            break
        if s == '' and not hdr:
            i += 1
        else:
            break
    if hdr > MAX_HEADER:
        errs.append(f"{path}:1: R2 header comment is {hdr} lines (max {MAX_HEADER})")
    decl = DECL_SWIFT
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('//'):
            size, j = comment_block(lines, i, '//')
            if j < len(lines) and decl.match(lines[j].strip()) and size > MAX_BLOCK:
                errs.append(f"{path}:{i+1}: R3 {size}-line comment before declaration "
                            f"(max {MAX_BLOCK})")
            i = j
        else:
            i += 1
    return errs


def check_py(path):
    errs = []
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    if len(lines) > MAX_FILE:
        errs.append(f"{path}:1: R1 file is {len(lines)} lines (max {MAX_FILE})")
    i, hdr = (1, 0) if lines and lines[0].startswith('#!') else (0, 0)
    if i:
        while i < len(lines) and lines[i].strip().startswith('#'):
            hdr += 1
            i += 1
    if hdr > MAX_HEADER:
        errs.append(f"{path}:1: R2 header comment is {hdr} lines (max {MAX_HEADER})")
    while i < len(lines):
        if lines[i].strip().startswith('#'):
            size, j = comment_block(lines, i, '#')
            if j < len(lines) and DECL_PY.match(lines[j].strip()) and size > MAX_BLOCK:
                errs.append(f"{path}:{i+1}: R3 {size}-line comment before declaration "
                            f"(max {MAX_BLOCK})")
            i = j
        else:
            i += 1
    return errs


CHECKS = {'.swift': check_c_like, '.metal': check_c_like, '.metalh': check_c_like,
          '.py': check_py}
errs = []
for root_dir in (sys.argv[1:] or ['Sources', 'Tests', 'Scripts']):
    for root, dirs, fs in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(fs):
            path = os.path.join(root, f)
            for ext, check in CHECKS.items():
                if f.endswith(ext):
                    errs += (check(path, ext) if check is check_c_like else check(path))
                    break
for e in errs:
    print(e)
print(f"style: {len(errs)} violation(s)")
sys.exit(1 if errs else 0)

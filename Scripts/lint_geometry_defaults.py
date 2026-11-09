#!/usr/bin/env python3
"""The geometry-default gate.

A helper that takes a `QWModelGeometry` must be TOLD which geometry. A
default is how a 27B-shaped number sneaks back in: the call compiles, the
constants are the 27B's, and the model runs with the wrong numbers. Task
4.3g removed the defaults that were there and this gate keeps them gone.

The rule: a parameter of type `QWModelGeometry` carries NO default value.
"""
import os
import re
import sys

# A parameter declaration of type QWModelGeometry with a default value.
# The declaration may span lines, so the pattern is matched over the whole
# file text and the line number is recovered from the match offset.
PARAM_RE = re.compile(r"\b[A-Za-z_]\w*\s*:\s*QWModelGeometry\s*=\s*[^,)\n]+")

problems = 0
roots = sys.argv[1:] or ["Sources"]
for root in roots:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".swift"):
                continue
            path = os.path.join(dirpath, fn)
            text = open(path, encoding="utf-8", errors="replace").read()
            for m in PARAM_RE.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                snippet = " ".join(m.group(0).split())
                print(f"{path}:{line}: a QWModelGeometry parameter carries a "
                      f"default — {snippet}")
                problems += 1

if problems:
    print(f"geometry: {problems} violation(s)")
    sys.exit(1)
print("geometry: 0 violation(s)")

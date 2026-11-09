#!/usr/bin/env python3
"""Heuristic dead-symbol scan over Sources+Tests+Scripts. Prints CANDIDATES
only — a symbol declared and never referenced anywhere. Framework-called
names are allowlisted; still verify every hit by hand before deleting."""
import os, re, collections, sys

ALLOW = {'main', 'makeNSView', 'updateNSView', 'applicationDidFinishLaunching',
         'applicationShouldTerminate', 'applicationShouldTerminateAfterLastWindowClosed',
         'applicationWillTerminate', 'Qwg', 'QWMenuBarApp'}  # @main entry points
TYPE_RE = re.compile(r'^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public |internal |private |'
                     r'fileprivate |final |open |indirect )*(?:class|struct|enum|actor|'
                     r'protocol)\s+([A-Za-z_]\w*)', re.M)
FUNC_RE = re.compile(r'^\s*(?:public |internal |private |fileprivate |static |class |'
                     r'final )*func\s+([a-z_]\w*)\s*[(<]', re.M)
TEST_RE = re.compile(r'@Test(?:\(.*?\))?\s+func\s+([a-z_]\w*)\s*[(<]', re.M)

files = []
for r in (sys.argv[1:] or ['Sources', 'Tests', 'Scripts']):
    for root, _, fs in os.walk(r):
        files += [os.path.join(root, f) for f in fs if f.endswith('.swift')]
txt = {p: open(p, encoding='utf-8', errors='replace').read() for p in files}
decls: dict[str, list] = {}
test_funcs = {m.group(1) for t in txt.values() for m in TEST_RE.finditer(t)}
def get(name: str) -> list:
    if name not in decls: decls[name] = [0, set()]
    return decls[name]

for p, t in txt.items():
    for m in TYPE_RE.finditer(t):
        d = get(m.group(1)); d[0] += 1; d[1].add(p)
    for m in FUNC_RE.finditer(t):
        d = get(m.group(1)); d[0] += 1; d[1].add(p)
for name, (n, ps) in sorted(decls.items()):
    if name in ALLOW or name in test_funcs or name.startswith('test') or name.endswith('Tests'): continue
    total = sum(len(re.findall(r'\b' + re.escape(name) + r'\b', t)) for t in txt.values())
    if total <= n:
        print(f"dead? {name:35s} {sorted(ps)}")

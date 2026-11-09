#!/usr/bin/env python3
"""ASSET INHERITANCE for an EXISTING pack (release law, 2026-10-02):
create <pack>.assets/ from the official source directory — every
non-weight file (tokenizer, vocab/merges, chat template, preprocessor +
video preprocessor + generation config, LICENSE) copied with a manifest —
and rewrite the pack's provenance (assets_dir + manifest hash) by
rewriting the trailing header (data region bytes/offsets untouched, so
all planes stay valid). The pack + its assets dir then constitute the
complete ENGINE DIRECTORY; the source dir may be deleted.
"""
import sys, os, json, hashlib, shutil
from pathlib import Path

pack, src = sys.argv[1], sys.argv[2]
# A FLAT root keeps its assets beside the pack; the legacy layout uses the
# sibling `<pack>.assets/` directory. `--flat-root` selects the former.
flat_root = "--flat-root" in sys.argv[3:]
if flat_root:
    assets = Path(pack).resolve().parent
else:
    assets = Path(str(pack) + ".assets")
assets.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

manifest = {}
for name in sorted(os.listdir(src)):
    if name.endswith(".safetensors") or name.endswith(".index.json"):
        continue                       # weights belong to the pack, not assets
    if name.startswith(".") or name == "crc32.txt":
        continue                       # HF download artifacts
    sp = os.path.join(src, name)
    if not os.path.isfile(sp):
        continue
    shutil.copy2(sp, os.path.join(assets, name))
    manifest[name] = {"sha256": sha256(sp), "bytes": os.path.getsize(sp)}
with open(assets / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=1, sort_keys=True)
assets_hash = hashlib.sha256((assets / "manifest.json").read_bytes()).hexdigest()

# provenance rewrite (header lives at file tail + size at offset 4; the
# 44 GB payload is NOT touched — only the tail region)
with open(pack, "r+b") as f:
    assert f.read(4) == b"QWP1"
    n = int.from_bytes(f.read(8), "little")
    f.seek(-n, 2)
    hdr = json.loads(f.read(n))
    hdr["provenance"]["assets_dir"] = str(assets)
    hdr["provenance"]["assets_manifest_sha256"] = assets_hash
    new = json.dumps(hdr).encode()
    f.seek(0, 2)
    f.write(new)                       # append new header…
    f.seek(4)
    f.write(len(new).to_bytes(8, "little"))   # …and the size prefix
print(f"assets: {len(manifest)} files inherited → {assets}")
print("manifest sha256:", assets_hash)
for k in sorted(manifest):
    print(f"  {k:36s} {manifest[k]['bytes']:>10} B")

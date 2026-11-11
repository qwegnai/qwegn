#!/usr/bin/env python3
"""
Standalone release-bundle audit for Qwen3.8-Flash-Next (HQ4_G32_M5 + PLE N-gram).
Verifies pack, sidecar, manifest, PLE locations, ngram.bin, and router constants.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from pack import (NGRAM_BIN_NAME, NGRAM_ROW_BYTES, PLE_LOCATION_NAME,
                   audit_ngram_bin, resolve_assets_dir)
from verify import audit_pack

REQUIRED_ASSETS = ("config.json", "tokenizer.json", "chat_template.jinja")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def validate_router_constants(rc: dict) -> dict:
    layers = rc.get("layers") if isinstance(rc, dict) else None
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"L7 router_constants: missing/empty layers: {rc}")
    summary = {}
    for lk, entry in layers.items():
        if not lk.isdigit():
            raise ValueError(f"L7 layer {lk!r}: non-numeric layer key")
        for req in ("layer_multipliers", "ngram_heads_offsets",
                    "ngram_heads_vocab_sizes"):
            if req not in entry or not isinstance(entry[req], list) or not entry[req]:
                raise ValueError(f"L7 layer {lk}: missing/empty {req}")
        mults = entry["layer_multipliers"]
        offs = entry["ngram_heads_offsets"]
        vocs = entry["ngram_heads_vocab_sizes"]
        if len(offs) != len(vocs):
            raise ValueError(
                f"L7 layer {lk}: offsets len {len(offs)} != vocabs len {len(vocs)}")
        for i, m in enumerate(mults):
            if not isinstance(m, int) or m <= 0 or (m % 2 == 0):
                raise ValueError(
                    f"L7 layer {lk}: multiplier[{i}]={m} not a positive odd integer")
        cum = 0
        for h, (o, v) in enumerate(zip(offs, vocs)):
            if not isinstance(v, int) or v < 2 or not _is_prime(v):
                raise ValueError(
                    f"L7 layer {lk}: vocab_sizes[{h}]={v} is not prime")
            if o != cum:
                raise ValueError(
                    f"L7 layer {lk}: offsets[{h}]={o} != cumulative {cum}")
            cum += v
        summary[int(lk)] = {
            "order": len(mults) + 1, "heads": len(vocs), "vocab_sum": cum,
        }
    return summary


def audit_release(release: Path, checkpoint: Path | None = None,
                  post_pruning: bool = False) -> dict:
    release = Path(release)
    pack = release / "model.qwp"
    sidecar = release / "model.qwp.sha256"
    assets = resolve_assets_dir(release)
    manifest_p = assets / "manifest.json"
    loc_p = assets / PLE_LOCATION_NAME
    ngram_p = assets / NGRAM_BIN_NAME

    if not pack.is_file():
        raise ValueError(f"L1: missing {pack}")
    header, counts = audit_pack(pack)
    prov = header.get("provenance", {})

    if not sidecar.is_file():
        raise ValueError(f"L2: missing {sidecar}")
    recorded_pack_sha = sidecar.read_text().strip().split()[0]
    actual_pack_sha = _sha256(pack)
    if recorded_pack_sha != actual_pack_sha:
        raise ValueError(
            f"L2: model.qwp.sha256 sidecar {recorded_pack_sha} != {actual_pack_sha}")

    if not manifest_p.is_file():
        raise ValueError(f"L3: missing {manifest_p}")
    manifest = json.loads(manifest_p.read_text())
    cfg_p = assets / "config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.is_file() else {}
    tc = cfg.get("text_config", cfg)
    has_ple = bool(
        tc.get("ple_embed_dim")
        or tc.get("ple_layer_ids")
        or tc.get("ple_module_indices")
        or PLE_LOCATION_NAME in manifest
        or NGRAM_BIN_NAME in manifest
    )
    req_assets = (
        (*REQUIRED_ASSETS, PLE_LOCATION_NAME, NGRAM_BIN_NAME)
        if has_ple else REQUIRED_ASSETS
    )
    for req in req_assets:
        if req not in manifest:
            raise ValueError(f"L3: manifest missing required entry {req!r}")
    for name, meta in sorted(manifest.items()):
        p = assets / name
        if not p.is_file():
            raise ValueError(f"L3: manifest asset {name!r} missing on disk")
        sz = p.stat().st_size
        if sz != meta.get("bytes"):
            raise ValueError(
                f"L3: {name}: disk size {sz} != manifest {meta.get('bytes')}")
        digest = _sha256(p)
        if digest != meta.get("sha256"):
            raise ValueError(
                f"L3: {name}: sha256 {digest} != manifest {meta.get('sha256')}")
    on_disk = {p.name for p in assets.iterdir() if p.is_file()} - {"manifest.json"}
    extra = sorted(on_disk - set(manifest))
    if extra:
        raise ValueError(f"L3: unlisted file(s) in assets/: {extra}")

    manifest_sha = _sha256(manifest_p)
    if prov.get("assets_manifest_sha256") != manifest_sha:
        raise ValueError(
            f"L4: pack provenance assets_manifest_sha256 "
            f"{prov.get('assets_manifest_sha256')} != {manifest_sha} "
            f"(stale provenance)")
    qr = prov.get("quant_recorded")
    valid_qr = isinstance(qr, str) and (
        qr in ("hq4_g32_m5", "q4g", "hq8_hq4_hybrid_v1")
        or qr.startswith(("hq4_", "hq6_", "hq8_", "dense_hq", "shvq", "dense_shvq"))
    )
    if not valid_qr:
        raise ValueError(
            f"L4: unexpected quant_recorded={qr!r}")

    ver = None
    shards = []
    ng = None
    router_summary = None
    if has_ple:
        loc = json.loads(loc_p.read_text())
        ver = loc.get("version")
        if ver not in (1, 2):
            raise ValueError(f"L5: unsupported location version {ver!r}")
        if loc.get("row_bytes") != NGRAM_ROW_BYTES:
            raise ValueError(f"L5: row_bytes {loc.get('row_bytes')} != 320")
        shards = loc.get("shards", [])
        if not shards:
            raise ValueError("L5: empty shards[]")
        if [s.get("shard") for s in shards] != list(range(len(shards))):
            raise ValueError("L5: shards not dense 0..N-1")
        cum = 0
        src_dir = Path(checkpoint) if checkpoint else Path(prov.get("source", ""))
        for s in shards:
            shape = s.get("shape")
            if (s.get("dtype") != "BF16" or s.get("row_bytes") != NGRAM_ROW_BYTES
                    or not isinstance(shape, list) or len(shape) != 2
                    or shape[1] != 160 or shape[0] <= 0):
                raise ValueError(f"L5: malformed shard descriptor {s}")
            if s.get("first_row") != cum:
                raise ValueError(
                    f"L5: shard {s['shard']} first_row {s.get('first_row')} != {cum}")
            cum += shape[0]
            if not post_pruning and src_dir and src_dir.is_dir():
                sf = src_dir / s["file"]
                if not sf.is_file():
                    raise ValueError(f"L5: missing checkpoint shard {sf}")
                need_end = s["base_off"] + shape[0] * NGRAM_ROW_BYTES
                if sf.stat().st_size < need_end:
                    raise ValueError(f"L5: {sf.name} truncated (< {need_end})")
        if cum != loc.get("total_rows"):
            raise ValueError(
                f"L5: sum(rows)={cum} != total_rows={loc.get('total_rows')}")

        ng = audit_ngram_bin(ngram_p)
        if ng["entryCount"] != loc["total_rows"]:
            raise ValueError(
                f"L6: ngram.bin entryCount {ng['entryCount']} != "
                f"ple_ngram_locations total_rows {loc['total_rows']}")

        if ver == 2:
            if "router_constants" not in loc:
                raise ValueError("L7: v2 location file missing router_constants")
            router_summary = validate_router_constants(loc["router_constants"])

    return {
        "ok": True,
        "pack_sha256": actual_pack_sha,
        "manifest_sha256": manifest_sha,
        "counts": counts,
        "assets_count": len(manifest),
        "location_version": ver,
        "ple_shards": len(shards),
        "ngram": ng,
        "router": router_summary,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--post-pruning", action="store_true")
    args = ap.parse_args(argv)
    try:
        info = audit_release(args.release, args.checkpoint, args.post_pruning)
    except Exception as e:
        print(f"AUDIT FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

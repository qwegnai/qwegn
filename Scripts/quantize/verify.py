"""Audit HQ4_G32_M5 packs (2D + 3D expert stacks) and compare retained plane bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from hq_blocks import VALID_BITS, VALID_GROUPS, hq_block_bytes
from pack import Q4G_ALIGNMENT
from shvq import SHVQ_VALID_BITS, SHVQ_VALID_GROUPS, shvq_block_bytes

GROUP = 32
BLOCK_BYTES = 20


def _resolve_hq_layout(entry):
    hq = entry.get("hq")
    if hq is None:
        return GROUP, BLOCK_BYTES, "hq4_g32_m5"
    if not isinstance(hq, dict):
        raise ValueError(f"invalid hq metadata: {hq!r}")
    if hq.get("vq") == "shvq1":
        if set(hq) != {"bits", "group", "vq"}:
            raise ValueError(f"invalid shvq metadata keys: {hq!r}")
        bits, group = hq["bits"], hq["group"]
        if type(bits) is not int or type(group) is not int:
            raise ValueError(f"invalid shvq metadata types: {hq!r}")
        if bits not in SHVQ_VALID_BITS or group not in SHVQ_VALID_GROUPS:
            raise ValueError(f"unsupported shvq metadata: {hq!r}")
        return group, shvq_block_bytes(bits, group), f"shvq{bits}_g{group}"
    if set(hq) != {"bits", "group"}:
        raise ValueError(f"invalid hq metadata: {hq!r}")
    bits, group = hq["bits"], hq["group"]
    if type(bits) is not int or type(group) is not int:
        raise ValueError(f"invalid hq metadata types: {hq!r}")
    if bits not in VALID_BITS or group not in VALID_GROUPS:
        raise ValueError(f"unsupported hq metadata: {hq!r}")
    return group, hq_block_bytes(bits, group), f"hq{bits}_g{group}"


def read_header(path):
    with Path(path).open("rb") as f:
        if f.read(4) != b"QWP1":
            raise ValueError(f"bad pack magic: {path}")
        size = int.from_bytes(f.read(8), "little")
        length = f.seek(0, 2)
        if not 0 < size <= length - 12:
            raise ValueError(f"invalid header length: {path}")
        start = length - size
        f.seek(start)
        return json.loads(f.read(size)), start


def planes(entry):
    shape = entry["shape"]
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"invalid tensor shape: {shape}")
    if "raw_off" in entry:
        rkind = entry.get("kind")
        if rkind == "bf16":
            return [("raw_off", math.prod(shape) * 2, "raw")]
        if rkind == "f32":
            return [("raw_off", math.prod(shape) * 4, "raw")]
        raise ValueError("raw tensor must be BF16 or F32")
    if "experts" in entry:
        grp, bbytes, _ = _resolve_hq_layout(entry)
        if len(shape) != 3 or shape[1] % 2 or shape[2] % grp:
            raise ValueError(f"invalid expert stack shape: {shape} for group {grp}")
        E, N, K = shape
        if entry["experts"] != E:
            raise ValueError("experts count does not match shape[0]")
        e_hq4 = N * (K // grp) * bbytes
        if entry.get("e_hq4") != e_hq4:
            raise ValueError(
                f"expert e_hq4 {entry.get('e_hq4')} != {e_hq4} (not contiguous)")
        return [("hq4_off", E * e_hq4, "hq4")]
    if "hq8_off" in entry:
        if len(shape) != 2 or shape[1] % GROUP:
            raise ValueError(f"invalid HQ8_G32_M5 tensor shape: {shape}")
        if "hq4_off" in entry:
            raise ValueError("hq8 tensor must not carry an hq4 plane")
        n, k = shape
        return [("hq8_off", n * k, "hq8"),
                ("hq8_s_off", n * (k // GROUP) * 2, "hq8")]
    grp, bbytes, _ = _resolve_hq_layout(entry)
    if len(shape) != 2 or shape[1] % grp:
        raise ValueError(f"invalid HQ tensor shape: {shape} for group {grp}")
    n, k = shape
    return [("hq4_off", n * (k // grp) * bbytes, "hq4")]


def _check_gap(f, gap_start, gap_len):
    if gap_len == 0:
        return
    f.seek(gap_start)
    step = 1 << 16
    left = gap_len
    while left:
        block = f.read(min(step, left))
        if block != b"\0" * len(block):
            raise ValueError("overlap or unused payload in pack (non-zero pad)")
        left -= len(block)


def audit_pack(path, expected_shapes=None):
    header, start = read_header(path)
    tensors = header["tensors"]
    if not tensors:
        raise ValueError("empty pack")
    if expected_shapes is not None:
        if {name: entry["shape"] for name, entry in tensors.items()} != expected_shapes:
            raise ValueError("packed tensors do not match source names and shapes")
    counts = {"hq4_g32_m5": 0, "bf16": 0}
    for name, entry in tensors.items():
        if "raw_off" in entry:
            allowed = {"shape", "kind", "raw_off"}
            kind = entry.get("kind", "bf16")
        elif "experts" in entry:
            allowed = {"shape", "experts", "e_hq4", "hq4_off"} | ({"hq"} if "hq" in entry else set())
            _, _, kind = _resolve_hq_layout(entry)
        elif "hq8_off" in entry:
            allowed = {"shape", "hq8_off", "hq8_s_off"}
            kind = "hq8_g32_m5"
        else:
            allowed = {"shape", "hq4_off"} | ({"hq"} if "hq" in entry else set())
            _, _, kind = _resolve_hq_layout(entry)
        if set(entry) != allowed:
            raise ValueError(
                f"{name}: unexpected or missing planes (only hq4_off or raw_off allowed)")
        counts[kind] = counts.get(kind, 0) + 1
        for key, length, _ in planes(entry):
            off = entry[key]
            if type(off) is not int or off < 256 or off % 256 or off + length > start:
                raise ValueError(f"{name}: invalid {key} extent")
    with Path(path).open("rb") as f:
        def layout_error(cadence):
            q4g_align, raw_align = cadence
            end = 12
            prev_align = 256
            for name, entry in sorted(
                    tensors.items(),
                    key=lambda kv: kv[1].get("raw_off",
                                             kv[1].get("hq4_off",
                                                       kv[1].get("hq8_off", 0)))):
                for key, length, kind in planes(entry):
                    if kind == "raw":
                        my = raw_align
                    elif kind == "hq8":
                        my = 256
                    else:
                        my = q4g_align
                    off = entry[key]
                    want = (end + prev_align - 1) // prev_align * prev_align
                    want = (want + my - 1) // my * my
                    if off != want:
                        return "overlap or unused payload in pack (not contiguous)"
                    _check_gap(f, end, off - end)
                    end = off + length
                    prev_align = my
            tail = (end + prev_align - 1) // prev_align * prev_align
            if start != tail:
                return "unused trailing payload in pack"
            _check_gap(f, end, start - end)
            return None
        err = layout_error((Q4G_ALIGNMENT, 256))
        if err is not None:
            legacy = layout_error((256, 256))
            if legacy is not None:
                raise ValueError(err)
    return header, counts


def hash_region(f, off, length):
    f.seek(off)
    h = hashlib.sha256()
    while length:
        data = f.read(min(length, 1 << 20))
        if not data:
            raise ValueError("truncated plane")
        h.update(data)
        length -= len(data)
    return h.hexdigest()


def compare_planes(path, reference, allow_requantized=()):
    header, _ = audit_pack(path)
    reference = Path(reference)
    if reference.is_dir():
        reference = reference / "model.qwp"
    old, old_start = read_header(reference)
    if header["tensors"].keys() != old["tensors"].keys():
        raise ValueError("reference tensor names differ")
    allowed = set(allow_requantized)
    unknown = allowed - header["tensors"].keys()
    if unknown:
        raise ValueError(f"unknown requantized tensor(s): {', '.join(sorted(unknown))}")
    compared = 0
    skipped = []
    with Path(path).open("rb") as new_file, reference.open("rb") as old_file:
        for name, entry in header["tensors"].items():
            previous = old["tensors"][name]
            if entry["shape"] != previous["shape"]:
                raise ValueError(f"{name}: reference shape differs")
            new_planes = planes(entry)
            old_planes = planes(previous)
            if new_planes != old_planes:
                if name not in allowed:
                    raise ValueError(f"{name}: reference plane schema differs")
                skipped.append(name)
                continue
            if name in allowed:
                raise ValueError(
                    f"{name}: allowed requantization did not change plane schema")
            for key, length, _ in new_planes:
                off = previous.get(key)
                if type(off) is not int or off < 12 or off + length > old_start:
                    raise ValueError(
                        f"{name}: missing or truncated reference plane {key}")
                if hash_region(new_file, entry[key], length) != hash_region(old_file, off, length):
                    raise ValueError(f"{name}: {key} bytes differ")
                compared += length
    if allowed != set(skipped):
        raise ValueError(
            f"requantized tensor(s) do not have a changed plane schema: "
            f"{', '.join(sorted(allowed - set(skipped)))}")
    return compared, skipped


def verify_assets(directory, header):
    from pack import sha256, resolve_assets_dir
    assets = resolve_assets_dir(directory)
    manifest_path = assets / "manifest.json"
    if sha256(manifest_path) != header["provenance"]["assets_manifest_sha256"]:
        raise ValueError("assets manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    if not {"config.json", "tokenizer.json", "chat_template.jinja"} <= manifest.keys():
        raise ValueError("required assets missing from manifest")
    actual = {p.name for p in assets.iterdir()}
    # A flat root keeps the pack beside the assets, so the pack's own files
    # live in the assets directory and are not assets.
    actual -= {"model.qwp", "model.qwp.sha256"}
    if actual != set(manifest) | {"manifest.json"}:
        raise ValueError("unexpected or missing assets")
    for name, entry in manifest.items():
        if Path(name).name != name or name.endswith((".safetensors", ".index.json")):
            raise ValueError(f"invalid asset name: {name}")
        path = assets / name
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"asset hash/size differs: {name}")
    checksum = (directory / "model.qwp.sha256").read_text().split()
    if len(checksum) != 2 or checksum[1] != "model.qwp":
        raise ValueError("invalid weights checksum file")
    if sha256(directory / "model.qwp") != checksum[0]:
        raise ValueError("weights checksum differs")
    return len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="HQ4_G32_M5 model directory")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--allow-requantized", action="append", default=[])
    args = parser.parse_args()
    weights = args.model / "model.qwp"
    header, counts = audit_pack(weights)
    assets = verify_assets(args.model, header)
    compared, skipped = (compare_planes(weights, args.reference, args.allow_requantized)
                         if args.reference else (None, []))
    print(json.dumps({"status": "PASS", "bytes": weights.stat().st_size,
                      "tensors": counts, "assets": assets,
                      "reference_bytes_matched": compared,
                      "reference_requantized_skipped": skipped}, sort_keys=True))

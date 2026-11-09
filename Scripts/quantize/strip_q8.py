#!/usr/bin/env python3
"""Q8-MIRROR STRIP for an EXISTING QWP1 pack (2026-10-04).

The q8_0 plane was a 1.016 B/param second quantization of every q4_g
tensor — 57% of pack bytes, read by NO engine kernel (only
QWPack.dequantQ8, the legacy inspect escape hatch). This tool rebuilds a
pack with ONLY the q4_g planes and the verbatim bf16 planes.

LAW: pure memcpy of q4_g/bf16 plane BYTES — no requantization, no math —
so every stored byte that the engine reads is bit-identical to the input
pack. Goldens, oracle parity, and engine parity are unaffected. Offsets
are pack-native file offsets (header at the tail); the reader resolves
tensors by NAME, so a rebuilt header is valid.

Layout preserved exactly as pack.py writes it:
  "QWP1" | u64 header-size (patched at the end) | pad to 256 |
  per tensor (any order; 256 B aligned):
    raw plane   : bf16 bytes                        {shape, raw_off, kind}
    q4g plane   : nibbles | s f16 | min f16         {shape, q4g_*_off}
  | trailing JSON header {provenance, tensors}

Usage: python3 strip_q8.py <in.qwp> <out.qwp>
Refuses an input without q8 planes (idempotence guard).
"""
import json, os, struct, sys

GROUP = 128
CHUNK = 1 << 26  # 64 MiB copy grain

def main(inp, outp):
    with open(inp, "rb") as f:
        assert f.read(4) == b"QWP1", "not a QWP1 pack"
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(-n, 2)
        hdr = json.loads(f.read(n))
    idx = hdr["tensors"]
    with open(inp, "rb") as f:
        n_q8 = sum(1 for e in idx.values() if "q8_off" in e)
        if n_q8 == 0:
            sys.exit(f"refused: {inp} carries no q8 planes (already stripped?)")
        print(f"input: {len(idx)} tensors, {n_q8} with q8 mirror")

        out = open(outp, "wb")
        out.write(b"QWP1")
        out.write(b"\0" * 8)           # header-size placeholder
        off = out.tell()
        def pad256():
            nonlocal off
            off = out.tell()
            p = (-off) % 256
            if p:
                out.write(b"\0" * p)
                off += p
        pad256()

        def copy_plane(src_off, size):
            """memcpy src_off..+size from the input into the output at `off`."""
            nonlocal off
            pad256()
            dest_off = off
            f.seek(src_off)
            remaining = size
            while remaining:
                b = f.read(min(CHUNK, remaining))
                if not b:
                    sys.exit(f"truncated input at {src_off}+{size}")
                out.write(b)
                remaining -= len(b)
            off = out.tell()
            return dest_off

        new_idx = {}
        # deterministic order: name-sorted (reader is name-keyed anyway)
        for name in sorted(idx):
            e = idx[name]
            shape = e["shape"]
            cnt = 1
            for d in shape: cnt *= d
            if "raw_off" in e:
                entry = {"shape": shape, "kind": "bf16"}
                entry["raw_off"] = copy_plane(e["raw_off"], cnt * 2)
            else:
                assert len(shape) == 2, f"{name}: non-2-D quant plane"
                N, K = shape; g = K // GROUP
                entry = {"shape": shape}
                entry["q4g_off"]   = copy_plane(e["q4g_off"],   N * g * 64)
                entry["q4g_s_off"] = copy_plane(e["q4g_s_off"], N * g * 2)
                entry["q4g_m_off"] = copy_plane(e["q4g_m_off"], N * g * 2)
                # q8_off / q8_s_off DROPPED by design (q8-removal law)
            new_idx[name] = entry
        prov = dict(hdr["provenance"])
        prov["planes_stripped"] = "q8_0 (strip_q8.py 2026-10-04; q4_g + bf16 only)"
        out_header = json.dumps({"provenance": prov, "tensors": new_idx}).encode()
        out.seek(4); out.write(len(out_header).to_bytes(8, "little"))
        out.seek(0, 2); out.write(out_header)
        out.close()
    print(f"output: {outp}")
    print(f"  size: {os.path.getsize(inp)/1e9:.2f} GB -> {os.path.getsize(outp)/1e9:.2f} GB")

if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: strip_q8.py <in.qwp> <out.qwp>"
    main(sys.argv[1], sys.argv[2])

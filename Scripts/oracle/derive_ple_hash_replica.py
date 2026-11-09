#!/usr/bin/env python3
"""Re-derive Qwen3.8-Flash-Next PLE hash math and validate against THE
CHECKPOINT, then emit golden vectors for the Swift loader.

Reference semantics: transformers 5.17.0 models/qwen4_exp/modeling_qwen4_exp.py
(splitmix64 / _build_layer_multipliers / _find_nth_prime_after /
Qwen4ExpTextNGramEmbedding.forward). Seed: config json has no `seed` key ->
dataclass default 1234; VERIFIED 2026-10-10 via HF range reads of
layers.1.ple.ple_embedding.layer_multipliers =
[23703573157769, 20109073645365, 8052911324071] (stored I64) — replica with
seed=1234 reproduces all three exactly.

Offline mode validates the derived prime chain + head tables against the
pinned inventory (rows arithmetic). --verify-live re-reads the checkpoint
multipliers over HTTP range requests (network; no GPU) and fails closed on
any drift.
"""
import json, math, struct, sys, urllib.request

_MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
PRIME_1 = 10007
SEED = 1234
REPO = "Qwen/Qwen3.8-Flash-Next"

def splitmix64(v: int) -> int:
    v = (v + GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * M1) & _MASK64
    v = ((v ^ (v >> 27)) * M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64

def layer_multipliers(unigram_vocab, ngram_size, ple_layer_index, seed=SEED):
    half = max(1, (((1 << 63) - 1) // max(unigram_vocab, 1)) // 2)
    base = seed + PRIME_1 * ple_layer_index
    return [2 * (splitmix64((base + GAMMA * (i + 1)) & _MASK64) % half) + 1
            for i in range(ngram_size)]

def primes_after(base: int, count: int):
    """First `count` primes > base via segmented Eratosthenes."""
    lo, out = base + 1, []
    span = 4_000_000
    small = [p for p in range(2, math.isqrt(base + count * span // max(count, 1) + span)) ]
    small = [p for p in range(2, math.isqrt(base + span) + 1) if all(p % q for q in range(2, math.isqrt(p) + 1))]
    while len(out) < count:
        hi = lo + span - 1
        sieve = bytearray(hi - lo + 1)
        for p in small:
            start = ((lo + p - 1) // p) * p
            for m in range(start - lo, hi - lo + 1, p):
                sieve[m] = 1
        for i in range(hi - lo + 1):
            v = lo + i
            if v >= 2 and not sieve[i]:
                out.append(v)
                if len(out) >= count:
                    return out[:count]
        lo = hi + 1
    return out

def head_tables(primes, heads_per_layer, n_layers, divisible_by):
    sizes, offsets, total = [], [], 0
    for h in range(heads_per_layer * n_layers):
        sizes.append(primes[h]); offsets.append(total); total += primes[h]
    padded = math.ceil(total / divisible_by) * divisible_by
    return sizes, offsets, total, padded

def fetch_stored_multipliers(cfg_inv):
    entry = cfg_inv["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"]
    def get(url, rng):
        req = urllib.request.Request(url, headers={"Range": rng})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    url = f"https://huggingface.co/{REPO}/resolve/main/{entry['file']}"
    hdr_len = struct.unpack("<Q", get(url, "bytes=0-7"))[0]
    hdr = json.loads(get(url, f"bytes=8-{8 + hdr_len - 1}"))
    lo, hi = hdr["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"]["data_offsets"]
    raw = get(url, f"bytes={8 + hdr_len + lo}-{8 + hdr_len + hi - 1}")
    return list(struct.unpack("<3q", raw))

def main() -> int:
    live = "--verify-live" in sys.argv
    tc = json.load(open("Scripts/oracle/config_flashnext_golden.json"))["text_config"]
    inv = json.load(open("Scripts/oracle/flashnext_tensor_inventory.json"))
    ngram_size, hpn = tc["ngram_size"], tc["heads_per_ngram"]
    heads_per_layer = (ngram_size - 1) * hpn                    # 16
    rows_key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    rows_per_part, head_dim = inv[rows_key]["shape"]
    parts = sorted({k.rsplit("shard_", 1)[1].split(".")[0] for k in inv if "ngram_embedding.shard_" in k})
    total_padded = len(parts) * rows_per_part
    print(f"inventory: {len(parts)} parts × {rows_per_part:,} rows × {head_dim} = {total_padded:,} padded rows")

    mult0 = layer_multipliers(tc["vocab_size"], ngram_size, 0)
    if live:
        stored = fetch_stored_multipliers(inv)
        print("stored   :", stored)
        print("replica  :", mult0)
        if stored != mult0:
            print("FAIL: stored multipliers drift from replica (seed changed upstream?)"); return 1
        print("live multiplier verification: PASS")
    else:
        # anchors verified via HF range reads 2026-10-10 (fail-closed copy):
        anchor = [23703573157769, 20109073645365, 8052911324071]
        assert mult0 == anchor, "replica drifted from 2026-10-10 verified anchors"
        print("offline: replica matches 2026-10-10 verified multiplier anchors")

    # prime chain sized so padded total equals the inventory's rows
    target_heads = heads_per_layer * 3  # 3 PLE layers expected from table arithmetic
    primes = primes_after(tc["ngram_vocab_size_base"], target_heads + 4)
    for n_layers in (1, 2, 3, 4):
        sizes, offs, cum, padded = head_tables(primes, heads_per_layer, n_layers, tc["make_ngram_vocab_size_divisible_by"])
        if padded == total_padded:
            print(f"prime chain over {n_layers} PLE layers ({heads_per_layer*n_layers} heads): raw {cum:,} -> padded {padded:,} MATCH")
            break
    else:
        print("FAIL: no whole-layer prime chain reproduces the inventory padded rows"); return 1
    derived_layers = (heads_per_layer * n_layers) // heads_per_layer

    # golden head-row vectors (eos-aware shift replicated from forward())
    eos = tc["eos_token_id"][0] if isinstance(tc.get("eos_token_id"), list) else tc["eos_token_id"]
    def shift_right_ignore_eos(toks, shift):
        pos = list(range(len(toks)))
        eospos = [p if t == eos else -1 for p, t in enumerate(toks)]
        prev_inc = []; m = -1
        for p in pos:
            m = max(m, eospos[p]); prev_inc.append(m)
        seg_start = [pe + 1 for pe in [-1] + prev_inc[:-1]]
        # position_in_segment = i - seg_start[i]; the early (2026-10-11)
        # `seg_start[i] >= shift` form was WRONG at segment starts (torch checks
        # the position WITHIN the segment). Caught by differential-parity:
        # post-eos tokens gathered pre-eos context instead of eos.
        return [toks[i - shift] if (i - seg_start[i] >= shift and i - shift >= 0) else eos for i in pos]
    tokens = [101, 20200, 35, eos, 7, 999, 20201, 64]
    ctx = ngram_size - 1
    history = [eos] * ctx + tokens
    shifted = [shift_right_ignore_eos(history, s) for s in range(ngram_size)]
    vectors = []
    for t in range(len(tokens)):
        ids = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * hpn
            for h in range(hpn):
                gi = start + h
                mixed = (shifted[0][ctx + t] * mult0[0]) & _MASK64
                for p in range(1, ngram):
                    mixed ^= (shifted[p][ctx + t] * mult0[p]) & _MASK64
                ids.append(mixed % sizes[gi] + offs[gi])
        vectors.append({"token": tokens[t], "head_row_ids": ids})
    out = {
        "note": "golden PLE hash vectors; reference transformers 5.17.0 qwen4_exp; multipliers byte-verified against checkpoint via HF range read 2026-10-10 (seed=1234 config default)",
        "seed": SEED, "eos_token_id": eos, "vocab_size": tc["vocab_size"],
        "ngram_size": ngram_size, "heads_per_ngram": hpn,
        "heads_layer0": sizes[:heads_per_layer], "offsets_layer0": offs[:heads_per_layer],
        "layer_multipliers_layer0": [str(x) for x in mult0],
        "derived_ple_layers": derived_layers,
        "raw_rows": cum, "total_padded_rows": total_padded,
        "head_dim": head_dim, "parts": len(parts),
        "tokens": tokens, "head_ids_by_token": vectors,
    }
    path = "Scripts/oracle/flashnext_ple_hash_golden.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")
    print("PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())

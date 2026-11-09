"""Teacher-forced KL of simulated Flash-Next packs against the BF16 checkpoint.

The BF16 source is streamed one decoder layer at a time through the reference
`qwen4_exp` layers. Every variant (a choice of trunk, routed-expert and n-gram
table formats, or the MLX checkpoint) carries its own hidden state, so each
layer's source weights are read once and reused by all variants. Quantized
weights come from `hq_fakequant.fake_quant`, which reproduces the packer's
reconstruction bit for bit. The output is full-vocabulary KL(teacher||variant)
per token, top-1 agreement and next-token NLL, stratified by corpus kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hq_blocks import hq_block_bytes, parse_quant_spec, resolve_group_size
from hq_fakequant import _m5_rot, fake_quant
from hq_frontier import pareto_frontier, routed_expert
from pack import is_raw_plane
from pack_ngram import _NG_SECTOR, _row_bytes, _rows_per_sector
from verify import planes, read_header

PREFIX = "model.language_model."
MLX_PREFIX = "language_model.model."
SEQ = 512
NGRAM_DIM = 160

VARIANTS = {
    "teacher": ("bf16", "bf16", "bf16", "bf16"),
    "incumbent": ("hq4_g32_m5", "hq4_g32_m5", "hq4_g32_m5", "q8"),
    "incumbent_bf16_ngram": ("hq4_g32_m5", "hq4_g32_m5", "hq4_g32_m5", "bf16"),
    "trunk_only_m5": ("hq4_g32_m5", "bf16", "bf16", "bf16"),
    "experts_only_m5": ("bf16", "hq4_g32_m5", "hq4_g32_m5", "bf16"),
    "t8_e_m5": ("hq8_g32", "hq4_g32_m5", "hq4_g32_m5", "q8"),
    "t6_e_m5": ("hq6_g32", "hq4_g32_m5", "hq4_g32_m5", "q8"),
    "t8_e4g32": ("hq8_g32", "hq4_g32", "hq4_g32", "q8"),
    "t8_e4g64": ("hq8_g32", "hq4_g64", "hq4_g64", "q8"),
    "t8_e4g128": ("hq8_g32", "hq4_g128", "hq4_g128", "q8"),
    "t8_e3g32": ("hq8_g32", "hq3_g32", "hq3_g32", "q8"),
    "t8_e3g64": ("hq8_g32", "hq3_g64", "hq3_g64", "q8"),
    "t8_m5_d6": ("hq8_g32", "hq4_g32_m5", "hq6_g32", "q8"),
    "t8_gu3_dm5": ("hq8_g32", "hq3_g32", "hq4_g32_m5", "q8"),
    "incumbent_q4_ngram": ("hq4_g32_m5", "hq4_g32_m5", "hq4_g32_m5", "q4"),
    "t8_e_m5_q4": ("hq8_g32", "hq4_g32_m5", "hq4_g32_m5", "q4"),
    "t8_e4g64_q4": ("hq8_g32", "hq4_g64", "hq4_g64", "q4"),
    "t8_e3g32_q4": ("hq8_g32", "hq3_g32", "hq3_g32", "q4"),
    "mlx": ("mlx", "mlx", "mlx", "mlx4"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_quantized_name(name: str, shape) -> bool:
    """Mirror of the packer's choice: every matrix that is not a raw plane is HQ-quantized."""
    return len(shape) >= 2 and not is_raw_plane(name) and "ngram_embedding" not in name


def tensor_spec(name: str, variant: tuple[str, str, str, str]) -> str:
    trunk, gate_up, down, _ = variant
    if name.endswith("mlp.experts.gate_up_proj"):
        return gate_up
    if name.endswith("mlp.experts.down_proj"):
        return down
    return trunk


class SafetensorsIndex:
    """Lazy, memory-mapped access to a sharded safetensors checkpoint."""

    def __init__(self, root: Path, index_name: str = "model.safetensors.index.json"):
        self.root = root
        index_path = root / index_name
        if index_path.exists():
            files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in root.glob("model*.safetensors"))
        self.entries: dict[str, tuple[Path, dict, int]] = {}
        for fname in files:
            path = root / fname
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for key, meta in header.items():
                if key != "__metadata__":
                    self.entries[key] = (path, meta, 8 + n)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def numpy(self, key: str) -> np.ndarray:
        path, meta, base = self.entries[key]
        dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
                 "U32": np.uint32, "I64": np.int64}[meta["dtype"]]
        start, end = meta["data_offsets"]
        count = (end - start) // np.dtype(dtype).itemsize
        arr = np.memmap(path, dtype=dtype, mode="r", offset=base + start, shape=(count,))
        return arr.reshape(meta["shape"])

    def float(self, key: str) -> torch.Tensor:
        _, meta, _ = self.entries[key]
        return to_float(self.numpy(key), meta["dtype"])

    def bf16(self, key: str) -> torch.Tensor:
        """Zero-copy BF16 view of a memory-mapped tensor."""
        _, meta, _ = self.entries[key]
        if meta["dtype"] != "BF16":
            raise ValueError(f"{key} is {meta['dtype']}, not BF16")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(self.numpy(key)).view(torch.bfloat16)


def to_float(arr: np.ndarray, dtype: str) -> torch.Tensor:
    if dtype == "BF16":
        return torch.from_numpy((np.ascontiguousarray(arr).astype(np.uint32) << 16).view(np.float32))
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))


def mlx_dequant(words: np.ndarray, scales: torch.Tensor, biases: torch.Tensor, k: int) -> torch.Tensor:
    """Dequantize an MLX affine tensor packed little-endian into u32 words along the last axis."""
    bits = words.shape[-1] * 32 // k
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported MLX bits {bits}")
    group = k // scales.shape[-1]
    per = 32 // bits
    w = torch.from_numpy(np.ascontiguousarray(words).view(np.int32))
    shifts = torch.arange(per, dtype=torch.int32) * bits
    q = ((w.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], k).float()
    q = q.reshape(*words.shape[:-1], k // group, group)
    return (q * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(*words.shape[:-1], k)


class MLXCheckpoint:
    def __init__(self, root: Path):
        self.st = SafetensorsIndex(root)

    def key(self, source_name: str) -> str:
        if source_name == "lm_head.weight":
            return "language_model.lm_head.weight"
        return MLX_PREFIX + source_name[len(PREFIX):]

    def tensor(self, source_name: str, fallback: torch.Tensor) -> torch.Tensor:
        """MLX's reconstruction of `source_name`; unquantized MLX tensors fall back to the source values."""
        if source_name.endswith("mlp.experts.gate_up_proj"):
            base = self.key(source_name).replace("experts.gate_up_proj", "switch_mlp.")
            k = fallback.shape[-1]
            return torch.cat([self._dequant(base + "gate_proj", k), self._dequant(base + "up_proj", k)], dim=1)
        if source_name.endswith("mlp.experts.down_proj"):
            base = self.key(source_name).replace("experts.down_proj", "switch_mlp.down_proj")
            return self._dequant(base, fallback.shape[-1])
        stem = self.key(source_name).removesuffix(".weight")
        if stem + ".scales" in self.st:
            return self._dequant(stem, fallback.shape[-1])
        return fallback

    def _dequant(self, stem: str, k: int) -> torch.Tensor:
        return mlx_dequant(self.st.numpy(stem + ".weight"), self.st.float(stem + ".scales"),
                           self.st.float(stem + ".biases"), k)


class NgramTables:
    """Fetches PLE n-gram rows by id from the source shards, the q8 pack law, or the MLX 4-bit table."""

    def __init__(self, source: SafetensorsIndex, mlx_table: Path | None):
        names = sorted((k for k in source.entries if "ngram_embedding.shard_" in k),
                       key=lambda k: int(k.split("shard_")[1].split(".")[0]))
        self.shards = [source.numpy(k) for k in names]
        self.rows_per_shard = self.shards[0].shape[0]
        if any(s.shape != (self.rows_per_shard, NGRAM_DIM) for s in self.shards):
            raise ValueError("PLE n-gram shards must share one [rows, 160] shape")
        self.total_rows = self.rows_per_shard * len(self.shards)
        self.mlx = None
        if mlx_table is not None:
            st = SafetensorsIndex.__new__(SafetensorsIndex)
            st.root = mlx_table.parent
            st.entries = {}
            with open(mlx_table, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for key, meta in header.items():
                if key != "__metadata__":
                    st.entries[key] = (mlx_table, meta, 8 + n)
            self.mlx = st

    def source_rows(self, ids: np.ndarray) -> torch.Tensor:
        if ids.min() < 0 or ids.max() >= self.total_rows:
            raise ValueError("n-gram id outside the source table")
        out = np.empty((len(ids), NGRAM_DIM), dtype=np.uint16)
        shard = ids // self.rows_per_shard
        for s in np.unique(shard):
            sel = shard == s
            out[sel] = self.shards[s][ids[sel] % self.rows_per_shard]
        return to_float(out, "BF16")

    def mlx_rows(self, ids: np.ndarray) -> torch.Tensor:
        words = self.mlx.numpy("weight")[ids]
        scales = to_float(self.mlx.numpy("scales")[ids], self.mlx.entries["scales"][1]["dtype"])
        biases = to_float(self.mlx.numpy("biases")[ids], self.mlx.entries["biases"][1]["dtype"])
        return mlx_dequant(words, scales, biases, NGRAM_DIM)


def q8_rows(rows: torch.Tensor) -> torch.Tensor:
    """The Q8_G32 symmetric law of `pack_ngram._encode_q8`: code = rint(x / scale32), value = code * f16(scale)."""
    g = rows.reshape(rows.shape[0], -1, 32)
    scale32 = g.abs().amax(dim=-1) / 127.0
    scale16 = scale32.half().float()
    live = scale16 > 0
    safe = torch.where(live, scale32, torch.ones_like(scale32))
    codes = torch.round(g / safe.unsqueeze(-1)).clamp(-127, 127)
    codes = torch.where(live.unsqueeze(-1), codes, torch.zeros_like(codes))
    return (codes * scale16.unsqueeze(-1)).reshape(rows.shape)


def q4_rows(rows: torch.Tensor) -> torch.Tensor:
    """The Q4_G32 n-gram law of `pack_ngram._encode_q4`: unrotated HQ4_G32 M5 blocks."""
    return _m5_rot(rows)


def ngram_file_bytes(fmt: str, rows: int) -> int:
    """Bytes of a sector-packed `ngram.bin` in `fmt`, header sector included."""
    rps = _rows_per_sector(_row_bytes(fmt, NGRAM_DIM))
    return _NG_SECTOR * (1 + (rows + rps - 1) // rps)


class RowTable(nn.Module):
    """Stand-in for the 205 GB `ngram_embedding`: looks up a precomputed set of rows by id."""

    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.lookup: torch.Tensor | None = None
        self.rows: torch.Tensor | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(self.lookup, ids.reshape(-1))
        if not bool((self.lookup[pos] == ids.reshape(-1)).all()):
            raise RuntimeError("n-gram id missing from the prefetched rows")
        return self.rows[pos].reshape(*ids.shape, -1)


def build_corpus(tokenizer, repo: Path, per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    def windows(texts: list[tuple[str, str]], kind: str) -> list[dict]:
        rng.shuffle(texts)
        out = []
        for label, text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < SEQ:
                continue
            start = rng.randrange(0, len(ids) - SEQ + 1)
            out.append({"stratum": kind, "source": label, "ids": ids[start:start + SEQ]})
            if len(out) == per_stratum:
                return out
        raise ValueError(f"not enough {kind} text for {per_stratum} windows")

    def read(paths) -> list[tuple[str, str]]:
        return [(str(p.relative_to(repo)), p.read_text(errors="replace")) for p in sorted(paths)]

    code = read([p for p in (repo / "Sources").rglob("*.swift") if p.stat().st_size > 6000][:400]
                + [p for p in (repo / "Scripts").rglob("*.py") if p.stat().st_size > 6000])
    prose = read([p for p in repo.rglob("*.md") if ".build" not in p.parts and ".venv" not in str(p)
                  and p.stat().st_size > 6000])
    json_texts = []
    for p in sorted((repo / "Scripts").rglob("*.json")) + sorted((repo / "Tests").rglob("*.json")):
        if 4000 < p.stat().st_size < 2_000_000:
            try:
                json_texts.append((str(p.relative_to(repo)),
                                   json.dumps(json.loads(p.read_text()), indent=2)))
            except ValueError:
                continue
    corpus = windows(code, "code") + windows(prose, "prose") + windows(json_texts, "json")
    corpus += chat_windows(tokenizer, per_stratum, rng)
    return corpus


def chat_windows(tokenizer, count: int, rng: random.Random) -> list[dict]:
    tools = [
        {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                          "parameters": {"type": "object", "properties": {
                                              "city": {"type": "string"}, "unit": {"type": "string",
                                                                                   "enum": ["c", "f"]}},
                                              "required": ["city"]}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 file.",
                                          "parameters": {"type": "object", "properties": {
                                              "path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "search_docs", "description": "Search documentation.",
                                          "parameters": {"type": "object", "properties": {
                                              "query": {"type": "string"}, "limit": {"type": "integer"}},
                                              "required": ["query"]}}},
    ]
    cities = ["Lisbon", "Osaka", "Nairobi", "Denver", "Tallinn", "Hanoi", "Quito", "Perth"]
    files = ["Package.swift", "README.md", "Scripts/quantize/pack.py", "config.json"]
    topics = ["prefix cache eviction", "speculative decoding", "tool call schema", "KV quantization"]
    out = []
    for i in range(count):
        messages = [{"role": "system", "content": "You are a careful assistant. Use tools when they help and "
                                                  "answer with short, exact JSON when asked."}]
        ids: list[int] = []
        turn = 0
        while len(ids) < SEQ:
            city, path, topic = rng.choice(cities), rng.choice(files), rng.choice(topics)
            kind = (i + turn) % 3
            if kind == 0:
                ask, call, result = (f"What's the weather in {city} in celsius?", "get_weather",
                                     {"city": city, "unit": "c"})
                reply = json.dumps({"city": city, "temp_c": rng.randint(-5, 35), "sky": "clear"})
                answer = f"It is currently {json.loads(reply)['temp_c']} °C and clear in {city}."
            elif kind == 1:
                ask, call, result = (f"Open {path} and tell me what it declares.", "read_file", {"path": path})
                reply = f"// {path}\nlet name = \"qwegine\"\nlet version = \"0.9.{turn}\"\n"
                answer = f"{path} declares `name` and `version` (0.9.{turn})."
            else:
                ask, call, result = (f"Find docs about {topic}, at most 3 hits.", "search_docs",
                                     {"query": topic, "limit": 3})
                reply = json.dumps({"hits": [f"docs/{topic.replace(' ', '-')}-{j}.md" for j in range(3)]})
                answer = json.dumps({"topic": topic, "documents": 3})
            messages += [
                {"role": "user", "content": ask},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": call, "arguments": result}}]},
                {"role": "tool", "content": reply},
                {"role": "assistant", "content": answer},
            ]
            text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            turn += 1
        out.append({"stratum": "chat", "source": f"synthetic-tool-chat-{i}", "ids": ids[:SEQ]})
    return out


def corpus_digest(corpus: list[dict]) -> str:
    h = hashlib.sha256()
    for row in corpus:
        h.update(np.asarray(row["ids"], dtype=np.int64).tobytes())
    return h.hexdigest()


def variant_cost(header: dict, variant: tuple[str, str, str, str], ngram_bytes: dict) -> dict:
    """Bytes of a pack written with this variant, from the incumbent header's tensor inventory."""
    payload = 0
    for name, entry in header["tensors"].items():
        count = math.prod(entry["shape"])
        if "raw_off" in entry:
            payload += sum(length for _, length, _ in planes(entry))
            continue
        if routed_expert(name, entry):
            spec = variant[2] if "down" in name else variant[1]
        else:
            spec = variant[0]
        if spec == "bf16":
            payload += count * 2
            continue
        bits, requested, _ = parse_quant_spec(spec)
        group = resolve_group_size(entry["shape"][-1], requested)
        payload += count // group * hq_block_bytes(bits, group)
    return {"model_bytes": payload, "ngram_bytes": ngram_bytes[variant[3]],
            "total_bytes": payload + ngram_bytes[variant[3]],
            "total_gib": (payload + ngram_bytes[variant[3]]) / 2**30}


class Harness:
    def __init__(self, args):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen4_exp import modeling_qwen4_exp as mq

        self.mq = mq
        self.args = args
        self.device = torch.device(args.device)
        self.source = SafetensorsIndex(args.source)
        cfg = AutoConfig.from_pretrained(args.source)
        self.cfg = cfg.text_config
        self.cfg._attn_implementation = "sdpa"
        if SEQ > self.cfg.indexer_budget:
            raise ValueError("the indexer bypass is exact only while the sequence fits the token budget")
        self.tokenizer = AutoTokenizer.from_pretrained(args.source)
        self.variants = {k: VARIANTS[k] for k in args.variants}
        self.mlx = MLXCheckpoint(args.mlx) if "mlx" in self.variants else None
        self.ngram = NgramTables(self.source, args.mlx / "ngram_table.bin" if self.mlx else None)

    def run(self) -> dict:
        args = self.args
        corpus = build_corpus(self.tokenizer, args.repo, args.per_stratum, args.seed)
        ids = torch.tensor([row["ids"] for row in corpus], dtype=torch.long)
        log(f"corpus: {ids.shape[0]} x {ids.shape[1]} tokens, sha256 {corpus_digest(corpus)[:16]}")
        hidden = self.forward_layers(ids)
        if args.hidden_cache:
            torch.save(hidden, args.hidden_cache)
        metrics = self.score(hidden, ids, corpus)
        header, _ = read_header(args.pack)
        rows = self.ngram.total_rows
        ngram_bytes = {fmt: ngram_file_bytes(fmt, rows) for fmt in ("bf16", "q8", "q4")}
        ngram_bytes["mlx4"] = (args.mlx / "ngram_table.bin").stat().st_size if args.mlx else 0
        for name, variant in self.variants.items():
            if variant[0] == "mlx":
                model_bytes = sum(p.stat().st_size for p in args.mlx.glob("model*.safetensors"))
                metrics[name]["size"] = {"model_bytes": model_bytes, "ngram_bytes": ngram_bytes["mlx4"],
                                         "total_bytes": model_bytes + ngram_bytes["mlx4"],
                                         "total_gib": (model_bytes + ngram_bytes["mlx4"]) / 2**30}
            else:
                metrics[name]["size"] = variant_cost(header, variant, ngram_bytes)
        teacher_sha = header.get("provenance", {}).get("sha256_source", "unknown")
        digest = corpus_digest(corpus)
        frontier = pareto_frontier([
            {"candidate": k, "pack_bytes": m["size"]["total_bytes"], "mean_kl": m["kl_mean"],
             "teacher_sha256": teacher_sha, "corpus_sha256": digest}
            for k, m in metrics.items()])
        return {
            "teacher": {"source": str(args.source), "sha256_source": teacher_sha,
                        "layers": args.layers or self.cfg.num_hidden_layers},
            "pack": str(args.pack),
            "corpus": {"sha256": digest, "sequences": len(corpus), "seq_len": SEQ,
                       "sources": [{"stratum": r["stratum"], "source": r["source"]} for r in corpus]},
            "variants": {k: {"spec": list(v), **metrics[k]} for k, v in self.variants.items()},
            "frontier": [row["candidate"] for row in frontier],
        }

    def layer_weights(self, layer: int) -> dict[str, torch.Tensor]:
        prefix = f"{PREFIX}layers.{layer}."
        return {k[len(prefix):]: (self.source.bf16(k) if ".experts." in k else self.source.float(k))
                for k in self.source.entries if k.startswith(prefix) and "ple_embedding." not in k}

    def variant_tensor(self, full_name: str, src: torch.Tensor, variant) -> torch.Tensor:
        spec = tensor_spec(full_name, variant)
        if spec == "mlx":
            return self.mlx.tensor(full_name, src).to(self.device)
        if not is_quantized_name(full_name, src.shape) or spec == "bf16":
            return src.to(self.device)
        return fake_quant(src.to(self.device), spec)

    def trunk_state(self, prefix: str, weights: dict, variant) -> dict[str, torch.Tensor]:
        return {k: self.variant_tensor(prefix + k, v, variant)
                for k, v in weights.items() if ".experts." not in k}

    def forward_layers(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        mq, cfg, dev = self.mq, self.cfg, self.device
        batch, seq = ids.shape
        ids_dev = ids.to(dev)
        embed = self.source.float(PREFIX + "embed_tokens.weight")
        hidden = {}
        for name, variant in self.variants.items():
            table = self.variant_tensor(PREFIX + "embed_tokens.weight", embed, variant)
            hidden[name] = table[ids].to(dev).repeat(1, 1, cfg.hc_count)
        del embed
        rotary = mq.Qwen4ExpTextRotaryEmbedding(config=cfg).to(dev)
        pos = torch.arange(seq, device=dev).view(1, 1, -1).expand(3, batch, -1)
        position_embeddings = rotary(hidden[next(iter(hidden))], pos)
        mask = torch.ones(seq, seq, dtype=torch.bool, device=dev).tril().view(1, 1, seq, seq).expand(batch, 1, -1, -1)
        layers = self.args.layers or cfg.num_hidden_layers
        for i in range(layers):
            t0 = time.time()
            hidden = self.forward_layer(i, hidden, ids_dev, position_embeddings, mask)
            log(f"layer {i + 1}/{layers} ({cfg.layer_types[i]}) {time.time() - t0:.1f}s")
        mixer = mq.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        src = {k[len(PREFIX + "hyper_connection_mixer."):]: self.source.float(k) for k in self.source.entries
               if k.startswith(PREFIX + "hyper_connection_mixer.")}
        out = {}
        with torch.no_grad():
            for name, variant in self.variants.items():
                mixer.load_state_dict(self.trunk_state(PREFIX + "hyper_connection_mixer.", src, variant),
                                      strict=True, assign=True)
                out[name] = mixer(hidden[name]).cpu()
        return out

    def make_layer(self, i: int):
        mq = self.mq
        with torch.device("meta"):
            layer = mq.Qwen4ExpTextDecoderLayer(self.cfg, i)
        table = None
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            table = RowTable(self.device)
            emb.ngram_embedding = table
            emb.layer_multipliers = mq._build_layer_multipliers(
                emb.unigram_vocab_size, emb.ngram_size, emb.ple_layer_index, emb.seed).to(self.device)
            emb.ngram_heads_vocab_sizes = torch.tensor(emb.head_vocab_sizes, dtype=torch.long, device=self.device)
            emb.ngram_heads_offsets = torch.tensor(emb.head_offsets, dtype=torch.long, device=self.device)
        if layer.layer_type != "linear_attention":
            layer.self_attn.indexer.forward = lambda hs, pe, attention_mask, pkv: attention_mask
        return layer, table

    def prefetch_ngram(self, layer, ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        emb = layer.ple.ple_embedding
        captured = {}
        probe = RowTable(self.device)

        def capture(x):
            captured["ids"] = x
            return torch.zeros(*x.shape, NGRAM_DIM, device=x.device)

        probe.forward = capture
        emb.ngram_embedding = probe
        emb(ids, None)
        unique = torch.unique(captured["ids"].reshape(-1)).cpu().numpy()
        src = self.ngram.source_rows(unique)
        tables = {"bf16": src, "q8": q8_rows(src), "q4": q4_rows(src)}
        if self.ngram.mlx is not None:
            tables["mlx4"] = self.ngram.mlx_rows(unique)
        lookup = torch.from_numpy(unique).to(self.device)
        return {k: (lookup, v.to(self.device)) for k, v in tables.items()}

    @torch.no_grad()
    def forward_layer(self, i, hidden, ids, position_embeddings, mask):
        layer, table = self.make_layer(i)
        weights = self.layer_weights(i)
        prefix = f"{PREFIX}layers.{i}."
        ngram = self.prefetch_ngram(layer, ids) if table is not None else None
        if table is not None:
            layer.ple.ple_embedding.ngram_embedding = table
        mb = self.args.micro_batch
        pending = {}
        for name, variant in self.variants.items():
            state = self.trunk_state(prefix, weights, variant)
            missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
            missing = [k for k in missing if ".experts." not in k and "ple_embedding." not in k]
            if missing or unexpected:
                raise RuntimeError(f"layer {i}: missing {missing[:4]} unexpected {unexpected[:4]}")
            if table is not None:
                table.lookup, table.rows = ngram[variant[3]]
            parts = []
            for b0 in range(0, hidden[name].shape[0], mb):
                parts.append(self.pre_moe(layer, hidden[name][b0:b0 + mb], ids[b0:b0 + mb],
                                          tuple(x[b0:b0 + mb] for x in position_embeddings), mask[b0:b0 + mb]))
            x = torch.cat([p[0] for p in parts])
            hyper = torch.cat([p[1] for p in parts])
            inj = torch.cat([p[2] for p in parts])
            flat = x.reshape(-1, x.shape[-1])
            shared = layer.mlp.shared_expert(flat) * torch.sigmoid(layer.mlp.shared_expert_gate(flat))
            _, route_w, route_idx = layer.mlp.gate(flat)
            pending[name] = {"hyper": hyper, "x": flat, "inj": inj, "out": shared,
                             "w": route_w, "idx": route_idx}
            hidden[name] = None
        self.routed_experts(prefix, weights, pending)
        for name, p in pending.items():
            moe = p["out"].reshape(*p["hyper"].shape[:-1], -1)
            hidden[name] = p["hyper"] + (moe.unsqueeze(-2) * p["inj"].unsqueeze(-1)).flatten(-2)
        return hidden

    def pre_moe(self, layer, h, ids, position_embeddings, mask):
        if layer.ple is not None:
            h = h + layer.ple(h, ids, None, conv_mask=None)
        x, hyper, inj = layer.attn_hyper_connection(h)
        if layer.layer_type == "linear_attention":
            x = layer.linear_attn(x, cache_params=None, attention_mask=None)
        else:
            x, _ = layer.self_attn(x, position_embeddings, attention_mask=mask, past_key_values=None)
        h = hyper + (x.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        return layer.mlp_hyper_connection(h)

    def routed_experts(self, prefix, weights, pending):
        gu_src, down_src = weights["mlp.experts.gate_up_proj"], weights["mlp.experts.down_proj"]
        experts = gu_src.shape[0]
        groups: dict[tuple[str, str], list[str]] = {}
        for name, variant in self.variants.items():
            groups.setdefault((variant[1], variant[2]), []).append(name)
        plans = {}
        for name, p in pending.items():
            flat = p["idx"].reshape(-1)
            order = torch.argsort(flat, stable=True)
            counts = torch.bincount(flat, minlength=experts).cpu().tolist()
            plans[name] = (order, counts)
        chunk = self.args.expert_chunk
        for (gu_spec, down_spec), names in groups.items():
            for e0 in range(0, experts, chunk):
                e1 = min(e0 + chunk, experts)
                gu = self.expert_slice(prefix + "mlp.experts.gate_up_proj", gu_src, gu_spec, e0, e1)
                dn = self.expert_slice(prefix + "mlp.experts.down_proj", down_src, down_spec, e0, e1)
                for name in names:
                    p = pending[name]
                    order, counts = plans[name]
                    start = sum(counts[:e0])
                    for e in range(e0, e1):
                        n = counts[e]
                        if n == 0:
                            continue
                        sel = order[start:start + n]
                        start += n
                        tok, slot = sel // p["idx"].shape[1], sel % p["idx"].shape[1]
                        g, u = F.linear(p["x"][tok], gu[e - e0]).chunk(2, dim=-1)
                        y = F.linear(F.silu(g) * u, dn[e - e0]) * p["w"][tok, slot, None]
                        p["out"].index_add_(0, tok, y)

    def expert_slice(self, name, src, spec, e0, e1):
        if spec == "mlx":
            key = self.mlx.key(name)
            if name.endswith("gate_up_proj"):
                base = key.replace("experts.gate_up_proj", "switch_mlp.")
                k = src.shape[-1]
                parts = [mlx_dequant(self.mlx.st.numpy(base + p + ".weight")[e0:e1],
                                     self.mlx.st.float(base + p + ".scales")[e0:e1],
                                     self.mlx.st.float(base + p + ".biases")[e0:e1], k)
                         for p in ("gate_proj", "up_proj")]
                return torch.cat(parts, dim=1).to(self.device)
            stem = key.replace("experts.down_proj", "switch_mlp.down_proj")
            return mlx_dequant(self.mlx.st.numpy(stem + ".weight")[e0:e1],
                               self.mlx.st.float(stem + ".scales")[e0:e1],
                               self.mlx.st.float(stem + ".biases")[e0:e1], src.shape[-1]).to(self.device)
        return fake_quant(src[e0:e1].to(self.device).float(), spec)

    @torch.no_grad()
    def score(self, hidden: dict[str, torch.Tensor], ids: torch.Tensor, corpus: list[dict]) -> dict:
        dev = self.device
        head_src = self.source.float("lm_head.weight")
        teacher_head = head_src.to(dev)
        n_seq, seq = ids.shape
        targets = ids[:, 1:].reshape(-1)
        strata = np.repeat([r["stratum"] for r in corpus], seq - 1)
        teacher = hidden.pop("teacher", None)
        if teacher is None:
            raise ValueError("the variant list must include the 'teacher' baseline")
        teacher = teacher[:, :-1].reshape(-1, teacher.shape[-1])
        results: dict[str, dict] = {}
        heads: dict[str, list[str]] = {}
        for name in hidden:
            heads.setdefault(self.variants[name][0], []).append(name)
            hidden[name] = hidden[name][:, :-1].reshape(-1, teacher.shape[-1])
        teacher_nll = torch.empty(teacher.shape[0])
        for spec, names in heads.items():
            head = self.variant_tensor("lm_head.weight", head_src, self.variants[names[0]]).to(dev)
            per = {n: {"kl": [], "top1": [], "nll": []} for n in names}
            for s in range(0, teacher.shape[0], self.args.score_chunk):
                t_logp = F.log_softmax(teacher[s:s + self.args.score_chunk].to(dev) @ teacher_head.T, dim=-1)
                tgt = targets[s:s + self.args.score_chunk].to(dev)
                teacher_nll[s:s + len(tgt)] = -t_logp.gather(1, tgt[:, None]).squeeze(1).cpu()
                t_top = t_logp.argmax(-1)
                for name in names:
                    h = hidden[name][s:s + self.args.score_chunk].to(dev)
                    v_logp = F.log_softmax(h @ head.T, dim=-1)
                    kl = (t_logp.exp() * (t_logp - v_logp)).sum(-1).clamp_min(0)
                    per[name]["kl"].append(kl.cpu())
                    per[name]["top1"].append((v_logp.argmax(-1) == t_top).cpu())
                    per[name]["nll"].append(-v_logp.gather(1, tgt[:, None]).squeeze(1).cpu())
            for name in names:
                kl = torch.cat(per[name]["kl"]).double().numpy()
                top1 = torch.cat(per[name]["top1"]).numpy()
                nll = torch.cat(per[name]["nll"]).double().numpy()
                results[name] = summarize(kl, top1, nll, teacher_nll.double().numpy(), strata)
            del head
        results["teacher"] = {"kl_mean": 0.0, "top1": 1.0,
                              "nll_teacher": float(teacher_nll.mean()),
                              "per_stratum": {s: {"nll": float(teacher_nll.numpy()[strata == s].mean())}
                                              for s in np.unique(strata)}}
        return results


def summarize(kl, top1, nll, teacher_nll, strata) -> dict:
    def stats(sel):
        return {"kl_mean": float(kl[sel].mean()), "kl_p95": float(np.percentile(kl[sel], 95)),
                "kl_p99": float(np.percentile(kl[sel], 99)), "kl_max": float(kl[sel].max()),
                "top1": float(top1[sel].mean()), "nll": float(nll[sel].mean()),
                "nll_delta": float((nll[sel] - teacher_nll[sel]).mean()), "tokens": int(sel.sum())}

    out = stats(np.ones_like(kl, dtype=bool))
    out["nll_teacher"] = float(teacher_nll.mean())
    out["per_stratum"] = {s: stats(strata == s) for s in np.unique(strata)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True, help="BF16 Qwen3.8-Flash-Next snapshot directory")
    ap.add_argument("--pack", type=Path, required=True, help="incumbent model.qwp (for the size inventory)")
    ap.add_argument("--mlx", type=Path, help="MLX checkpoint directory (enables the 'mlx' variant)")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--per-stratum", type=int, default=4)
    ap.add_argument("--layers", type=int, default=0, help="stop after N layers (smoke tests only)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--expert-chunk", type=int, default=32)
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260517)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-cache", type=Path, help="also save final hidden states (torch.save) before scoring")
    args = ap.parse_args(argv)
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown}")
    if "mlx" in args.variants and args.mlx is None:
        ap.error("the 'mlx' variant needs --mlx")
    args.variants = ["teacher"] + [v for v in args.variants if v != "teacher"]
    report = Harness(args).run()
    args.out.write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Quantization Formats & Quality Tooling

## Supported formats

By default `qwg quantize` writes `HQ4_G32_M5`. It applies an orthonormal Walsh–Hadamard rotation to each 32-weight group, then stores 4-bit codes plus a scale and bias (5 effective bits per weight) in a layout the Metal kernels read directly. The engine also supports non-incumbent rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` and standard-basis affine `AQ{2,3,4,6,8}_G{32,64,128,256}` packs, with row-paired batched verify (`M >= 4`), 4-row fused-FWHT gate/up, and batched `lm_head` kernels active across both incumbent `HQ4_G32` and non-incumbent `HQ4_G32` / `AQ4_G32` containers.

The full matrix — Hadamard-rotated `HQ{2,3,4,6,8}_G{32,64,128,256}`, affine `AQ{2,3,4,6,8}_G{32,64,128,256}`, `HQ4_G32_M5`, and `BF16` — supports mixed-format packs such as:

- `HQ4_G32_M5` dense/shared trunk + `HQ3_G64` (`3.5 bits/weight`) routed MoE experts (`3.59 bits/param` overall on `GLM-5.3-Flash`).
- `HQ8_G32` dense trunk + `HQ4_G64` (`4.5 bits/weight`) routed experts (`4.76 bits/param`, `71.30 GiB` on `Qwen3.8-Flash-Next-hq8g32-hq4g64` — `5.16 GiB` smaller with `2.62×` lower KLD vs BF16 than uniform `HQ4_G32`).

## Native HQ quality investigation

`Scripts/probe_quality.py` replays the marker-enum and nested-array JSON tests
against a user-managed loopback server. It records request settings, output hashes,
finish reasons, and exact shape failures; failed checks exit nonzero. The prompt-only
test does not request `response_format`, and no probe requests logprobs (which would
disable Flash-Next speculation and change the execution being tested).

```bash
python3 Scripts/probe_quality.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen3.8-Flash-Next-hq8g32-hq4g64 --case marker \
  --output /tmp/marker-replay.json

PYTHONPATH=Scripts:Scripts/quantize .venv-oracle/bin/python \
  Scripts/quantize/hq_frontier.py \
  --pack ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --output /tmp/hq-size-frontier.json
```

The HQ tool forecasts trunk/expert bit-width and group-size combinations from
actual tensor headers, preserving BF16 exceptions. Estimates preserve the original
padding/header budget; they are not measured final pack sizes, KL scores, or a
quality recommendation. External PLE tables are unchanged and excluded from
`model.qwp` byte counts. `--source` checks a downloaded safetensors index and
payload extents and reports missing shards. Add `--verify-source-hash` to stream all
indexed shards in the packer's order and compare the aggregate checksum with the
incumbent pack's provenance. This explicitly reports checksum mismatches (exit 2);
the default extent-only audit does not certify checksums.
`--measurements` accepts measured bytes/KL with matching teacher and scoring-corpus
identities to calculate a two-objective Pareto frontier; it never selects a winner.

`Scripts/quantize/score_teacher_logits.py` scores verified full-vocabulary `.npy`
logit captures with matching integer token-prefix matrices (right padding is `-1`).
It reports `KL(teacher || candidate)` in nats/token, mean/p95/max, top-1 agreement,
and optional target-token NLL delta. It uses float64 CPU accumulation and rejects
diverged contexts, non-finite logits, or invalid targets. This is a capture scorer,
not a teacher forward pass: source completeness and oracle correctness must be
verified separately. Use `--help` for capture paths and optional candidate pack-byte
accounting. All three tools refuse to overwrite existing reports.

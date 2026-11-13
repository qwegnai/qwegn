#!/usr/bin/env python3
"""Synthetic golden fixtures for ERNIE-5.5 hybrid primitives (zero model weights).
Generates Scripts/spike/out/ernie55_synthetic_goldens.json for XCTest parity gates.
"""
import json
import math
from pathlib import Path
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fwht_ortho(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    d = y.shape[-1]
    h = 1
    while h < d:
        y = y.reshape(*y.shape[:-1], d // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*x.shape[:-1], d)
        h *= 2
    return (y / math.sqrt(d)).astype(np.float32)


def situ_and_mul(gate: np.ndarray, up: np.ndarray, beta: float = 4.0, linear_beta: float = 25.0) -> np.ndarray:
    g = beta * np.tanh(gate / beta) * sigmoid(gate)
    u = linear_beta * np.tanh(up / linear_beta)
    return (g * u).astype(np.float32)


def multimax(logits: np.ndarray, ranges: np.ndarray, ts: np.ndarray) -> np.ndarray:
    r0, r1, r2, r3 = [float(v) for v in ranges]
    t0, t1, t2, t3 = [float(v) for v in ts]
    x = logits.astype(np.float32)
    return (
        x
        + t0 * np.maximum(r0 - x, 0.0)
        + t1 * np.maximum(x - r1, 0.0)
        + t2 * np.square(np.maximum(r2 - x, 0.0))
        + t3 * np.square(np.maximum(x - r3, 0.0))
    ).astype(np.float32)


def dual_split_router(
    x: np.ndarray, w0: np.ndarray, w1: np.ndarray, bias: np.ndarray, scale: np.ndarray, topk: int
) -> tuple[np.ndarray, np.ndarray]:
    s0 = sigmoid(x @ w0.T)
    s1 = sigmoid(x @ w1.T)
    scores = s0 + s1
    biased = scores + bias[None, :]
    ids = np.argsort(-biased, axis=-1, kind="stable")[:, :topk].astype(np.int32)
    picked = np.take_along_axis(scores, ids, axis=-1)
    normed = picked / np.maximum(picked.sum(axis=-1, keepdims=True), 1e-20)
    weights = (normed * scale[ids]).astype(np.float32)
    return weights, ids


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    out = xf * rstd
    if weight is not None:
        out = out * weight.astype(np.float32)
    return out.astype(np.float32)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True)
    diff = xf - mean
    rstd = 1.0 / np.sqrt(np.mean(diff * diff, axis=-1, keepdims=True) + eps)
    return (diff * rstd * weight.astype(np.float32) + bias.astype(np.float32)).astype(np.float32)


def compute_yarn_inv_freq(
    rotary_dim: int = 64,
    base: float = 160000.0,
    factor: float = 16.0,
    original_max_pos: int = 65536,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> np.ndarray:
    pairs = rotary_dim // 2
    pos_freqs = base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim))
    inv_ex = 1.0 / pos_freqs
    inv_in = 1.0 / (np.float32(factor) * pos_freqs)

    def corr_dim(nr: float) -> float:
        return (rotary_dim * math.log(original_max_pos / (nr * 2.0 * math.pi))) / (2.0 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0.0)
    high = min(math.ceil(corr_dim(beta_slow)), float(rotary_dim - 1))
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(pairs, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0.0, 1.0)
    eef = 1.0 - ramp
    return (inv_in * (1.0 - eef) + inv_ex * eef).astype(np.float32)


def rope_gptj(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray, inverse: bool = False) -> np.ndarray:
    xf = x.astype(np.float32)
    pairs = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 2, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    while angles.ndim < even.ndim:
        angles = angles[:, None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    if inverse:
        re = even * c + odd * s
        ro = odd * c - even * s
    else:
        re = even * c - odd * s
        ro = odd * c + even * s
    return np.stack([re, ro], axis=-1).reshape(xf.shape).astype(np.float32)


def rope_neox(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    half = xf.shape[-1] // 2
    x0 = xf[..., :half]
    x1 = xf[..., half:]
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    while angles.ndim < x0.ndim:
        angles = angles[:, None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], axis=-1).astype(np.float32)


def mhc_pre_additive_eps(
    residual: np.ndarray,
    fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    rms_eps: float = 0.25,
    hc_eps: float = 1e-6,
    post_alpha: float = 2.0,
    iters: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, c, h = residual.shape
    x = residual.reshape(t, c * h).astype(np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True)) + rms_eps
    mixes = (x @ fn.T) / rms
    pre_mix = sigmoid(mixes[:, :c] * hc_scale[0] + hc_base[:c])
    post_mix = sigmoid(mixes[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c]) * post_alpha
    comb_raw = (mixes[:, 2 * c :] * hc_scale[2] + hc_base[2 * c :]).reshape(t, c, c)
    rm = np.max(comb_raw, axis=-1, keepdims=True)
    ex = np.exp(comb_raw - rm)
    comb = ex / np.sum(ex, axis=-1, keepdims=True) + hc_eps
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (np.sum(comb, axis=-1, keepdims=True) + hc_eps)
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + hc_eps)
    layer_in = np.sum((pre_mix + hc_eps)[:, :, None] * residual.astype(np.float32), axis=1)
    normed = rms_norm(layer_in, norm_weight, rms_eps)
    return post_mix.astype(np.float32), comb.astype(np.float32), layer_in.astype(np.float32), normed


def mhc_exit_head_ref(
    residual: np.ndarray,
    head_fn: np.ndarray,
    head_scale: float,
    head_base: np.ndarray,
    norm_weight: np.ndarray,
    rms_eps: float = 0.25,
    hc_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    t, c, h = residual.shape
    x = residual.reshape(t, c * h).astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + rms_eps)
    logits = (x @ head_fn.T) * rstd * head_scale + head_base[None, :]
    pre = sigmoid(logits) + hc_eps
    contracted = np.sum(pre[:, :, None] * residual.astype(np.float32), axis=1)
    normed = rms_norm(contracted, norm_weight, rms_eps)
    return contracted.astype(np.float32), normed


def vha_fold_and_apply(o: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heads = u.shape[0]
    m = np.eye(heads, dtype=np.float32) + (v.astype(np.float32) @ u.astype(np.float32).T)
    mixed = np.einsum("hg,tgd->thd", m, o.astype(np.float32))
    return m, mixed.astype(np.float32)


def make_lowrank_f16(rng: np.random.Generator, rows: int, cols: int, rank: int = 2, scale: float = 0.15):
    a = (rng.standard_normal((rows, rank)) * scale).astype(np.float16)
    b = (rng.standard_normal((rank, cols)) * scale).astype(np.float16)
    w = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return a, b, w


def sim_hq8_g32_rot(w: np.ndarray) -> np.ndarray:
    n, k = w.shape
    out = np.zeros((n, k), dtype=np.float32)
    for r in range(n):
        row = fwht_ortho(w[r].reshape(-1, 32)).reshape(-1)
        for grp in range(0, k, 32):
            blk = row[grp : grp + 32]
            mn, mx = float(blk.min()), float(blk.max())
            sc = float(np.float16(max((mx - mn) / 255.0, 1e-8)))
            bi = float(np.float16(mn))
            q = np.clip(np.round((blk - bi) / sc), 0.0, 255.0)
            row[grp : grp + 32] = q * sc + bi
        out[r] = fwht_ortho(row.reshape(-1, 32)).reshape(-1)
    return out


def sim_mla_q4_kv_rot(kv_raw: np.ndarray) -> np.ndarray:
    t_len, kv_lora = kv_raw.shape
    out = np.zeros((t_len, kv_lora), dtype=np.float32)
    for t in range(t_len):
        row = fwht_ortho(kv_raw[t].reshape(-1, 32)).reshape(-1)
        for grp in range(0, kv_lora, 128):
            blk = row[grp : grp + 128]
            mn_h = float(np.float16(blk.min()))
            mx = float(blk.max())
            sc_h = float(np.float16(max((mx - mn_h) / 15.0, 1e-8)))
            q = np.clip(np.round((blk - mn_h) / sc_h), 0.0, 15.0)
            row[grp : grp + 128] = q * sc_h + mn_h
        out[t] = fwht_ortho(row.reshape(-1, 32)).reshape(-1)
    return out


def eval_mla_dsa_forward(
    qb_eff: np.ndarray,
    kvb_eff: np.ndarray,
    q_c: np.ndarray,
    kv_eff: np.ndarray,
    k_pe: np.ndarray,
    k_idx: np.ndarray,
    idx_weights: np.ndarray,
    vha_m: np.ndarray,
    sink: np.ndarray,
    positions: np.ndarray,
    inv_freq: np.ndarray,
    q_heads: int,
    idx_heads: int,
    idx_dim: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
    kv_lora: int,
    topk: int,
    window: int,
) -> tuple[list[list[int]], np.ndarray]:
    T = q_c.shape[0]
    q_channels = q_heads * (nope_dim + rope_dim + v_dim)
    q_full = (q_c.astype(np.float32) @ qb_eff.astype(np.float32).T).astype(np.float16)
    n_qnope = q_heads * nope_dim
    n_qrope = q_heads * rope_dim
    n_ogate = q_heads * v_dim
    q_nope = q_full[:, :n_qnope].reshape(T, q_heads, nope_dim)
    q_rope = rope_gptj(
        q_full[:, n_qnope : n_qnope + n_qrope].reshape(T, q_heads, rope_dim), positions, inv_freq
    ).astype(np.float16)
    o_gate = q_full[:, n_qnope + n_qrope : n_qnope + n_qrope + n_ogate].reshape(T, q_heads, v_dim)
    q_idx_raw = q_full[:, q_channels:].reshape(T, idx_heads, idx_dim).astype(np.float32)
    q_idx = np.concatenate(
        [rope_neox(q_idx_raw[:, :, :rope_dim], positions, inv_freq), q_idx_raw[:, :, rope_dim:]], axis=-1
    )
    q_idx = fwht_ortho(q_idx).astype(np.float16)

    scale = 1.0 / math.sqrt(nope_dim + rope_dim)
    idx_scale = 1.0 / math.sqrt(idx_dim)
    selected_rows: list[list[int]] = []
    u_lat = np.zeros((T, q_heads, kv_lora), dtype=np.float32)
    for t in range(T):
        seq_len = t + 1
        dense_lim = topk + window
        if seq_len <= dense_lim:
            sel = list(range(seq_len))
        else:
            n_mid = seq_len - window
            dots = np.einsum("hd,md->hm", q_idx[t].astype(np.float32), k_idx[:n_mid].astype(np.float32)) * idx_scale
            sc = np.sum(idx_weights[t, :, None] * np.maximum(dots, 0.0), axis=0)
            top = np.sort(np.argsort(-sc, kind="stable")[:topk]).tolist()
            sel = top + list(range(n_mid, seq_len))
        selected_rows.append(sel)
        for h in range(q_heads):
            w_uk = kvb_eff[h * (nope_dim + v_dim) : h * (nope_dim + v_dim) + nope_dim].astype(np.float32)
            q_abs = q_nope[t, h].astype(np.float32) @ w_uk
            logits = (
                kv_eff[sel].astype(np.float32) @ q_abs
                + k_pe[sel].astype(np.float32) @ q_rope[t, h].astype(np.float32)
            ) * scale
            m = max(float(np.max(logits)), float(sink[h]))
            ex = np.exp(logits - m)
            denom = math.exp(float(sink[h]) - m) + float(np.sum(ex))
            p = ex / denom
            u_lat[t, h] = p @ kv_eff[sel].astype(np.float32)

    y = np.zeros((T, q_heads, v_dim), dtype=np.float32)
    for h in range(q_heads):
        w_uv = kvb_eff[h * (nope_dim + v_dim) + nope_dim : (h + 1) * (nope_dim + v_dim)].astype(np.float32)
        y[:, h, :] = u_lat[:, h, :].astype(np.float16).astype(np.float32) @ w_uv.T
    y_vha = np.einsum("hg,tgd->thd", vha_m.astype(np.float32), y.astype(np.float16).astype(np.float32))
    out = (y_vha * sigmoid(o_gate.astype(np.float32))).reshape(T, q_heads * v_dim).astype(np.float32)
    return selected_rows, out


def build_mla_dsa_fixture(rng: np.random.Generator) -> dict:
    T, hidden, q_heads, idx_heads, idx_dim = 8, 128, 16, 64, 128
    q_lora, kv_lora, nope_dim, rope_dim, v_dim = 64, 512, 192, 64, 256
    topk, window = 4, 2
    rms_eps = 0.25
    rope_theta = 8_000_000.0
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / np.float32(rope_dim)))
    positions = np.arange(T, dtype=np.int32)

    ln1 = (rng.standard_normal((T, hidden)) * 0.4).astype(np.float16)
    lat_rows = q_lora + kv_lora + rope_dim + idx_dim
    lat = (rng.standard_normal((T, lat_rows)) * 0.4).astype(np.float16)
    qa_w = (1.0 + rng.uniform(-0.1, 0.1, size=(q_lora,))).astype(np.float16)
    kva_w = (1.0 + rng.uniform(-0.1, 0.1, size=(kv_lora,))).astype(np.float16)
    kn_w = (1.0 + rng.uniform(-0.1, 0.1, size=(idx_dim,))).astype(np.float16)
    kn_b = (rng.uniform(-0.05, 0.05, size=(idx_dim,))).astype(np.float16)
    wg_w = (rng.standard_normal((idx_heads, hidden)) * 0.12).astype(np.float16)

    q_channels = q_heads * (nope_dim + rope_dim + v_dim)
    qb_rows = q_channels + idx_heads * idx_dim
    qb_a, qb_b, qb_w = make_lowrank_f16(rng, qb_rows, q_lora, rank=2, scale=0.2)
    kvb_a, kvb_b, kvb_w = make_lowrank_f16(rng, q_heads * (nope_dim + v_dim), kv_lora, rank=2, scale=0.18)
    vha_m = (np.eye(q_heads, dtype=np.float32) + rng.standard_normal((q_heads, q_heads)) * 0.05).astype(np.float16)
    sink = rng.uniform(-1.5, 0.2, size=(q_heads,)).astype(np.float32)

    q_c_f32 = rms_norm(lat[:, :q_lora], qa_w, rms_eps)
    kv_c_f32 = rms_norm(lat[:, q_lora : q_lora + kv_lora], kva_w, rms_eps)
    q_c = q_c_f32.astype(np.float16)
    kv_c = kv_c_f32.astype(np.float16)
    k_pe = rope_gptj(lat[:, q_lora + kv_lora : q_lora + kv_lora + rope_dim], positions, inv_freq).astype(np.float16)
    k_idx_ln = layer_norm(lat[:, q_lora + kv_lora + rope_dim :], kn_w, kn_b, 1e-6)
    k_idx = np.concatenate([rope_neox(k_idx_ln[:, :rope_dim], positions, inv_freq), k_idx_ln[:, rope_dim:]], axis=-1)
    k_idx = fwht_ortho(k_idx).astype(np.float16)
    idx_weights = (ln1.astype(np.float32) @ wg_w.astype(np.float32).T) / math.sqrt(idx_heads)

    selected_rows, out = eval_mla_dsa_forward(
        qb_w, kvb_w, q_c, kv_c, k_pe, k_idx, idx_weights, vha_m, sink,
        positions, inv_freq, q_heads, idx_heads, idx_dim, nope_dim, rope_dim, v_dim, kv_lora, topk, window
    )
    qb_hq8 = sim_hq8_g32_rot(qb_w.astype(np.float32))
    kvb_hq8 = sim_hq8_g32_rot(kvb_w.astype(np.float32))
    q_c_rot = fwht_ortho(fwht_ortho(q_c_f32.reshape(T, -1, 32)).astype(np.float16).reshape(T, -1, 32)).reshape(T, q_lora)
    kv_c_rot_f16 = fwht_ortho(fwht_ortho(kv_c_f32.reshape(T, -1, 32)).astype(np.float16).reshape(T, -1, 32)).reshape(T, kv_lora)
    kv_c_rot_q4 = sim_mla_q4_kv_rot(kv_c_f32)
    selected_hq8, out_hq8_f16 = eval_mla_dsa_forward(
        qb_hq8, kvb_hq8, q_c_rot, kv_c_rot_f16, k_pe, k_idx, idx_weights, vha_m, sink,
        positions, inv_freq, q_heads, idx_heads, idx_dim, nope_dim, rope_dim, v_dim, kv_lora, topk, window
    )
    _, out_hq8_q4 = eval_mla_dsa_forward(
        qb_hq8, kvb_hq8, q_c_rot, kv_c_rot_q4, k_pe, k_idx, idx_weights, vha_m, sink,
        positions, inv_freq, q_heads, idx_heads, idx_dim, nope_dim, rope_dim, v_dim, kv_lora, topk, window
    )

    return {
        "T": T,
        "hidden": hidden,
        "q_heads": q_heads,
        "idx_heads": idx_heads,
        "idx_dim": idx_dim,
        "q_lora": q_lora,
        "kv_lora": kv_lora,
        "nope_dim": nope_dim,
        "rope_dim": rope_dim,
        "v_dim": v_dim,
        "topk": topk,
        "window": window,
        "rms_eps": rms_eps,
        "rope_theta": rope_theta,
        "ln1": ln1.astype(np.float32).tolist(),
        "latent": lat.astype(np.float32).tolist(),
        "qa_w": qa_w.astype(np.float32).tolist(),
        "kva_w": kva_w.astype(np.float32).tolist(),
        "kn_w": kn_w.astype(np.float32).tolist(),
        "kn_b": kn_b.astype(np.float32).tolist(),
        "wg_w": wg_w.astype(np.float32).tolist(),
        "qb_a": qb_a.astype(np.float32).tolist(),
        "qb_b": qb_b.astype(np.float32).tolist(),
        "kvb_a": kvb_a.astype(np.float32).tolist(),
        "kvb_b": kvb_b.astype(np.float32).tolist(),
        "vha_m": vha_m.astype(np.float32).tolist(),
        "sink": sink.tolist(),
        "selected": selected_hq8,
        "out": out.tolist(),
        "out_hq8_f16": out_hq8_f16.tolist(),
        "out_hq8_q4": out_hq8_q4.tolist(),
    }


def build_hca_fixture(rng: np.random.Generator) -> dict:
    T, q_heads, o_groups, o_lora = 10, 16, 2, 128
    head_dim, nope_dim, rope_dim, q_lora = 512, 448, 64, 128
    compress_ratio, window = 4, 4
    rms_eps = 0.25
    inv_freq = compute_yarn_inv_freq(rotary_dim=rope_dim, base=160000.0, factor=16.0, original_max_pos=65536)
    positions = np.arange(T, dtype=np.int32)

    fused_in = (rng.standard_normal((T, q_lora + 3 * head_dim)) * 0.35).astype(np.float16)
    q_norm_w = (1.0 + rng.uniform(-0.1, 0.1, size=(q_lora,))).astype(np.float16)
    kv_norm_w = (1.0 + rng.uniform(-0.1, 0.1, size=(head_dim,))).astype(np.float16)
    comp_norm_w = (1.0 + rng.uniform(-0.1, 0.1, size=(head_dim,))).astype(np.float16)
    ape = (rng.standard_normal((compress_ratio, head_dim)) * 0.2).astype(np.float32)

    qup_rows = q_heads * head_dim + o_groups * o_lora
    qup_a, qup_b, qup_w = make_lowrank_f16(rng, qup_rows, q_lora, rank=2, scale=0.2)
    heads_per_group = q_heads // o_groups
    group_in = heads_per_group * head_dim
    woa_a, woa_b, woa_w = make_lowrank_f16(rng, o_groups * o_lora, group_in, rank=2, scale=0.15)
    vha_m = (np.eye(q_heads, dtype=np.float32) + rng.standard_normal((q_heads, q_heads)) * 0.05).astype(np.float16)
    sink = rng.uniform(-1.5, 0.2, size=(q_heads,)).astype(np.float32)

    qr = rms_norm(fused_in[:, :q_lora], q_norm_w, rms_eps).astype(np.float16)
    kv_raw = rms_norm(fused_in[:, q_lora : q_lora + head_dim], kv_norm_w, rms_eps)
    swa_kv = np.concatenate(
        [kv_raw[:, :nope_dim], rope_gptj(kv_raw[:, nope_dim:], positions, inv_freq)], axis=-1
    ).astype(np.float16)

    comp_kv_part = fused_in[:, q_lora + head_dim : q_lora + 2 * head_dim].astype(np.float32)
    comp_score_part = (
        fused_in[:, q_lora + 2 * head_dim :].astype(np.float32) + ape[positions % compress_ratio]
    )

    num_total_comp = T // compress_ratio
    comp_kv = np.zeros((num_total_comp, head_dim), dtype=np.float16)
    for m in range(num_total_comp):
        s_blk = comp_score_part[m * compress_ratio : (m + 1) * compress_ratio]
        k_blk = comp_kv_part[m * compress_ratio : (m + 1) * compress_ratio]
        sm = np.max(s_blk, axis=0, keepdims=True)
        ex = np.exp(s_blk - sm)
        probs = ex / np.sum(ex, axis=0, keepdims=True)
        c_sum = np.sum(probs * k_blk, axis=0, keepdims=True)
        c_norm = rms_norm(c_sum, comp_norm_w, rms_eps)
        c_pos = np.array([m * compress_ratio], dtype=np.int32)
        c_rot = np.concatenate(
            [c_norm[:, :nope_dim], rope_gptj(c_norm[:, nope_dim:], c_pos, inv_freq)], axis=-1
        )
        comp_kv[m] = c_rot[0].astype(np.float16)

    qup_out = (qr.astype(np.float32) @ qup_w.astype(np.float32).T).astype(np.float16)
    q_raw = qup_out[:, : q_heads * head_dim].reshape(T, q_heads, head_dim)
    o_gate = qup_out[:, q_heads * head_dim :]
    q_normed = rms_norm(q_raw, None, rms_eps)
    q = np.concatenate(
        [q_normed[:, :, :nope_dim], rope_gptj(q_normed[:, :, nope_dim:], positions, inv_freq)], axis=-1
    ).astype(np.float16)

    scale = 1.0 / math.sqrt(head_dim)
    att_o = np.zeros((T, q_heads, head_dim), dtype=np.float32)
    for t in range(T):
        n_comp = (t + 1) // compress_ratio
        swa_start = max(0, t - (window - 1))
        keys_list = []
        if n_comp > 0:
            keys_list.append(comp_kv[:n_comp].astype(np.float32))
        keys_list.append(swa_kv[swa_start : t + 1].astype(np.float32))
        keys = np.concatenate(keys_list, axis=0)
        for h in range(q_heads):
            logits = (keys @ q[t, h].astype(np.float32)) * scale
            m = max(float(np.max(logits)), float(sink[h]))
            ex = np.exp(logits - m)
            denom = math.exp(float(sink[h]) - m) + float(np.sum(ex))
            p = ex / denom
            att_o[t, h] = p @ keys

    o_vha = np.einsum("hg,tgd->thd", vha_m.astype(np.float32), att_o.astype(np.float16).astype(np.float32))
    o_inv = np.concatenate(
        [o_vha[:, :, :nope_dim], rope_gptj(o_vha[:, :, nope_dim:], positions, inv_freq, inverse=True)],
        axis=-1,
    ).astype(np.float16)
    o_grp = o_inv.reshape(T, o_groups, group_in).astype(np.float32)
    z = np.zeros((T, o_groups * o_lora), dtype=np.float32)
    for g in range(o_groups):
        wg = woa_w[g * o_lora : (g + 1) * o_lora].astype(np.float32)
        z[:, g * o_lora : (g + 1) * o_lora] = o_grp[:, g, :] @ wg.T
    z_gated = (z * sigmoid(o_gate.astype(np.float32))).astype(np.float32)

    return {
        "T": T,
        "q_heads": q_heads,
        "o_groups": o_groups,
        "o_lora": o_lora,
        "head_dim": head_dim,
        "nope_dim": nope_dim,
        "rope_dim": rope_dim,
        "q_lora": q_lora,
        "compress_ratio": compress_ratio,
        "window": window,
        "rms_eps": rms_eps,
        "fused_in": fused_in.astype(np.float32).tolist(),
        "q_norm_w": q_norm_w.astype(np.float32).tolist(),
        "kv_norm_w": kv_norm_w.astype(np.float32).tolist(),
        "comp_norm_w": comp_norm_w.astype(np.float32).tolist(),
        "ape": ape.tolist(),
        "qup_a": qup_a.astype(np.float32).tolist(),
        "qup_b": qup_b.astype(np.float32).tolist(),
        "woa_a": woa_a.astype(np.float32).tolist(),
        "woa_b": woa_b.astype(np.float32).tolist(),
        "vha_m": vha_m.astype(np.float32).tolist(),
        "sink": sink.tolist(),
        "comp_kv": comp_kv.astype(np.float32).tolist(),
        "out": z_gated.tolist(),
    }


def main() -> int:
    rng = np.random.default_rng(20261112)
    out_dir = Path("Scripts/spike/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    gate = rng.uniform(-12.0, 12.0, size=(4, 32)).astype(np.float32)
    up = rng.uniform(-40.0, 40.0, size=(4, 32)).astype(np.float32)
    situ_out = situ_and_mul(gate, up, beta=4.0, linear_beta=25.0)

    logits = np.array([-8.0, -2.5, 0.0, 1.75, 4.5, 9.25], dtype=np.float32)
    mm_ranges = np.array([-3.0, 2.0, -5.0, 4.0], dtype=np.float32)
    mm_ts = np.array([0.25, 0.5, 0.05, 0.1], dtype=np.float32)
    mm_out = multimax(logits, mm_ranges, mm_ts)

    rx = rng.standard_normal((3, 64)).astype(np.float32) * 0.5
    rw0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.25
    rw1 = rng.standard_normal((32, 64)).astype(np.float32) * 0.25
    rbias = rng.uniform(-0.1, 0.1, size=(32,)).astype(np.float32)
    rscale = rng.uniform(2.0, 4.5, size=(32,)).astype(np.float32)
    r_weights, r_ids = dual_split_router(rx, rw0, rw1, rbias, rscale, topk=4)

    res = rng.standard_normal((2, 4, 64)).astype(np.float32) * 0.2
    fn = rng.standard_normal((24, 256)).astype(np.float32) * 0.05
    hc_scale = np.array([0.8, 1.1, 0.9], dtype=np.float32)
    hc_base = rng.standard_normal((24,)).astype(np.float32) * 0.1
    norm_w = (1.0 + rng.uniform(-0.1, 0.1, size=(64,))).astype(np.float16).astype(np.float32)
    post_mix, comb, layer_in, layer_normed = mhc_pre_additive_eps(
        res, fn, hc_scale, hc_base, norm_w, rms_eps=0.25, hc_eps=1e-6, post_alpha=2.0, iters=20
    )

    head_fn = rng.standard_normal((4, 256)).astype(np.float32) * 0.05
    head_scale = 1.15
    head_base = rng.standard_normal((4,)).astype(np.float32) * 0.1
    exit_raw, exit_normed = mhc_exit_head_ref(
        res, head_fn, head_scale, head_base, norm_w, rms_eps=0.25, hc_eps=1e-6
    )

    vo = rng.standard_normal((2, 8, 16)).astype(np.float32) * 0.3
    vu = rng.standard_normal((8, 4)).astype(np.float32) * 0.1
    vv = rng.standard_normal((8, 4)).astype(np.float32) * 0.1
    vm, vmixed = vha_fold_and_apply(vo, vu, vv)

    yarn_inv_freq = compute_yarn_inv_freq()
    yarn_x = rng.standard_normal((2, 64)).astype(np.float32)
    yarn_pos = np.array([300_000, 999_999], dtype=np.int32)
    yarn_fwd = rope_gptj(yarn_x, yarn_pos, yarn_inv_freq, inverse=False)
    yarn_roundtrip = rope_gptj(yarn_fwd, yarn_pos, yarn_inv_freq, inverse=True)

    mla_dsa = build_mla_dsa_fixture(rng)
    hca = build_hca_fixture(rng)

    payload = {
        "situ": {"gate": gate.tolist(), "up": up.tolist(), "out": situ_out.tolist()},
        "multimax": {
            "logits": logits.tolist(),
            "ranges": mm_ranges.tolist(),
            "ts": mm_ts.tolist(),
            "out": mm_out.tolist(),
        },
        "router": {
            "x": rx.tolist(),
            "w0": rw0.tolist(),
            "w1": rw1.tolist(),
            "bias": rbias.tolist(),
            "scale": rscale.tolist(),
            "weights": r_weights.tolist(),
            "ids": r_ids.tolist(),
        },
        "mhc": {
            "residual": res.tolist(),
            "fn": fn.tolist(),
            "scale": hc_scale.tolist(),
            "base": hc_base.tolist(),
            "norm_weight": norm_w.tolist(),
            "rms_eps": 0.25,
            "hc_eps": 1e-6,
            "post_mix": post_mix.tolist(),
            "comb": comb.tolist(),
            "layer_in": layer_in.tolist(),
            "layer_normed": layer_normed.tolist(),
        },
        "mhc_exit_head": {
            "head_fn": head_fn.tolist(),
            "head_scale": head_scale,
            "head_base": head_base.tolist(),
            "contracted": exit_raw.tolist(),
            "normed": exit_normed.tolist(),
        },
        "vha": {"o": vo.tolist(), "u": vu.tolist(), "v": vv.tolist(), "m": vm.tolist(), "out": vmixed.tolist()},
        "yarn_rope": {
            "inv_freq": yarn_inv_freq.tolist(),
            "x": yarn_x.tolist(),
            "positions": yarn_pos.tolist(),
            "fwd": yarn_fwd.tolist(),
            "roundtrip": yarn_roundtrip.tolist(),
        },
        "mla_dsa": mla_dsa,
        "hca": hca,
    }
    (out_dir / "ernie55_synthetic_goldens.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

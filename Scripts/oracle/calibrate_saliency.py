#!/usr/bin/env python3
"""SwiGLU Saliency Calibration Script (Phase 1 Math Acceleration).

Profiles intermediate SwiGLU activation magnitudes across layers or token
sequences to determine the static or semi-static top-K salient groups of 128
channels for each layer (or globally).

Outputs:
  - mlp_active_groups.bin: uint32 array of active group indices [0..135]
"""
import argparse
import pathlib
import numpy as np

def calibrate_saliency(activations_h: np.ndarray, keep_ratio: float = 0.65) -> np.ndarray:
    """Computes group-wise L1 energy and selects the top keep_ratio groups.
    
    activations_h: shape [N_tokens, 17408]
    keep_ratio: fraction of groups to retain (e.g. 0.65 => 88 of 136 groups)
    returns: sorted uint32 array of active group indices
    """
    assert activations_h.ndim == 2 and activations_h.shape[1] == 17408
    n_tokens, dim = activations_h.shape
    num_groups = dim // 128  # 136

    # Reshape to [N_tokens, 136, 128]
    grouped = np.abs(activations_h.reshape(n_tokens, num_groups, 128))
    # Aggregate L1 energy across tokens and channels within group
    group_energies = np.sum(grouped, axis=(0, 2))

    k = int(np.round(num_groups * keep_ratio))
    top_indices = np.argsort(group_energies)[::-1][:k]
    sorted_top = np.sort(top_indices).astype(np.uint32)
    return sorted_top

def main():
    parser = argparse.ArgumentParser(description="Calibrate SwiGLU group saliency")
    parser.add_argument("--ratio", type=float, default=0.65, help="Ratio of active groups to keep (default 0.65)")
    parser.add_argument("--out", type=str, default="mlp_active_groups.bin", help="Output path for active groups binary")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        out_path = repo_root / "Scripts" / "spike" / "out" / args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If mlp_h_M8.bin exists, use it as sample activations
    spike_out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
    sample_file = spike_out / "mlp_h_M8.bin"

    if sample_file.exists():
        data = np.fromfile(sample_file, dtype=np.float16).astype(np.float32)
        h = data.reshape(-1, 17408)
    else:
        rng = np.random.default_rng(42)
        h = rng.standard_normal((16, 17408), dtype=np.float32)

    active_groups = calibrate_saliency(h, keep_ratio=args.ratio)
    active_groups.tofile(out_path)
    print(f"Calibrated {len(active_groups)}/{17408//128} active groups ({args.ratio*100:.1f}% density) -> {out_path}")

if __name__ == "__main__":
    main()

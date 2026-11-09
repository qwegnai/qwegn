#!/usr/bin/env python3
"""Score full-vocabulary logit captures on identical teacher-forced token prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def log_softmax(row):
    values = np.asarray(row, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("logits require at least two finite vocabulary entries")
    try:
        with np.errstate(over="raise", invalid="raise"):
            shifted = values - np.max(values)
    except FloatingPointError as error:
        raise ValueError("logit range exceeds float64 log-softmax capacity") from error
    return shifted - np.log(np.exp(shifted).sum())


def score(teacher, candidate, teacher_contexts, candidate_contexts, targets=None):
    if (teacher.ndim != 2 or teacher.shape != candidate.shape or teacher.shape[0] == 0
            or teacher.shape[1] < 2 or not np.issubdtype(teacher.dtype, np.floating)
            or not np.issubdtype(candidate.dtype, np.floating)):
        raise ValueError("teacher/candidate logits must have identical nonempty [positions, vocabulary] shape")
    if (teacher_contexts.ndim != 2 or teacher_contexts.shape[0] != teacher.shape[0]
            or teacher_contexts.shape[1] == 0 or teacher_contexts.shape != candidate_contexts.shape
            or not np.issubdtype(teacher_contexts.dtype, np.integer)
            or not np.issubdtype(candidate_contexts.dtype, np.integer)
            or not np.array_equal(teacher_contexts, candidate_contexts)):
        raise ValueError("teacher and candidate must use identical integer token prefixes at every position")
    if (np.any(teacher_contexts < -1) or np.any(teacher_contexts >= teacher.shape[1])
            or np.any(np.all(teacher_contexts == -1, axis=1))):
        raise ValueError("prefix token IDs must be in vocabulary; -1 denotes right padding only")
    for row in teacher_contexts:
        padding = np.flatnonzero(row == -1)
        if padding.size and np.any(row[padding[0]:] != -1):
            raise ValueError("prefix padding must occur only at the right edge")
    if targets is not None:
        if (targets.shape != (teacher.shape[0],)
                or not np.issubdtype(targets.dtype, np.integer)
                or np.any(targets < 0) or np.any(targets >= teacher.shape[1])):
            raise ValueError("targets must contain one in-vocabulary integer token per scored position")
    divergences = []
    agreements = 0
    nll_deltas = []
    for index, (reference, student) in enumerate(zip(teacher, candidate)):
        log_p = log_softmax(reference)
        log_q = log_softmax(student)
        kl = float(np.sum(np.exp(log_p) * (log_p - log_q)))
        if not np.isfinite(kl) or kl < -1e-10:
            raise ValueError(f"invalid KL accumulation at position {index}")
        divergences.append(max(0.0, kl))
        agreements += int(np.argmax(reference) == np.argmax(student))
        if targets is not None:
            target = targets[index]
            nll_deltas.append(float(log_p[target] - log_q[target]))
    result = {
        "positions": len(divergences), "vocabulary": teacher.shape[1],
        "mean_kl": float(np.mean(divergences)),
        "p95_kl": float(np.percentile(divergences, 95)),
        "max_kl": float(np.max(divergences)),
        "top1_agreement": agreements / len(divergences),
        "kl_by_position": divergences,
        "precision": "float64 CPU log-softmax and accumulation",
        "direction": "KL(teacher || candidate), full vocabulary, nats/token",
    }
    if nll_deltas:
        result["mean_nll_delta"] = float(np.mean(nll_deltas))
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-logits", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--teacher-contexts", type=Path, required=True)
    parser.add_argument("--candidate-contexts", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True, help="verified source revision or teacher artifact identity")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-pack", type=Path,
                        help="optional validated model.qwp for measured frontier byte accounting")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.teacher_id.strip() or not args.candidate_id.strip():
        parser.error("teacher and candidate identities must be nonempty")
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        def load(path):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(array, np.ndarray):
                array.close()
                raise ValueError("captures must be .npy arrays, not archives")
            return array
        teacher = load(args.teacher_logits)
        candidate = load(args.candidate_logits)
        teacher_contexts = load(args.teacher_contexts)
        candidate_contexts = load(args.candidate_contexts)
        targets = load(args.targets) if args.targets else None
        result = score(teacher, candidate, teacher_contexts, candidate_contexts, targets)
        teacher_digest = file_digest(args.teacher_logits)
        contexts_digest = file_digest(args.teacher_contexts)
        result.update({
            "version": 1, "teacher_id": args.teacher_id, "candidate": args.candidate_id,
            "teacher_logits_sha256": teacher_digest,
            "teacher_sha256": hashlib.sha256(
                (args.teacher_id + "\0" + teacher_digest).encode("utf-8")).hexdigest(),
            "candidate_logits_sha256": file_digest(args.candidate_logits),
            "contexts_sha256": contexts_digest,
            "corpus_sha256": contexts_digest,
            "capture_caveat": "This scorer validates alignment and distributions, not the forward-pass "
                              "implementation or source completeness. Supply verified oracle captures.",
        })
        if args.targets:
            result["targets_sha256"] = file_digest(args.targets)
        if args.candidate_pack:
            from verify import read_header
            read_header(args.candidate_pack)
            result["pack_bytes"] = args.candidate_pack.stat().st_size
        encoded = json.dumps(result, indent=2, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
        print(f"KL={result['mean_kl']:.6g} nats/token, top-1={result['top1_agreement']:.2%}, "
              f"positions={result['positions']} -> {args.output}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"score_teacher_logits: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

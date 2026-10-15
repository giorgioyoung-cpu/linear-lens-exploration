"""Fit a LINEAR lens (no intercept) using cached residuals; do not load Qwen.

Equal weight per passage, uniform over matching positions 0 <= j = i < n within a passage.
Run with no arguments to use Giorgio's completed layer-12 collection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

DEFAULT_RUN = Path.home() / "jspace-data/activations/layer12_20261014T191345_397425Z"


def sequence_moments(U, Y):
    """Return S=E[UU^T], C=E[YU^T], q=E[||Y||^2] for one passage.

    Rows are token positions; each matching pair has weight 1/n.
    """
    U = np.asarray(U, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if U.ndim != 2 or Y.shape != U.shape or U.shape[0] == 0:
        raise ValueError("Expected matching nonempty [n,d] residual arrays.")
    if not np.isfinite(U).all() or not np.isfinite(Y).all():
        raise ValueError("Residuals contain non-finite values.")
    n = U.shape[0]
    S = U.T @ U / n
    C = Y.T @ U / n
    q = float(np.sum(Y * Y) / n)
    return S, C, q


def fit_linear(S, C, rtol=1e-10):
    """Compute A=C S^dagger with an explicit relative eigenvalue cutoff.

    This is a numerical truncated pseudoinverse, not ridge regression.
    The cutoff is on S's eigenvalues (squared design singular values).
    """
    if not 0 <= rtol < 1:
        raise ValueError("rtol must be in [0,1).")
    S = (S + S.T) / 2
    eigenvalues, V = np.linalg.eigh(S)
    largest = max(float(eigenvalues[-1]), 0.0)
    if eigenvalues[0] < -1e-10 * max(largest, np.finfo(float).tiny):
        raise ValueError("Second-moment matrix is not positive semidefinite.")
    threshold = rtol * largest
    keep = eigenvalues > threshold
    if np.any(keep):
        Vk = V[:, keep]
        A = ((C @ Vk) / eigenvalues[keep]) @ Vk.T
    else:
        A = np.zeros_like(C)
    info = {
        "retained_rank": int(keep.sum()), "dimension": int(S.shape[0]),
        "relative_cutoff": rtol, "absolute_cutoff": threshold,
        "largest_eigenvalue": largest,
        "smallest_retained_eigenvalue": float(eigenvalues[keep][0]) if keep.any() else None,
    }
    if not np.isfinite(A).all():
        raise ValueError("Fit produced non-finite coefficients.")
    return A, eigenvalues, info


def prediction_error(A, S, C, q):
    """Equal-passage mean squared EUCLIDEAN error, not per-coordinate MSE."""
    cross = float(np.sum(A * C))
    pred_energy = float(np.sum((A @ S) * A))
    error = float(q - 2 * cross + pred_energy)
    scale = max(abs(q), 2 * abs(cross), abs(pred_energy), np.finfo(float).tiny)
    if error < -1e-9 * scale:
        raise ValueError("Negative squared error beyond numerical roundoff.")
    return max(error, 0.0)


def collect_moments(run, split, metadata):
    """Read only the requested split, one saved sequence at a time."""
    import torch
    files = sorted((run / split).glob("sequence_*.pt"))
    expected_count = metadata["splits"][split]["sequences"]
    if len(files) != expected_count or not files:
        raise ValueError(f"{split}: found {len(files)} files, expected {expected_count}.")
    d = metadata["hidden_size"]
    S, C, q = np.zeros((d, d)), np.zeros((d, d)), 0.0
    target_mean = np.zeros(d)
    tokens = 0
    file_hashes = {}
    for k, path in enumerate(files):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if (record["split"] != split or record["layer"] != metadata["layer"]
                or record["final_layer"] != metadata["final_layer"]):
            raise ValueError(f"Layer or split mismatch in {path}.")
        U = record["h_mid"].to(torch.float64).numpy()
        Y = record["h_final"].to(torch.float64).numpy()
        if U.ndim != 2 or U.shape[1] != d:
            raise ValueError(f"Unexpected dimensions in {path}.")
        target_mean += Y.mean(axis=0)
        n = len(U)
        if len(record["input_ids"]) != n or not np.array_equal(
                record["positions"].numpy(), np.arange(n)):
            raise ValueError(f"Token/position mismatch in {path}.")
        Sk, Ck, qk = sequence_moments(U, Y)
        S += Sk
        C += Ck
        q += qk
        tokens += n
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if (k + 1) % 20 == 0 or k + 1 == len(files):
            print(f"{split}: {k + 1}/{len(files)} sequences", flush=True)
    if tokens != metadata["splits"][split]["tokens"]:
        raise ValueError(f"{split}: token count disagrees with collection metadata.")
    count = len(files)
    return (S / count, C / count, q / count), {
        "sequences": count, "tokens": tokens, "file_sha256": file_hashes,
        "target_mean": (target_mean / count).tolist(),
    }


def score(A, moments):
    S, C, q = moments
    mse = prediction_error(A, S, C, q)
    identity_mse = prediction_error(np.eye(A.shape[0]), S, C, q)
    return {
        "mean_squared_euclidean_error": mse,
        "mean_squared_error_per_coordinate": mse / A.shape[0],
        "relative_error_vs_zero": mse / q if q > 0 else None,
        "zero_predictor_error": q,
        "identity_predictor_error": identity_mse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--pinv-rtol", type=float, default=1e-10)
    args = parser.parse_args()
    if not 0 <= args.pinv_rtol < 1:
        parser.error("--pinv-rtol must be in [0,1).")
    run = args.run_dir.expanduser().resolve()
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete" or metadata.get("format_version") != 1:
        raise ValueError("Expected a complete format-version-1 activation collection.")
    started = perf_counter()
    print("Fitting a linear map: no intercept, no centering, no ridge.", flush=True)
    print("Uniform passages; uniform pairs 0 <= j = i < n per passage.", flush=True)
    train, train_info = collect_moments(run, "train", metadata)
    A, spectrum, solver = fit_linear(train[0], train[1], args.pinv_rtol)
    print(f"Retained rank: {solver['retained_rank']}/{solver['dimension']}", flush=True)
    validation, val_info = collect_moments(run, "validation", metadata)
    results = {"train": score(A, train), "validation": score(A, validation)}
    mean = np.asarray(train_info["target_mean"])
    for name, info in (("train", train_info), ("validation", val_info)):
        metrics = results[name]
        mean_error = (metrics["zero_predictor_error"]
                      - 2 * mean @ np.asarray(info["target_mean"]) + mean @ mean)
        metrics["training_mean_predictor_error"] = float(mean_error)
        reduction = 1 - metrics["mean_squared_euclidean_error"] / mean_error if mean_error > 0 else None
        metrics["reduction_vs_training_mean"] = reduction
        reduction_text = f"{100*reduction:.2f}%" if reduction is not None else "undefined"
        print(f"{name}: training-mean error = {mean_error:.6f}; reduction = {reduction_text}", flush=True)
    for name, metrics in results.items():
        relative = metrics["relative_error_vs_zero"]
        rel_text = f"{relative:.6f}" if relative is not None else "undefined"
        print(f"{name}: squared error = {metrics['mean_squared_euclidean_error']:.6f}; "
              f"relative to zero = {rel_text}; "
              f"identity error = {metrics['identity_predictor_error']:.6f}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = run / f"same_position_fit_{stamp}"
    out.mkdir(exist_ok=False)
    mean_dir = out / f"mean_comparison_{stamp}"
    mean_dir.mkdir()
    np.save(mean_dir / "training_target_mean.npy", mean)
    np.savez(out / "linear_lens.npz", A=A, S_train=train[0], C_train=train[1],
             q_train=np.array(train[2]), eigenvalues=spectrum)
    report = {
        "format_version": 1, "created_utc": stamp, "activation_run": str(run),
        "collection_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "model_name": metadata["model_name"], "model_revision": metadata.get("model_revision"),
        "source_layer": metadata["layer"], "target_layer": metadata["final_layer"],
        "target": "final residual before normalization", "intercept": False,
        "centered": False, "ridge": False, "pair_set": "0 <= j = i < n",
        "weighting": "uniform passages, then uniform pairs within each passage",
        "orientation": "column vectors: prediction=A@u; row batches: prediction=U@A.T",
        "computation_dtype": "float64", "device": "cpu", "solver": solver,
        "metrics": results, "data": {"train": train_info, "validation": val_info},
        "test_evaluated": False, "elapsed_seconds": perf_counter() - started,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "torch")},
    }
    (out / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved fit to: {out}", flush=True)
    print("Test activations were not read or evaluated.", flush=True)


if __name__ == "__main__":
    main()

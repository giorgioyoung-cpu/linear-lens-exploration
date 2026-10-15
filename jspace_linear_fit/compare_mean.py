"""Compare the saved linear lens with the constant TRAINING target mean.

Run: python compare_mean.py
Uses the latest saved fit in the completed layer-12 activation collection.
Override with --fit-dir /path/to/linear_fit_... if desired.
No Qwen loading, refitting, or test-set evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

DEFAULT_RUN = Path.home() / "jspace-data/activations/layer12_20261014T191345_397425Z"


def sequence_target_moments(Y):
    """Uniform pairs j<=i give target row i weight (i+1)/[n(n+1)/2]."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or len(Y) == 0 or not np.isfinite(Y).all():
        raise ValueError("Expected finite, nonempty [n,d] target residuals.")
    n = len(Y)
    weights = np.arange(1, n + 1, dtype=np.float64) / (n * (n + 1) / 2)
    return weights @ Y, float(weights @ np.sum(Y * Y, axis=1))


def constant_error(c, mean_Y, q):
    value = float(q - 2 * np.dot(c, mean_Y) + np.dot(c, c))
    if value < -1e-9 * max(q, float(np.dot(c, c)), np.finfo(float).tiny):
        raise ValueError("Unexpected negative squared error.")
    return max(value, 0.0)


def load_target_moments(run, split, report):
    import torch
    hashes = report["data"][split]["file_sha256"]
    if len(hashes) != report["data"][split]["sequences"] or not hashes:
        raise ValueError("Invalid saved file inventory.")
    observed = {p.name for p in (run / split).glob("sequence_*.pt")}
    if observed != set(hashes):
        raise ValueError(f"{split}: files differ from those used in the fit.")
    total_mean, total_q = None, 0.0
    for name, expected_hash in sorted(hashes.items()):
        if Path(name).name != name:
            raise ValueError("Invalid sequence filename in report.")
        data = (run / split / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"{split}/{name} changed since fitting.")
        record = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        Y = record["h_final"].to(torch.float64).numpy()
        mean, q = sequence_target_moments(Y)
        total_mean = mean.copy() if total_mean is None else total_mean + mean
        total_q += q
    N = len(hashes)
    mean, q = total_mean / N, total_q / N
    np.testing.assert_allclose(q, report["metrics"][split]["zero_predictor_error"],
                               rtol=1e-10, atol=1e-8)
    return mean, q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.fit_dir is None:
        candidates = sorted(p.parent for p in DEFAULT_RUN.glob("linear_fit_*/fit_report.json"))
        if not candidates:
            raise FileNotFoundError("No saved fit found. Run fit_affine.py first.")
        fit_dir = candidates[-1]
    else:
        fit_dir = args.fit_dir.expanduser().resolve()
    print(f"Using saved fit: {fit_dir}", flush=True)
    report = json.loads((fit_dir / "fit_report.json").read_text())
    if (report.get("format_version") != 1
            or report.get("pair_set") != "0 <= j <= i < n"
            or report.get("weighting") != "uniform passages, then uniform pairs within each passage"
            or report.get("intercept") is not False or report.get("centered") is not False):
        raise ValueError("This baseline script requires our uncentered, uniform-passage linear fit.")
    run = Path(report["activation_run"])
    meta_bytes = (run / "metadata.json").read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != report["collection_metadata_sha256"]:
        raise ValueError("Collection metadata changed since fitting.")
    print("Reading training targets...", flush=True)
    mean_train, q_train = load_target_moments(run, "train", report)
    print("Reading validation targets...", flush=True)
    mean_val, q_val = load_target_moments(run, "validation", report)
    scores = {}
    for split, mean, q in (("train", mean_train, q_train), ("validation", mean_val, q_val)):
        baseline = constant_error(mean_train, mean, q)
        fitted = report["metrics"][split]["mean_squared_euclidean_error"]
        reduction = 1 - fitted / baseline if baseline > 0 else None
        scores[split] = {"training_mean_predictor_error": baseline,
                         "linear_lens_error": fitted,
                         "fractional_reduction_vs_training_mean": reduction}
        gain = f"{100 * reduction:.2f}%" if reduction is not None else "undefined"
        print(f"{split}: mean predictor error = {baseline:.6f}; "
              f"linear lens error = {fitted:.6f}; reduction = {gain}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = fit_dir / f"mean_comparison_{stamp}"
    out.mkdir(exist_ok=False)
    np.save(out / "training_target_mean.npy", mean_train, allow_pickle=False)
    result = {"predictor": "constant target mean estimated on training only",
              "weighting": report["weighting"], "pair_set": report["pair_set"],
              "source_fit": str(fit_dir), "test_evaluated": False, "metrics": scores}
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved comparison to: {out}", flush=True)
    print("Positive reduction means the linear lens beats the training-mean baseline.")
    print("The test set was not read.")


if __name__ == "__main__":
    main()

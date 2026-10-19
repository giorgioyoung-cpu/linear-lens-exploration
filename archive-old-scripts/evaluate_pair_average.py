"""Read lenses at held-out WikiText validation positions using cached residuals.

Keep beside model_utils.py and evaluate.py. Run: python evaluate_pair_average.py
Checks midpoint and final position of each validation sequence. Prints the
first three sequences and saves all results. No test-set access or refitting.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from evaluate import rank_tokens, DEFAULT_RUN

def resolve_fit(explicit):
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = sorted(DEFAULT_RUN.glob('pair_average_fit_*/fit_report.json'))
    if not candidates:
        raise FileNotFoundError('Run fit_pair_average.py first.')
    return candidates[-1].parent


def positions_to_inspect(n):
    if n < 1:
        raise ValueError("Empty sequence.")
    return sorted({n // 2, n - 1})


def main():
    import torch
    from model_utils import MODEL_NAME, load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--print-sequences", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1 or args.print_sequences < 0:
        parser.error("Require positive top-k and nonnegative print-sequences.")
    fit_dir = resolve_fit(args.fit_dir)
    report = json.loads((fit_dir / "fit_report.json").read_text())
    if (report["model_name"] != MODEL_NAME or report["intercept"]
            or report["centered"]):
        raise ValueError("Expected the uncentered linear Qwen fit.")
    run = Path(report["activation_run"])
    meta_bytes = (run / "metadata.json").read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != report["collection_metadata_sha256"]:
        raise ValueError("Collection metadata changed since fitting.")
    with np.load(fit_dir / "linear_lens.npz", allow_pickle=False) as saved:
        A = saved["A"].astype(np.float64)
        pooled_A = saved['pooled_A'].astype(np.float64)
        same_A = saved['same_position_A'].astype(np.float64)
        same_mean = saved['same_training_mean'].astype(np.float64)
    comparisons = sorted(fit_dir.glob("mean_comparison_*/training_target_mean.npy"))
    if not comparisons:
        raise FileNotFoundError("Run fit_pair_average.py first.")
    mean = np.load(comparisons[-1], allow_pickle=False)
    print(f"Using fit: {fit_dir}", flush=True)
    print("Loading cached model for normalization and unembedding only...", flush=True)
    model, tokenizer, device = load_model()
    d = model.config.hidden_size
    revision = getattr(model.config, "_commit_hash", None)
    if report.get("model_revision") and revision != report["model_revision"]:
        raise ValueError("Model revision differs from fitting.")
    if (A.shape != (d, d) or mean.shape != (d,)
            or report["target_layer"] != model.config.num_hidden_layers
            or not np.isfinite(A).all() or not np.isfinite(mean).all()):
        raise ValueError("Invalid model dimensions or saved arrays.")

    def decode(residual):
        h = torch.as_tensor(residual, dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = model.lm_head(model.model.norm(h))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits.")
        return rank_tokens(logits, tokenizer, min(args.top_k, logits.shape[-1]))

    constant_tokens = decode(mean)
    hashes = report["data"]["validation"]["file_sha256"]
    if not hashes or {p.name for p in (run / "validation").glob("sequence_*.pt")} != set(hashes):
        raise ValueError("Validation files differ from those used in fitting.")
    examples = []
    for seq_index, (filename, expected_hash) in enumerate(sorted(hashes.items())):
        if Path(filename).name != filename:
            raise ValueError("Invalid sequence filename.")
        raw = (run / "validation" / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Validation file changed: {filename}")
        record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        ids = record["input_ids"].tolist()
        U = record["h_mid"].to(torch.float64).numpy()
        Y = record["h_final"].to(torch.float64).numpy()
        if (U.shape != (len(ids), d) or Y.shape != U.shape
                or record["layer"] != report["source_layer"]
                or record["split"] != "validation"):
            raise ValueError(f"Invalid sequence: {filename}")
        for position in positions_to_inspect(len(ids)):
            context = tokenizer.decode(ids[max(0, position - 23):position + 1],
                                       skip_special_tokens=False)
            methods = {
                "pair_average_lens": decode(A @ U[position]),
                "matched_pooled_lens": decode(pooled_A @ U[position]),
                "matched_same_position_lens": decode(same_A @ U[position]),
                "same_position_training_mean": decode(same_mean),
                "ordinary_logit_lens": decode(U[position]),
                "final_layer_readout": decode(Y[position]),
                "constant_training_mean": constant_tokens,
            }
            example = {
                "sequence_file": filename, "wikitext_row": record["row_id"],
                "position": position, "sequence_length": len(ids),
                "context_last_24_tokens": context,
                "full_prefix": tokenizer.decode(ids[:position+1], skip_special_tokens=False),
                "current_token": tokenizer.decode([ids[position]], skip_special_tokens=False),
                "readouts": methods,
            }
            examples.append(example)
            if seq_index < args.print_sequences:
                print(f"\n{filename}, position {position}/{len(ids)-1}", flush=True)
                print(f"Context ending here: {context!r}", flush=True)
                for name, tokens in methods.items():
                    print(name + ": " + ", ".join(repr(t["token"]) for t in tokens), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = fit_dir / f"plain_text_readouts_{stamp}"
    out.mkdir(exist_ok=False)
    output = {
        "fit_directory": str(fit_dir), "split": "validation", "test_used": False,
        "model_name": MODEL_NAME, "model_revision": revision,
        "source_layer": report["source_layer"], "final_layer": report["target_layer"],
        "positions": "midpoint n//2 and final n-1 of every saved validation sequence",
        "selection": "fixed before inspecting results; no claim these positions are independent",
        "inference": "cached residuals decoded with final RMSNorm and unembedding; no transformer forward pass",
        "interpretation": "Final-layer readout predicts the next token at this position. Pair-average map averages pair-specific regressions; pooled and same-position controls use the identical training subset.",
        "examples": examples,
    }
    (out / "readouts.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Held-out plain-text readouts", "",
             "Validation only. Midpoint and final position of each sequence. No refitting.",
             "The final-layer readout is the model's next-token readout; all three fitted maps use the same training subset, with different fitting/aggregation objectives.", ""]
    for example in examples:
        lines += [f"## {example['sequence_file']} / position {example['position']}", "",
                  "Context (last 24 tokens, ending at the inspected position):", "",
                  "```text", example["context_last_24_tokens"], "```", ""]
        for name, tokens in example["readouts"].items():
            lines += ["**" + name + "**", "", "```text",
                      ", ".join(repr(t["token"]) for t in tokens), "```", ""]
    (out / "readouts.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {len(examples)} position readouts to: {out}", flush=True)
    print("Only the first few sequences were printed. All validation examples are in readouts.md.")


if __name__ == "__main__":
    main()

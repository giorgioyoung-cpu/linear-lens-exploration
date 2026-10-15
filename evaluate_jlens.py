"""Read lenses at held-out WikiText validation positions using cached residuals.

Keep beside model_utils.py and evaluate.py. Run: python evaluate_jlens.py
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
    candidates = sorted(DEFAULT_RUN.glob('jlens_fit_*/fit_report.json'))
    if not candidates:
        raise FileNotFoundError('No full-passage J-lens estimate exists. Run fit_jlens.py --fit first.')
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
    print(f'J-lens based on {report["n_prompts"]} complete training passages ({report["status"]}).', flush=True)
    run = Path(report["activation_run"])
    meta_bytes = (run / "metadata.json").read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != report["collection_metadata_sha256"]:
        raise ValueError("Collection metadata changed since fitting.")
    with np.load(fit_dir / "linear_lens.npz", allow_pickle=False) as saved:
        A = saved["A"].astype(np.float64)
    comparisons = sorted(fit_dir.glob("mean_comparison_*/training_target_mean.npy"))
    if not comparisons:
        raise FileNotFoundError("Missing J-lens training mean; finish one estimation passage.")
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
    controls = {}
    control_paths = sorted(run.glob('pair_average_fit_*/fit_report.json'))
    if control_paths:
        control_dir = control_paths[-1].parent
        cr = json.loads(control_paths[-1].read_text())
        if (cr['source_layer'] != report['source_layer'] or cr['target_layer'] != report['target_layer']
                or cr['collection_metadata_sha256'] != report['collection_metadata_sha256']):
            raise ValueError('Regression control does not match J-lens model/layers/data.')
        with np.load(control_dir/'linear_lens.npz', allow_pickle=False) as z:
            controls = {name:z[key].astype(np.float64) for name,key in
                        [('pair_average_lens','A'),('pooled_lens','pooled_A'),('same_position_lens','same_position_A')]}
        print(f'Regression controls: {control_dir}', flush=True)
    else:
        control_dir = None
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run/'validation').glob('sequence_*.pt'))}
    meta = json.loads(meta_bytes)
    if len(hashes) != meta['splits']['validation']['sequences']:
        raise ValueError('Validation file count mismatch.')
    if not hashes or {p.name for p in (run / "validation").glob("sequence_*.pt")} != set(hashes):
        raise ValueError("Validation files differ from those used in fitting.")
    examples = []
    from fit_affine import sequence_moments, prediction_error
    causal = [np.zeros((d,d)),np.zeros((d,d)),0.0]
    diagonal = [np.zeros((d,d)),np.zeros((d,d)),0.0]
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
        cm = sequence_moments(U,Y)
        dm = (U.T@U/len(U),Y.T@U/len(U),float(np.sum(Y*Y)/len(U)))
        for k in range(3):
            causal[k] += cm[k]/len(hashes)
            diagonal[k] += dm[k]/len(hashes)
        for position in positions_to_inspect(len(ids)):
            context = tokenizer.decode(ids[max(0, position - 23):position + 1],
                                       skip_special_tokens=False)
            methods = {
                "jacobian_lens": decode(A @ U[position]),
                "ordinary_logit_lens": decode(U[position]),
                "final_layer_readout": decode(Y[position]),
                "constant_training_mean": constant_tokens,
            }
            for name, B in controls.items():
                methods[name] = decode(B @ U[position])
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
    summary = {}
    for name in examples[0]['readouts']:
        if name == 'final_layer_readout':
            continue
        overlaps=[]; hits=0; agreements=0
        for e in examples:
            pred=[t['token_id'] for t in e['readouts'][name]]
            target=[t['token_id'] for t in e['readouts']['final_layer_readout']]
            overlaps.append(len(set(pred)&set(target)))
            hits += int(target[0] in pred)
            agreements += int(pred[0] == target[0])
        summary[name] = {'mean_top_k_overlap':float(np.mean(overlaps)),
                         'final_top1_in_top_k':hits,'top1_agreement':agreements,'positions':len(examples)}
    residual_errors = {}
    for name,B in {'jacobian_lens':A, **controls, 'ordinary_logit_lens':np.eye(d)}.items():
        residual_errors[name] = {'all_causal_pairs':prediction_error(B,*causal),
                                'same_position':prediction_error(B,*diagonal),
                                'matrix_frobenius_norm':float(np.linalg.norm(B))}
    output = {
        "fit_directory": str(fit_dir), "split": "validation", "test_used": False,
        "model_name": MODEL_NAME, "model_revision": revision,
        "source_layer": report["source_layer"], "final_layer": report["target_layer"],
        "positions": "midpoint n//2 and final n-1 of every saved validation sequence",
        "selection": "fixed before inspecting results; no claim these positions are independent",
        "inference": "cached residuals decoded with final RMSNorm and unembedding; no transformer forward pass",
        "interpretation": "Final-layer readout predicts the next token at this position. J-lens readout uses a Jacobian estimate. Output agreement is a secondary diagnostic, not its objective.",
        "examples": examples, "summary":summary, "residual_errors":residual_errors,
        "residual_error_caveat":"Raw J-lens scale follows sum-target convention, not a residual reconstruction objective. Losses are secondary diagnostics.",
        "n_jacobian_training_passages": report['n_prompts'],
        "regression_control_directory": str(control_dir) if control_dir else None,
        "validation_file_sha256":hashes,
    }
    (out / "readouts.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Held-out plain-text readouts", "",
             "Validation only. Midpoint and final position of each sequence. No refitting.",
             "The final-layer readout is the model's next-token readout; J-lens uses the reference positional estimator, and regression controls have their own objectives and training sample sizes.", ""]
    for example in examples:
        lines += [f"## {example['sequence_file']} / position {example['position']}", "",
                  "Context (last 24 tokens, ending at the inspected position):", "",
                  "```text", example["context_last_24_tokens"], "```", ""]
        for name, tokens in example["readouts"].items():
            lines += ["**" + name + "**", "", "```text",
                      ", ".join(repr(t["token"]) for t in tokens), "```", ""]
    (out / "readouts.md").write_text("\n".join(lines), encoding="utf-8")
    lines += ['## Aggregate readout agreement', '', '```json', json.dumps(summary,indent=2), '```', '',
              '## Raw residual errors', '',
              'J-lens has sum-target scaling and does not optimize these losses. Interpret as secondary diagnostics.',
              '```json', json.dumps(residual_errors,indent=2), '```', '']
    (out / 'readouts.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('Raw residual errors: '+json.dumps(residual_errors,indent=2), flush=True)
    print(f"\nSaved {len(examples)} position readouts to: {out}", flush=True)
    print("Only the first few sequences were printed. All validation examples are in readouts.md.")


if __name__ == "__main__":
    main()

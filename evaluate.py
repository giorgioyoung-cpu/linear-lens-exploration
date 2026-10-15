"""Inspect top-token readouts before an answer. No fitting or interventions.

Keep this file beside model_utils.py. Run: python evaluate.py
Optional: python evaluate.py --prompt 'Name one fruit. Answer in one word.'
This is a qualitative readout check, not a J-space decomposition or J-lens replication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

DEFAULT_RUN = Path.home() / "jspace-data/activations/layer12_20261014T191345_397425Z"
DEFAULT_PROMPTS = [
    "Name one sport. Answer in one word.",
    "Name one fruit. Answer in one word.",
    "What is the capital of the country shaped like a boot? Answer in one word.",
    "What is 17 + 25? Answer with only the number.",
]


def resolve_fit(explicit):
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = sorted(p.parent for p in DEFAULT_RUN.glob("linear_fit_*/fit_report.json"))
    if not candidates:
        raise FileNotFoundError("Run fit_affine.py first, or supply --fit-dir.")
    return candidates[-1]


def rank_tokens(logits, tokenizer, k):
    scores, ids = logits.float().topk(k)
    return [{"rank": rank, "token_id": token_id,
             "token": tokenizer.decode([token_id], skip_special_tokens=False),
             "logit": score}
            for rank, (token_id, score) in enumerate(
                zip(ids.cpu().tolist(), scores.cpu().tolist()), start=1)]


def main():
    import torch
    from model_utils import load_model, MODEL_NAME

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    fit_dir = resolve_fit(args.fit_dir)
    fit_report = json.loads((fit_dir / "fit_report.json").read_text())
    if (fit_report.get("model_name") != MODEL_NAME
            or fit_report.get("intercept") is not False
            or fit_report.get("centered") is not False):
        raise ValueError("Expected our uncentered linear Qwen fit.")
    lens_path = fit_dir / "linear_lens.npz"
    with np.load(lens_path, allow_pickle=False) as saved:
        A = saved["A"].astype(np.float64)
    print(f"Using fit: {fit_dir}", flush=True)
    print("Loading cached Qwen...", flush=True)
    model, tokenizer, device = load_model()
    d = model.config.hidden_size
    layer = fit_report["source_layer"]
    if A.shape != (d, d) or not np.isfinite(A).all():
        raise ValueError("Invalid fitted matrix.")
    if not 1 <= layer < model.config.num_hidden_layers:
        raise ValueError("Invalid intermediate layer.")
    if fit_report["target_layer"] != model.config.num_hidden_layers:
        raise ValueError("Final layer mismatch.")
    revision = getattr(model.config, "_commit_hash", None)
    if fit_report.get("model_revision") and revision != fit_report["model_revision"]:
        raise ValueError("Cached model revision differs from the fitted model.")
    k = min(args.top_k, model.config.vocab_size)

    mean = None
    mean_path = None
    comparisons = sorted(fit_dir.glob("mean_comparison_*/training_target_mean.npy"))
    if comparisons:
        mean_path = comparisons[-1]
        mean = np.load(mean_path, allow_pickle=False)
        if mean.shape != (d,) or not np.isfinite(mean).all():
            raise ValueError("Invalid saved training target mean.")

    def decode_residual(residual):
        vector = torch.as_tensor(residual, dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = model.lm_head(model.model.norm(vector))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite readout logits.")
        return logits

    all_results = []
    for prompt in args.prompt or DEFAULT_PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        captured = {}

        def save_mid(module, inputs_to_layer, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["mid"] = hidden[0, -1].detach().cpu().float().numpy().copy()

        hook = model.model.layers[layer - 1].register_forward_hook(save_mid)
        try:
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, return_dict=True)
                actual_logits = result.logits[0, -1].float()
        finally:
            hook.remove()
        h = captured["mid"]
        if h.shape != (d,) or not np.isfinite(h).all():
            raise ValueError("Invalid captured intermediate residual.")
        # Apply A to a column vector on CPU in float64, then decode in float32.
        transported = A @ h.astype(np.float64)
        logits_by_method = {
            "linear_lens": decode_residual(transported),
            "ordinary_logit_lens": decode_residual(h),
            "model_next_token": actual_logits,
        }
        if mean is not None:
            logits_by_method["constant_training_mean"] = decode_residual(mean)
        actual_id = int(actual_logits.argmax().item())
        readouts = {}
        print(f"\nPrompt: {prompt}", flush=True)
        for name, logits in logits_by_method.items():
            top = rank_tokens(logits, tokenizer, k)
            # Ties receive the best rank among tied tokens.
            actual_rank = 1 + int((logits > logits[actual_id]).sum().item())
            readouts[name] = {"top_tokens": top, "model_top1_token_rank": actual_rank}
            print(f"{name}: " + ", ".join(repr(item["token"]) for item in top), flush=True)
        ids = inputs["input_ids"][0].cpu().tolist()
        all_results.append({
            "prompt": prompt, "formatted_prompt": text, "input_ids": ids,
            "readout_position": len(ids) - 1,
            "readout_token": tokenizer.decode([ids[-1]], skip_special_tokens=False),
            "model_top1_token_id": actual_id, "readouts": readouts,
            "source_residual_norm": float(np.linalg.norm(h)),
            "transported_residual_norm": float(np.linalg.norm(transported)),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = fit_dir / f"readouts_{stamp}"
    out.mkdir(exist_ok=False)
    report = {
        "model_name": MODEL_NAME, "model_revision": revision,
        "fit_directory": str(fit_dir),
        "matrix_sha256": hashlib.sha256(lens_path.read_bytes()).hexdigest(),
        "mean_path": str(mean_path) if mean_path is not None else None,
        "source_layer": layer, "device": str(device),
        "readout": "lm_head(final_RMSNorm(A @ h)); no intercept",
        "position": "last formatted prompt token, before any answer is generated",
        "purpose": "qualitative next-token readout sanity check",
        "caveat": "Fit on plain WikiText; evaluated on chat-formatted prompts. No J-lens comparison yet.",
        "test_split_used": False,
        "packages": {p: version(p) for p in ("torch", "numpy", "transformers")},
        "examples": all_results,
    }
    (out / "readouts.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Initial lens readouts", "",
             "Read at the final prompt position, before the answer. Tokens may be word fragments or special tokens.",
             "Scores are raw logits, not probabilities. Model agreement is a diagnostic, not proof of conceptual recovery.", ""]
    for item in all_results:
        lines += ["## " + item["prompt"], ""]
        for name, values in item["readouts"].items():
            lines += ["### " + name, "", "```text"]
            for entry in values["top_tokens"]:
                lines.append(f"{entry['rank']:2d}  {entry['token']!r}  (logit {entry['logit']:.4f})")
            lines += ["```", ""]
    (out / "readouts.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved readouts to: {out}", flush=True)
    print("No lens fitting or WikiText test evaluation was performed.")


if __name__ == "__main__":
    main()

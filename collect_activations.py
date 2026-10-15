"""Save per-position residuals from WikiText. Default: a small pilot.

Run: python collect_activations.py
No generation, model training, derivatives, or regression is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
import random
from pathlib import Path
from time import perf_counter

from datasets import load_from_disk
import torch

from model_utils import DATA_DIR, MODEL_NAME, load_model


def select_sequences(split, tokenizer, count, max_tokens, min_tokens, seed):
    """Sample distinct prose rows, taking each row's first max_tokens tokens.

    Each row is a separate context; rows are NOT concatenated. Blank rows and
    WikiText headings are excluded. This is a paragraph-prefix sampling rule,
    not uniform sampling over documents or tokens.
    """
    indices = list(range(len(split)))
    random.Random(seed).shuffle(indices)
    selected = []
    for row_id in indices:
        text = split[row_id]["text"].strip()
        if not text or text.startswith("="):
            continue
        ids = tokenizer(
            text, add_special_tokens=False,
            truncation=True, max_length=max_tokens,
        )["input_ids"]
        if len(ids) < min_tokens:
            continue
        selected.append({
            "row_id": row_id,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input_ids": torch.tensor(ids, dtype=torch.long),
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"Requested {count} sequences; only found {len(selected)}.")


def capture_residuals(model, input_ids, layer, device):
    """Return CPU float32 tensors of shape [sequence_length, model_dimension].

    h_mid is the output of transformer block `layer` (one-based).
    h_final is the input to the final RMSNorm: after all transformer blocks.
    """
    captured = {}

    def save_mid(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h_mid"] = hidden.detach().clone()

    def save_final(module, args):
        captured["h_final"] = args[0].detach().clone()

    middle_hook = model.model.layers[layer - 1].register_forward_hook(save_mid)
    final_hook = model.model.norm.register_forward_pre_hook(save_final)
    try:
        with torch.inference_mode():
            # Run the transformer without the vocabulary projection to save work.
            result = model.model(
                input_ids=input_ids.unsqueeze(0).to(device),
                use_cache=False, return_dict=True,
            )
            if set(captured) != {"h_mid", "h_final"}:
                raise RuntimeError("A residual hook did not fire.")
            # Check that we captured BEFORE final normalization.
            reconstructed = model.model.norm(captured["h_final"])
            torch.testing.assert_close(
                reconstructed, result.last_hidden_state, rtol=1e-5, atol=1e-6,
            )
    finally:
        middle_hook.remove()
        final_hook.remove()

    expected = (1, len(input_ids), model.config.hidden_size)
    output = {}
    for name, value in captured.items():
        if tuple(value.shape) != expected:
            raise RuntimeError(f"Unexpected {name} shape: {tuple(value.shape)}")
        value = value[0].to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in {name}.")
        output[name] = value
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=2)
    parser.add_argument("--test-sequences", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--layer", type=int, default=12,
                        help="One-based block number: 12 means after 12 blocks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    counts = (args.train_sequences, args.validation_sequences, args.test_sequences)
    if min(counts) < 1 or not 1 <= args.min_tokens <= args.max_tokens:
        parser.error("Counts must be positive and 1 <= min-tokens <= max-tokens.")
    return args


def main():
    args = parse_args()
    dataset_path = DATA_DIR / "wikitext-2"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Run download_data.py first; missing {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    print("Loading cached Qwen model...", flush=True)
    model, tokenizer, device = load_model()
    if not 1 <= args.layer < model.config.num_hidden_layers:
        raise ValueError("Choose an intermediate block between 1 and L-1.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (args.output or DATA_DIR / "activations" / f"layer{args.layer}_{stamp}").expanduser()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; output: {run_dir}", flush=True)

    counts = {"train": args.train_sequences,
              "validation": args.validation_sequences, "test": args.test_sequences}
    metadata = {
        "format_version": 1, "status": "incomplete", "created_utc": stamp,
        "model_name": MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_config": model.config.to_dict(),
        "dataset_path": str(dataset_path),
        "dataset_fingerprints": {s: dataset[s]._fingerprint for s in counts},
        "layer": args.layer, "final_layer": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size, "dtype": "float32",
        "device": str(device), "seed": args.seed,
        "max_tokens": args.max_tokens, "min_tokens": args.min_tokens,
        "requested_sequences": counts,
        "sampling": "seeded shuffled prose rows; prefix truncation; no concatenation",
        "chat_template": False, "add_special_tokens": False,
        "positions": "zero-based within each independent sequence; none discarded",
        "pair_weighting": "not selected yet; all positions retained for later fitting",
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("torch", "transformers", "datasets")},
        "splits": {},
    }
    manifest = run_dir / "metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_number, (split_name, count) in enumerate(counts.items()):
        split_dir = run_dir / split_name
        split_dir.mkdir()
        samples = select_sequences(
            dataset[split_name], tokenizer, count, args.max_tokens,
            args.min_tokens, args.seed + split_number,
        )
        started = perf_counter()
        n_tokens = 0
        for i, sample in enumerate(samples):
            arrays = capture_residuals(model, sample["input_ids"], args.layer, device)
            n = len(sample["input_ids"])
            record = dict(sample, **arrays, positions=torch.arange(n),
                          split=split_name, sequence_id=i, layer=args.layer,
                          final_layer=model.config.num_hidden_layers)
            target = split_dir / f"sequence_{i:05d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(record, temporary)
            temporary.replace(target)
            n_tokens += n
            print(f"{split_name}: {i + 1}/{count}, {n} tokens", flush=True)
        elapsed = perf_counter() - started
        metadata["splits"][split_name] = {
            "sequences": count, "tokens": n_tokens,
            "collection_and_save_seconds": elapsed,
        }
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{split_name}: {n_tokens} tokens in {elapsed:.1f} seconds", flush=True)

    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nCollection complete. Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

# Collecting WikiText residuals

Copy `model_utils.py` and `collect_activations.py` into Jspace-Extensions,
alongside your existing Python and TeX files. Keep them in the same folder.
They do not import or run download_model.py.

## Pilot

With your jspace environment active and Terminal in Jspace-Extensions:

```bash
python collect_activations.py
```

Or run collect_activations.py with VS Code's Python: Run Python File in Terminal.
The default is 8 training sequences, 2 validation sequences, and 2 test sequences.
Each has 32–128 tokens. This pilot checks collection, not regression quality.

The model and dataset are loaded from the locations already used on your Mac:

- Model cache: ~/jspace-data/models
- Dataset: ~/jspace-data/wikitext-2
- Output: ~/jspace-data/activations/layer12_<UTC timestamp>/

Model loading is local-only. Every run gets a new output folder; it does not
overwrite earlier runs. The model is frozen and runs in float32 on MPS when
available. The CPU fallback is slower. No new dependencies are needed beyond
your installed torch, transformers, and datasets.

## What is collected

Each sequence has one .pt file with:

| Field | Shape / meaning |
|---|---|
| input_ids | [n], actual input token IDs |
| positions | [n], zero-based token positions |
| h_mid | [n, 896], residual after block 12 |
| h_final | [n, 896], residual after block 24, before final RMSNorm |
| row_id | Original WikiText row index within the recorded split |
| text_sha256 | Hash of the stripped source row, before truncation |
| split, sequence_id | Identity within this collection |
| layer, final_layer | One-based block counts |

No padding is used. We run one sequence at a time, without generating answers
or computing vocabulary logits. Stored residuals are CPU float32 tensors.
The normalization check verifies that applying Qwen's final RMSNorm to the
captured final residual reproduces the base model's returned hidden state.
Shapes and finite values are checked before saving.

For a sequence x, row t of h_mid is h_12^(t)(x) with Python's zero-based t.
Row t_prime of h_final is h_24^(t_prime)(x). Both are from the same forward pass.
To form admissible future-position pairs later, use t <= t_prime within the
same sequence. Do not pair tokens across sequence files.

No position is discarded during collection. Burn-in, horizon, pair sampling,
and pair weights remain choices for fit_affine.py. Expanding all pairs now
would duplicate saved vectors unnecessarily.

## Sampling choices

The official train, validation, and test splits remain separate. Within each
split, row indices are shuffled using a fixed seed; blank rows and rows
beginning with '=' (WikiText headings) are skipped. We keep the first 128
tokens of each qualifying row, requiring at least 32. Each selected row is
an independent context, with no chat template or added special tokens.

This is a simple paragraph-prefix sampling distribution, not uniform sampling
over articles or tokens and not an exact replication of Anthropic's corpus.
Different rows can belong to the same article; observations are correlated.
Changing this preparation changes the distribution defining the affine fit.
Only the first 128 tokens are retained from long rows.

metadata.json records the sampling rule, model revision/config, package
versions, dataset fingerprints, counts, and timings. Status remains incomplete
if a run fails; do not treat that folder as a completed collection. Completed
sequence files remain on disk, but automatic resumption is not implemented.

## Larger run, after checking the pilot

```bash
python collect_activations.py --train-sequences 200 --validation-sequences 40 --test-sequences 40
```

At 128 tokens per sequence this saves at most 25,600 training token positions.
For 280 sequences, the two residual arrays together take at most about 257 MB
(decimal), plus small metadata and token-ID overhead. Intermediate model
memory is additional. Runtime should be judged from the pilot.

## Reading a sequence later

```python
record = torch.load(path_to_sequence_file, map_location="cpu", weights_only=True)
X = record["h_mid"]
Y = record["h_final"]
```

Check metadata.json status and model/layer conventions before combining files.
Keep the scripts and this guide in GitHub; activation files stay outside the
project directory. The scripts collect data only; they do not fit a lens.

## Validation and sources

The files have been syntax-checked and their sampling function tested with
synthetic rows. A full Qwen run and MPS compatibility have not been tested in
the authoring environment; the pilot performs the real model checks on your Mac.

- Qwen implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
- Dataset loading: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict.load_from_disk

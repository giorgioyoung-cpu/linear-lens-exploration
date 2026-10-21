# Affine-slope lenses: pooled, same, paired

Fits affine approximations to cached Qwen residuals, then discards the intercept
for token readout. These are new experiments, not the earlier uncentered results.

## Pause at 32 passages before deciding whether to continue

Add `--stop-after 32` to the 118-passage fitting command. This is a TOTAL count,
not 32 additional passages. On a multi-layer root, each of the five layers runs
up to 32 passages, then the program exits after the last layer. It does not proceed
to 64 or 118 automatically. All layers remain at 32 for the comparison.

```bash
python fit_jlens.py --run-dir "PATH" --fit --prompts 118 --save-at 8 32 64 118 --stop-after 32
```

Each layer's fit directory contains `stability.md` and `stability.json`, comparing
successive retained matrices (8 versus 32 for this stage). The report gives
relative Frobenius change, matrix cosine and norm ratio. No thresholds are imposed;
these are matrix diagnostics, not a test that readouts have converged. Share the
five Markdown reports and compare fixed validation readouts before deciding.

To continue a SINGLE layer to 64 total passages, reuse its exact fit directory:

```bash
python fit_jlens.py --run-dir "PATH/layer_12" --fit --prompts 118 --save-at 8 32 64 118 --stop-after 64 --resume "PATH/layer_12/jlens_fit_TIMESTAMP"
```

Repeat for each layer chosen for continuation. For completion to 118, omit
`--stop-after` but retain `--resume` and the original other options. The stop count
is deliberately excluded from the fit configuration, so changing it does not
invalidate the resume. Already completed passages are reused. Running the root
command again WITHOUT resume starts new fits; it is not a continuation.

This update changes only `fit_jlens.py`, `test_lenses.py` and this README; keep the
other files from the preceding checkpoint update.

## J-lens sample-size checkpoints

Full fitting now requires an explicit `--prompts` argument. For the agreed
118-passage experiment, use `--fit --prompts 118 --save-at 8 32 64 118`.
The same ordered eligible training passages are used at each layer; each smaller
estimate is a nested prefix of the larger estimate. The estimator itself is
unchanged. Only the sample count and retained outputs change.

Each `jlens_fit_TIMESTAMP` contains `snapshots/prompts_0008`, `prompts_0032`,
`prompts_0064`, and `prompts_0118`. Each snapshot has its own matrix, training
mean, exact passage list and report. Snapshots are retained permanently; the
ordinary top-level matrix remains the running estimate. The final passage count
is always retained, even if omitted from `--save-at`.

Both evaluators accept `--jlens-checkpoint 8` (or 32, 64, 118). In multi-layer
counting mode this selects that sample count in each layer's newest J-lens fit.
A missing requested snapshot raises an error, rather than silently falling back.
For an exact single-layer run, combine `--jlens-dir FIT_PATH` with
`--jlens-checkpoint N`; without the checkpoint option, `--jlens-dir` can also
point directly at an individual snapshot directory. Evaluation output records
the actual selected paths, counts and matrix hashes.

A checkpoint marked complete means all N passage contributions are computed;
it does NOT establish statistical convergence. Compare the same prompts and
readout positions across sample counts. The nested estimates are dependent,
not independent replications. Stability across them is useful evidence but not
a guarantee that the population-average lens has been accurately estimated.

Recovery state is saved after every complete passage. If interrupted, resume
ONE layer with the exact original options and its printed fit directory:

```bash
python fit_jlens.py --run-dir "PATH/layer_12" --fit --prompts 118 --save-at 8 32 64 118 --resume "PATH/layer_12/jlens_fit_TIMESTAMP"
```

Use the layer actually interrupted, not necessarily layer 12. A partial passage
must be recomputed; completed passages in the checkpoint are reused. Do not use
`--resume` on the multi-layer root. After resuming one layer, run remaining layer
directories individually to avoid refitting layers already completed. Existing
old eight-passage fits cannot be extended with this command, since their saved
configuration differs. Start this experiment as a new fit; old results remain.

## Multi-layer counting experiment (new)

The updated files retain the original single-layer commands and cache schema.
Replace the files of the same names inside your existing `affine_lenses` folder;
do not delete saved activations or fit directories. No new dependencies are added.

1. Run `python test_lenses.py`.
2. Collect the same WikiText sampling scheme at five source layers in ONE forward
   pass per passage (the final residual is stored in each layer directory to keep
   old tools compatible):

```bash
python collect_activations.py --layers 4 8 12 16 20 --train-sequences 200 --validation-sequences 40 --test-sequences 40 --max-tokens 128 --min-tokens 32 --seed 42
```

Copy the printed `multilayer_TIMESTAMP` directory and use it in place of PATH below:

```bash
python fit_lenses.py --run-dir "PATH"
python fit_jlens.py --run-dir "PATH"
python fit_jlens.py --run-dir "PATH" --fit --prompts 118 --save-at 8 32 64 118
python evaluate_concepts.py --task counting --run-dir "PATH"
```

The first J-lens command benchmarks each layer; the second fits each. They run
sequentially to limit memory use. Earlier layers may take longer than layer 12.
A multi-layer root has `layers.json` and `layer_04`, `layer_08`, etc.; each child
is a complete ordinary activation run with its own fits. To resume or rerun a
single layer, pass its child directory. A repeated root fitting command refits
all layers. Old layer-12 fits are preserved, but this new run refits layer 12
for a consistent collection and provenance. Do not copy an old J-lens into a new
run: the collection-hash checks deliberately reject mismatched provenance.

Counting generates one greedy response to "Count to five and introspect deeply."
(default cap: 64 new tokens). It then replays that exact token sequence in one
causal forward pass, capturing every requested source layer and the final residual.
No counting prompt is used for regression fitting. Inspect the generated answer
for task compliance; the small model may not follow the instruction as intended.
Use `--max-new-tokens 128` if the output is truncated. Overrides and EOS status are
recorded. `--prompt "..."` permits a control such as "Count to five."

Readouts cover the final prompt token and every generated non-EOS token. At
position j, the residual includes token j; its final-layer readout predicts j+1.
There is no automatic 17..n-1 evaluation mask on these counting positions: that
mask belongs to WikiText fitting. Each layer uses its own fitted matrices.
All methods see the SAME response generated by the unmodified model.

Outputs in a new `counting_TIMESTAMP` folder:

- `activations.pt`: exact input/response token IDs, prompt length, all captured
  source residuals and final residuals at every position (CPU float32).
- `readouts.md` and `readouts.json`: top 10 tokens and fixed concept-candidate ranks
  for each layer/position/method; JSON also records exact fit paths and provenance.
- `concept_ranks.csv`: tidy rows for plotting layer/position trajectories.

Candidates include counting, halfway, done, thoughts, AI, consciousness and
one through five. Only complete single-token spellings are ranked; missing variants
are unavailable. These are exploratory candidates, not an exhaustive concept set.
No plot or causal intervention is performed by this script.

Cached J-lens is loaded separately for each layer. If absent, that layer's readout
contains regression lenses and controls only; the console says so. Run the J-lens
fit command above before making a comparison to J-lens at all five layers.

The arithmetic mode remains the default. With a multi-layer root it repeats the
existing arithmetic probe for each layer, saving separate results. For WikiText
readouts, give `evaluate_lenses.py` a single child layer directory with `--run-dir`.

### Choices relative to the paper

We use its lens readout form: a layer-specific matrix, final normalization, then
unembedding (softmax is unnecessary for ranks). The counting/introspection task
is inspired by Figure 5. Our small Qwen model, five-layer sampling, chat template,
generation cap, candidate list, WikiText fitting data and sample counts are our
choices. This is an adaptation, not an exact reproduction. We retain the final
block as the target for all lenses, whereas the paper's default Sonnet Jacobian
recipe targets the penultimate block. Our earlier midpoint/penultimate TOKEN
selection is a separate WikiText display choice and is not used for counting.
Reference: https://www.transformer-circuits.pub/2026/workspace/index.html

Modified: `collect_activations.py`, `fit_lenses.py`, `fit_jlens.py`,
`evaluate_concepts.py`, `lens_core.py`, `test_lenses.py`, and this README.
The remaining bundled files are unchanged.

## Run with the existing cache

Extract this entire folder into your project. Keep its files together and run
commands from inside `affine_lenses`. It is self-contained; no legacy evaluation
or fitting scripts need to be copied into it. Activate your existing environment:

```bash
source ~/venvs/jspace/bin/activate
python test_lenses.py
python fit_lenses.py
python evaluate_lenses.py
python evaluate_concepts.py
```

No new installation or downloads are normally needed in your existing environment.
The default activation directory is:
`~/jspace-data/activations/layer12_20261014T191345_397425Z`.
Fits are saved in new `affine_fit_TIMESTAMP` directories there, leaving old results
intact. Evaluators automatically choose the newest affine fit under that run.
For reproducibility, use the exact directory printed by fitting:

```bash
python evaluate_lenses.py --fit-dir '/absolute/path/to/affine_fit_TIMESTAMP'
python evaluate_concepts.py --fit-dir '/absolute/path/to/affine_fit_TIMESTAMP'
```

`evaluate_lenses.py` compares pooled, same, paired, ordinary logit lens, final-layer
readout and the two training-mean baselines. If present, it also loads the newest
cached `jlens_fit_*` from the same activation run. To choose it explicitly, pass
`--jlens-dir '/absolute/path/to/jlens_fit_TIMESTAMP'` to either evaluator.
An existing eight-passage J-lens can still be loaded, but its sample-size stability is untested; the 118-passage experiment above is intended to assess this limitation.

Default WikiText readouts use the midpoint and last position of the **retained**
interval. To compare on exactly the old midpoint/final-token selection:

```bash
python evaluate_lenses.py --positions legacy
```

The latter is a readout choice only; it does not change the training mask.
Arithmetic defaults to the six existing chat-formatted addition problems at the
final prompt token. For the plain expression format:

```bash
python evaluate_concepts.py --format plain
```

## Files and replacements

| File | Purpose | Supersedes for these new experiments |
|---|---|---|
| `fit_lenses.py` | Fits all three affine slopes and scores reconstruction | `fit_affine.py`, `fit_same_position.py`, `fit_pair_average.py`, `compare_mean.py` |
| `evaluate_lenses.py` | WikiText readouts for all three slopes and cached J-lens | `evaluate_plain_text.py`, `evaluate_same_position.py`, `evaluate_pair_average.py`, `evaluate_jlens.py` |
| `evaluate_concepts.py` | Arithmetic probes for all three slopes and cached J-lens | `evaluate_arithmetic.py`, `evaluate_jlens_arithmetic.py` |
| `lens_core.py` | Shared regression, data validation and readout helpers | New dependency; removes dependence on old `evaluate.py` helpers |
| `test_lenses.py` | Checks against explicit weighted least squares and pair fits | New numerical verification |
| `model_utils.py` | Offline model loading | Included unchanged |
| `fit_jlens.py` | Original Jacobian benchmark/fit | Extended to dispatch multi-layer runs; old single-layer mode retained |
| `collect_activations.py` | Collects residuals | Extended to dispatch multi-layer runs; old single-layer mode retained |
| `download_data.py` | Downloads model and WikiText | Optional first-time setup |
| `METHODS.tex` | Mathematical definitions and solutions | New reference |

Keep the old scripts if you want to reproduce the old uncentered experiments.
The generic custom-prompt `evaluate.py` is not replaced by this package; these
new evaluators cover WikiText and the arithmetic experiments.

## What is fitted

- **pooled:** one affine map minimizing mean squared residual error across
  passages and retained causal pairs, with equal passage and uniform pair weights.
- **same:** one affine map minimizing error at matching retained positions, with
  equal passage and uniform position weights.
- **paired:** independently centered affine regressions for each positional pair,
  followed by a uniform average of their slope matrices.

All three use the **same eligible training passages**, truncated to 128 tokens.
Shorter passages are excluded from all three fits. Your existing collection should
therefore supply 118 training passages. This differs from the original 200-passage
same-position experiment. Training positions are 17 through 127 inclusive in
one-based notation (Python slice `16:127`). Both source and target obey this mask.
The paired implementation deliberately uses common-length passages; it does NOT
implement the variable-length eligible-passage extension discussed mathematically.

Validation uses all 40 saved validation passages, each with positions 17 through
its own penultimate token. The saved test set is never read. No new forward passes
are needed for fitting or WikiText readouts. Arithmetic requires forward passes.

Pooled and same center using their objective-weighted global means, not separate
passage means. Paired centers separately across passages at each fixed position.
All calculations use float64 NumPy. Pooled/same covariance eigenvalues below
`1e-10` times the largest are discarded. Paired uses the equivalent relative
design singular-value cutoff `1e-5`. There is no ridge penalty. Since centering
removes one sample degree of freedom, paired ranks are at most 117/896 with 118
passages. Finite numerical cutoffs approximate the Moore–Penrose inverse.

The shared suffix-target calculation aggregates pair-specific slopes/intercepts
exactly without storing thousands of matrices. `paired_b` is the average of those
intercepts, not the optimal intercept for the final averaged slope.

## Outputs and interpretation

Each fit saves:

- `lenses.npz`: `pooled_A`, `pooled_b`, `pooled_mean`, `same_A`, `same_b`,
  `same_mean`, `paired_A`, `paired_b`.
- `fit_report.json`: dataset hashes, excluded files, mask, model revision, rank
  diagnostics, timings and train/validation reconstruction scores.

For each final map, reconstruction is scored under both pooled and same-position
objectives, separately for `A h` and `A h + b`. The constant baseline always uses
that objective's training target mean. The averaged paired affine map is not a
minimizer of either aggregate objective. Loss is squared Euclidean error, averaged
over positions/pairs/passages, **not divided by residual dimension**.

Both evaluators use **only `A h`**, followed by the model's final RMSNorm and
unembedding. They save `readouts.md` and `readouts.json` in new timestamped folders.
Printed and saved token positions are one-based. Arithmetic scores complete
single-token numeral and word variants plus every digit 0–9 for specificity;
these exploratory cases have already been examined, and are not a fresh test set.

Older J-lens runs used eight passages; new runs record the explicitly requested
count. J-lens uses sum-target/mean-source normalization. Regression fits use 118
passages and average causal pairs.
Matching masks does not make sample sizes or estimators identical. Neither raw
reconstruction error nor agreement with final-layer tokens establishes conceptual
interpretability. No intermediate arithmetic result has yet been demonstrated by
these new affine fits: they must be run and inspected.

## Optional reproduction from a fresh cache

Use a Python environment compatible with PyTorch/Transformers. Requirements list
direct dependencies, not a tested version lock. Preserve your working environment;
record it with `python -m pip freeze > environment-lock.txt` when publishing results.

```bash
python -m pip install -r requirements.txt
python download_data.py
python collect_activations.py --train-sequences 200 --validation-sequences 40 --test-sequences 40 --max-tokens 128 --min-tokens 32 --layer 12 --seed 42
```

Use the new activation directory printed by collection for every subsequent step:

```bash
python fit_lenses.py --run-dir '/absolute/path/to/new_activation_run'
python fit_jlens.py --run-dir '/absolute/path/to/new_activation_run'
python fit_jlens.py --run-dir '/absolute/path/to/new_activation_run' --fit --prompts 8
python evaluate_lenses.py --run-dir '/absolute/path/to/new_activation_run'
python evaluate_concepts.py --run-dir '/absolute/path/to/new_activation_run'
```

The first `fit_jlens.py` invocation runs its synthetic/finite-difference benchmark;
`--fit` computes full matrices. Its checkpoint resume requires the original fit
configuration. The adapter is independent code implementing the estimator in
https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py.
Model: Qwen/Qwen2.5-0.5B-Instruct; data: Salesforce/wikitext, wikitext-2-raw-v1.
Downloads use the available Hub revision; collection records the model revision.
Byte-identical reproduction across fresh Hub/software versions is not guaranteed.

## Verification

Implementation verification. Numerical tests cover weighted affine least squares for variable sequence lengths, pairwise regression in full-rank and underdetermined settings, translation invariance of paired slopes, recovery of known affine maps, zero-covariance cases, and agreement of reconstruction formulas. Python syntax checks also pass.


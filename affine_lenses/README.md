#A more complete write-up of the theory and motivation may be found in L2 minimizing Jlens.pdf


# Affine-slope lenses: pooled, same, paired

Fits affine approximations to cached Qwen residuals, then discards the intercept
for token readout. These are new experiments, not the earlier uncentered results.

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
The existing eight-passage J-lens does not need to be recomputed.

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
| `fit_jlens.py` | Original Jacobian benchmark/fit | Included unchanged; optional to rerun |
| `collect_activations.py` | Collects residuals | Included unchanged; optional to rerun |
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

J-lens retains its original eight-passage sample and sum-target/mean-source
normalization. The regression fits use 118 passages and average causal pairs.
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

Author-side checks: explicit weighted affine least squares for variable sequence
lengths; explicit pairwise fits in full-rank and underdetermined regimes;
translation invariance of paired slopes; known affine recovery; zero-covariance
cases; reconstruction formula agreement; Python syntax checks.
The new Qwen evaluation paths have not been executed in the author environment.
Run the supplied scripts in your existing Mac environment to obtain actual results.

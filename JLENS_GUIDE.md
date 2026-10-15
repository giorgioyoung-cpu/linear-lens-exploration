# Jacobian baseline for the existing Qwen project

Place all three Python scripts beside model_utils.py, evaluate.py, and
fit_affine.py in your MAIN project folder. The existing environment suffices.
This is an independent adapter for the existing cached Qwen model, implementing
the reference Jacobian estimator. It does not install the Anthropic repository.

## Start with the benchmark

```bash
python fit_jlens.py
```

This computes eight output-coordinate rows on one 128-token training passage,
with two replicated sequences per backward pass. It prints a rough estimate
for a full matrix and an eight-passage pilot. It does NOT save a usable lens.
Send the terminal output before deciding how many passages to estimate.

It first runs a small CPU autograd test with a known causal linear sequence map.
It then compares actual source and target residuals against the saved cache,
and checks one aggregate Jacobian row against central finite differences through
Qwen. The finite-difference check is deliberately approximate for float32.
A failure needs investigation before running a fit.

If the MPS device reports an unsupported operation, the explicit alternative is:

```bash
python fit_jlens.py --device cpu
```

For memory pressure, use --dim-batch 1. Keep the same setting when fitting.
Timing is only an extrapolation; thermal throttling and overhead can change it.

## Estimator and conventions

Reference inspected: https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py

For a sequence of n tokens, V={16,...,n-2} in zero-based indices. The estimator is

J = mean_x [ (1/|V|) sum_{j in V} sum_{i in V, i>=j}
             partial h_final^(i)(x) / partial h_layer^(j)(x) ].

Both source and target exclude the first 16 positions and the final position.
Targets are SUMMED, sources and passages are AVERAGED. Gradients include the
attention pattern's dependence on the activation; attention is not frozen.
Each output-coordinate row is calculated exactly by reverse-mode autograd
up to floating-point arithmetic, rather than by a random-projection estimate.
The corpus average is a finite-sample estimate.

Our source is after block 12 (reference zero-based block index 11). The target
is after the final block, before the final RMSNorm. Source activations become
independent gradient leaves. Model weights are frozen; no optimizer or training
updates are performed. Only the suffix after the source enters the backward graph.
The model is loaded in float32 with eager attention for a transparent gradient
path, and matrix accumulation is in CPU float64.

Selected inputs are the first N cached training passages having exactly 128
tokens. Their IDs and hashes are saved. No validation or test inputs enter
estimation. This adapts the estimator to our model and WikiText data; it is not
a replication of the paper's model, corpus, or sample size.

Our previous regressions used different position exclusions and reductions.
In particular, the sum-target scale of this Jacobian is not a residual-prediction
scale. Raw squared errors should not be used as a ranking of interpretability.
The comparison readouts are explicitly exploratory with unmatched estimation
sample sizes and positional conventions.

## After the benchmark: small full-matrix pilot

```bash
python fit_jlens.py --fit --prompts 8
```

This uses eight full training passages. Eight is a compute pilot, not a claim
of a converged average. Each passage requires 896 Jacobian rows. The script
prints progress and saves after each complete passage in jlens_fit_TIMESTAMP.
Running averages are usable only as preliminary lenses; convergence needs
checking with more passages. The fit directory and completed count are recorded.

To resume after interruption, repeat the SAME options and add:

```bash
python fit_jlens.py --fit --prompts 8 --resume /absolute/path/to/jlens_fit_TIMESTAMP
```

The current incomplete passage is recomputed. Changing the passage count or
device requires a new fit directory with this implementation. Existing regression
fits and activation files are never overwritten.

## Compare held-out readouts

```bash
python evaluate_jlens.py
```

Uses the latest estimate with at least one complete passage (prints its count).
Decodes the same midpoint and final positions from all 40 validation passages.
If a pair_average_fit directory exists, loads its three regression matrices as
controls. Also includes ordinary logit lens, final-layer readout, and a constant
mean from the Jacobian training passages. No new parameters are fitted.

It saves all readouts, top-k overlap/top-1 agreement by token ID, and raw residual
errors. J-lens is decoded using final RMSNorm and unembedding exactly like our
regression maps. The comparison script imports mathematical loss helpers from
fit_affine.py, without running its fitting procedure.

## Compare the same arithmetic cases

```bash
python evaluate_jlens_arithmetic.py
```

This runs the six simple addition cases from the latest regression arithmetic
probe, in the same chat format and at the same layer-12 final-prompt position.
It reports full numeral ranks for every digit, variant ranks, and generated
answers. Use --format plain for a separately labeled plain-expression run.
It does not establish causal use of the parenthesized intermediate or progression
across layers. No change has been made to the arithmetic task to favor J-lens.

Both evaluators accept --fit-dir /absolute/path/to/jlens_fit_TIMESTAMP to select
a particular checkpoint. Results are saved in timestamped subdirectories.

## Validation status

All delivered scripts passed Python syntax checks in the authoring environment.
PyTorch and Qwen were unavailable there, so the actual autograd and model checks
run on your Mac as the first stage of the benchmark. Report failures rather than
treating an unverified or incomplete matrix as a baseline.

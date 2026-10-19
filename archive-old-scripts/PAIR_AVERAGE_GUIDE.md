# Pair-specific regression followed by averaging

Place the two new scripts beside your existing fit_affine.py, evaluate.py,
and model_utils.py in the main project folder. Do not leave them in a subfolder.

Run in your activated jspace environment:

```bash
python fit_pair_average.py
python evaluate_pair_average.py
```

The first command uses the existing layer-12 activation collection. It selects
training passages with at least 128 tokens and uses their first 128 positions.
Shorter passages are excluded and listed in fit_report.json. It reports the
number retained before fitting. No model inference is needed to fit these maps.

For each pair 0 <= j <= i < 128, the mathematical definition is the
minimum-Frobenius-norm least-squares map A_ij from source position j to final
position i, fitted uniformly across eligible training passages. The final map
is the uniform average over those 8,256 pair-specific matrices. There is no
intercept, centering, ridge penalty, or position exclusion.

For each fixed j the design matrix is shared by all i. The code applies its
pseudoinverse to the sum of future target matrices, then divides the final sum
by 8,256. This is algebraically identical to computing and averaging all A_ij;
it requires just 128 design factorizations and does not store all pair maps.
The relative singular-value cutoff is 1e-5, corresponding to the earlier
1e-10 eigenvalue cutoff on the second-moment matrix.

## Interpret this as a pilot

With at most 200 passages and 896 input coordinates, each position's regression
is underdetermined. The pseudoinverse chooses the minimum-norm solution and
sets its action on unobserved input directions to zero. Averaging can produce
a higher-rank final map, but does not make the separate regressions well determined.
Per-position ranks are printed and saved. Positive or negative pilot results
will need checking with more data or a separately specified regularization scheme.

This construction implements the proposed order of regression and aggregation.
It is not a replication of original J-lens: in particular, it includes all causal
pairs without the original implementation's position exclusions. Uniform pair
weighting is retained here to isolate the change from pooled regression.

## Matched controls and readouts

The script also refits the previous pooled all-future map and the same-position
map on exactly the same selected passages. This avoids confusing a change in
training data with a change in the estimator. Both training-mean baselines
are saved. It evaluates the three final matrices under both residual losses
on training and validation data. The pooled fit optimizes the training causal
loss and the same-position fit optimizes the training diagonal loss; their
advantages on those losses are expected, not evidence of better concepts.

The readout script uses all the original validation sequences, at their midpoint
and last token, for continuity with earlier results. It compares the averaged
map, matched pooled and same-position maps, ordinary logit lens, the final
residual readout, and the two training means. It loads the cached model only
for normalization and unembedding. It does not fit on validation or access test data.

New results are saved under pair_average_fit_TIMESTAMP; existing fits remain
untouched. Readouts are saved as readouts.md and readouts.json in the printed
subfolder. Upload the new readouts and the per-position rank summary for review.

An alternative activation collection can be selected with --run-dir PATH.
Then pass its resulting fit explicitly to the readout script with --fit-dir PATH.
--length changes the common training prefix length and thus the experiment.

## Verification

The included test_pair_average.py verifies the optimized estimator against
explicit fitting and averaging of every pair, including underdetermined and
rank-deficient designs. It also checks zero inputs and the single-position case.
These synthetic tests and script syntax were checked before delivery. Actual
PyTorch activation loading and Qwen readout must run on your Mac.

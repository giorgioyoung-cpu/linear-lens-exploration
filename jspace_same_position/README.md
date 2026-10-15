# Same-position comparison

Place both Python scripts alongside your existing evaluate.py and model_utils.py.
Activate your usual jspace environment, then run from the project folder:

```bash
python fit_same_position.py
python evaluate_same_position.py
```

The first command fits A without an intercept or centering, with equal weight
per training passage and uniform weight per position within each passage.
It uses only pairs i=j, using the same cached layer-12 and final residuals,
float64 solver and eigenvalue cutoff as the earlier fit. It reports training
and validation error, including comparison with the TRAINING target mean.
Test data are untouched. These errors use a different objective from the
all-future fit and should not be directly compared as if they were the same loss.

Results go into a new same_position_fit_TIMESTAMP folder under the existing
activation collection. Earlier fits are preserved.

The second command selects the latest same-position fit and decodes the same
80 validation positions as before. It uses the cached model's final normalization
and unembedding, with no transformer forward pass. Its constant baseline is
the training mean for the new objective. All readouts are saved in readouts.md
and readouts.json in the printed output directory.

For a specific fit, use:
python evaluate_same_position.py --fit-dir /absolute/path/to/same_position_fit_TIMESTAMP

Dependencies are those already installed for the project. The fitting algebra
was checked against direct weighted least squares on synthetic unequal-length
passages; the full Qwen run must be performed on your Mac.

# First linear fit

Copy fit_affine.py and test_fit_affine.py into Jspace-Extensions. The name
fit_affine.py follows our planned file structure, but this version fits ONLY
a linear map: there is no intercept, centering, or ridge penalty.

## Run

With the jspace Python environment active:

```bash
python fit_affine.py
```

Running the file with VS Code's Python: Run Python File in Terminal also works.
Its default activation folder is your completed collection:

```text
~/jspace-data/activations/layer12_20261014T191345_397425Z
```

For another collection:

```bash
python fit_affine.py --run-dir /absolute/path/to/activation_folder
```

No model download or forward pass is needed. Computation uses CPU NumPy in
float64. Only train and validation sequence files are read; test stays untouched.

## Exact objective and orientation

For each passage k of length n_k, let U[j] be the layer-12 residual at source j
and Y[i] the final, pre-normalization residual at target i. Positions are
zero-based. The admissible set is ALL pairs 0 <= j <= i < n_k, including i=j.
There is no initial-position exclusion and no horizon cutoff in this first fit.
This is our chosen pair set, not a claim to reproduce the published J-lens.

Each passage has weight 1/N and each of its m_k=n_k(n_k+1)/2 pairs has weight
1/m_k within that passage. The objective is:

    (1/N) sum_k (1/m_k) sum_{i>=j} ||Y_k[i] - A U_k[j]||^2.

The two uncentered moments are S=E[U U^T] and C=E[Y U^T]. We compute
A=C S^dagger. Arrays store vectors as ROWS: applying the fitted map to a row
batch U is U @ A.T. Applying it to a column vector u is A @ u.

For one passage, the implementation uses exactly equivalent sums:

    S_k = sum_j (n-j) U[j] U[j]^T / m_k
    C_k = sum_j (sum_{i=j}^{n-1} Y[i]) U[j]^T / m_k
    q_k = sum_i (i+1) ||Y[i]||^2 / m_k

It then averages S_k, C_k and q_k equally over passages. Suffix sums avoid
building a quadratic-sized expanded pair dataset. Pair observations remain
correlated; their count is not an independent sample size.

## Pseudoinverse convention

S is symmetrized, then diagonalized. Eigenvalues <= 1e-10 times its largest
eigenvalue are discarded. This is an explicit numerical rank cutoff, not ridge.
If nonzero directions fall below the cutoff, the computed map is a truncated
approximation to the mathematical Moore-Penrose solution. Retained rank and
thresholds are reported. The default is a starting numerical choice, not a
validated optimal cutoff. It applies to eigenvalues of S, not singular values
of the expanded design matrix. No cutoff selection is automated.

The optional --pinv-rtol argument changes it. Decide such changes using
validation results, not test performance. No activations are centered.

## Outputs

A new linear_fit_<UTC timestamp> folder is created inside the activation run:

- linear_lens.npz: A, training moments S_train and C_train, q_train, and spectrum.
- fit_report.json: train/validation scores, numerical rank, exact pair rule,
  model/layer identity, source file hashes, package versions, and timings.

Earlier fits and residuals are not overwritten.

Reading the fitted map:

```python
import numpy as np
with np.load('/path/to/linear_lens.npz', allow_pickle=False) as fit:
    A = fit['A']
```

## Scores

- Squared error is the passage-weighted mean of squared Euclidean residual
  prediction errors, summed over all 896 coordinates.
- Per-coordinate MSE divides that number by 896.
- Relative error divides it by E[||Y||^2]. A value below 1 beats predicting zero.
  This is NOT centered R-squared.
- Identity error uses A=I for the SAME cross-position pairs; it predicts Y[i]
  directly from U[j]. It is a raw-residual baseline, not a token-readout score.

All scores use the same passage/pair weighting. Error is evaluated from the
moments as q - 2*sum(A*C) + trace(A*S*A.T). Tiny negative roundoff is clamped
to zero; larger negative values raise an error.

Good residual prediction is not sufficient evidence of useful conceptual
readouts. Those experiments come after this fit.

## Checks

```bash
python test_fit_affine.py
```

The three included tests passed in the authoring environment. They compare
the efficient moment formulas with explicit pair enumeration and SVD-based
weighted least squares, including unequal passage lengths, singular inputs,
zero inputs, and error evaluation for several matrices. This validates the
numerical core. Loading your actual .pt files and the full 896-dimensional fit
have not been run here; the first run on your Mac checks that integration.

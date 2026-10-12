# SLTS + Kibria-Lukman Hybrid Estimator

A robust, regularized regression pipeline for high-dimensional data with
**both** multicollinearity and outlier contamination, built by extending the
SLTS+LASSO estimator from my undergraduate thesis
([*Robust Hybrid Algorithms for Regularization and Variable Selection in the
Presence of Outliers and Multicollinearity*](#)) with a second-stage ridge-type
estimator from the Kibria-Lukman family (Kibria & Lukman, 2020, *Scientifica*),
the estimator family studied in Dr. Kayode Ayinde's own published research.

## Why this project

My thesis combined a robust first-stage estimator (Sparse Least Trimmed
Squares, SLTS) with a sparse second-stage predictor (LASSO) to handle data
that is simultaneously multicollinear and outlier-contaminated. LASSO is a
natural second stage when the true underlying signal is sparse (only a few
predictors actually matter). Ridge-type estimators are the natural second
stage when the signal is dense (many correlated predictors each contribute a
little) — which is the regime most directly addressed by the biased/ridge-type
estimator literature, including the Kibria-Lukman estimator.

This project:
1. Implements the Kibria-Lukman ridge-type estimator from first principles
   (`src/estimators.py`), including a documented fix for a known instability
   in its closed-form shrinkage factor (see below).
2. Swaps it in as an alternative second stage after the same SLTS robust
   screening step used in the thesis, producing a new hybrid: **SLTS+KLE**.
3. Benchmarks SLTS+KLE against OLS, Ridge, LASSO, and the original
   SLTS+LASSO across a grid of outlier-contamination levels, under both a
   sparse-truth and a dense-truth simulation regime, so the comparison is not
   tilted toward whichever method is being proposed.

## A genuine numerical issue, found and fixed

The Kibria-Lukman estimator applies, to each coefficient in the eigenbasis of
`X'X`, the shrinkage factor `(lambda_i - k) / (lambda_i + k)`. For
eigen-directions where the eigenvalue `lambda_i` is *smaller* than the
shrinkage parameter `k` — i.e. exactly the near-collinear directions the
estimator is meant to stabilize — this raw factor goes negative and **flips
the sign** of an already-unstable OLS coefficient rather than suppressing it.
In initial runs this produced visibly worse RMSE than plain ridge regression,
which should never happen for a correctly-tuned ridge-type method.

The fix implemented here (`kibria_lukman_fit` in `src/estimators.py`) clips
the shrinkage factor to `[0, 1]`, a standard non-negative-shrinkage
correction: weak or collinear directions are shrunk toward zero instead of
sign-flipped. The shrinkage parameter `k` itself is chosen by k-fold
cross-validation over a grid scaled to the largest eigenvalue of `X'X`,
rather than via the closed-form plug-in formula in the original paper, which
assumes an overdetermined, well-conditioned OLS fit and is unreliable once
the robustly-screened sample size is only modestly larger than the number of
predictors (exactly the regime this project operates in).

## Results

Simulations use a 70-observation, 35-predictor design built from a 5-factor
low-rank model (strong multicollinearity by construction, in the spirit of
correlated descriptor blocks in QSAR-type data), with a controllable fraction
of y-outliers. Full code: `src/run_experiment.py`. Raw results:
`results/*.csv`, plots: `results/*.png`.

**Sparse-truth regime** (10 of 35 coefficients active — the regime closest to
the thesis's own QSAR evaluation): SLTS+LASSO outperforms SLTS+KLE at nearly
every contamination level, as expected — LASSO has a structural advantage
when the truth really is sparse, since ridge-type shrinkage never zeroes out
irrelevant coefficients.

**Dense-truth regime** (28 of 35 coefficients active, smaller and more
homogeneous effects — the regime ridge-type estimators are built for): the
two hybrids are comparable at low contamination, and **SLTS+KLE overtakes
SLTS+LASSO at the higher contamination levels** (mean coefficient RMSE 5.70
vs. 6.19 at 20% contamination; 6.75 vs. 8.37 at 25% contamination).

| Contamination | SLTS+LASSO (dense) | SLTS+KLE (dense) |
|---:|---:|---:|
| 0.00 | 0.95 | 1.17 |
| 0.10 | 1.62 | 1.78 |
| 0.20 | 6.19 | 5.70 |
| 0.25 | 8.37 | 6.75 |

This is the expected and honest pattern, not a result engineered to favor
either method: which second-stage estimator wins depends on whether the
underlying signal is sparse or dense, which is a standard and well-documented
trade-off between L1 and ridge-type shrinkage. Reporting both regimes, rather
than only the one that favors this project's own contribution, is
intentional.

## Connection to proposed endeavor

Industrial sensor and manufacturing-process data (e.g. cutting-stock
material-usage telemetry) is rarely sparse in the way the original QSAR
benchmark was — many correlated process variables typically contribute jointly
to an outcome like material waste or equipment wear, which is closer to the
dense-truth regime above. This project is a direct, tested step toward
extending the thesis's robust-regularized framework into that regime, and
feeds the companion project,
[`robust-cutting-stock`](../robust-cutting-stock), which uses this kind of
robust estimator to drive an actual cutting-stock optimization pipeline.

## Reproducing

```bash
pip install numpy scipy scikit-learn pandas matplotlib
cd src
python3 run_experiment.py
```

## Files

```
src/estimators.py        Core estimator implementations (SLTS, Kibria-Lukman, hybrids)
src/run_experiment.py    Simulation study: sparse and dense regimes, 6 contamination levels
results/                 CSV results and plots from the runs reported above
```

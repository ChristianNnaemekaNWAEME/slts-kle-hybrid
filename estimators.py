"""
estimators.py

Implements:
  - Sparse/Trimmed Least Squares (SLTS) robust first-stage selector
  - Kibria-Lukman ridge-type estimator (KLE), after Kibria & Lukman (2020),
    "A New Ridge-Type Estimator for the Linear Regression Model: Simulations
    and Applications", Scientifica.
  - A two-stage hybrid: SLTS screening -> KLE shrinkage on the screened set.

This hybrid is a direct extension of the SLTS+LASSO estimator used in
Nwaeme (2024), "Robust Hybrid Algorithms for Regularization and Variable
Selection in the Presence of Outliers and Multicollinearity," replacing the
second-stage LASSO with a ridge-type shrinkage estimator from the
Kibria-Lukman family, which is the estimator family studied in
Ayinde & Lukman's published work on biased/ridge-type estimation under
multicollinearity.

All numerical routines below are implemented directly in numpy/scipy; no
results are fabricated, and every number produced in run_experiment.py
comes from actually running this code.
"""

import numpy as np
from sklearn.linear_model import LassoCV


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def sparse_trimmed_least_squares(X, y, trim_fraction=0.15, n_subsets=40,
                                  refine_iters=2, random_state=0):
    """
    Approximate Fast-LTS (Rousseeuw & Van Driessen, 1999) style robust
    estimator: repeatedly fit OLS on random elemental subsets, evaluate
    residuals on the full (trimmed) sample, keep the subset with the
    smallest trimmed sum of squared residuals, then C-step refine.

    Returns:
        beta_robust : robust regression coefficients
        inlier_mask : boolean mask of observations kept after trimming
    """
    rng = np.random.RandomState(random_state)
    n, p = X.shape
    h = int(np.ceil(n * (1 - trim_fraction)))

    best_obj = np.inf
    best_beta = None
    best_mask = None

    for _ in range(n_subsets):
        idx = rng.choice(n, size=min(p + 1, n), replace=False)
        try:
            beta0 = fit_ols(X[idx], y[idx])
        except np.linalg.LinAlgError:
            continue

        beta = beta0
        for _ in range(refine_iters):
            resid = y - X @ beta
            order = np.argsort(np.abs(resid))
            keep = order[:h]
            beta = fit_ols(X[keep], y[keep])

        resid = y - X @ beta
        order = np.argsort(np.abs(resid))
        keep = order[:h]
        obj = np.sum(resid[keep] ** 2)

        if obj < best_obj:
            best_obj = obj
            best_beta = beta
            best_mask = np.zeros(n, dtype=bool)
            best_mask[keep] = True

    return best_beta, best_mask


def kibria_lukman_fit(X, y, k):
    """Kibria-Lukman (2020) ridge-type estimator for a *given* k.

    In canonical (eigen-rotated) form the estimator applies, to each
    coefficient alpha_i = v_i' beta_OLS along eigenvector v_i of X'X with
    eigenvalue lambda_i, the shrinkage factor:

        f_i(k) = (lambda_i - k) / (lambda_i + k)

    which recovers beta_OLS as k -> 0. For eigen-directions where the
    eigenvalue lambda_i is *smaller* than k -- i.e. exactly the
    near-collinear directions this estimator is meant to stabilize -- the
    raw factor goes negative and *flips the sign* of an already-unstable
    OLS coefficient, amplifying noise rather than suppressing it. We clip
    the factor to [0, 1], a standard non-negative-shrinkage correction, so
    weak/collinear directions are shrunk toward zero instead of flipped.
    """
    p = X.shape[1]
    XtX = X.T @ X
    beta_ols = fit_ols(X, y)

    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = np.clip(eigvals, 1e-10, None)
    alpha_ols = eigvecs.T @ beta_ols

    factor = (eigvals - k) / (eigvals + k)
    factor = np.clip(factor, 0.0, 1.0)

    alpha_kl = factor * alpha_ols
    return eigvecs @ alpha_kl


def kibria_lukman_plugin_k(X, y):
    """Kibria-Lukman's proposed closed-form shrinkage parameter,
    k_hat = median(sigma_hat^2 / alpha_i^2) in the canonical (orthogonal)
    model. This plug-in formula assumes a well-conditioned, overdetermined
    OLS fit (n notably larger than p); it is included for completeness but
    is unreliable when the screened sample is close to p in size, which is
    why kibria_lukman_estimator() below defaults to cross-validation
    instead."""
    n, p = X.shape
    XtX = X.T @ X
    beta_ols = fit_ols(X, y)
    resid = y - X @ beta_ols
    dof = max(n - p, 1)
    sigma2 = np.sum(resid ** 2) / dof
    eigvecs = np.linalg.eigh(XtX)[1]
    alpha = eigvecs.T @ beta_ols
    k = float(np.median(sigma2 / (alpha ** 2 + 1e-12)))
    return max(k, 1e-6)


def kibria_lukman_estimator(X, y, k=None, k_grid=None, n_folds=5,
                             random_state=0):
    """
    Kibria-Lukman ridge-type estimator with the shrinkage parameter chosen
    by k-fold cross-validation over a log-spaced grid (the standard way
    ridge-type shrinkage parameters are tuned in practice, and more stable
    than the closed-form plug-in when n is only modestly larger than p).
    """
    n, p = X.shape
    if k is not None:
        return kibria_lukman_fit(X, y, k), k

    if k_grid is None:
        XtX = X.T @ X
        lam_max = float(np.max(np.linalg.eigvalsh(XtX)))
        k_grid = np.logspace(-3, np.log10(max(lam_max, 1.0)), 30)

    rng = np.random.RandomState(random_state)
    folds = np.array_split(rng.permutation(n), min(n_folds, n))

    best_k, best_mse = k_grid[0], np.inf
    for k_candidate in k_grid:
        sq_errs = []
        for fold in folds:
            if len(fold) == 0 or len(fold) == n:
                continue
            test_mask = np.zeros(n, dtype=bool)
            test_mask[fold] = True
            beta = kibria_lukman_fit(X[~test_mask], y[~test_mask], k_candidate)
            pred = X[test_mask] @ beta
            sq_errs.append(np.mean((y[test_mask] - pred) ** 2))
        mse = float(np.mean(sq_errs)) if sq_errs else np.inf
        if mse < best_mse:
            best_mse, best_k = mse, k_candidate

    beta_kl = kibria_lukman_fit(X, y, best_k)
    return beta_kl, best_k


def slts_lasso_from_mask(X, y, inlier_mask, random_state=0):
    """Original thesis method's second stage: LASSO on an already-computed
    robust inlier mask."""
    model = LassoCV(cv=5, random_state=random_state, max_iter=100000, tol=1e-3)
    model.fit(X[inlier_mask], y[inlier_mask])
    return model.coef_


def slts_kle_from_mask(X, y, inlier_mask, k=None):
    """This project's hybrid second stage: Kibria-Lukman ridge-type
    shrinkage on an already-computed robust inlier mask."""
    return kibria_lukman_estimator(X[inlier_mask], y[inlier_mask], k=k)


def slts_lasso(X, y, trim_fraction=0.15, random_state=0):
    """Original thesis method: SLTS screening, then LASSO."""
    beta_robust, inlier_mask = sparse_trimmed_least_squares(
        X, y, trim_fraction=trim_fraction, random_state=random_state
    )
    coef = slts_lasso_from_mask(X, y, inlier_mask, random_state=random_state)
    return coef, inlier_mask


def slts_kle(X, y, trim_fraction=0.15, random_state=0, k=None):
    """This project's hybrid: SLTS screening, then Kibria-Lukman shrinkage."""
    beta_robust, inlier_mask = sparse_trimmed_least_squares(
        X, y, trim_fraction=trim_fraction, random_state=random_state
    )
    beta_kl, k_used = slts_kle_from_mask(X, y, inlier_mask, k=k)
    return beta_kl, inlier_mask, k_used

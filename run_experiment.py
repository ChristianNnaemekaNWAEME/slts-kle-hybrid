"""
run_experiment.py

Simulation study comparing OLS, Ridge, LASSO, SLTS+LASSO (Nwaeme's thesis
method), and SLTS+KLE (this project's hybrid, using the Kibria-Lukman
ridge-type estimator) under controlled multicollinearity and outlier
contamination -- the two conditions both Nwaeme's thesis and Ayinde's own
ridge-type estimator research separately address.

Run with:  python3 src/run_experiment.py
Outputs:   results/rmse_vs_contamination.csv
           results/rmse_vs_contamination.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV

from estimators import (
    sparse_trimmed_least_squares, slts_lasso_from_mask, slts_kle_from_mask, fit_ols
)

RANDOM_STATE = 42


def make_multicollinear_data(n=70, p=35, n_factors=5, noise_sd=1.0,
                              contamination=0.0, outlier_scale=15.0,
                              n_active=10, coef_scale=(1.5, 3.0),
                              random_state=0):
    """Simulate a high-dimensional, strongly multicollinear design (via a
    low-rank factor model, similar in spirit to correlated molecular
    descriptor blocks in QSAR data), with a controllable true coefficient
    vector (sparse when n_active is small, dense when n_active is close to
    p), and a controllable fraction of y-outliers.

    Sparse truth (n_active small) favors LASSO-type second stages, since
    they can zero out irrelevant predictors. Dense truth (n_active close
    to p) favors ridge-type shrinkage, which never zeroes coefficients but
    stabilizes all of them jointly -- this is the regime Kibria-Lukman-type
    estimators are actually designed for, so both regimes are reported
    rather than only the one that favors the new hybrid."""
    rng = np.random.RandomState(random_state)

    factors = rng.normal(size=(n, n_factors))
    loadings = rng.normal(size=(n_factors, p))
    X = factors @ loadings + 0.3 * rng.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)

    beta_true = np.zeros(p)
    active = rng.choice(p, size=n_active, replace=False)
    beta_true[active] = rng.uniform(*coef_scale, size=n_active) * rng.choice([-1, 1], size=n_active)

    y = X @ beta_true + noise_sd * rng.normal(size=n)

    n_out = int(np.round(contamination * n))
    if n_out > 0:
        out_idx = rng.choice(n, size=n_out, replace=False)
        y[out_idx] += outlier_scale * rng.choice([-1, 1], size=n_out)

    return X, y, beta_true


def rmse(beta_hat, beta_true):
    return float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))


def run_regime(regime_name, n_active, coef_scale):
    contamination_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    n_repeats = 5
    rows = []

    for c in contamination_levels:
        for rep in range(n_repeats):
            seed = 1000 * int(c * 100) + rep + (500 if regime_name == "dense" else 0)
            X, y, beta_true = make_multicollinear_data(
                contamination=c, n_active=n_active, coef_scale=coef_scale,
                random_state=seed,
            )

            beta_ols = fit_ols(X, y)

            ridge = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5).fit(X, y)
            beta_ridge = ridge.coef_

            lasso = LassoCV(cv=5, random_state=seed, max_iter=100000, tol=1e-3).fit(X, y)
            beta_lasso = lasso.coef_

            # Robust screening computed once, shared by both hybrid estimators
            _, inlier_mask = sparse_trimmed_least_squares(X, y, random_state=seed)
            beta_slts_lasso = slts_lasso_from_mask(X, y, inlier_mask, random_state=seed)
            beta_slts_kle, k_used = slts_kle_from_mask(X, y, inlier_mask)

            rows.append({
                "contamination": c, "rep": rep,
                "OLS": rmse(beta_ols, beta_true),
                "Ridge": rmse(beta_ridge, beta_true),
                "LASSO": rmse(beta_lasso, beta_true),
                "SLTS+LASSO": rmse(beta_slts_lasso, beta_true),
                "SLTS+KLE": rmse(beta_slts_kle, beta_true),
                "k_used": k_used,
            })
            print(f"[{regime_name}] contamination={c:.2f} rep={rep} done (k={k_used:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(f"results/rmse_raw_{regime_name}.csv", index=False)

    summary = df.groupby("contamination")[
        ["OLS", "Ridge", "LASSO", "SLTS+LASSO", "SLTS+KLE"]
    ].mean()
    summary.to_csv(f"results/rmse_vs_contamination_{regime_name}.csv")
    print(f"\n=== [{regime_name}] Mean RMSE by contamination level ===")
    print(summary.round(4))

    plt.figure(figsize=(7, 5))
    for col in ["OLS", "Ridge", "LASSO", "SLTS+LASSO", "SLTS+KLE"]:
        plt.plot(summary.index, summary[col], marker="o", label=col)
    plt.xlabel("Outlier contamination fraction")
    plt.ylabel("Coefficient RMSE (mean over 5 reps)")
    plt.title(f"Robustness under multicollinearity + outliers ({regime_name} truth)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/rmse_vs_contamination_{regime_name}.png", dpi=150)
    print(f"Saved results/rmse_vs_contamination_{regime_name}.png and .csv\n")
    return summary


def main():
    # Regime 1: sparse true coefficients (10 of 35 active) -- the setting
    # Nwaeme's thesis was evaluated in, where LASSO-based second stages
    # have a structural advantage.
    sparse_summary = run_regime("sparse", n_active=10, coef_scale=(1.5, 3.0))

    # Regime 2: dense true coefficients (28 of 35 active, smaller, more
    # homogeneous effects) -- the setting ridge-type / Kibria-Lukman
    # estimators are specifically designed for.
    dense_summary = run_regime("dense", n_active=28, coef_scale=(0.4, 1.2))


if __name__ == "__main__":
    main()

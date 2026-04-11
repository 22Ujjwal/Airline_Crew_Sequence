"""
Statistical significance layer for pair risk scores.

Methods:
  - Wilson confidence interval for observed bad rates (exact binomial CI)
  - Chi-square test: is a pair's bad rate significantly above baseline?
  - FDR correction (Benjamini-Hochberg) for multiple testing
  - Minimum sample size filter (n >= MIN_N)
  - Effect size: risk ratio vs baseline

Only pairs that pass all filters are reported as "significantly high risk".
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

MIN_N   = 30      # minimum sequences to report a pair
ALPHA   = 0.05    # significance level (before FDR correction)


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(successes: np.ndarray, n: np.ndarray, alpha: float = ALPHA):
    """
    Vectorised Wilson score interval for proportions.
    Returns (lower, upper) arrays.
    """
    z     = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (center - margin).clip(0, 1), (center + margin).clip(0, 1)


# ---------------------------------------------------------------------------
# Chi-square test against baseline
# ---------------------------------------------------------------------------

def chisq_vs_baseline(n_bad: int, n_total: int, baseline_rate: float) -> tuple[float, float]:
    """
    Tests H0: pair bad rate == baseline_rate.
    Uses chi-square goodness-of-fit.
    Returns (chi2_stat, p_value).
    """
    if n_total < MIN_N:
        return np.nan, np.nan
    expected_bad  = n_total * baseline_rate
    expected_good = n_total * (1 - baseline_rate)
    observed_bad  = n_bad
    observed_good = n_total - n_bad
    if expected_bad < 5 or expected_good < 5:
        return np.nan, np.nan
    chi2 = (observed_bad - expected_bad)**2 / expected_bad + \
           (observed_good - expected_good)**2 / expected_good
    p = stats.chi2.sf(chi2, df=1)
    return chi2, p


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def fdr_correction(p_values: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns boolean array: True = reject H0 (significant after correction).
    """
    n  = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    # Sort p-values, track original indices
    order    = np.argsort(p_values)
    p_sorted = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    # Find largest k where p_k <= (k/n)*alpha
    below    = p_sorted <= thresholds
    if not below.any():
        reject_sorted = np.zeros(n, dtype=bool)
    else:
        cutoff        = np.where(below)[0].max()
        reject_sorted = np.arange(n) <= cutoff
    # Restore original order
    result              = np.empty(n, dtype=bool)
    result[order]       = reject_sorted
    return result


# ---------------------------------------------------------------------------
# Main significance pipeline
# ---------------------------------------------------------------------------

def compute_significance(pair_scores: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Annotates pair_scores with:
      - ci_lower, ci_upper      : Wilson 95% CI on observed_bad_rate
      - chi2_stat, p_value      : test vs baseline
      - p_value_fdr             : BH-corrected p-value threshold
      - significant             : bool, passes all filters
      - risk_ratio              : observed_bad_rate / baseline_rate
      - effect_size             : risk_ratio - 1 (0 = no difference)
    """
    df = pair_scores.copy()

    # Need n_sequences and observed_bad_rate
    df["n_bad"] = (df["observed_bad_rate"] * df["n_sequences"]).round().astype(int)

    # Wilson CI
    ci_lo, ci_hi = wilson_ci(df["n_bad"].values.astype(float),
                             df["n_sequences"].values.astype(float), alpha)
    df["ci_lower"] = ci_lo
    df["ci_upper"] = ci_hi

    # Baseline: overall bad rate across all pairs
    total_bad = df["n_bad"].sum()
    total_n   = df["n_sequences"].sum()
    baseline  = total_bad / total_n
    df["baseline_rate"] = baseline

    # Chi-square test per pair×month
    chisq_results = [
        chisq_vs_baseline(row.n_bad, row.n_sequences, baseline)
        for row in df.itertuples()
    ]
    df["chi2_stat"] = [r[0] for r in chisq_results]
    df["p_value"]   = [r[1] for r in chisq_results]

    # FDR correction on valid (non-NaN) p-values
    valid_mask = df["p_value"].notna()
    df["significant_fdr"] = False
    if valid_mask.any():
        p_valid = df.loc[valid_mask, "p_value"].values
        df.loc[valid_mask, "significant_fdr"] = fdr_correction(p_valid, alpha)

    # Effect size
    df["risk_ratio"]  = df["observed_bad_rate"] / baseline
    df["effect_size"] = df["risk_ratio"] - 1.0

    # Final significance flag: passes all criteria
    df["significant"] = (
        (df["n_sequences"] >= MIN_N) &           # sufficient sample size
        df["significant_fdr"] &                   # survives FDR correction
        (df["observed_bad_rate"] > baseline) &    # actually worse than baseline
        (df["ci_lower"] > baseline)               # CI lower bound above baseline
    )

    print(f"Baseline bad rate:       {baseline:.3f} ({baseline:.1%})")
    print(f"Pairs tested:            {len(df):,}")
    print(f"Pairs with n >= {MIN_N}:    {(df['n_sequences'] >= MIN_N).sum():,}")
    print(f"Significant (FDR-adj):   {df['significant'].sum():,}")
    print(f"Significant high-risk:   {(df['significant'] & (df['risk_ratio'] > 1)).sum():,}")

    return df


def top_significant_pairs(pair_scores_with_sig: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Return top n statistically significant high-risk pairs."""
    sig = pair_scores_with_sig[
        pair_scores_with_sig["significant"] &
        (pair_scores_with_sig["risk_ratio"] > 1)
    ].copy()
    return (
        sig.groupby(["airport_A", "airport_B"])
        .agg(
            avg_risk_score     = ("avg_risk_score",    "mean"),
            avg_bad_rate       = ("observed_bad_rate", "mean"),
            avg_ci_lower       = ("ci_lower",          "mean"),
            avg_ci_upper       = ("ci_upper",          "mean"),
            risk_ratio         = ("risk_ratio",        "mean"),
            n_significant_months = ("significant",     "sum"),
            total_sequences    = ("n_sequences",       "sum"),
        )
        .sort_values("avg_risk_score", ascending=False)
        .head(n)
        .reset_index()
    )

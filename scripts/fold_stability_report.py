"""
Fold-Level Reporting for the Thesis Revision
=============================================
Reads a svm_results_<timestamp>.json produced by the PATCHED
svm_classifier.py (which now persists fold_results, not just the
aggregated summary) and produces exactly the numbers the supervisor's
comments #19-21, #41, and #57-58 ask for:

  1. Per-fold retained-feature counts for every Path B method
     (comments #19, #20, #21).
  2. A 95% confidence interval for each metric on GSE42568, computed
     from the 5 real fold values rather than just mean +/- std
     (comment #41).
  3. Cross-fold feature-selection stability (mean pairwise Jaccard
     overlap of the selected-feature index sets across the 5 folds)
     for every Path B method (comments #57, #58).

Usage
-----
    python fold_stability_report.py path/to/svm_results_<timestamp>.json

Writes two CSVs next to the input file:
    <input>_fold_feature_counts.csv
    <input>_stability_and_ci.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd


def _t_multiplier_95(n: int) -> float:
    """
    Approximate two-sided 95% t multiplier for small n (no scipy
    dependency required). Falls back to the standard table for n<=30;
    good enough for the n=5 fold case used throughout this thesis.
    """
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }
    return table.get(n, 1.96)


def compute_ci(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for a 95% CI over `values`."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    t_mult = _t_multiplier_95(n)
    margin = t_mult * se
    return mean, mean - margin, mean + margin


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mean_pairwise_jaccard(fold_index_sets: list[set]) -> float:
    pairs = list(itertools.combinations(range(len(fold_index_sets)), 2))
    if not pairs:
        return float("nan")
    scores = [jaccard(fold_index_sets[i], fold_index_sets[j]) for i, j in pairs]
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path, help="svm_results_<timestamp>.json from the patched svm_classifier.py")
    args = parser.parse_args()

    with open(args.results_json) as fh:
        results = json.load(fh)

    feature_count_rows = []
    stability_ci_rows = []

    for path_name, data in results.items():
        fold_results = data.get("fold_results", [])
        if not fold_results:
            print(f"[skip] {path_name}: no fold_results present "
                  f"(re-run with the patched svm_classifier.py to populate this)")
            continue

        # --- Per-fold feature counts (comments #19, #20, #21) ---
        n_features_per_fold = [fr.get("n_features") for fr in fold_results if "n_features" in fr]
        if n_features_per_fold:
            row = {"path": path_name}
            for i, n in enumerate(n_features_per_fold, 1):
                row[f"fold_{i}_n_features"] = n
            row["mean"] = sum(n_features_per_fold) / len(n_features_per_fold)
            row["std"] = (
                (sum((n - row["mean"]) ** 2 for n in n_features_per_fold) / max(1, len(n_features_per_fold) - 1)) ** 0.5
                if len(n_features_per_fold) > 1 else 0.0
            )
            feature_count_rows.append(row)

        # --- 95% CIs for each metric, from the actual fold values (comment #41) ---
        metric_keys = ["accuracy", "specificity", "sensitivity", "mcc", "roc_auc"]
        ci_row = {"path": path_name}
        for key in metric_keys:
            vals = [fr[key] for fr in fold_results if fr.get(key) is not None]
            if len(vals) == len(fold_results) and vals:
                mean, lo, hi = compute_ci(vals)
                ci_row[f"{key}_mean"] = round(mean, 4)
                ci_row[f"{key}_95ci_low"] = round(lo, 4)
                ci_row[f"{key}_95ci_high"] = round(hi, 4)

        # --- Cross-fold feature-selection stability (comments #57, #58) ---
        index_sets = [
            set(fr["selected_feature_indices"])
            for fr in fold_results
            if "selected_feature_indices" in fr
        ]
        if len(index_sets) == len(fold_results) and len(index_sets) > 1:
            ci_row["mean_pairwise_jaccard_stability"] = round(mean_pairwise_jaccard(index_sets), 4)

        stability_ci_rows.append(ci_row)

    out_base = args.results_json.with_suffix("")
    if feature_count_rows:
        fc_path = Path(f"{out_base}_fold_feature_counts.csv")
        pd.DataFrame(feature_count_rows).to_csv(fc_path, index=False)
        print(f"Wrote {fc_path}")
    else:
        print("No per-fold feature counts found (nothing to write).")

    if stability_ci_rows:
        st_path = Path(f"{out_base}_stability_and_ci.csv")
        pd.DataFrame(stability_ci_rows).to_csv(st_path, index=False)
        print(f"Wrote {st_path}")
    else:
        print("No stability/CI rows found (nothing to write).")


if __name__ == "__main__":
    main()
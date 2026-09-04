#!/usr/bin/env python
"""Regenerate the GSE42568 Welch t-test extension table from fresh CV runs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.svm_classifier import SVMClassifierWithCV


def summary_row(summary: dict[str, float]) -> dict[str, str]:
    fields = ("accuracy", "recall", "specificity", "f1", "mcc", "roc_auc")
    return {
        field: f"{summary[f'{field}_mean']:.4f} ± {summary[f'{field}_std']:.4f}"
        for field in fields
    }


def run_variant(classifier: SVMClassifierWithCV, label: str, **kwargs) -> dict:
    classifier.train_path_b_optimized(
        feature_method="filter_ttest",
        n_features=20,
        p_value=0.05,
        **kwargs,
    )

    if kwargs.get("tune_threshold"):
        key = "path_b_filter_ttest_threshold_tuned"
    elif kwargs.get("tune_k_candidates") is not None:
        key = "path_b_filter_ttest_k_tuned"
    elif kwargs.get("tune_c"):
        key = "path_b_filter_ttest_c_tuned"
    elif kwargs.get("alternative_classifier"):
        key = f"path_b_filter_ttest_{kwargs['alternative_classifier']}"
    else:
        key = "path_b_filter_ttest"

    result = classifier.results[key]
    row = summary_row(result["summary"])
    if label in {"baseline", "threshold_tuned", "logreg"}:
        row["fold_audit"] = [
            {
                "fold": fold_number,
                "tn": int(fold["tn"]),
                "fp": int(fold["fp"]),
                "fn": int(fold["fn"]),
                "tp": int(fold["tp"]),
                **(
                    {"youden_threshold": float(fold["youden_threshold"])}
                    if "youden_threshold" in fold
                    else {}
                ),
            }
            for fold_number, fold in enumerate(result["fold_results"], 1)
        ]
    if kwargs.get("tune_k_candidates") is not None:
        row["selected_k_per_fold"] = [
            int(fold["selected_k"]) for fold in result["fold_results"]
        ]
    print(f"{label}: {json.dumps(row)}")
    return row


def main() -> None:
    classifier = SVMClassifierWithCV(
        dataset_name="GSE42568",
        n_splits=5,
        random_state=42,
        n_features=20,
        p_value=0.05,
    )
    variants = [
        ("baseline", dict()),
        ("smote", dict(apply_smote=True)),
        ("threshold_tuned", dict(tune_threshold=True)),
        ("c_tuned", dict(tune_c=True)),
        ("logreg", dict(alternative_classifier="logistic_regression")),
        ("rf", dict(alternative_classifier="random_forest")),
        ("k_tuned", dict(tune_k_candidates=[10, 20, 50, 100])),
    ]

    results = {
        label: run_variant(classifier, label, **kwargs)
        for label, kwargs in variants
    }
    output_path = Path("results") / "GSE42568_breast_cancer" / "table41_regenerated.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n=== LaTeX rows ===")
    labels = {
        "baseline": "Baseline (default threshold, $k=20$)",
        "smote": "SMOTE inside CV folds",
        "threshold_tuned": "Threshold tuned (Youden $J$)",
        "c_tuned": "$C$ tuned by inner CV",
        "logreg": "Logistic regression check",
        "rf": "Random-forest check",
        "k_tuned": "$k$ tuned by inner CV",
    }
    for label, row in results.items():
        print(
            f"{labels[label]} & {row['accuracy']} & {row['recall']} & "
            f"{row['specificity']} & {row['f1']} & {row['mcc']} & "
            f"{row['roc_auc']} " + r"\\\\"
        )


if __name__ == "__main__":
    main()

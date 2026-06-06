"""
SVM Classifier with Nested Cross-Validation
=============================================
Implements dual-track training:
  Path A — Baseline: SVM on all features (no feature selection)
  Path B — Optimised: SVM with feature selection inside each training fold

Feature selection is applied INSIDE each training fold to prevent data leakage.

Fix log (vs original)
---------------------
- Removed stale "path_b": {} from self.results __init__ dict (Issue 5).
  train_path_b_optimized writes to "path_b_{method}" keys; the orphaned
  "path_b" key was polluting JSON/CSV output with an empty entry.
- DATASET_INFO is no longer built at module import time (Issue 7).
  build_dataset_info() is now called lazily inside SVMClassifierWithCV.__init__
  so importing this module does not touch the filesystem or call cwd().
- _setup_logging() no longer clears root handlers (Issue 2).
  It now attaches handlers only to the module-level logger, leaving the
  root logger intact so feature_selection and preprocessing log normally.
- run_full_pipeline() now reads the method list from
  FeatureSelectionPipeline.ALL_METHODS instead of a hardcoded list (Issue 3),
  so adding/renaming a method in feature_selection.py propagates automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# Ensure sibling scripts are importable regardless of working directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from feature_selection import FeatureSelector, FeatureSelectionPipeline
from preprocessing import GenomicDataProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata helper — called lazily, not at import time
# ---------------------------------------------------------------------------
def build_dataset_info(project_dir: Optional[str] = None) -> dict:
    """
    Build dataset display/folder metadata from GenomicDataProcessor.DATASET_CONFIG.

    Called lazily inside SVMClassifierWithCV.__init__ so that importing this
    module never touches the filesystem or calls Path.cwd().

    Parameters
    ----------
    project_dir : str, optional
        Passed to GenomicDataProcessor so it uses the correct project root.
    """
    processor = GenomicDataProcessor(project_dir=project_dir or "")
    dataset_info: dict = {}

    for dataset_name, config in processor.DATASET_CONFIG.items():
        total = config["n_cancer"] + config["n_normal"]
        balance = "balanced" if config["n_cancer"] == config["n_normal"] else "imbalanced"
        cancer_type = config.get("cancer_type", "Unknown Cancer")

        dataset_info[dataset_name] = {
            "name": cancer_type,
            "description": (
                f"{dataset_name} - {cancer_type} "
                f"({balance}: {config['n_cancer']}/{config['n_normal']}, {total} total)"
            ),
            "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        }

    return dataset_info


# ---------------------------------------------------------------------------
# SVMClassifierWithCV
# ---------------------------------------------------------------------------
class SVMClassifierWithCV:
    """SVM with nested cross-validation for robust evaluation."""

    def __init__(
        self,
        dataset_name: str = "GSE19804",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.random_state = random_state

        self.base_dir = Path(__file__).resolve().parent.parent

        # Build metadata lazily here instead of at module import time,
        # so the project root is always resolved from __file__, not cwd().
        dataset_info = build_dataset_info(str(self.base_dir))
        info = dataset_info.get(dataset_name, {})
        self.dataset_folder = info.get("folder", dataset_name)
        self.dataset_info = dataset_info  # retained for run_full_pipeline header

        self.results_dir = self.base_dir / "results" / self.dataset_folder
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self._load_data()

        # Only Path A is pre-allocated; Path B keys are created dynamically
        # by train_path_b_optimized() as "path_b_{method}".
        self.results: dict = {"path_a": {}}

    # ------------------------------------------------------------------
    # Logging — attaches to the module logger only, not the root logger,
    # so other modules' loggers are unaffected.
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        log_file = self.results_dir / "svm_training.log"

        # Remove only this logger's own handlers before adding new ones
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        # Prevent messages bubbling up to the root logger and being double-printed
        logger.propagate = False

        logger.info(f"Logging initialised → {log_file}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        logger.info("=" * 70)
        logger.info("LOADING GENOMIC EXPERIMENT DATA")
        logger.info("=" * 70)

        processor = GenomicDataProcessor(self.dataset_name, str(self.base_dir))

        try:
            self.X, self.y = processor.load_preprocessed_data()
            logger.info("Loaded from preprocessed cache.")
        except Exception as exc:
            logger.warning(
                f"Cache load failed ({exc}). Running full preprocessing pipeline …"
            )
            self.X, self.y = processor.preprocess_complete()
            processor.save_preprocessed_data()

        logger.info(f"Data shape: {self.X.shape}")
        logger.info(f"Class distribution: {np.bincount(self.y).tolist()}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute a comprehensive set of binary classification metrics."""
        metrics: dict = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "mcc":       matthews_corrcoef(y_true, y_pred),
        }

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception:
                metrics["roc_auc"] = None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        })

        return metrics

    # ------------------------------------------------------------------
    # Path A — Baseline (no feature selection)
    # ------------------------------------------------------------------
    def train_path_a_baseline(self) -> None:
        """Evaluate SVM on the full feature set as a baseline."""
        logger.info("\n" + "=" * 70)
        logger.info("PATH A: BASELINE — All features (no feature selection)")
        logger.info("=" * 70)

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        fold_results = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y), 1):
            logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            X_train = self.X.iloc[train_idx]
            X_test  = self.X.iloc[test_idx]
            y_train = self.y[train_idx]
            y_test  = self.y[test_idx]

            svm = SVC(
                kernel="linear", C=1.0,
                random_state=self.random_state,
                class_weight="balanced",
                probability=True,
            )
            svm.fit(X_train, y_train)

            y_pred       = svm.predict(X_test)
            y_pred_proba = svm.predict_proba(X_test)

            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)
            metrics["n_features"] = self.X.shape[1]
            fold_results.append(metrics)

            logger.info(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"MCC: {metrics['mcc']:.4f} | F1: {metrics['f1']:.4f}"
            )

        self.results["path_a"]["fold_results"] = fold_results
        self._aggregate_cv_results(fold_results, "path_a")

        logger.info("\n" + "-" * 70)
        logger.info("BASELINE SUMMARY")
        logger.info("-" * 70)
        self._print_cv_summary("path_a")

    # ------------------------------------------------------------------
    # Path B — Optimised (feature selection inside each fold)
    # ------------------------------------------------------------------
    def train_path_b_optimized(self, feature_method: str = "filter_ttest") -> None:
        """
        Evaluate SVM with feature selection applied strictly inside each
        training fold to prevent data leakage.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"PATH B: OPTIMISED — Feature selection: {feature_method}")
        logger.info("=" * 70)

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        fold_results = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y), 1):
            logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            X_train = self.X.iloc[train_idx]
            X_test  = self.X.iloc[test_idx]
            y_train = self.y[train_idx]
            y_test  = self.y[test_idx]

            # Feature selection fitted ONLY on the training fold
            selector = FeatureSelector(method=feature_method, n_features=20)
            X_train_sel = selector.fit_transform(X_train.values, y_train)
            X_test_sel  = X_test.iloc[:, selector.selected_features].values

            logger.info(f"  Reduced to {X_train_sel.shape[1]} features")

            svm = SVC(
                kernel="linear", C=1.0,
                random_state=self.random_state,
                class_weight="balanced",
                probability=True,
            )
            svm.fit(X_train_sel, y_train)

            y_pred       = svm.predict(X_test_sel)
            y_pred_proba = svm.predict_proba(X_test_sel)

            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)
            metrics["n_features"]     = len(selector.selected_features)
            metrics["feature_method"] = feature_method
            fold_results.append(metrics)

            logger.info(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"MCC: {metrics['mcc']:.4f} | F1: {metrics['f1']:.4f}"
            )

        # Key pattern: "path_b_{method}" — no orphan "path_b" key
        key = f"path_b_{feature_method}"
        self.results[key] = {"fold_results": fold_results}
        self._aggregate_cv_results(fold_results, key)

        logger.info("\n" + "-" * 70)
        logger.info(f"OPTIMISED SUMMARY ({feature_method})")
        logger.info("-" * 70)
        self._print_cv_summary(key)

    # ------------------------------------------------------------------
    # Aggregation & display
    # ------------------------------------------------------------------
    def _aggregate_cv_results(self, fold_results: list[dict], path_name: str) -> None:
        metrics_df = pd.DataFrame(fold_results)
        summary: dict = {}
        skip_cols = {"n_features", "feature_method", "tn", "fp", "fn", "tp"}

        for col in metrics_df.columns:
            if col not in skip_cols:
                summary[f"{col}_mean"] = float(metrics_df[col].mean())
                summary[f"{col}_std"]  = float(metrics_df[col].std())

        self.results[path_name]["summary"] = summary

    def _print_cv_summary(self, path_name: str) -> None:
        s = self.results[path_name]["summary"]
        logger.info(f"  Accuracy  : {s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}")
        logger.info(f"  Precision : {s['precision_mean']:.4f} ± {s['precision_std']:.4f}")
        logger.info(f"  Recall    : {s['recall_mean']:.4f} ± {s['recall_std']:.4f}")
        logger.info(f"  F1-Score  : {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")
        logger.info(f"  MCC       : {s['mcc_mean']:.4f} ± {s['mcc_std']:.4f}")
        roc_mean = s.get("roc_auc_mean")
        roc_std  = s.get("roc_auc_std")
        if roc_mean is not None and roc_std is not None:
            logger.info(f"  ROC-AUC   : {roc_mean:.4f} ± {roc_std:.4f}")
        else:
            logger.info("  ROC-AUC   : N/A")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def compare_paths(self) -> dict:
        """Log and return the relative improvement of each Path B vs Path A."""
        logger.info("\n" + "=" * 70)
        logger.info("PATH A vs PATH B COMPARISON")
        logger.info("=" * 70)

        comparison: dict = {}
        summary_a = self.results["path_a"].get("summary", {})

        for key, value_b in self.results.items():
            if not key.startswith("path_b_") or "summary" not in value_b:
                continue
            summary_b = value_b["summary"]

            acc_a = summary_a.get("accuracy_mean", 0)
            mcc_a = summary_a.get("mcc_mean", 0)
            f1_a  = summary_a.get("f1_mean", 0)
            acc_b = summary_b.get("accuracy_mean", 0)
            mcc_b = summary_b.get("mcc_mean", 0)
            f1_b  = summary_b.get("f1_mean", 0)

            improvement = {
                "accuracy": ((acc_b - acc_a) / acc_a * 100) if acc_a != 0 else 0.0,
                "mcc":      ((mcc_b - mcc_a) / mcc_a * 100) if mcc_a != 0 else 0.0,
                "f1":       ((f1_b  - f1_a)  / f1_a  * 100) if f1_a  != 0 else 0.0,
            }
            comparison[key] = improvement

            logger.info(f"\n{key}:")
            logger.info(f"  Accuracy shift : {improvement['accuracy']:+.2f}%")
            logger.info(f"  MCC shift      : {improvement['mcc']:+.2f}%")
            logger.info(f"  F1 shift       : {improvement['f1']:+.2f}%")

        return comparison

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self) -> None:
        """Write JSON and CSV summaries to the results directory."""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING RESULTS")
        logger.info("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build JSON — skip any key with an empty dict (shouldn't happen now
        # that the orphan "path_b" key is removed, but defensive)
        results_json = {
            path: {
                "summary": data.get("summary", {}),
                "n_folds": len(data.get("fold_results", [])),
            }
            for path, data in self.results.items()
            if data
        }

        json_path = self.results_dir / f"svm_results_{timestamp}.json"
        with open(json_path, "w") as fh:
            json.dump(results_json, fh, indent=2)
        logger.info(f"JSON written to: {json_path}")

        summaries = []
        for path, data in self.results.items():
            if data and "summary" in data:
                row = data["summary"].copy()
                row["path"] = path
                summaries.append(row)

        csv_path = self.results_dir / f"svm_summary_{timestamp}.csv"
        pd.DataFrame(summaries).to_csv(csv_path, index=False)
        logger.info(f"CSV written to: {csv_path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self) -> None:
        """Run Path A, all Path B methods, compare, and save."""
        info = self.dataset_info.get(self.dataset_name, {})

        logger.info("\n" + "=" * 70)
        logger.info("FULL PIPELINE STARTED")
        logger.info("=" * 70)
        logger.info(f"  Dataset     : {info.get('name', self.dataset_name)}")
        logger.info(f"  Description : {info.get('description', '')}")
        logger.info(f"  Output dir  : {self.results_dir}")
        logger.info("=" * 70)

        self.train_path_a_baseline()

        # Derive method list from FeatureSelectionPipeline so it stays in
        # sync automatically when feature_selection.py is updated.
        for method in FeatureSelectionPipeline.ALL_METHODS:
            try:
                self.train_path_b_optimized(feature_method=method)
            except Exception as exc:
                logger.error(f"Path B ({method}) failed: {exc}")

        self.compare_paths()
        self.save_results()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    classifier = SVMClassifierWithCV(dataset_name="GSE42568", n_splits=5)
    classifier.run_full_pipeline()


if __name__ == "__main__":
    main()

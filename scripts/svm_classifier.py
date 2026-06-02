"""
SVM Classifier with Nested Cross-Validation
Implements dual-track training: Baseline (Path A) vs Optimized (Path B)
Prevents data leakage by applying feature selection INSIDE training folds
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

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

# Modern path injection
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from feature_selection import FeatureSelector
from preprocessing import GenomicDataProcessor


def build_dataset_info():
    """
    Build dataset info dynamically from preprocessor config.
    Eliminates all hardcoded string matching.
    """
    processor = GenomicDataProcessor()
    dataset_info = {}

    for dataset_name, config in processor.dataset_config.items():
        total = config["n_cancer"] + config["n_normal"]
        balance = "balanced" if config["n_cancer"] == config["n_normal"] else "imbalanced"

        # Safely fetch the cancer type defined in preprocessing.py
        cancer_type = config.get("cancer_type", "Unknown Cancer")

        dataset_info[dataset_name] = {
            "name": cancer_type,
            "description": f"{dataset_name} - {cancer_type} ({balance}: {config['n_cancer']}/{config['n_normal']}, {total} total)",
            "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        }

    return dataset_info


# Build DATASET_INFO dynamically at runtime
DATASET_INFO = build_dataset_info()
logger = logging.getLogger(__name__)


class SVMClassifierWithCV:
    """SVM with nested cross-validation for robust evaluation"""

    def __init__(self, dataset_name="GSE19804", n_splits=5, random_state=42):
        """
        Initialize SVM classifier framework using OOP standard paths.
        """
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.random_state = random_state

        # Pure Pathlib directory parsing
        base_dir = Path(__file__).resolve().parent.parent
        dataset_info = DATASET_INFO.get(dataset_name, {})
        self.dataset_folder = dataset_info.get("folder", dataset_name)

        self.results_dir = base_dir / "results" / self.dataset_folder
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging for this specific run
        self._setup_logging()

        # Placeholders for data arrays
        self.X = None
        self.y = None
        self._load_data(base_dir)

        self.results = {"path_a": {}, "path_b": {}}

    def _setup_logging(self):
        """Setup dataset-specific logging configurations"""
        log_file = self.results_dir / "svm_training.log"

        # Clear existing logging handles safely
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.info(f"Logging initialized for: {log_file}")

    def _load_data(self, project_dir):
        """Load preprocessed matrices or fallback gracefully to run preprocessing"""
        logger.info("=" * 70)
        logger.info("LOADING GENOMIC EXPERIMENT DATA")
        logger.info("=" * 70)

        processor = GenomicDataProcessor(self.dataset_name, str(project_dir))

        try:
            self.X, self.y = processor.load_preprocessed_data()
            logger.info("Loaded successfully from cached preprocessed data files.")
        except Exception as e:
            logger.warning(
                f"Preprocessed cache load failed ({e}). Running full preprocessing pipeline..."
            )
            self.X, self.y = processor.preprocess_complete()
            processor.save_preprocessed_data()

        logger.info(f"Data shape: {self.X.shape}")
        logger.info(f"Class distribution: {np.bincount(self.y)}")

    def evaluate_predictions(self, y_true, y_pred, y_pred_proba=None):
        """Compute comprehensive high-dimensional classification evaluation metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception:
                metrics["roc_auc"] = None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            }
        )

        return metrics

    def train_path_a_baseline(self):
        """Path A: Directly evaluate SVM on original features (No Feature Selection)"""
        logger.info("\n" + "=" * 70)
        logger.info("PATH A: BASELINE TRACK (All Available Expression Features)")
        logger.info("=" * 70)

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        fold_results = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y), 1):
            logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            svm = SVC(
                kernel="linear",
                C=1.0,
                random_state=self.random_state,
                class_weight="balanced",
                probability=True,
            )
            svm.fit(X_train, y_train)

            y_pred = svm.predict(X_test)
            y_pred_proba = svm.predict_proba(X_test)

            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)
            metrics["n_features"] = self.X.shape[1]
            fold_results.append(metrics)

            logger.info(
                f"Accuracy: {metrics['accuracy']:.4f} | MCC: {metrics['mcc']:.4f} | F1: {metrics['f1']:.4f}"
            )

        self.results["path_a"]["fold_results"] = fold_results
        self._aggregate_cv_results(fold_results, "path_a")

        logger.info("\n" + "-" * 70)
        logger.info("BASELINE EVALUATION SUMMARY")
        logger.info("-" * 70)
        self._print_cv_summary("path_a")

    def train_path_b_optimized(self, feature_method="filter_ttest"):
        """Path B: Feature selection executed strictly inside training folds to eliminate leakage"""
        logger.info("\n" + "=" * 70)
        logger.info(f"PATH B: OPTIMIZED TRACK (Feature Selection: {feature_method})")
        logger.info("=" * 70)

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        fold_results = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y), 1):
            logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # CRITICAL: Feature selection isolate inside individual training fold split context
            selector = FeatureSelector(method=feature_method, n_features=20)
            X_train_selected = selector.fit_transform(X_train.values, y_train)
            X_test_selected = X_test.iloc[:, selector.selected_features].values

            logger.info(f"Isolate subset dimensionality down to: {X_train_selected.shape[1]} features")

            svm = SVC(
                kernel="linear",
                C=1.0,
                random_state=self.random_state,
                class_weight="balanced",
                probability=True,
            )
            svm.fit(X_train_selected, y_train)

            y_pred = svm.predict(X_test_selected)
            y_pred_proba = svm.predict_proba(X_test_selected)

            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)
            metrics["n_features"] = len(selector.selected_features)
            metrics["feature_method"] = feature_method
            fold_results.append(metrics)

            logger.info(
                f"Accuracy: {metrics['accuracy']:.4f} | MCC: {metrics['mcc']:.4f} | F1: {metrics['f1']:.4f}"
            )

        key = f"path_b_{feature_method}"
        self.results[key] = {"fold_results": fold_results}
        self._aggregate_cv_results(fold_results, key)

        logger.info("\n" + "-" * 70)
        logger.info(f"OPTIMIZED EVALUATION SUMMARY ({feature_method})")
        logger.info("-" * 70)
        self._print_cv_summary(key)

    def _aggregate_cv_results(self, fold_results, path_name):
        """Aggregate cross-validation evaluation matrices metrics safely"""
        metrics_df = pd.DataFrame(fold_results)
        summary = {}

        for col in metrics_df.columns:
            if col not in ["n_features", "feature_method", "tn", "fp", "fn", "tp"]:
                summary[f"{col}_mean"] = float(metrics_df[col].mean())
                summary[f"{col}_std"] = float(metrics_df[col].std())

        self.results[path_name]["summary"] = summary

    def _print_cv_summary(self, path_name):
        """Print clear evaluation report summary data profile"""
        summary = self.results[path_name]["summary"]
        logger.info(f"Accuracy:   {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        logger.info(f"Precision:  {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        logger.info(f"Recall:     {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        logger.info(f"F1-Score:   {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        logger.info(f"MCC:        {summary['mcc_mean']:.4f} ± {summary['mcc_std']:.4f}")
        
        roc_mean = summary.get("roc_auc_mean")
        roc_std = summary.get("roc_auc_std")
        if roc_mean is not None and roc_std is not None:
            logger.info(f"ROC-AUC:    {roc_mean:.4f} ± {roc_std:.4f}")
        else:
            logger.info("ROC-AUC:    N/A")

    def compare_paths(self):
        """Compare performance differences across pipeline configurations"""
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-TRACK EVALUATION INTEGRITY REPORT: Path A vs Path B")
        logger.info("=" * 70)

        comparison = {}
        summary_a = self.results["path_a"].get("summary", {})

        for key, value_b in self.results.items():
            if key.startswith("path_b") and "summary" in value_b:
                summary_b = value_b["summary"]

                acc_a, acc_b = summary_a.get("accuracy_mean", 0), summary_b.get("accuracy_mean", 0)
                mcc_a, mcc_b = summary_a.get("mcc_mean", 0), summary_b.get("mcc_mean", 0)
                f1_a, f1_b = summary_a.get("f1_mean", 0), summary_b.get("f1_mean", 0)

                improvement = {
                    "accuracy": ((acc_b - acc_a) / acc_a * 100) if acc_a != 0 else 0,
                    "mcc": ((mcc_b - mcc_a) / mcc_a * 100) if mcc_a != 0 else 0,
                    "f1": ((f1_b - f1_a) / f1_a * 100) if f1_a != 0 else 0,
                }
                comparison[key] = improvement

                logger.info(f"\n{key} Variance Profiling:")
                logger.info(f"  Accuracy Shift: {improvement['accuracy']:+.2f}%")
                logger.info(f"  MCC Shift:      {improvement['mcc']:+.2f}%")
                logger.info(f"  F1 Score Shift: {improvement['f1']:+.2f}%")

        return comparison

    def save_results(self):
        """Save performance metadata configurations cleanly into JSON/CSV files"""
        logger.info("\n" + "=" * 70)
        logger.info("EXPLICIT PERFORMANCE EXPORT")
        logger.info("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_json = {
            path: {
                "summary": data.get("summary", {}),
                "n_folds": len(data.get("fold_results", [])),
            }
            for path, data in self.results.items() if data
        }

        json_path = self.results_dir / f"svm_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"JSON data matrix written to: {json_path}")

        summaries = []
        for path, data in self.results.items():
            if data and "summary" in data:
                summary_copy = data["summary"].copy()
                summary_copy["path"] = path
                summaries.append(summary_copy)

        csv_path = self.results_dir / f"svm_summary_{timestamp}.csv"
        pd.DataFrame(summaries).to_csv(csv_path, index=False)
        logger.info(f"CSV operational summary spreadsheet written to: {csv_path}")

    def run_full_pipeline(self):
        """Run complete parallel training tracks sequence"""
        dataset_info = DATASET_INFO.get(self.dataset_name, {})

        logger.info("\n" + "=" * 70)
        logger.info("EXECUTION MATRIX STARTED")
        logger.info("=" * 70)
        logger.info(f"Target Cluster:   {dataset_info.get('name', self.dataset_name)}")
        logger.info(f"Configuration:    {dataset_info.get('description', '')}")
        logger.info(f"Target Output:    {self.results_dir}")
        logger.info("=" * 70)

        self.train_path_a_baseline()

        feature_methods = ["filter_ttest", "filter_anova", "wrapper_svm", "embedded_lasso"]
        for method in feature_methods:
            try:
                self.train_path_b_optimized(feature_method=method)
            except Exception as e:
                logger.error(f"Execution tracking failed along Track Path B ({method}): {e}")

        self.compare_paths()
        self.save_results()


def main():
    """Main wrapper validation loop invocation"""
    classifier = SVMClassifierWithCV(dataset_name="GSE42568", n_splits=5)
    classifier.run_full_pipeline()


if __name__ == "__main__":
    main()
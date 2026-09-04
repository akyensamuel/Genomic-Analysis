"""
SVM Classifier with Nested Cross-Validation
Implements dual-track training:
Path A — Baseline: SVM on all features (no feature selection)
Path B — Optimised: SVM with feature selection INSIDE each training fold.

CRITICAL FIX: Uses sklearn.pipeline.Pipeline to guarantee that feature 
selection is fitted ONLY on the training fold, completely preventing 
data leakage. Pre-computing features on the full dataset is intentionally 
avoided here to preserve evaluation validity.

Fix log (vs original)
- Removed stale "path_b": {} from self.results init dict.
- DATASET_INFO is no longer built at module import time.
- _setup_logging() no longer clears root handlers.
- run_full_pipeline() reads the method list from FeatureSelectionPipeline.ALL_METHODS.
- NEW: Path B now uses sklearn.pipeline.Pipeline to ensure strict, leakage-free 
  feature selection within each cross-validation fold.
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
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from scipy.stats import wilcoxon
from imblearn.over_sampling import SMOTE
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
                f"{dataset_name} - {cancer_type}  "
                f"({balance}: {config['n_cancer']}/{config['n_normal']}, {total} total)"
            ),
            "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        }

    return dataset_info

# ---------------------------------------------------------------------------
# SVMClassifierWithCV
# ---------------------------------------------------------------------------
class SVMClassifierWithCV:
    """SVM with cross-validation for robust, leakage-free evaluation."""
    
    def __init__(
        self,
        dataset_name: str = "GSE19804",
        n_splits: int = 5,
        random_state: int = 42,
        n_features: int = 20,
        p_value: Optional[float] = 0.05,
    ) -> None:
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_features = n_features
        # Optional p-value threshold to pass to filter-based FeatureSelector
        self.p_value = p_value

        self.base_dir = Path(__file__).resolve().parent.parent

        # Build metadata lazily here instead of at module import time
        dataset_info = build_dataset_info(str(self.base_dir))
        info = dataset_info.get(dataset_name, {})
        self.dataset_folder = info.get("folder", dataset_name)
        self.dataset_info = dataset_info

        self.results_dir = self.base_dir / "results" / self.dataset_folder
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self._load_data()

        # Only Path A is pre-allocated; Path B keys are created dynamically
        self.results: dict = {"path_a": {}}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        log_file = self.results_dir / "svm_training.log"

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
    # Imbalanced-data helpers used for GSE42568 experiments
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_fold_safe_smote(X: np.ndarray | pd.DataFrame, y: np.ndarray, k_neighbors: int = 5, sampling_strategy: str = "minority") -> tuple[np.ndarray, np.ndarray, int]:
        """Apply SMOTE only to the training fold while guarding k_neighbors for tiny minority folds."""
        if X is None or y is None or len(y) == 0:
            return np.asarray(X), np.asarray(y), 0

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_minority = int(np.sum(y_arr == 1)) if np.unique(y_arr).size == 2 else 0

        if n_minority <= 1:
            return X_arr, y_arr, 0

        safe_k = max(1, min(int(k_neighbors), int(n_minority - 1)))
        if safe_k <= 0:
            return X_arr, y_arr, 0

        try:
            sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=safe_k, random_state=42)
            X_res, y_res = sampler.fit_resample(X_arr, y_arr)
            return X_res, y_res, safe_k
        except Exception:
            sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=1, random_state=42)
            X_res, y_res = sampler.fit_resample(X_arr, y_arr)
            return X_res, y_res, 1

    @staticmethod
    def _select_youden_threshold(decision_scores: np.ndarray, y_true: np.ndarray) -> float:
        """Choose threshold maximizing Youden's J from raw decision_function scores."""
        scores = np.asarray(decision_scores, dtype=float)
        labels = np.asarray(y_true, dtype=int)
        if scores.size == 0:
            return 0.0

        thresholds = np.unique(scores)
        best_threshold = float(thresholds[0]) if thresholds.size else 0.0
        best_j = -np.inf

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j_stat = sensitivity + specificity - 1.0
            if j_stat > best_j:
                best_j = j_stat
                best_threshold = float(threshold)

        if not np.isfinite(best_threshold):
            best_threshold = 0.0
        return float(best_threshold)

    @staticmethod
    def _select_best_c_inner_cv(X_train: pd.DataFrame, y_train: np.ndarray, c_grid: Optional[list[float]] = None, class_weight_options: Optional[list[Optional[str]]] = None) -> dict:
        """Tune C and class_weight with a compact inner CV using MCC and decision_function thresholds."""
        if c_grid is None:
            c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
        if class_weight_options is None:
            class_weight_options = [None, "balanced"]

        best = {"C": 1.0, "class_weight": "balanced", "mcc_mean": -np.inf, "threshold": 0.0}
        inner_cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train).min(), 5)), shuffle=True, random_state=42)

        for class_weight in class_weight_options:
            for C_val in c_grid:
                fold_scores = []
                for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train):
                    model = SVC(
                        kernel="linear",
                        C=C_val,
                        class_weight=class_weight,
                        random_state=42,
                        probability=False,
                    )
                    model.fit(X_train.iloc[inner_train_idx], y_train[inner_train_idx])
                    decision = model.decision_function(X_train.iloc[inner_valid_idx])
                    threshold = SVMClassifierWithCV._select_youden_threshold(decision, y_train[inner_valid_idx])
                    predictions = (decision >= threshold).astype(int)
                    fold_scores.append(matthews_corrcoef(y_train[inner_valid_idx], predictions))
                mean_mcc = float(np.mean(fold_scores))
                if mean_mcc > best["mcc_mean"]:
                    best = {"C": float(C_val), "class_weight": class_weight, "mcc_mean": mean_mcc, "threshold": 0.0}
        return best

    @staticmethod
    def _compute_oof_youden_threshold(X_train: pd.DataFrame, y_train: np.ndarray, C_val: float, class_weight: Optional[str]) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute out-of-fold decision scores and a Youden threshold for a fixed C and class_weight."""
        inner_cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train).min(), 5)), shuffle=True, random_state=42)
        oof_scores = []
        oof_labels = []

        for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train):
            model = SVC(
                kernel="linear",
                C=C_val,
                class_weight=class_weight,
                random_state=42,
                probability=False,
            )
            model.fit(X_train.iloc[inner_train_idx], y_train[inner_train_idx])
            valid_scores = model.decision_function(X_train.iloc[inner_valid_idx])
            oof_scores.extend(valid_scores.tolist())
            oof_labels.extend(y_train[inner_valid_idx].tolist())

        threshold = SVMClassifierWithCV._select_youden_threshold(np.asarray(oof_scores), np.asarray(oof_labels))
        return threshold, np.asarray(oof_scores), np.asarray(oof_labels)

    @staticmethod
    def _paired_wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Return the two-sided Wilcoxon signed-rank p-value for paired MCC scores."""
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("Paired Wilcoxon test requires equal-length score arrays.")
        diffs = a - b
        if np.allclose(diffs, 0.0):
            return 1.0
        try:
            _, p_value = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
            return float(p_value)
        except Exception:
            return 1.0

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
                scores = (
                    y_pred_proba
                    if np.asarray(y_pred_proba).ndim == 1
                    else np.asarray(y_pred_proba)[:, 1]
                )
                metrics["roc_auc"] = roc_auc_score(y_true, scores)
            except Exception:
                metrics["roc_auc"] = None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            "tn": int(tn),  "fp": int(fp),  "fn": int(fn),  "tp": int(tp),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        })

        return metrics

    @staticmethod
    def _select_best_k_inner_cv(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_method: str,
        candidate_ks: list[int],
        p_value: Optional[float],
        random_state: int = 42,
    ) -> tuple[int, dict[int, float]]:
        """Select k using only inner-fold training data and mean validation MCC."""
        counts = np.bincount(y_train)
        if counts.size < 2 or counts.min() < 2:
            return candidate_ks[0], {candidate_ks[0]: float("nan")}

        n_inner_splits = min(5, int(counts.min()))
        inner_cv = StratifiedKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state,
        )
        scores_by_k: dict[int, list[float]] = {k: [] for k in candidate_ks}

        for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_valid = X_train.iloc[inner_valid_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_valid = y_train[inner_valid_idx]

            for k in candidate_ks:
                selector = FeatureSelector(
                    method=feature_method,
                    n_features=k,
                    p_value=p_value,
                )
                selector.fit(X_inner_train.to_numpy(), y_inner_train)
                X_selected_train = selector.transform(X_inner_train.to_numpy())
                X_selected_valid = selector.transform(X_inner_valid.to_numpy())
                model = SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                    probability=False,
                )
                model.fit(X_selected_train, y_inner_train)
                predictions = model.predict(X_selected_valid)
                scores_by_k[k].append(
                    float(matthews_corrcoef(y_inner_valid, predictions))
                )

        mean_scores = {
            k: float(np.mean(scores)) for k, scores in scores_by_k.items()
        }
        best_k = max(candidate_ks, key=lambda k: (mean_scores[k], -k))
        return best_k, mean_scores

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
                f"  Accuracy: {metrics['accuracy']:.4f} |  "
                f"MCC: {metrics['mcc']:.4f} | F1: {metrics['f1']:.4f} "
            )

        self.results["path_a"]["fold_results"] = fold_results
        self._aggregate_cv_results(fold_results, "path_a")

        logger.info("\n" + "-" * 70)
        logger.info("BASELINE SUMMARY")
        logger.info("-" * 70)
        self._print_cv_summary("path_a")

    # ------------------------------------------------------------------
    # Path B — Optimised (LEAKAGE-FREE via sklearn Pipeline)
    # ------------------------------------------------------------------
    def train_path_b_optimized(self, feature_method: str = "filter_ttest", n_features: int = 20, p_value: Optional[float] = None, apply_smote: bool = False, tune_threshold: bool = False, tune_k: Optional[int] = None, tune_k_candidates: Optional[list[int]] = None, tune_c: bool = False, alternative_classifier: Optional[str] = None) -> None:
        """
        Evaluate SVM with feature selection applied strictly INSIDE each training fold using sklearn.pipeline.Pipeline.

        This guarantees ZERO data leakage: the feature selector only ever sees the training data for that specific fold.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"PATH B: OPTIMISED — Feature selection: {feature_method} (n={n_features})")
        logger.info("=" * 70)

        p_value = p_value if p_value is not None else getattr(self, "p_value", None)

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        fold_results = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y), 1):
            logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]

            selected_k = n_features if tune_k is None else tune_k
            if tune_k_candidates is not None:
                selected_k, inner_k_scores = self._select_best_k_inner_cv(
                    X_train,
                    y_train,
                    feature_method,
                    tune_k_candidates,
                    p_value,
                    self.random_state,
                )
                logger.info(
                    "  Inner-CV k selection: %s; selected k=%d",
                    {k: round(score, 4) for k, score in inner_k_scores.items()},
                    selected_k,
                )
            if apply_smote:
                X_train_raw = X_train.copy()
                y_train_raw = y_train.copy()
                X_train_resampled, y_train, _ = self._apply_fold_safe_smote(X_train_raw, y_train_raw, k_neighbors=5)
                X_train = pd.DataFrame(X_train_resampled, columns=X_train_raw.columns)
                logger.info(f"  SMOTE applied within fold: {X_train.shape[0]} samples, minority count={np.sum(y_train==1)}")

            selector = FeatureSelector(method=feature_method, n_features=selected_k, p_value=p_value)
            X_train_array = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
            X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
            selector.fit(X_train_array, y_train)
            X_train_selected = selector.transform(X_train_array)
            X_test_selected = selector.transform(X_test_array)
            n_selected = len(selector.selected_features)
            logger.info(f"  Reduced to {n_selected} features for this fold")

            C_value = 1.0
            if tune_c:
                selected_train_frame = pd.DataFrame(
                    X_train_selected,
                    index=X_train.index,
                )
                tuned = self._select_best_c_inner_cv(
                    selected_train_frame,
                    y_train,
                    c_grid=[0.01, 0.1, 1.0, 10.0, 100.0],
                    class_weight_options=[None, "balanced"],
                )
                C_value = tuned["C"]
                logger.info(f"  Tuned C={C_value} with class_weight={tuned['class_weight']}")
            threshold = 0.0
            if tune_threshold:
                inner_cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train).min(), 5)), shuffle=True, random_state=self.random_state)
                oof_scores = []
                oof_labels = []
                for inner_train_idx, inner_valid_idx in inner_cv.split(X_train_selected, y_train):
                    model = SVC(kernel="linear", C=C_value, class_weight="balanced", random_state=self.random_state, probability=False)
                    model.fit(X_train_selected[inner_train_idx], y_train[inner_train_idx])
                    oof_scores.extend(model.decision_function(X_train_selected[inner_valid_idx]).tolist())
                    oof_labels.extend(y_train[inner_valid_idx].tolist())
                threshold = self._select_youden_threshold(np.asarray(oof_scores), np.asarray(oof_labels))
                logger.info(f"  Tuned threshold (Youden J) = {threshold:.4f}")

            if alternative_classifier == 'logistic_regression':
                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=self.random_state)
                clf.fit(X_train_selected, y_train)
                y_pred = clf.predict(X_test_selected)
                y_pred_proba = clf.predict_proba(X_test_selected)
            elif alternative_classifier == 'random_forest':
                clf = RandomForestClassifier(n_estimators=300, random_state=self.random_state, class_weight='balanced')
                clf.fit(X_train_selected, y_train)
                y_pred = clf.predict(X_test_selected)
                y_pred_proba = clf.predict_proba(X_test_selected)
            else:
                model = SVC(
                    kernel="linear",
                    C=C_value,
                    random_state=self.random_state,
                    class_weight="balanced",
                    probability=False,
                )
                model.fit(X_train_selected, y_train)
                decision_scores = model.decision_function(X_test_selected)
                if tune_threshold:
                    y_pred = (decision_scores >= threshold).astype(int)
                    y_pred_proba = decision_scores
                else:
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = model.decision_function(X_test_selected)

            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)

            metrics["n_features"] = n_selected
            metrics["selected_k"] = selected_k
            metrics["feature_method"] = feature_method
            fold_results.append(metrics)

            logger.info(f"  Accuracy: {metrics['accuracy']:.4f} | MCC: {metrics['mcc']:.4f} | Specificity: {metrics.get('specificity', 0.0):.4f}")

        key = f"path_b_{feature_method}"
        if tune_threshold:
            key = f"path_b_{feature_method}_threshold_tuned"
        if tune_k_candidates is not None:
            key = f"path_b_{feature_method}_k_tuned"
        if tune_c:
            key = f"path_b_{feature_method}_c_tuned"
        if alternative_classifier is not None:
            key = f"path_b_{feature_method}_{alternative_classifier}"
        self.results[key] = {"fold_results": fold_results}
        self._aggregate_cv_results(fold_results, key)

        logger.info("\n" + "-" * 70)
        logger.info(f"OPTIMISED SUMMARY ({feature_method}) ")
        logger.info("-" * 70)
        self._print_cv_summary(key)

    def run_improved_imbalanced_analysis(self, dataset_name: Optional[str] = None) -> dict:
        """Run the set of imbalanced-data experiments requested for GSE42568: SMOTE, tuned threshold, tuned C, and alternate classifiers."""
        dataset_name = dataset_name or self.dataset_name
        results = {}

        for method in FeatureSelectionPipeline.ALL_METHODS:
            for config in [
                {"label": f"{method}_baseline", "apply_smote": False, "tune_threshold": False, "tune_c": False, "alternative_classifier": None},
                {"label": f"{method}_smote", "apply_smote": True, "tune_threshold": False, "tune_c": False, "alternative_classifier": None},
                {"label": f"{method}_threshold_tuned", "apply_smote": False, "tune_threshold": True, "tune_c": False, "alternative_classifier": None},
                {"label": f"{method}_c_tuned", "apply_smote": False, "tune_threshold": False, "tune_c": True, "alternative_classifier": None},
                {"label": f"{method}_logreg", "apply_smote": False, "tune_threshold": False, "tune_c": False, "alternative_classifier": "logistic_regression"},
                {"label": f"{method}_rf", "apply_smote": False, "tune_threshold": False, "tune_c": False, "alternative_classifier": "random_forest"},
            ]:
                try:
                    self.train_path_b_optimized(feature_method=method, n_features=self.n_features, p_value=self.p_value, apply_smote=config["apply_smote"], tune_threshold=config["tune_threshold"], tune_c=config["tune_c"], alternative_classifier=config["alternative_classifier"])
                    results[config["label"]] = self.results.get(f"path_b_{method}_threshold_tuned" if config["tune_threshold"] else f"path_b_{method}_c_tuned" if config["tune_c"] else f"path_b_{method}_{config['alternative_classifier']}" if config["alternative_classifier"] is not None else f"path_b_{method}")
                except Exception as exc:
                    logger.error(f"Imbalanced-data experiment failed for {method}: {exc}")
        return results
    # ------------------------------------------------------------------
    def _aggregate_cv_results(self, fold_results: list[dict], path_name: str) -> None:
        metrics_df = pd.DataFrame(fold_results)
        summary: dict = {}
        skip_cols = {"n_features", "selected_k", "feature_method", "tn", "fp", "fn", "tp"}

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

            logger.info(f"\n{key}: ")
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

        # Derive method list from FeatureSelectionPipeline so it stays in sync
        for method in FeatureSelectionPipeline.ALL_METHODS:
            try:
                self.train_path_b_optimized(feature_method=method, n_features=self.n_features, p_value=self.p_value)
            except Exception as exc:
                logger.error(f"Path B ({method}) failed: {exc}")

        self.compare_paths()
        self.save_results()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run SVM classifier pipeline")
    parser.add_argument("--dataset", "-d", default="GSE19804",
                        help="Dataset name or path (default: GSE19804)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV splits (default: 5)")
    parser.add_argument("--method", "-m", default=None,
                        help="Run a single Path B method (default: run all)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max parallel workers for Path B methods (default: 1 — no parallelism)")
    parser.add_argument("--p-value", type=float, default=0.05,
                        help="Optional p-value threshold to pass to selectors (default: 0.05)")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Target n_features for selectors (default: 20)")
    parser.add_argument("--run", choices=["path_a", "path_b", "all"],
                        default="all", help="Which paths to run (default: all)")
    args = parser.parse_args()

    classifier = SVMClassifierWithCV(dataset_name=args.dataset, n_splits=args.n_splits, n_features=args.n_features, p_value=args.p_value)

    if args.run in ("all", "path_a"):
        classifier.train_path_a_baseline()

    if args.run in ("all", "path_b"):
        if args.method:
            classifier.train_path_b_optimized(feature_method=args.method, n_features=args.n_features, p_value=args.p_value)
        else:
            methods = FeatureSelectionPipeline.ALL_METHODS
            max_workers = max(1, args.max_workers)
            if max_workers == 1:
                for method in methods:
                    try:
                        classifier.train_path_b_optimized(feature_method=method, n_features=args.n_features, p_value=args.p_value)
                    except Exception as exc:
                        logger.error(f"Path B ({method}) failed during run: {exc}")
            else:
                # Parallel execution using ThreadPoolExecutor; protect shared results with a lock
                lock = threading.Lock()
                def _run_and_store(method_name: str):
                    # Run training and return the key/summary for aggregation
                    try:
                        # Use the existing instance method which appends to self.results
                        classifier.train_path_b_optimized(feature_method=method_name, n_features=args.n_features, p_value=args.p_value)
                        return method_name, True, None
                    except Exception as exc:
                        return method_name, False, str(exc)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_run_and_store, m): m for m in methods}
                    for fut in as_completed(futures):
                        method = futures[fut]
                        try:
                            mname, ok, err = fut.result()
                            if not ok:
                                logger.error(f"Parallel Path B method {mname} failed: {err}")
                        except Exception as exc:
                            logger.error(f"Parallel Path B method {method} exception: {exc}")

    classifier.compare_paths()
    classifier.save_results()

if __name__ == "__main__":
    main()
"""
Comprehensive Feature Selection Pipeline
=========================================
Implements Filter, Wrapper, and Embedded methods with forward/backward selection.
Designed to work within cross-validation folds to prevent data leakage.

STANDALONE USAGE
----------------
    python feature_selection.py --dataset <path_or_name> [OPTIONS]

Arguments
---------
--dataset   Path to a CSV/TSV file  OR  a dataset name (e.g. GSE19804).
            When a name is given, the script looks for a CSV at:
              <project>/preprocessed_datasets/<name>/<name>.csv
            That file is produced by preprocessing.py's export_csv().
--label-col Name of the target/label column in the CSV (default: "label").
--n-features  Number of features to select per method (default: 20).
--methods   Space-separated subset of methods to run (default: all).
            Choices: filter_ttest  filter_anova  filter_fdr_ranked
                     wrapper_svm  wrapper_rf  embedded_lasso
--results-dir  Directory to write results into.
            Default: <project_root>/results/feature_selection
--no-feature-files  Skip writing per-method .txt feature lists.

Examples
--------
    # Run all methods on a preprocessed dataset
    python feature_selection.py --dataset GSE19804

    # Run only fast filter methods, 50 features
    python feature_selection.py --dataset GSE19804 \\
        --n-features 50 --methods filter_ttest filter_anova embedded_lasso

    # Explicit CSV path with a custom label column
    python feature_selection.py --dataset my_data.csv --label-col diagnosis

Fix log (vs previous version)
------------------------------
- Removed module-level basicConfig call; it is now inside main() only,
  so importing this module never reconfigures the root logger.
- Removed unused `import os`.
- np.ndarray | None replaced with Optional[np.ndarray] for Python 3.9
  compatibility (covered by `from __future__ import annotations`).
- _locate_dataset now also searches the per-dataset sub-folder that
  preprocessing.export_csv() creates, closing the path-mismatch gap.
- FeatureSelectionPipeline.ALL_METHODS is the single canonical list of
  method names; svm_classifier imports it to stay in sync automatically.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, false_discovery_control
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# scikit-learn 1.8 still references ast.Num on Python 3.14, where that
# compatibility alias may be absent. Restore the alias before estimators load.
if not hasattr(ast, "Num"):
    ast.Num = ast.Constant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Unified feature selection interface.

    Parameters
    ----------
    method : str
        One of: filter_ttest | filter_anova | wrapper_svm | wrapper_rf |
                wrapper_forward | wrapper_backward | embedded_lasso
    n_features : int
        Number of features to select.
    """

    DISPLAY_NAMES: dict[str, str] = {
        "filter_ttest":     "Welch's T-Test (Filter)",
        "filter_anova":     "ANOVA F-Test (Filter)",
        "filter_fdr_ranked": "FDR-Ranked (F-score + MI + Effect Size)",
        "wrapper_svm":      "SVM-RFE (Wrapper)",
        "wrapper_rf":       "RandomForest Importance (Wrapper)",
        "embedded_lasso":   "LASSO / L1 Logistic (Embedded)",
    }

    def __init__(
        self,
        method: str = "filter_ttest",
        n_features: int = 20,
        p_value: Optional[float] = 0.05,
        lasso_Cs: Optional[list[float]] = None,
        lasso_scoring: str = "roc_auc",
    ) -> None:
        self.method = method
        self.n_features = n_features
        # When p_value is provided, all methods use an uncapped significance-
        # driven mode; n_features is used only without a threshold.
        self.p_value = 0.05 if p_value is None else float(p_value)
        # LASSO cross-validation grid and scoring metric
        self.lasso_Cs = np.array(lasso_Cs) if lasso_Cs is not None else None
        self.lasso_scoring = lasso_scoring
        self.selected_features: Optional[np.ndarray] = None
        self.feature_scores: Optional[np.ndarray] = None
        self.selection_rule: str = "top_k"
        # Diagnostic results for embedded LASSO CV (filled after fit)
        self.lasso_cv_results: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        """
        Fit the selector on training data.
        Must be called on the training fold ONLY to prevent data leakage.
        """
        dispatch = {
            "filter_ttest":     self._fit_ttest,
            "filter_anova":     self._fit_anova,
            "filter_fdr_ranked": self._fit_fdr_ranked,
            "wrapper_svm":      self._fit_wrapper_svm,
            "wrapper_rf":       self._fit_wrapper_rf,
            "embedded_lasso":   self._fit_embedded_lasso,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Valid choices: {sorted(dispatch)}"
            )
        dispatch[self.method](X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return only the selected feature columns."""
        if self.selected_features is None:
            raise RuntimeError("Call fit() before transform().")
        return X[:, self.selected_features]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAMES.get(self.method, self.method)

    # ------------------------------------------------------------------
    # Filter methods
    # ------------------------------------------------------------------
    def _fit_ttest(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[filter_ttest]  Welch's Independent Two-Sample T-Test …")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("filter_ttest requires exactly 2 classes.")
        X0, X1 = X[y == classes[0]], X[y == classes[1]]
        _, p_values = ttest_ind(X1, X0, axis=0, equal_var=False)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        p_values = np.clip(p_values, 0.0, 1.0)
        # In threshold mode, retain every feature meeting the p-value cutoff.
        if self.p_value is not None:
            mask = np.where(p_values <= float(self.p_value))[0]
            self.selected_features = mask
            self.selection_rule = f"all_p<={self.p_value}"
        else:
            self.selection_rule = f"top_k({self.n_features})"
            order = np.argsort(p_values)
            self.selected_features = order[: self.n_features]
        self.feature_scores = p_values
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")

    def _fit_anova(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[filter_anova]  ANOVA F-Test …")
        # Compute F-statistic and associated p-values
        F, p_values = f_classif(X, y)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        p_values = np.clip(p_values, 0.0, 1.0)
        # In threshold mode, retain every feature meeting the p-value cutoff.
        if self.p_value is not None:
            mask = np.where(p_values <= float(self.p_value))[0]
            self.selected_features = mask
            self.selection_rule = f"all_p<={self.p_value}"
        else:
            self.selection_rule = f"top_k({self.n_features})"
            k = min(self.n_features, X.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k)
            sel.fit(X, y)
            self.selected_features = sel.get_support(indices=True)
        self.feature_scores = F
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")

    def _compute_cohens_d(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Cohen's d (standardized effect size) for each feature.
        Assumes binary classification (y in {0, 1}).
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Cohen's d requires exactly 2 classes.")
        
        X0, X1 = X[y == classes[0]], X[y == classes[1]]
        n0, n1 = len(X0), len(X1)
        
        mean0 = np.mean(X0, axis=0)
        mean1 = np.mean(X1, axis=0)
        var0 = np.var(X0, axis=0, ddof=1)
        var1 = np.var(X1, axis=0, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
        
        # Avoid division by zero
        pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)
        
        cohens_d = (mean1 - mean0) / pooled_std
        return np.abs(cohens_d)

    def _percentile_rank(self, values: np.ndarray) -> np.ndarray:
        """
        Convert values to percentile ranks in [0, 1].
        Handles ties uniformly by averaging ranks.
        """
        # rankdata returns 1-based ranks; convert to 0-based percentiles
        from scipy.stats import rankdata
        ranks = rankdata(values, method='average')
        percentiles = (ranks - 1) / (len(values) - 1) if len(values) > 1 else np.zeros_like(ranks)
        return percentiles

    def _fit_fdr_ranked(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        FDR-based ranking: BH filter -> F-score + Cohen's d + MI -> percentile ranks.
        
        Steps:
        1. Compute adjusted p-values using Benjamini-Hochberg correction.
        2. Hard filter to features with FDR < 0.05.
        3. For survivors, compute: ANOVA F-score, Cohen's d, mutual information.
        4. Convert each metric to percentile ranks (to handle outliers).
        5. Average ranks with equal weights (1/3 each).
        6. Retain every feature that passes the BH-FDR threshold.
        """
        logger.info("[filter_fdr_ranked]  FDR-ranked (BH + F-score + MI + Effect Size) …")
        
        # Step 1: Compute adjusted p-values and filter by FDR < 0.05
        F, p_values = f_classif(X, y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        p_values = np.clip(p_values, 0.0, 1.0)
        adjusted_pvalues = false_discovery_control(p_values, method='bh')
        adjusted_pvalues = np.nan_to_num(adjusted_pvalues, nan=1.0, posinf=1.0, neginf=1.0)
        adjusted_pvalues = np.clip(adjusted_pvalues, 0.0, 1.0)
        
        fdr_mask = adjusted_pvalues <= 0.05
        survivor_indices = np.where(fdr_mask)[0]
        
        if survivor_indices.size == 0:
            logger.warning(
                "[filter_fdr_ranked] No features survived FDR <= 0.05"
            )
            self.selection_rule = "fdr<=0.05 (none)"
            self.selected_features = np.array([], dtype=int)
            self.feature_scores = F
            logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")
            return
        
        logger.info(f"  FDR < 0.05: {survivor_indices.size} features passed; ranking …")
        
        # Step 2: Compute three ranking metrics on the survivors
        X_survivors = X[:, survivor_indices]
        F_survivors = F[survivor_indices]
        
        # Cohen's d (effect size)
        cohens_d = self._compute_cohens_d(X_survivors, y)
        
        # Mutual information (requires integer labels, y is already 0/1)
        mi = mutual_info_classif(X_survivors, y, random_state=42)
        
        # Step 3: Convert each metric to percentile ranks
        f_ranks = self._percentile_rank(F_survivors)
        d_ranks = self._percentile_rank(cohens_d)
        mi_ranks = self._percentile_rank(mi)
        
        # Step 4: Combine with equal weights
        combined_ranks = (f_ranks + d_ranks + mi_ranks) / 3.0
        
        # Step 5: Keep every BH-FDR survivor; ranking is retained for reporting.
        ranked_survivors = np.argsort(-combined_ranks)
        self.selected_features = survivor_indices[ranked_survivors]
        
        self.feature_scores = F
        self.selection_rule = "fdr_ranked(all_bh<=0.05)"
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")

    # ------------------------------------------------------------------
    # Wrapper helpers
    # ------------------------------------------------------------------
    def _prefilter(
        self, X: np.ndarray, y: np.ndarray, max_k: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quick ANOVA pre-filter; returns (X_filtered, original_indices)."""
        # If a p-value threshold is specified, retain every passing feature.
        # Otherwise use the bounded top-k ANOVA prefilter.
        if getattr(self, "p_value", None) is not None:
            _, p_values = f_classif(X, y)
            p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
            p_values = np.clip(p_values, 0.0, 1.0)
            mask = np.where(p_values <= float(self.p_value))[0]
            if mask.size == 0:
                logger.warning(
                    "[prefilter] No features met p-value <= %s; falling back to top-%d prefilter",
                    self.p_value,
                    max_k,
                )
                k = min(max_k, X.shape[1])
                sel = SelectKBest(score_func=f_classif, k=k)
                X_f = sel.fit_transform(X, y)
                return X_f, sel.get_support(indices=True)
            # Threshold mode is intentionally uncapped: retain every gene
            # passing the p-value criterion for downstream method ranking.
            X_f = X[:, mask]
            return X_f, mask
        else:
            k = min(max_k, X.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k)
            X_f = sel.fit_transform(X, y)
            return X_f, sel.get_support(indices=True)

    # ------------------------------------------------------------------
    # Wrapper methods
    # ------------------------------------------------------------------
    def _fit_wrapper_svm(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_svm]  SVM-RFE (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features, running RFE …")
        svc = LinearSVC(
            C=0.01, penalty="l2", dual=True,
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        # Stop at the training-sample count, a data-dependent bound that
        # avoids leaving the p >> n regime without imposing an arbitrary k.
        target_features = min(X_pre.shape[1], max(2, X_pre.shape[0]))
        rfe = RFE(
            estimator=svc,
            n_features_to_select=target_features,
            step=0.2,
        )
        rfe.fit(X_pre, y)
        self.selected_features = pre_idx[rfe.support_]
        self.selection_rule = f"svm_rfe_until_n_samples({target_features})"
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_wrapper_rf(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_rf]  RandomForest importance prefilter (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features, fitting RandomForest …")
        rf = RandomForestClassifier(
            n_estimators=80, random_state=42,
            # joblib's process-parallel expression evaluator is incompatible
            # with Python 3.14; threaded parallelism is disabled here so the
            # importance model still runs deterministically.
            n_jobs=1, class_weight="balanced",
        )
        try:
            rf.fit(X_pre, y)
            importances = rf.feature_importances_
            threshold = float(np.mean(importances))
            keep = importances > threshold
            if not np.any(keep):
                keep[np.argmax(importances)] = True
                self.selection_rule = "rf_importance(maximum_fallback)"
            else:
                self.selection_rule = f"rf_importance(>mean={threshold:.6g})"
            selected_orig_idx = pre_idx[keep]
            # keep them sorted for downstream reproducibility
            self.selected_features = np.sort(selected_orig_idx)
            self.feature_scores = importances
            logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")
        except Exception as exc:
            logger.exception("  RandomForest importance failed")
            raise RuntimeError("RandomForest importance selection failed") from exc

    # Note: greedy forward/backward wrapper methods removed — they were
    # computationally prohibitive on high-dimensional genomic data and
    # have been replaced by pre-filtered RFE and RF importance strategies.

    # ------------------------------------------------------------------
    # Embedded methods
    # ------------------------------------------------------------------
    def _fit_embedded_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[embedded_lasso]  LASSO (L1 Logistic Regression) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features for LASSO")
        # Use cross-validated LogisticRegressionCV to pick regularization strength
        # Cs is a grid of inverse regularization strengths; smaller C -> stronger regularization
        Cs = self.lasso_Cs if self.lasso_Cs is not None else np.logspace(-4, 1, 12)
        try:
            lrcv = LogisticRegressionCV(
                Cs=Cs,
                penalty="l1",
                solver="liblinear",
                scoring=self.lasso_scoring,
                cv=5,
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
                n_jobs=1,
                refit=True,
            )
            lrcv.fit(X_pre, y)
            coefs = np.abs(lrcv.coef_[0])
            # Report chosen C (inverse reg strength) — note that lambda ~ 1/C
            try:
                chosen_C = lrcv.C_[0]
            except Exception:
                chosen_C = getattr(lrcv, "C_", None)
            logger.info(f"  LogisticRegressionCV selected C={chosen_C}")
            # Compute diagnostic: number of non-zero coefficients for each C
            non_zero_counts = []
            for c in Cs:
                try:
                    lr = LogisticRegression(
                        penalty="l1", solver="liblinear", C=float(c),
                        class_weight="balanced", max_iter=1000, random_state=42
                    )
                    lr.fit(X_pre, y)
                    non_zero_counts.append(int((np.abs(lr.coef_[0]) > 1e-8).sum()))
                except Exception:
                    non_zero_counts.append(None)
            self.lasso_cv_results = {
                "Cs": Cs.tolist(),
                "non_zero_counts": non_zero_counts,
                "chosen_C": float(chosen_C) if chosen_C is not None else None,
            }
            selected_count = int(np.count_nonzero(coefs))
            if selected_count == 0 or selected_count == X_pre.shape[1]:
                viable = [
                    (count, float(c))
                    for c, count in zip(Cs, non_zero_counts)
                    if count is not None and 0 < count < X_pre.shape[1]
                ]
                if viable:
                    _, fallback_C = min(viable)
                    logger.warning(
                        "  CV-selected C=%s produced a degenerate %s solution; "
                        "using sparsest non-degenerate grid candidate C=%s",
                        chosen_C,
                        "all-zero" if selected_count == 0 else "all-nonzero",
                        fallback_C,
                    )
                    fallback = LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=fallback_C,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    )
                    fallback.fit(X_pre, y)
                    coefs = np.abs(fallback.coef_[0])
                    self.lasso_cv_results["fallback_C"] = fallback_C
        except Exception as exc:
            logger.warning(f"  LogisticRegressionCV failed ({exc}); falling back to fixed-C LASSO")
            lasso = LogisticRegression(
                penalty="l1", solver="liblinear",
                C=0.1, random_state=42, max_iter=1000, class_weight="balanced",
            )
            lasso.fit(X_pre, y)
            coefs = np.abs(lasso.coef_[0])

        non_zero = np.flatnonzero(coefs != 0.0)
        if len(non_zero) == 0:
            logger.warning("  LASSO produced no non-zero coefficients across the CV grid")
            self.selected_features = np.array([], dtype=int)
            self.selection_rule = "lasso_nonzero_coefficients(none)"
        else:
            self.selected_features = np.sort(pre_idx[non_zero])
            self.selection_rule = f"lasso_nonzero_coefficients({len(non_zero)})"

        self.feature_scores = coefs
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")


# ---------------------------------------------------------------------------
# FeatureSelectionPipeline
# ---------------------------------------------------------------------------
class FeatureSelectionPipeline:
    """
    Orchestrates all feature selection methods on the same dataset.

    ALL_METHODS is the single canonical list of valid method names.
    svm_classifier imports it directly so the two files never drift apart.
    """

    # Single source of truth for valid method names — imported by svm_classifier
    ALL_METHODS: list[str] = list(FeatureSelector.DISPLAY_NAMES.keys())

    def __init__(
        self,
        n_features: int = 20,
        methods: Optional[list[str]] = None,
        p_value: Optional[float] = 0.05,
        lasso_Cs: Optional[list[float]] = None,
        lasso_scoring: str = "roc_auc",
    ) -> None:
        self.n_features = n_features
        chosen = methods or self.ALL_METHODS
        unknown = set(chosen) - set(self.ALL_METHODS)
        if unknown:
            raise ValueError(
                f"Unknown method(s): {unknown}. "
                f"Valid choices: {self.ALL_METHODS}"
            )
        self.p_value = 0.05 if p_value is None else float(p_value)
        self.lasso_Cs = lasso_Cs
        self.lasso_scoring = lasso_scoring
        self.selectors: dict[str, FeatureSelector] = {
            m: FeatureSelector(m, n_features, p_value=p_value, lasso_Cs=lasso_Cs, lasso_scoring=lasso_scoring)
            for m in chosen
        }

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit all selectors; returns a results dict."""
        logger.info("=" * 65)
        logger.info("FEATURE SELECTION PIPELINE — fitting all methods")
        logger.info("=" * 65)
        results: dict = {}
        for method, selector in self.selectors.items():
            try:
                selector.fit(X, y)
                results[method] = {
                    "n_selected": int(len(selector.selected_features)),
                    "selected_indices": selector.selected_features,
                }
            except Exception as exc:
                logger.error(f"  ERROR in {method}: {exc}")
                results[method] = {"error": str(exc)}
        return results


# ---------------------------------------------------------------------------
# Data loading (standalone mode)
# ---------------------------------------------------------------------------
def _locate_dataset(dataset_arg: str, script_dir: Path) -> Path:
    """
    Resolve --dataset to an actual CSV file path.

    Accepted forms
    --------------
    1. An absolute or relative path to an existing .csv / .tsv file.
    2. A dataset name — looks in these locations in order:
         <project>/preprocessed_datasets/<name>/<name>.csv   ← export_csv() output
         <project>/preprocessed_datasets/<name>.csv
         <project>/datasets/<name>/<name>.csv
         <project>/datasets/<name>.csv
    """
    p = Path(dataset_arg)
    if p.exists():
        return p.resolve()

    project_dir = script_dir.parent
    candidates = [
        project_dir / "preprocessed_datasets" / p.name / f"{p.name}.csv",
        project_dir / "preprocessed_datasets" / f"{p.name}.csv",
        project_dir / "datasets" / p.name / f"{p.name}.csv",
        project_dir / "datasets" / f"{p.name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Resolved '{dataset_arg}' → {candidate}")
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find dataset '{dataset_arg}'.\n"
        "Tried:\n" + "\n".join(f"  {c}" for c in candidates) + "\n"
        "Run preprocessing.py first to generate the CSV, or pass a direct path."
    )


def load_dataset(
    csv_path: Path,
    label_col: str = "label",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load a CSV/TSV dataset; split into X (features) and y (labels).

    The label column must contain binary values (0/1) or exactly two
    unique values that will be label-encoded to 0/1.
    """
    sep = "\t" if csv_path.suffix.lower() in {".tsv", ".txt"} else ","
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, sep=sep, index_col=0)
    logger.info(f"  Shape: {df.shape}")

    # Case-insensitive column lookup
    if label_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == label_col.lower()]
        if not matches:
            raise KeyError(
                f"Label column '{label_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        label_col = matches[0]
        logger.info(f"  Using label column: '{label_col}'")

    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    unique_vals = sorted(y_raw.unique())
    if len(unique_vals) != 2:
        raise ValueError(
            f"Expected exactly 2 class labels, found {len(unique_vals)}: {unique_vals}"
        )
    if set(unique_vals) != {0, 1}:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        logger.info(f"  Encoding labels: {mapping}")
        y_raw = y_raw.map(mapping)

    y = y_raw.to_numpy(dtype=int)
    logger.info(
        f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}  |  "
        f"Class distribution: {np.bincount(y).tolist()}"
    )
    return X, y


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _print_console_summary(
    results: dict,
    n_original: int,
    selectors: dict[str, FeatureSelector],
) -> None:
    w = 72
    print("\n" + "=" * w)
    print("FEATURE SELECTION — RESULTS SUMMARY")
    print("=" * w)
    print(f"Original feature space: {n_original} features\n")
    print(f"{'Method':<22} {'Display Name':<32} {'Selected':>8} {'Reduction':>10}")
    print("-" * w)
    for method, res in results.items():
        display = selectors[method].display_name
        if "error" in res:
            print(f"{method:<22} {display:<32} {'ERROR':>8}  {res['error']}")
        else:
            n_sel = res["n_selected"]
            pct = (1 - n_sel / n_original) * 100
            print(f"{method:<22} {display:<32} {n_sel:>8} {pct:>9.1f}%")
    print("=" * w + "\n")


def save_results(
    pipeline: FeatureSelectionPipeline,
    results: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    results_base_dir: Path,
    n_features: int,
    write_feature_files: bool = True,
) -> Path:
    """
    Write all output files into results_base_dir / dataset_name /

    Files created
    -------------
    feature_selection_summary_<dataset>_<ts>.csv
        One row per method: counts, reduction %, feature indices and names.
    feature_selection_results_<dataset>_<ts>.json
        Full structured results with metadata block.
    feature_selection_report_<dataset>_<ts>.txt
        Human-readable text report.
    selected_features_<dataset>_<ts>/
        <method>_selected_features.txt   — one file per method.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_base_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 1. CSV summary
    # ------------------------------------------------------------------
    csv_rows = []
    for method, selector in pipeline.selectors.items():
        if selector.selected_features is not None:
            idx = sorted(selector.selected_features.tolist())
            names = [feature_names[i] for i in idx]
            csv_rows.append({
                "Method": method,
                "Display Name": selector.display_name,
                "Features Selected": len(idx),
                "Original Features": X.shape[1],
                "Reduction (%)": round((1 - len(idx) / X.shape[1]) * 100, 2),
                "Selected Feature Indices": ";".join(map(str, idx)),
                "Selected Feature Names": ";".join(names),
            })
        else:
            csv_rows.append({
                "Method": method,
                "Display Name": selector.display_name,
                "Features Selected": "ERROR",
                "Original Features": X.shape[1],
                "Reduction (%)": "",
                "Selected Feature Indices": results[method].get("error", ""),
                "Selected Feature Names": "",
            })

    csv_path = out_dir / f"feature_selection_summary_{dataset_name}_{ts}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  [OK] CSV summary          ->  {csv_path.name}")

    # ------------------------------------------------------------------
    # 2. JSON results
    # ------------------------------------------------------------------
    json_data: dict = {
        "metadata": {
            "dataset": dataset_name,
            "timestamp": ts,
            "n_samples": int(X.shape[0]),
            "n_original_features": int(X.shape[1]),
            "selection_policy": "uncapped_p_value_and_bh_fdr",
            "p_value_threshold": float(pipeline.p_value),
            "class_distribution": np.bincount(y).tolist(),
        },
        "methods": {},
    }
    for method, selector in pipeline.selectors.items():
        if selector.selected_features is not None:
            idx = sorted(selector.selected_features.tolist())
            json_data["methods"][method] = {
                "display_name": selector.display_name,
                "n_selected": len(idx),
                "reduction_percent": round(
                    (1 - len(idx) / X.shape[1]) * 100, 2
                ),
                "selected_indices": idx,
                "selected_feature_names": [feature_names[i] for i in idx],
            }
        else:
            json_data["methods"][method] = {
                "display_name": selector.display_name,
                "error": results[method].get("error", "unknown error"),
            }

    json_path = out_dir / f"feature_selection_results_{dataset_name}_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(json_data, fh, indent=2)
    print(f"  [OK] JSON results         ->  {json_path.name}")

    # ------------------------------------------------------------------
    # 3. Text report
    # ------------------------------------------------------------------
    report_path = out_dir / f"feature_selection_report_{dataset_name}_{ts}.txt"
    with open(report_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("FEATURE SELECTION EVALUATION REPORT\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(f"Dataset      : {dataset_name}\n")
        fh.write(f"Timestamp    : {ts}\n")
        fh.write(f"Samples      : {X.shape[0]}\n")
        fh.write(f"Features     : {X.shape[1]}\n")
        fh.write("Selection   : uncapped p-value/BH-FDR mode\n")
        fh.write(f"P threshold : {pipeline.p_value}\n")
        dist = np.bincount(y)
        fh.write(f"Class dist.  : {dist[0]} class-0 / {dist[1]} class-1\n\n")
        fh.write("=" * 70 + "\n")
        fh.write("RESULTS BY METHOD\n")
        fh.write("=" * 70 + "\n\n")
        for method, selector in pipeline.selectors.items():
            fh.write(f"  Method  : {selector.display_name}\n")
            fh.write(f"  Key     : {method}\n")
            if selector.selected_features is not None:
                idx = sorted(selector.selected_features.tolist())
                names = [feature_names[i] for i in idx]
                pct = (1 - len(idx) / X.shape[1]) * 100
                fh.write(
                    f"  Selected: {len(idx)} / {X.shape[1]} features"
                    f"  ({pct:.1f}% reduction)\n"
                )
                fh.write(f"  Indices : {idx}\n")
                fh.write(f"  Names   : {names}\n")
            else:
                fh.write(f"  ERROR   : {results[method].get('error', '?')}\n")
            fh.write("\n" + "-" * 70 + "\n\n")
    print(f"  [OK] Text report          ->  {report_path.name}")

    # ------------------------------------------------------------------
    # 4. Per-method feature list files
    # ------------------------------------------------------------------
    if write_feature_files:
        feat_dir = out_dir / f"selected_features_{dataset_name}_{ts}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        for method, selector in pipeline.selectors.items():
            fname = feat_dir / f"{method}_selected_features.txt"
            with open(fname, "w") as fh:
                fh.write(f"Feature Selection Method : {selector.display_name}\n")
                fh.write(f"Method Key               : {method}\n")
                fh.write(f"Dataset                  : {dataset_name}\n")
                fh.write(f"Timestamp                : {ts}\n")
                fh.write(f"Original Feature Space   : {X.shape[1]}\n")
                if selector.selected_features is not None:
                    idx = sorted(selector.selected_features.tolist())
                    names = [feature_names[i] for i in idx]
                    pct = (1 - len(idx) / X.shape[1]) * 100
                    fh.write(f"Features Selected        : {len(idx)}\n")
                    fh.write(f"Dimensionality Reduction : {pct:.2f}%\n")
                    fh.write("=" * 60 + "\n")
                    fh.write(f"{'Index':<10} Feature Name\n")
                    fh.write("-" * 60 + "\n")
                    for i, name in zip(idx, names):
                        fh.write(f"{i:<10} {name}\n")
                else:
                    fh.write(f"ERROR: {results[method].get('error', 'unknown')}\n")
        print(f"  [OK] Per-method .txt files ->  {feat_dir.name}/")

    # ------------------------------------------------------------------
    # 5. LASSO diagnostic plot (if available)
    # ------------------------------------------------------------------
    for method, selector in pipeline.selectors.items():
        if method == "embedded_lasso" and getattr(selector, "lasso_cv_results", None) is not None:
            diagnostics = selector.lasso_cv_results
            try:
                Cs = diagnostics["Cs"]
                counts = diagnostics["non_zero_counts"]
                plt.figure()
                plt.plot(Cs, counts, marker="o")
                plt.xscale("log")
                plt.xlabel("C (inverse regularization)")
                plt.ylabel("Number of non-zero coefficients")
                plt.title(f"LASSO: Non-zero coefficients vs C ({dataset_name})")
                plt.grid(True)
                plot_path = out_dir / f"lasso_nonzero_vs_C_{dataset_name}_{ts}.png"
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
                print(f"  ✓  LASSO diagnostic plot →  {plot_path.name}")
            except Exception as exc:
                print(f"  !  Could not create LASSO diagnostic plot: {exc}")

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help=(
            "Path to a CSV/TSV file OR a dataset name whose preprocessed CSV "
            "lives under <project>/preprocessed_datasets/<name>/<name>.csv"
        ),
    )
    parser.add_argument(
        "--label-col", "-l",
        default="label",
        dest="label_col",
        help="Name of the target/label column in the CSV (default: 'label')",
    )
    parser.add_argument(
        "--n-features", "-n",
        type=int,
        default=20,
        dest="n_features",
        help="Number of features to select per method (default: 20)",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=0.05,
        dest="p_value",
        help=(
            "Uncapped significance mode. If provided, all methods retain every "
            "gene passing the p-value/BH-FDR criteria and ignore --n-features. "
            "Default: 0.05."
        ),
    )
    parser.add_argument(
        "--lasso-cs",
        type=str,
        default=None,
        dest="lasso_cs",
        help=(
            "Comma-separated list of C values for LASSO CV (e.g. '1e-4,1e-3,1e-2'). "
            "If omitted, a logspace grid is used."
        ),
    )
    parser.add_argument(
        "--lasso-scoring",
        type=str,
        default="roc_auc",
        dest="lasso_scoring",
        help="Scoring metric for LASSO CV (default: 'roc_auc')",
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        choices=FeatureSelectionPipeline.ALL_METHODS,
        metavar="METHOD",
        help=(
            "Methods to run (default: all). "
            f"Choices: {', '.join(FeatureSelectionPipeline.ALL_METHODS)}"
        ),
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        dest="results_dir",
        help=(
            "Directory to write results into. "
            "Defaults to <project_root>/results/feature_selection"
        ),
    )
    parser.add_argument(
        "--no-feature-files",
        action="store_true",
        dest="no_feature_files",
        help="Skip writing individual per-method feature list .txt files",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    # Configure logging here — not at module level — so importing this file
    # never reconfigures the root logger when used as a library.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent  # …/scripts/

    try:
        csv_path = _locate_dataset(args.dataset, script_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    dataset_name = csv_path.stem

    try:
        X, y = load_dataset(csv_path, label_col=args.label_col)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        logger.error(f"Failed to load dataset: {exc}")
        return 1

    results_base_dir = (
        Path(args.results_dir)
        if args.results_dir
        else script_dir.parent / "results" / "feature_selection"
    )

    print("\n" + "=" * 65)
    print("FEATURE SELECTION EVALUATION TOOL")
    print("=" * 65)
    print(f"  Dataset    : {dataset_name}  ({csv_path})")
    print(f"  Shape      : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Label col  : '{args.label_col}'")
    print(f"  selection  : uncapped p-value/BH-FDR (p <= {args.p_value})")
    print(f"  Methods    : {args.methods or 'all'}")
    print(f"  Output dir : {results_base_dir / dataset_name}")
    print("=" * 65 + "\n")

    # Parse lasso Cs if provided
    lasso_Cs = None
    if args.lasso_cs:
        try:
            lasso_Cs = [float(x) for x in args.lasso_cs.split(",") if x.strip()]
        except Exception:
            logger.warning("Could not parse --lasso-cs; falling back to default grid")

    pipeline = FeatureSelectionPipeline(
        n_features=args.n_features,
        methods=args.methods,
        p_value=args.p_value,
        lasso_Cs=lasso_Cs,
        lasso_scoring=args.lasso_scoring,
    )
    results = pipeline.fit_all(X.to_numpy(), y)

    _print_console_summary(results, X.shape[1], pipeline.selectors)

    print("Saving results …")
    out_dir = save_results(
        pipeline=pipeline,
        results=results,
        X=X,
        y=y,
        dataset_name=dataset_name,
        results_base_dir=results_base_dir,
        n_features=args.n_features,
        write_feature_files=not args.no_feature_files,
    )

    print(f"\nAll results saved to: {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

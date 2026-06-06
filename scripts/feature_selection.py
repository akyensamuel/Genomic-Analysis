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
            Choices: filter_ttest  filter_anova  wrapper_svm  wrapper_rf
                     wrapper_forward  wrapper_backward  embedded_lasso
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
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------
class FeatureSelector:
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
        "wrapper_svm":      "SVM-RFE (Wrapper)",
        "wrapper_rf":       "RF-RFE (Wrapper)",
        "wrapper_forward":  "Forward Selection (Wrapper)",
        "wrapper_backward": "Backward Elimination (Wrapper)",
        "embedded_lasso":   "LASSO / L1 Logistic (Embedded)",
    }

    def __init__(self, method: str = "filter_ttest", n_features: int = 20) -> None:
        self.method = method
        self.n_features = n_features
        self.selected_features: Optional[np.ndarray] = None
        self.feature_scores: Optional[np.ndarray] = None

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
            "wrapper_svm":      self._fit_wrapper_svm,
            "wrapper_rf":       self._fit_wrapper_rf,
            "wrapper_forward":  self._fit_forward_selection,
            "wrapper_backward": self._fit_backward_elimination,
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
        order = np.argsort(p_values)
        self.selected_features = order[: self.n_features]
        self.feature_scores = p_values
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_anova(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[filter_anova]  ANOVA F-Test …")
        k = min(self.n_features, X.shape[1])
        sel = SelectKBest(score_func=f_classif, k=k)
        sel.fit(X, y)
        self.selected_features = sel.get_support(indices=True)
        self.feature_scores = sel.scores_
        logger.info(f"  → {len(self.selected_features)} features selected")

    # ------------------------------------------------------------------
    # Wrapper helpers
    # ------------------------------------------------------------------
    def _prefilter(
        self, X: np.ndarray, y: np.ndarray, max_k: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quick ANOVA pre-filter; returns (X_filtered, original_indices)."""
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
            C=0.01, penalty="l1", dual=False,
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        rfe = RFE(
            estimator=svc,
            n_features_to_select=min(self.n_features, X_pre.shape[1]),
            step=10,
        )
        rfe.fit(X_pre, y)
        self.selected_features = pre_idx[rfe.support_]
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_wrapper_rf(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_rf]  RF-RFE (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features, running RFE …")
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        rfe = RFE(
            estimator=rf,
            n_features_to_select=min(self.n_features, X_pre.shape[1]),
            step=10,
        )
        rfe.fit(X_pre, y)
        self.selected_features = pre_idx[rfe.support_]
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_forward_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_forward]  Greedy Forward Selection …")
        _, n_total = X.shape
        selected: list[int] = []
        remaining = set(range(n_total))
        svc = LinearSVC(
            C=0.01, penalty="l1", dual=False,
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        target = min(self.n_features, n_total)
        for step in range(target):
            best_feat, best_score = None, -np.inf
            for feat in remaining:
                X_cand = X[:, selected + [feat]]
                try:
                    svc.fit(X_cand, y)
                    score = svc.score(X_cand, y)
                    if score > best_score:
                        best_score, best_feat = score, feat
                except Exception:
                    continue
            if best_feat is None:
                logger.warning(f"  Step {step + 1}: no valid feature found, stopping.")
                break
            selected.append(best_feat)
            remaining.discard(best_feat)
            logger.info(
                f"  Step {step + 1}/{target}: added feature {best_feat}"
                f"  (acc={best_score:.4f})"
            )
        self.selected_features = np.array(selected)
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_backward_elimination(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_backward]  Backward Elimination (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        selected = list(range(X_pre.shape[1]))
        target = min(self.n_features, X_pre.shape[1])
        svc = LinearSVC(
            C=0.01, penalty="l1", dual=False,
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        logger.info(
            f"  Pre-filtered to {len(selected)} features, "
            f"eliminating down to {target} …"
        )
        while len(selected) > target:
            worst_idx, worst_score = None, np.inf
            for i in range(len(selected)):
                X_cand = X_pre[:, [f for j, f in enumerate(selected) if j != i]]
                try:
                    svc.fit(X_cand, y)
                    score = svc.score(X_cand, y)
                    if score < worst_score:
                        worst_score, worst_idx = score, i
                except Exception:
                    continue
            if worst_idx is None:
                logger.warning("  Could not find a feature to remove, stopping.")
                break
            removed = selected.pop(worst_idx)
            logger.info(
                f"  Removed feature {removed}  "
                f"(acc={worst_score:.4f}, remaining={len(selected)})"
            )
        self.selected_features = pre_idx[np.array(selected)]
        logger.info(f"  → {len(self.selected_features)} features selected")

    # ------------------------------------------------------------------
    # Embedded methods
    # ------------------------------------------------------------------
    def _fit_embedded_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[embedded_lasso]  LASSO (L1 Logistic Regression) …")
        lasso = LogisticRegression(
            penalty="l1", solver="liblinear",
            C=0.1, random_state=42, max_iter=1000, class_weight="balanced",
        )
        lasso.fit(X, y)
        coefs = np.abs(lasso.coef_[0])
        non_zero = np.where(coefs > 0)[0]
        if len(non_zero) <= self.n_features:
            logger.warning(
                f"  LASSO found only {len(non_zero)} non-zero coefficients "
                f"(target was {self.n_features}); keeping all non-zero."
            )
            self.selected_features = non_zero
        else:
            top = np.argsort(coefs)[-self.n_features:]
            self.selected_features = np.sort(top)
        self.feature_scores = coefs
        logger.info(f"  → {len(self.selected_features)} features selected")


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
    ) -> None:
        self.n_features = n_features
        chosen = methods or self.ALL_METHODS
        unknown = set(chosen) - set(self.ALL_METHODS)
        if unknown:
            raise ValueError(
                f"Unknown method(s): {unknown}. "
                f"Valid choices: {self.ALL_METHODS}"
            )
        self.selectors: dict[str, FeatureSelector] = {
            m: FeatureSelector(m, n_features) for m in chosen
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
    print(f"  ✓  CSV summary          →  {csv_path.name}")

    # ------------------------------------------------------------------
    # 2. JSON results
    # ------------------------------------------------------------------
    json_data: dict = {
        "metadata": {
            "dataset": dataset_name,
            "timestamp": ts,
            "n_samples": int(X.shape[0]),
            "n_original_features": int(X.shape[1]),
            "n_target_features": n_features,
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
    print(f"  ✓  JSON results         →  {json_path.name}")

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
        fh.write(f"Target n_feat: {n_features}\n")
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
    print(f"  ✓  Text report          →  {report_path.name}")

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
        print(f"  ✓  Per-method .txt files →  {feat_dir.name}/")

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
    print(f"  n_features : {args.n_features}")
    print(f"  Methods    : {args.methods or 'all'}")
    print(f"  Output dir : {results_base_dir / dataset_name}")
    print("=" * 65 + "\n")

    pipeline = FeatureSelectionPipeline(
        n_features=args.n_features,
        methods=args.methods,
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

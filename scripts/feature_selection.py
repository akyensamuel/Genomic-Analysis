"""
Feature Selection Pipeline
Implements Filter, Wrapper, and Embedded methods
Designed to work within cross-validation folds to prevent data leakage
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Unified feature selection interface"""

    def __init__(self, method="filter_ttest", n_features=20):
        """
        Initialize feature selector

        Args:
            method (str): "filter_ttest", "filter_anova", "wrapper_svm", "wrapper_rf", "embedded_lasso"
            n_features (int): Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = None

    def fit(self, X, y):
        """
        Fit feature selector on training data
        IMPORTANT: Must be called on training fold ONLY (not test fold)

        Args:
            X (array-like): Training features (n_samples, n_features)
            y (array-like): Training labels (n_samples,)
        """
        if self.method == "filter_ttest":
            return self._fit_ttest(X, y)
        elif self.method == "filter_anova":
            return self._fit_anova(X, y)
        elif self.method == "wrapper_svm":
            return self._fit_wrapper_svm(X, y)
        elif self.method == "wrapper_rf":
            return self._fit_wrapper_rf(X, y)
        elif self.method == "embedded_lasso":
            return self._fit_embedded_lasso(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_ttest(self, X, y):
        """Welch's Independent Two-Sample T-Test"""
        logger.info("Fitting Filter Method: Welch's T-Test")

        # Separate by class
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]

        # Compute t-statistic and p-value for each feature
        t_stats, p_values = ttest_ind(X_class1, X_class0, axis=0, equal_var=False)

        # Create results dataframe
        results = pd.DataFrame({
            'feature_idx': np.arange(X.shape[1]),
            'p_value': p_values,
            't_stat': np.abs(t_stats)
        }).sort_values('p_value')

        # Select top N features
        self.selected_features = results.head(self.n_features)['feature_idx'].values
        self.feature_scores = results['p_value'].values

        logger.info(f"Selected {len(self.selected_features)} features (T-Test)")
        return self

    def _fit_anova(self, X, y):
        """ANOVA F-Test"""
        logger.info("Fitting Filter Method: ANOVA F-Test")

        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        selector.fit(X, y)

        self.selected_features = selector.get_support(indices=True)
        self.feature_scores = selector.scores_

        logger.info(f"Selected {len(self.selected_features)} features (ANOVA)")
        return self

    def _fit_wrapper_svm(self, X, y):
        """SVM-based Recursive Feature Elimination"""
        logger.info("Fitting Wrapper Method: SVM-RFE")

        # Pre-filter to top 500 to save computation
        n_prefilter = min(500, X.shape[1])
        logger.info(f"Pre-filtering to top {n_prefilter} features...")

        # Quick ANOVA pre-filter
        selector_pre = SelectKBest(score_func=f_classif, k=n_prefilter)
        X_prefiltered = selector_pre.fit_transform(X, y)
        prefilter_indices = selector_pre.get_support(indices=True)

        # RFE on prefiltered set
        svc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000,
                       random_state=42, class_weight='balanced')
        rfe = RFE(estimator=svc, n_features_to_select=self.n_features, step=10)
        rfe.fit(X_prefiltered, y)

        # Map back to original feature indices
        self.selected_features = prefilter_indices[rfe.support_]

        logger.info(f"Selected {len(self.selected_features)} features (SVM-RFE)")
        return self

    def _fit_wrapper_rf(self, X, y):
        """Random Forest-based Recursive Feature Elimination"""
        logger.info("Fitting Wrapper Method: RF-RFE")

        # Pre-filter to top 500
        n_prefilter = min(500, X.shape[1])
        logger.info(f"Pre-filtering to top {n_prefilter} features...")

        selector_pre = SelectKBest(score_func=f_classif, k=n_prefilter)
        X_prefiltered = selector_pre.fit_transform(X, y)
        prefilter_indices = selector_pre.get_support(indices=True)

        # RFE with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                   n_jobs=-1, class_weight='balanced')
        rfe = RFE(estimator=rf, n_features_to_select=self.n_features, step=10)
        rfe.fit(X_prefiltered, y)

        # Map back to original feature indices
        self.selected_features = prefilter_indices[rfe.support_]

        logger.info(f"Selected {len(self.selected_features)} features (RF-RFE)")
        return self

    def _fit_embedded_lasso(self, X, y):
        """LASSO (L1-Regularized Logistic Regression)"""
        logger.info("Fitting Embedded Method: LASSO")

        lasso = LogisticRegression(penalty='l1', solver='liblinear',
                                  C=0.1, random_state=42, max_iter=1000,
                                  class_weight='balanced')
        lasso.fit(X, y)

        # Get non-zero coefficient indices
        coefs = np.abs(lasso.coef_[0])
        self.selected_features = np.where(coefs > 0)[0]

        # If fewer than n_features selected, take top N
        if len(self.selected_features) < self.n_features:
            logger.warning(f"LASSO selected only {len(self.selected_features)} "
                         f"features, keeping all non-zero")
        else:
            # Keep top n_features by coefficient magnitude
            top_indices = np.argsort(coefs)[-self.n_features:]
            self.selected_features = np.sort(top_indices)

        self.feature_scores = coefs

        logger.info(f"Selected {len(self.selected_features)} features (LASSO)")
        return self

    def transform(self, X):
        """
        Select features from data

        Args:
            X (array-like): Features (n_samples, n_features)

        Returns:
            array: Selected features (n_samples, n_selected_features)
        """
        if self.selected_features is None:
            raise ValueError("Fit the selector first using fit()")

        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        """Fit and transform in one call"""
        self.fit(X, y)
        return self.transform(X)


class FeatureSelectionPipeline:
    """Manages all feature selection methods"""

    METHODS = {
        "filter_ttest": "Welch's T-Test",
        "filter_anova": "ANOVA F-Test",
        "wrapper_svm": "SVM-RFE",
        "wrapper_rf": "RF-RFE",
        "embedded_lasso": "LASSO"
    }

    def __init__(self, n_features=20):
        self.n_features = n_features
        self.selectors = {
            method: FeatureSelector(method, n_features)
            for method in self.METHODS.keys()
        }

    def fit_all(self, X, y):
        """Fit all feature selection methods"""
        logger.info("="*60)
        logger.info("FITTING ALL FEATURE SELECTION METHODS")
        logger.info("="*60)

        results = {}
        for method, selector in self.selectors.items():
            selector.fit(X, y)
            results[method] = {
                'n_selected': len(selector.selected_features),
                'selected_indices': selector.selected_features
            }

        return results

    def get_selector(self, method):
        """Get a specific selector"""
        if method not in self.selectors:
            raise ValueError(f"Unknown method: {method}")
        return self.selectors[method]


# Example usage
if __name__ == "__main__":
    from preprocessing import GenomicDataProcessor

    # Load preprocessed data
    processor = GenomicDataProcessor("GSE19804", "/c/Users/surface/Documents/FINAL YEAR PROJECT")

    try:
        X, y = processor.load_preprocessed_data()
    except:
        X, y = processor.preprocess_complete()
        processor.save_preprocessed_data()

    # Test feature selection
    pipeline = FeatureSelectionPipeline(n_features=20)
    results = pipeline.fit_all(X.values, y)

    for method, result in results.items():
        print(f"{method}: Selected {result['n_selected']} features")

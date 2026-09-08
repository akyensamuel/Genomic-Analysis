import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Ensure scripts are importable
scripts_dir = str(Path(__file__).resolve().parent.parent / 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from feature_selection import FeatureSelector
from svm_classifier import SVMClassifierWithCV


def test_path_b_filter_ttest_one_fold():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label'])

    # Use a balanced split so each fold retains both classes.
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    split_0 = len(idx_0) // 2
    split_1 = len(idx_1) // 2
    train_idx = np.concatenate([idx_0[:split_0], idx_1[:split_1]])
    test_idx = np.concatenate([idx_0[split_0:], idx_1[split_1:]])

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    pipeline = Pipeline([
        ('selector', FeatureSelector(method='filter_ttest', n_features=10)),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42))
    ])

    # Should not raise
    pipeline.fit(X_train.values, y_train)
    y_pred = pipeline.predict(X_test.values)

    assert hasattr(pipeline.named_steps['selector'], 'selected_features')
    sel = pipeline.named_steps['selector'].selected_features
    assert sel is not None and len(sel) > 0
    assert len(y_pred) == len(y_test)


def test_selector_reports_pvalue_selection_rule():
    selector = FeatureSelector(method='filter_ttest', n_features=5, p_value=0.05)
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()
    selector.fit(X, y)
    assert 'p<=0.05' in selector.selection_rule


def test_wrapper_rf_fallback_returns_features():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()

    selector = FeatureSelector(method='wrapper_rf', n_features=5, p_value=0.05)
    selector.fit(X, y)

    assert selector.selected_features is not None
    assert len(selector.selected_features) > 0


def test_wrapper_xgb_returns_features():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()

    selector = FeatureSelector(method='wrapper_xgb', n_features=5, p_value=0.05)
    selector.fit(X, y)

    assert selector.selected_features is not None
    assert len(selector.selected_features) > 0


def test_xgb_classifier_alternative():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 8)))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    classifier = SVMClassifierWithCV.__new__(SVMClassifierWithCV)
    classifier.X = X
    classifier.y = y
    classifier.n_splits = 2
    classifier.random_state = 42
    classifier.n_features = 2
    classifier.p_value = 0.05
    classifier.results = {}

    classifier.train_path_b_optimized(feature_method='filter_ttest', n_features=2, p_value=1.0, alternative_classifier='xgboost')

    assert 'path_b_filter_ttest_xgboost' in classifier.results
    assert 'summary' in classifier.results['path_b_filter_ttest_xgboost']


def test_model_selectors_are_not_anova_aliases():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()

    # Use a permissive threshold so the ANOVA candidate pool is materially
    # larger than the sample-count RFE stopping bound.
    threshold = 1.0
    anova = FeatureSelector(method='filter_anova', n_features=5, p_value=threshold)
    anova.fit(X, y)
    anova_set = set(anova.selected_features.tolist())

    selected_sets = {}
    for method in ('wrapper_svm', 'wrapper_rf', 'wrapper_xgb', 'embedded_lasso'):
        selector = FeatureSelector(method=method, n_features=5, p_value=threshold)
        selector.fit(X, y)
        selected_sets[method] = set(selector.selected_features.tolist())

    assert all(features != anova_set for features in selected_sets.values())
    assert len(set(map(frozenset, selected_sets.values()))) == 4


def test_youden_threshold_uses_decision_scores():
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1, 1])

    threshold = SVMClassifierWithCV._select_youden_threshold(scores, labels)

    assert 0.0 <= threshold <= 2.0
    assert isinstance(threshold, float)


def test_safe_smote_k_neighbors_is_clamped():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X_resampled, y_resampled, k_neighbors = SVMClassifierWithCV._apply_fold_safe_smote(X, y)

    assert len(X_resampled) >= len(X)
    assert len(y_resampled) == len(X_resampled)
    assert k_neighbors <= 5
    assert k_neighbors >= 1


def test_train_path_b_optimized_handles_smote_without_dataframe_error():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 8)))
    X.iloc[:8, 0] += 2
    X.iloc[:8, 1] -= 1
    X.iloc[8:, 0] -= 2
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    classifier = SVMClassifierWithCV.__new__(SVMClassifierWithCV)
    classifier.X = X
    classifier.y = y
    classifier.n_splits = 2
    classifier.random_state = 42
    classifier.n_features = 2
    classifier.p_value = 0.05
    classifier.results = {}

    classifier.train_path_b_optimized(feature_method='filter_ttest', n_features=2, p_value=0.05, apply_smote=True)

    assert 'path_b_filter_ttest' in classifier.results
    assert 'summary' in classifier.results['path_b_filter_ttest']

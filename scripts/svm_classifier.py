"""
SVM Classifier with Nested Cross-Validation
Implements dual-track training: Baseline (Path A) vs Optimized (Path B)
Prevents data leakage by applying feature selection INSIDE training folds
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc,
    classification_report, make_scorer
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import GenomicDataProcessor
from feature_selection import FeatureSelector

# Build DATASET_INFO dynamically from preprocessing.dataset_config
def build_dataset_info():
    """Build dataset info from preprocessor config"""
    processor = GenomicDataProcessor()
    dataset_info = {}

    for dataset_name, config in processor.dataset_config.items():
        total = config['n_cancer'] + config['n_normal']
        balance = "balanced" if config['n_cancer'] == config['n_normal'] else "imbalanced"

        # Identify cancer type
        cancer_type = "Unknown"
        if "19804" in dataset_name:
            cancer_type = "Lung Cancer"
        elif "42568" in dataset_name:
            cancer_type = "Breast Cancer"

        dataset_info[dataset_name] = {
            "name": cancer_type,
            "description": f"{dataset_name} - {cancer_type} ({balance}: {config['n_cancer']}/{config['n_normal']}, {total} total)",
            "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}"
        }

    return dataset_info

DATASET_INFO = build_dataset_info()

# Setup logging (will be configured per dataset)
logger = logging.getLogger(__name__)


class SVMClassifierWithCV:
    """SVM with nested cross-validation for robust evaluation"""

    def __init__(self, dataset_name="GSE19804", n_splits=5, random_state=42):
        """
        Initialize SVM classifier

        Args:
            dataset_name (str): "GSE42568" or "GSE19804"
            n_splits (int): Number of CV folds
            random_state (int): Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.random_state = random_state

        # Create dataset-specific results directory
        base_results_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results'
        )
        os.makedirs(base_results_dir, exist_ok=True)

        # Get dataset info
        dataset_info = DATASET_INFO.get(dataset_name, {})
        self.dataset_folder = dataset_info.get('folder', dataset_name)

        self.results_dir = os.path.join(base_results_dir, self.dataset_folder)
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup logging for this dataset
        self._setup_logging()

        # Load data
        self.X = None
        self.y = None
        self._load_data()

        self.results = {
            'path_a': {},  # Baseline
            'path_b': {}   # Optimized
        }

    def _setup_logging(self):
        """Setup dataset-specific logging"""
        log_file = os.path.join(self.results_dir, 'svm_training.log')

        # Remove old handlers
        logger.handlers = []

        # Add new handlers for this dataset
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_data(self):
        """Load preprocessed data or preprocess if needed"""
        logger.info("="*70)
        logger.info("LOADING DATA")
        logger.info("="*70)

        project_dir = os.path.join(os.path.dirname(__file__), '..')
        processor = GenomicDataProcessor(self.dataset_name, project_dir)

        try:
            self.X, self.y = processor.load_preprocessed_data()
            logger.info("Loaded from cached preprocessed data")
        except:
            logger.info("Preprocessed data not found, running preprocessing pipeline...")
            self.X, self.y = processor.preprocess_complete()
            processor.save_preprocessed_data()

        logger.info(f"Data shape: {self.X.shape}")
        logger.info(f"Class distribution: {np.bincount(self.y)}")
        logger.info(f"Feature matrix type: {type(self.X)}")

    def evaluate_predictions(self, y_true, y_pred, y_pred_proba=None, fold_num=None):
        """
        Compute comprehensive evaluation metrics

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            fold_num: Fold number (for logging)

        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }

        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def train_path_a_baseline(self):
        """
        Path A: Baseline track
        Train SVM directly on raw preprocessed data (no feature selection)
        """
        logger.info("\n" + "="*70)
        logger.info("PATH A: BASELINE TRACK (No Feature Selection)")
        logger.info("="*70)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                            random_state=self.random_state)

        fold_results = []
        fold_num = 1

        for train_idx, test_idx in cv.split(self.X, self.y):
            logger.info(f"\n--- Fold {fold_num}/{self.n_splits} ---")

            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Train SVM on full feature set
            svm = SVC(kernel='linear', C=1.0, random_state=self.random_state,
                     class_weight='balanced', probability=True)
            svm.fit(X_train, y_train)

            # Predict
            y_pred = svm.predict(X_test)
            y_pred_proba = svm.predict_proba(X_test)

            # Evaluate
            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba, fold_num)
            metrics['n_features'] = self.X.shape[1]

            fold_results.append(metrics)

            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, "
                       f"MCC: {metrics['mcc']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")

            fold_num += 1

        # Aggregate results
        self.results['path_a']['fold_results'] = fold_results
        self._aggregate_cv_results(fold_results, 'path_a')

        logger.info("\n" + "-"*70)
        logger.info("BASELINE SUMMARY")
        logger.info("-"*70)
        self._print_cv_summary('path_a')

    def train_path_b_optimized(self, feature_method="filter_ttest"):
        """
        Path B: Optimized track
        Apply feature selection INSIDE training fold, then train SVM

        Args:
            feature_method (str): Feature selection method to use
        """
        logger.info("\n" + "="*70)
        logger.info(f"PATH B: OPTIMIZED TRACK (Feature Selection: {feature_method})")
        logger.info("="*70)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                            random_state=self.random_state)

        fold_results = []
        fold_num = 1

        for train_idx, test_idx in cv.split(self.X, self.y):
            logger.info(f"\n--- Fold {fold_num}/{self.n_splits} ---")

            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # CRITICAL: Apply feature selection ONLY to training fold
            selector = FeatureSelector(method=feature_method, n_features=20)
            X_train_selected = selector.fit_transform(X_train.values, y_train)

            # Transform test fold using same selected features
            X_test_selected = X_test.iloc[:, selector.selected_features].values

            logger.info(f"Selected {len(selector.selected_features)} features")

            # Train SVM on selected features
            svm = SVC(kernel='linear', C=1.0, random_state=self.random_state,
                     class_weight='balanced', probability=True)
            svm.fit(X_train_selected, y_train)

            # Predict
            y_pred = svm.predict(X_test_selected)
            y_pred_proba = svm.predict_proba(X_test_selected)

            # Evaluate
            metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba, fold_num)
            metrics['n_features'] = len(selector.selected_features)
            metrics['feature_method'] = feature_method

            fold_results.append(metrics)

            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, "
                       f"MCC: {metrics['mcc']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")

            fold_num += 1

        # Aggregate results
        key = f'path_b_{feature_method}'
        self.results[key] = {'fold_results': fold_results}
        self._aggregate_cv_results(fold_results, key)

        logger.info("\n" + "-"*70)
        logger.info(f"OPTIMIZED SUMMARY ({feature_method})")
        logger.info("-"*70)
        self._print_cv_summary(key)

    def _aggregate_cv_results(self, fold_results, path_name):
        """Aggregate cross-validation results across folds"""
        metrics_df = pd.DataFrame(fold_results)

        summary = {}
        for col in metrics_df.columns:
            if col not in ['n_features', 'feature_method', 'tn', 'fp', 'fn', 'tp']:
                summary[f'{col}_mean'] = metrics_df[col].mean()
                summary[f'{col}_std'] = metrics_df[col].std()

        self.results[path_name]['summary'] = summary

    def _print_cv_summary(self, path_name):
        """Print summary of cross-validation results"""
        summary = self.results[path_name]['summary']

        logger.info(f"Accuracy:   {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        logger.info(f"Precision:  {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        logger.info(f"Recall:     {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        logger.info(f"F1-Score:   {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        logger.info(f"MCC:        {summary['mcc_mean']:.4f} ± {summary['mcc_std']:.4f}")
        logger.info(f"ROC-AUC:    {summary.get('roc_auc_mean', 'N/A')}")

    def compare_paths(self):
        """Compare performance between Path A and Path B"""
        logger.info("\n" + "="*70)
        logger.info("COMPARATIVE ANALYSIS: Path A vs Path B")
        logger.info("="*70)

        comparison = {}
        summary_a = self.results['path_a'].get('summary', {})

        for key, value_b in self.results.items():
            if key.startswith('path_b') and 'summary' in value_b:
                summary_b = value_b['summary']

                # Safe division with fallback
                accuracy_improvement = ((summary_b['accuracy_mean'] - summary_a['accuracy_mean']) /
                                       summary_a['accuracy_mean'] * 100) if summary_a['accuracy_mean'] != 0 else 0
                mcc_improvement = ((summary_b['mcc_mean'] - summary_a['mcc_mean']) /
                                  summary_a['mcc_mean'] * 100) if summary_a['mcc_mean'] != 0 else 0
                f1_improvement = ((summary_b['f1_mean'] - summary_a['f1_mean']) /
                                 summary_a['f1_mean'] * 100) if summary_a['f1_mean'] != 0 else 0

                improvement = {
                    'accuracy': accuracy_improvement,
                    'mcc': mcc_improvement,
                    'f1': f1_improvement,
                }

                comparison[key] = improvement

                logger.info(f"\n{key}:")
                logger.info(f"  Accuracy improvement: {improvement['accuracy']:+.2f}%")
                logger.info(f"  MCC improvement:      {improvement['mcc']:+.2f}%")
                logger.info(f"  F1 improvement:       {improvement['f1']:+.2f}%")

        return comparison

    def save_results(self):
        """Save results to JSON and CSV"""
        logger.info("\n" + "="*70)
        logger.info("SAVING RESULTS")
        logger.info("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert results to JSON-serializable format
        results_json = {}
        for path, data in self.results.items():
            results_json[path] = {
                'summary': data.get('summary', {}),
                'n_folds': len(data.get('fold_results', []))
            }

        # Save JSON
        json_path = os.path.join(self.results_dir, f'svm_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"Saved JSON results: {json_path}")

        # Save CSV summary
        summaries = []
        for path, data in self.results.items():
            summary = data.get('summary', {})
            summary['path'] = path
            summaries.append(summary)

        csv_path = os.path.join(self.results_dir, f'svm_summary_{timestamp}.csv')
        pd.DataFrame(summaries).to_csv(csv_path, index=False)
        logger.info(f"Saved CSV summary: {csv_path}")

    def run_full_pipeline(self):
        """Run complete training pipeline"""
        dataset_info = DATASET_INFO.get(self.dataset_name, {})

        logger.info("\n" + "="*70)
        logger.info("STARTING SVM CLASSIFIER PIPELINE")
        logger.info("="*70)
        logger.info(f"Dataset: {dataset_info.get('name', self.dataset_name)}")
        logger.info(f"Description: {dataset_info.get('description', '')}")
        logger.info(f"Results Directory: {self.results_dir}")
        logger.info(f"Folds: {self.n_splits}-Fold Cross-Validation")
        logger.info("="*70)

        # Path A: Baseline
        self.train_path_a_baseline()

        # Path B: Optimized (test each feature method)
        feature_methods = ["filter_ttest", "filter_anova", "wrapper_svm", "embedded_lasso"]
        for method in feature_methods:
            try:
                self.train_path_b_optimized(feature_method=method)
            except Exception as e:
                logger.error(f"Error in Path B ({method}): {e}")

        # Comparison
        self.compare_paths()

        # Save results
        self.save_results()

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*70)


def main():
    """Main execution"""
    classifier = SVMClassifierWithCV(dataset_name="GSE42568", n_splits=5)
    classifier.run_full_pipeline()


if __name__ == "__main__":
    main()

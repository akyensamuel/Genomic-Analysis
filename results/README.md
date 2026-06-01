# SVM Classification Results

This directory contains organized results from SVM classifier training on different gene expression datasets.

## Directory Structure

```
results/
├── GSE19804_lung_cancer/
│   ├── svm_training.log              # Detailed training logs
│   ├── svm_results_*.json            # Complete results in JSON format
│   └── svm_summary_*.csv             # Summary metrics in CSV format
│
└── GSE42568_breast_cancer/
    ├── svm_training.log              # Detailed training logs
    ├── svm_results_*.json            # Complete results in JSON format
    └── svm_summary_*.csv             # Summary metrics in CSV format
```

## Dataset Descriptions

### GSE19804 - Lung Cancer (Balanced)
- **Samples:** 120 (60 cancer, 60 normal)
- **Features:** 54,675 genes
- **Class Distribution:** Perfectly balanced (50/50)
- **Focus:** Tests feature selection on balanced data
- **Key Challenge:** High-dimensional feature space (p >> n)

### GSE42568 - Breast Cancer (Imbalanced)
- **Samples:** 121 (104 cancer, 17 normal)
- **Features:** 54,675 genes
- **Class Distribution:** Highly imbalanced (~86/14)
- **Focus:** Tests feature selection robustness with class imbalance
- **Key Challenge:** Minority class underrepresentation + high dimensionality

## File Explanations

### svm_training.log
- Real-time training progress
- Per-fold performance metrics
- Feature selection details
- Data profiling information

### svm_results_YYYYMMDD_HHMMSS.json
```json
{
  "path_a": {
    "summary": {
      "accuracy_mean": 0.95,
      "accuracy_std": 0.07,
      "mcc_mean": 0.90,
      ...
    },
    "n_folds": 5
  },
  "path_b_filter_ttest": { ... },
  "path_b_filter_anova": { ... },
  ...
}
```

### svm_summary_YYYYMMDD_HHMMSS.csv
Spreadsheet with rows for each method (Path A, Path B variants) and columns for metrics:
- accuracy_mean, accuracy_std
- precision_mean, precision_std
- recall_mean, recall_std
- f1_mean, f1_std
- mcc_mean, mcc_std
- roc_auc_mean (if applicable)

## How to Compare Results

1. **Between Datasets:** Compare metrics in the two dataset folders
2. **Between Methods:** Look at different `path_b_*` entries (T-Test, ANOVA, SVM-RFE, LASSO)
3. **Imbalance Effect:** Compare baseline vs. optimized across both datasets

## Interpreting Results

**Key Metrics:**
- **MCC (Matthews Correlation Coefficient):** Best for imbalanced data; ranges -1 to +1
- **ROC-AUC:** Threshold-independent classification quality
- **Std Values:** Lower std = more stable/consistent across folds
- **Accuracy:** Can be misleading for imbalanced data; use with caution

## Timestamp Convention

Results are timestamped as `YYYYMMDD_HHMMSS` to prevent overwrites:
- `20260601_201051` = June 1, 2026 at 20:10:51
- Each run creates a new results file without deleting previous ones
- Allows tracking of algorithm improvements over time

# Effect of Feature Selection on Classification Accuracy in High-Dimensional Gene Expression Data

## Project Overview

This project investigates the comparative impact of different feature selection techniques on machine learning classification accuracy in high-dimensional gene expression datasets. The research addresses the **small *n*, large *p*** problem prevalent in genomics, where datasets contain tens of thousands of gene probes but relatively few samples.

**Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Kumasi  
**Department:** Mathematics  
**Degree:** BSc. Mathematics  
**Authors:** Darko Samuel and Akyen Samuel

## Research Objectives

1. **Analyze** high-dimensional gene expression data structure and characteristics
2. **Implement** three feature selection approaches:
   - Filter methods (Welch's T-Test, ANOVA F-Test)
   - Wrapper methods (SVM-RFE, Random Forest importance)
   - Embedded methods (LASSO / L1-regularized logistic regression)
3. **Train** an SVM classifier on selected features using leakage-free cross-validation
4. **Evaluate** classification performance across all feature subsets
5. **Determine** which feature selection technique most effectively improves classification accuracy

## Project Status

### Completed

- **Preprocessing pipeline** (`scripts/preprocessing.py`): GEO download, log2 transformation, z-score normalization, `.npy` cache + CSV export
- **Feature selection pipeline** (`scripts/feature_selection.py`): all methods with uncapped p-value/BH-FDR selection by default
- **Feature selection runs** on GSE19804 and GSE42568 (results under `results/feature_selection/`)
- **SVM evaluation framework** (`scripts/svm_classifier.py`, `scripts/svm_comparative.py`): Path A baseline + Path B with feature selection inside each CV fold
- **Integration tests** (`tests/test_svm_integration.py`)
- **Thesis draft**: Chapters 1–4 in LaTeX (`chapters/`)

### In Progress

- **SVM classifier evaluation**: run full pipeline on both datasets and populate Chapter 4 with quantitative results
- **Experimental consistency**: apply the same uncapped p-value/BH-FDR selection policy across datasets before final comparison

### Pending

- Chapter 5: Conclusions and Recommendations
- Final report PDF compilation
- Optional: Random Forest and Logistic Regression classifiers (out of current scope; thesis focuses on SVM)

## Datasets

Both datasets are registered in `GenomicDataProcessor.DATASET_CONFIG` and share the same high-dimensional microarray structure (~54,675 probes).

| Dataset | Cancer type | Samples | Class balance | p/n ratio |
|---------|-------------|---------|---------------|-----------|
| **GSE19804** | Lung | 120 (60 / 60) | Balanced | ~455:1 |
| **GSE42568** | Breast | 121 (104 / 17) | Imbalanced | ~452:1 |

**Source:** [NCBI Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/)

GSE19804 is the primary feature-selection dataset; GSE42568 provides a secondary benchmark under severe class imbalance.

## Pipeline Overview

The canonical workflow is script-based. The Jupyter notebook (`Genomic_data_analysis (3).ipynb`) is retained as early exploratory work but is no longer the primary pipeline.

```
preprocessing.py  →  feature_selection.py  →  svm_classifier.py
        │                      │                      │
        ▼                      ▼                      ▼
preprocessed_datasets/   results/feature_selection/   results/<dataset_folder>/
```

### Step 1 — Preprocessing

Downloads (or loads cached) GEO series matrices, applies log2(*x* + 1) and per-feature z-score normalization, and writes:

- NumPy cache: `preprocessed_datasets/<dataset>/`
- Labelled CSV: `preprocessed_datasets/<dataset>/<dataset>.csv`

### Step 2 — Feature Selection

Runs filter, wrapper, and embedded selectors. With `--p-value`, filters retain every qualifying gene; SVM-RFE stops at the training-sample bound, Random Forest keeps above-mean importances, and LASSO keeps nonzero coefficients.

### Step 3 — SVM Evaluation

- **Path A (baseline):** linear SVM on all features, no selection
- **Path B (optimised):** `FeatureSelector` → `SVC` inside a `sklearn.pipeline.Pipeline`, fitted only on each training fold

Evaluation uses **5-fold stratified cross-validation** with balanced class weights. Primary metric: **Matthews Correlation Coefficient (MCC)**.

## Experimental Settings

### Feature selection rules

The pipeline supports two selection modes, controlled by CLI flags:

| Mode | Flag | Behaviour |
|------|------|-----------|
| **P-value threshold** | `--p-value 0.05` | Retain **all** genes passing the p-value criterion; no top-*k* cap is applied |

**Methods implemented:** `filter_ttest`, `filter_anova`, `wrapper_svm`, `wrapper_rf`, `embedded_lasso`

The active wrapper methods are SVM-RFE and Random Forest importance.

### Current runs (August 2026)

| Dataset | Filter methods | Wrapper / LASSO | Notes |
|---------|----------------|-----------------|-------|
| GSE19804 | All genes passing the threshold | All genes passing the threshold | Use `--p-value` for uncapped statistical selection |
| GSE42568 | All genes passing the threshold | All genes passing the threshold | Use `--p-value` for uncapped statistical selection |

> **Note:** Both datasets are evaluated under identical selection parameters ($p \le 0.05, k = 20$), providing a direct cross-dataset and cross-method comparison.

### SVM settings

- Kernel: linear
- `C = 1.0`, `class_weight = 'balanced'`
- 5-fold stratified CV, `random_state = 42`
- Feature selection applied **inside each fold** (no data leakage)

## Directory Structure

```
GenomicsProj/
├── main.tex                              # LaTeX master document
├── chapters/
│   ├── chapter1_introduction.tex
│   ├── chapter2_methodology.tex
│   ├── chapter3_results.tex
│   └── chapter4_svm_evaluation.tex
├── scripts/
│   ├── preprocessing.py                  # Download, transform, cache, export CSV
│   ├── feature_selection.py              # Filter / wrapper / embedded selectors
│   ├── svm_classifier.py                 # Single-dataset SVM evaluation
│   └── svm_comparative.py                # Run SVM on all registered datasets
├── tests/
│   ├── test_svm_integration.py           # Pipeline smoke tests
│   ├── make_synthetic.py                 # Synthetic data generator
│   ├── run_svm_synthetic.py              # Quick SVM test on synthetic data
│   └── synthetic.csv
├── results/
│   ├── README.md                         # SVM results layout guide
│   └── feature_selection/                # Per-dataset feature selection outputs
│       ├── GSE19804/
│       └── GSE42568/
├── datasets/                             # Raw GEO downloads (gitignored)
├── preprocessed_datasets/                # Cached .npy + CSV (gitignored)
├── Genomic_data_analysis (3).ipynb       # Exploratory notebook
├── requirements.txt
└── venv_python_runner.bat                # Windows helper for venv Python
```

## Running the Analysis

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn imbalanced-learn
```

Or install the full pinned environment:

```bash
pip install -r requirements.txt
```

### 1. Preprocessing

#### Purpose
Downloads (or loads cached) GEO datasets, applies log₂ transformation and z-score normalization, and saves NumPy cache + CSV for downstream use.

#### Script: `scripts/preprocessing.py`

**Syntax:**
```bash
python scripts/preprocessing.py [dataset_name]
```

**Arguments:**
- `dataset_name` (positional, optional): Dataset key. Registered: `GSE19804`, `GSE42568`. Default: `GSE19804`

**Output:**
- NumPy cache: `preprocessed_datasets/<dataset_name>/X.npy`, `y.npy`
- CSV export: `preprocessed_datasets/<dataset_name>/<dataset_name>.csv`

**Examples:**

Preprocess GSE19804 (default):
```bash
python scripts/preprocessing.py
```

Preprocess GSE42568:
```bash
python scripts/preprocessing.py GSE42568
```

Preprocess both:
```bash
python scripts/preprocessing.py GSE19804
python scripts/preprocessing.py GSE42568
```

---

### 2. Feature Selection

#### Purpose
Apply filter, wrapper, and embedded feature-selection methods to rank / select genes based on univariate statistics, model-based importance, or regularization.

#### Script: `scripts/feature_selection.py`

**Syntax:**
```bash
python scripts/feature_selection.py --dataset DATASET [OPTIONS]
```

**Required Arguments:**
- `--dataset DATASET`, `-d DATASET`: CSV path or registered dataset name (e.g. `GSE19804`)

**Optional Arguments:**
| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--n-features N` | `-n` | int | 20 | Retained CLI option; ignored when statistical selection is active |
| `--p-value P` | — | float | 0.05 | Uncapped p-value/BH-FDR mode. All genes passing the relevant threshold are retained; `--n-features` is ignored |
| `--label-col COL` | `-l` | str | `label` | Name of the label column in CSV |
| `--methods M1 M2 ...` | `-m` | list | All | Methods to run. Choices: `filter_ttest`, `filter_anova`, `fdr_ranked`, `wrapper_svm`, `wrapper_rf`, `embedded_lasso` |
| `--lasso-cs C1,C2,...` | — | str | Logspace | Comma-separated C values for LASSO cross-validation tuning |
| `--lasso-scoring METRIC` | — | str | `roc_auc` | Scoring metric for LASSO CV selection |
| `--results-dir DIR` | `-r` | path | Auto | Output directory (default: `results/feature_selection`) |
| `--no-feature-files` | — | flag | False | Skip writing individual per-method feature `.txt` files |

**Output:**
- `feature_selection_summary_<TIMESTAMP>.csv` — Method names, feature counts, reduction %
- `feature_selection_results_<TIMESTAMP>.json` — Full structured results
- `feature_selection_report_<TIMESTAMP>.txt` — Human-readable report
- `selected_features_<TIMESTAMP>/` — Directory with per-method gene lists (one file per method)

**Examples:**

**Uncapped statistical selection (default):**
```bash
python scripts/feature_selection.py --dataset GSE19804 --n-features 20
python scripts/feature_selection.py --dataset GSE42568 --n-features 20
```

**P-value threshold selection (filters retain all significant probes):**
```bash
python scripts/feature_selection.py --dataset GSE19804 --p-value 0.05
```

**Run a subset of methods:**
```bash
python scripts/feature_selection.py --dataset GSE19804 \
    --methods filter_ttest filter_anova embedded_lasso
```

**Custom LASSO tuning grid:**
```bash
python scripts/feature_selection.py --dataset GSE19804 \
    --lasso-cs "0.0001,0.001,0.01,0.1,1.0"
```

**Save results to a custom directory:**
```bash
python scripts/feature_selection.py --dataset GSE19804 \
    --results-dir ./my_results --n-features 20
```

**Skip per-method feature files (faster, smaller output):**
```bash
python scripts/feature_selection.py --dataset GSE19804 \
    --no-feature-files --n-features 20
```

---

### 3. SVM Classification and Evaluation

#### Purpose
Train and evaluate SVM classifiers on single or multiple datasets. Supports Path A (baseline, no feature selection) and Path B (with feature selection inside each CV fold).

#### Script A: `scripts/svm_classifier.py` (single dataset)

**Syntax:**
```bash
python scripts/svm_classifier.py [OPTIONS]
```

**Optional Arguments:**
| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--dataset DATASET` | `-d` | str | `GSE19804` | Dataset name (e.g. `GSE19804`, `GSE42568`) |
| `--n-splits N` | — | int | 5 | Number of stratified cross-validation folds |
| `--n-features N` | — | int | 20 | Target number of features to select (for Path B) |
| `--p-value P` | — | float | 0.05 | P-value threshold for filter methods in Path B |
| `--method METHOD` | `-m` | str | None | Run a single Path B method. If None, run all methods. Choices: `filter_ttest`, `filter_anova`, `fdr_ranked`, `wrapper_svm`, `wrapper_rf`, `embedded_lasso` |
| `--run {path_a, path_b, all}` | — | enum | `all` | Which pipeline(s) to execute |
| `--max-workers N` | — | int | 1 | Number of parallel workers for Path B methods (default: no parallelism) |

**Output:**
- `results/<DATASET>_<CANCER_TYPE>/svm_training.log` — Per-fold training progress
- `results/<DATASET>_<CANCER_TYPE>/svm_results_<TIMESTAMP>.json` — Full results + summary statistics
- `results/<DATASET>_<CANCER_TYPE>/svm_summary_<TIMESTAMP>.csv` — Metrics table (accuracy, MCC, F1, etc.)

**Examples:**

**Full pipeline on GSE19804 (Path A + all Path B methods, 5-fold CV):**
```bash
python scripts/svm_classifier.py --dataset GSE19804
```

**Full pipeline on GSE42568 with custom feature settings:**
```bash
python scripts/svm_classifier.py --dataset GSE42568 --n-features 20 --p-value 0.05
```

**Path A baseline only (no feature selection):**
```bash
python scripts/svm_classifier.py --dataset GSE19804 --run path_a
```

**Path B with a single feature-selection method:**
```bash
python scripts/svm_classifier.py --dataset GSE19804 --run path_b --method filter_ttest
```

**Run Path B with all methods in parallel (4 workers):**
```bash
python scripts/svm_classifier.py --dataset GSE19804 --run path_b --max-workers 4
```

**10-fold cross-validation:**
```bash
python scripts/svm_classifier.py --dataset GSE19804 --n-splits 10
```

---

#### Script B: `scripts/svm_comparative.py` (all registered datasets)

**Purpose:**
Run the full SVM pipeline (Path A + all Path B methods) on every registered dataset in parallel.

**Syntax:**
```bash
python scripts/svm_comparative.py [OPTIONS]
```

**Optional Arguments:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n-splits N` | int | 5 | Number of stratified cross-validation folds |
| `--random-state SEED` | int | 42 | Random seed for reproducibility |

**Output:**
- One subdirectory per dataset under `results/` with same structure as `svm_classifier.py` output

**Examples:**

**Run full pipeline on all datasets (5-fold CV):**
```bash
python scripts/svm_comparative.py
```

**Run with 10-fold CV and custom random seed:**
```bash
python scripts/svm_comparative.py --n-splits 10 --random-state 123
```

---

### 4. Run Tests

**Unit and integration tests:**
```bash
python -m pytest tests/ -v
```

**Run a specific test file:**
```bash
python -m pytest tests/test_svm_integration.py -v
```

**Run tests with output capture disabled (show print statements):**
```bash
python -m pytest tests/ -v -s
```

**Run a specific test by name:**
```bash
python -m pytest tests/test_svm_integration.py::test_pipeline_smoke -v
```

---

### 5. Compile the Thesis

**Generate PDF from LaTeX:**
```bash
latexmk -pdf main.tex
```

**Or run pdflatex manually (twice for table of contents):**
```bash
pdflatex main.tex
pdflatex main.tex
```

**Clean up auxiliary files:**
```bash
latexmk -c
```

---

### Common Workflows

#### **Workflow 1: Full reproducible analysis (GSE19804, uncapped statistical selection)**

```bash
# 1. Preprocess
python scripts/preprocessing.py GSE19804

# 2. Run feature selection
python scripts/feature_selection.py --dataset GSE19804 --n-features 20

# 3. Train and evaluate SVM
python scripts/svm_classifier.py --dataset GSE19804 --n-features 20 --p-value 0.05

# 4. Compile thesis
latexmk -pdf main.tex
```

#### **Workflow 2: Compare filter methods only on imbalanced data**

```bash
python scripts/preprocessing.py GSE42568

python scripts/feature_selection.py --dataset GSE42568 \
    --methods filter_ttest filter_anova --n-features 20

python scripts/svm_classifier.py --dataset GSE42568 \
    --run path_b --method filter_ttest
python scripts/svm_classifier.py --dataset GSE42568 \
    --run path_b --method filter_anova
```

#### **Workflow 3: Benchmark all methods on both datasets**

```bash
# Preprocess both
python scripts/preprocessing.py GSE19804
python scripts/preprocessing.py GSE42568

# Feature selection on both
python scripts/feature_selection.py --dataset GSE19804 --n-features 20
python scripts/feature_selection.py --dataset GSE42568 --n-features 20

# SVM evaluation on both datasets in parallel
python scripts/svm_comparative.py --n-splits 5
```

#### **Workflow 4: P-value threshold exploration**

```bash
# Run feature selection with strict and permissive thresholds
python scripts/feature_selection.py --dataset GSE19804 --p-value 0.01
python scripts/feature_selection.py --dataset GSE19804 --p-value 0.05
python scripts/feature_selection.py --dataset GSE19804 --p-value 0.10

# With --p-value, wrappers apply their algorithmic stopping rules after the significance prefilter.
# (see --lasso-cs for LASSO tuning parameters)
```

## Outputs

### Feature selection (`results/feature_selection/<dataset>/`)

| File | Description |
|------|-------------|
| `feature_selection_summary_<timestamp>.csv` | Method, feature count, reduction %, selected gene names |
| `feature_selection_results_<timestamp>.json` | Full structured results |
| `feature_selection_report_<timestamp>.txt` | Human-readable report |
| `selected_features_<timestamp>/` | Per-method `.txt` gene lists |

### SVM evaluation (`results/<dataset>_<cancer_type>/`)

| File | Description |
|------|-------------|
| `svm_training.log` | Per-fold training progress |
| `svm_results_<timestamp>.json` | Path A + all Path B summaries |
| `svm_summary_<timestamp>.csv` | Metrics table (accuracy, MCC, F1, etc.) |

SVM result files are gitignored; generate them locally by running the classifier scripts.

## Evaluation Metrics

For each feature subset (Path A + five Path B methods), the pipeline reports:

- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Matthews Correlation Coefficient (MCC)** — primary metric, especially for GSE42568
- **ROC-AUC** (when probability estimates are available)
- **Sensitivity / Specificity** and confusion matrix counts (TP, TN, FP, FN)

All metrics are aggregated as mean ± std across CV folds.

## References

**Dataset source:** NCBI GEO — [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804), [GSE42568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568)

**Python libraries:** scikit-learn, pandas, numpy, scipy, matplotlib, seaborn

## Notes

- The small-*n*, large-*p* setting (p/n ≈ 450:1) makes overfitting inevitable without feature selection; baseline SVM on the full feature set is expected to perform poorly.
- Preprocessing (log2 + z-score) is applied to the full dataset before CV; the scaler is not refit per fold — a known simplification documented in the thesis.
- Filter methods with a permissive p-value threshold (e.g. 0.05) can retain thousands of probes; in threshold mode no method applies a top-*k* cap.
- Apply the same uncapped p-value/BH-FDR thresholds consistently across datasets before drawing final comparative conclusions.

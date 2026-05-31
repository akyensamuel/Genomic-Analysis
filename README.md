# Effect of Feature Selection on Classification Accuracy in High-Dimensional Gene Expression Data

## Project Overview

This project investigates the comparative impact of different feature selection techniques on machine learning classification accuracy in high-dimensional gene expression datasets. The research addresses the "Small n, Large p" problem prevalent in genomics, where datasets contain thousands of genes but relatively few samples.

**Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Kumasi  
**Department:** Mathematics  
**Degree:** BSc. Mathematics  
**Authors:** Darko Samuel and Akyen Samuel

## Research Objectives

1. **Analyze** high-dimensional gene expression data structure and characteristics
2. **Implement** three feature selection approaches:
   - Filter methods (Welch's T-Test, ANOVA F-Test)
   - Wrapper methods (SVM-RFE, Random Forest-RFE)
   - Embedded methods (LASSO L1-Regularization)
3. **Train** machine learning classifiers on selected features
4. **Evaluate** classification performance across all feature subsets
5. **Determine** which feature selection technique most effectively improves classification accuracy

## Project Status

### ✅ Completed

- **Data Preprocessing**: Log2 transformation, Z-score normalization
- **Feature Selection**: All three methods implemented
  - Filter: T-Test (54,675 genes → top 20)
  - Filter: ANOVA (54,675 genes → top 20)
  - Wrapper: SVM-RFE (top 500 pre-filtered → 20 genes)
  - Wrapper: RF-RFE (top 500 pre-filtered → 20 genes)
  - Embedded: LASSO (54,675 genes → 33 genes)
- **Feature Analysis**: Venn diagram overlaps, consensus biomarkers identified (6 universal genes)
- **Initial Evaluation**: 5-fold cross-validation with MCC scores (0.9370 - 0.9519)
- **Documentation**: Chapters 1-3 written in LaTeX (Introduction, Methodology, Feature Selection Results)

### 🔄 In Progress

- **Model Training**: Training 3 classifiers on each feature subset:
  - Model 1: Support Vector Machine (SVM) - Linear Kernel ← **STARTING HERE**
  - Model 2: Random Forest
  - Model 3: Logistic Regression

### ⏳ Pending

- Chapter 4: Classifier Evaluation Results
- Chapter 5: Conclusions and Recommendations
- Final Report PDF

## Dataset

**GSE19804 - Lung Cancer Gene Expression Dataset**
- **Source**: NCBI Gene Expression Omnibus (GEO)
- **Samples**: 120 (60 cancer, 60 normal) - balanced class distribution
- **Genes**: 54,675 probes
- **Dimensionality**: p/n ratio = 455:1 (extreme high-dimensional problem)

## Key Findings So Far

| Metric | Value |
|--------|-------|
| **Baseline (Full Set)** | MCC = -0.0344 (anti-predictive) |
| **Best Feature Selection** | Filter & Wrapper tied at MCC = 0.9519 |
| **Dimensionality Reduction** | 54,675 → 20 genes (99.96% reduction) |
| **Performance Improvement** | +3,233% improvement over baseline |
| **Universal Biomarkers** | 6 genes selected by all 4 methods |

## Directory Structure

```
FINAL YEAR PROJECT/
├── Genomic_data_analysis (3).ipynb      # Main analysis notebook (completed)
├── main.tex                             # LaTeX master document
├── chapters/
│   ├── chapter1_introduction.tex        # Background & objectives
│   ├── chapter2_methodology.tex         # Feature selection methods
│   └── chapter3_results.tex             # Feature selection results
├── scripts/                             # Python model training scripts (TO BE ADDED)
│   ├── svm_classifier.py                # SVM model training
│   ├── random_forest_classifier.py      # Random Forest training
│   └── logistic_regression_classifier.py # Logistic Regression training
├── Literature_Review (2).pdf            # Background literature
├── Project Overview and Methodological Framework.pdf
├── Synopsis_Article.pdf                 # Original project synopsis
└── knust_logo.png                       # University logo
```

## Feature Selection Summary

### Filter Methods (Univariate)
- **T-Test**: 54,675 genes → top 20 by p-value
- **ANOVA**: 54,675 genes → top 20 by F-score
- **Agreement**: 85% overlap (17/20 genes shared)
- **Speed**: $O(p)$ - extremely fast

### Wrapper Methods (Multivariate)
- **SVM-RFE**: Linear SVM with iterative elimination
- **RF-RFE**: Random Forest with Gini importance
- **Agreement**: 40% overlap (8/20 genes shared)
- **Cost**: $O(p^2 \cdot n)$ - more expensive but captures interactions

### Embedded Methods (Integrated)
- **LASSO**: L1-regularized logistic regression
- **Sparsity**: 54,675 genes → 33 genes retained
- **Advantage**: Automatic feature selection during model training

## Running the Analysis

### Prerequisites
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### Step 1: Feature Selection (Completed)
Open and run `Genomic_data_analysis (3).ipynb` to preprocess data and perform feature selection.

### Step 2: Model Training (In Progress)
Run individual model training scripts:
```bash
python scripts/svm_classifier.py
python scripts/random_forest_classifier.py
python scripts/logistic_regression_classifier.py
```

### Step 3: Generate Final Report
Compile LaTeX document:
```bash
pdflatex main.tex
pdflatex main.tex  # Run twice for table of contents
```

## Classification Evaluation Metrics

For each of 3 classifiers × 5 feature subsets, we compute:

- **Confusion Matrix**: TP, TN, FP, FN visualization
- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$
- **Precision**: $TP / (TP + FP)$ - how many predicted positives are correct
- **Recall/Sensitivity**: $TP / (TP + FN)$ - true positive rate
- **Specificity**: $TN / (TN + FP)$ - true negative rate
- **F1-Score**: $2 \times (Precision \times Recall) / (Precision + Recall)$ - harmonic mean
- **Matthews Correlation Coefficient (MCC)**: Balanced metric for imbalanced data
- **ROC Curve & AUC**: Receiver Operating Characteristic analysis

## Expected Outputs

For each model, we will generate:
1. **Performance Summary Table** - metrics across all feature sets
2. **Confusion Matrix Heatmaps** - TP/TN/FP/FN for each subset
3. **ROC Curves** - comparing feature selection methods
4. **Comparison Visualizations** - bar plots of metric performance

## Project Timeline

| Phase | Status | Deadline |
|-------|--------|----------|
| Data Preprocessing | ✅ Complete | May 14 |
| Feature Selection | ✅ Complete | May 20 |
| **Model 1 Training (SVM)** | 🔄 **IN PROGRESS** | June 1 |
| Model 2 Training (RF) | ⏳ Pending | June 2 |
| Model 3 Training (LR) | ⏳ Pending | June 3 |
| Chapter 4 Writing | ⏳ Pending | June 5 |
| Chapter 5 Writing | ⏳ Pending | June 7 |
| Final Report | ⏳ Pending | June 10 |

## Contact & References

**Dataset Source:** NCBI Gene Expression Omnibus (GEO) - GSE19804  
**Python Libraries:**
- scikit-learn: Machine learning
- pandas: Data manipulation
- numpy: Numerical computing
- scipy: Statistical analysis
- matplotlib/seaborn: Visualization

## Notes

- The "small n, large p" problem creates extreme overfitting risk (p/n = 455:1)
- Negative MCC on full dataset demonstrates necessity of feature selection
- 6 universal biomarkers identified by all methods suggest robust biological signal
- Feature selection methods capture different signal dimensions (filters vs. wrappers)

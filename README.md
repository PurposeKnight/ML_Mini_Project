# UE24CS352A Machine Learning Mini-Project
## Problem Statement 55: Predicting Mortgage-Backed Securities (MBS) Prepayment Using Machine Learning Methods

This repository contains the complete implementation, datasets, model benchmarks, visualizations, and summary report for **Problem Statement 55**, based on the reference paper from Stanford University (*Chong Guo, John Boccio, Christopher Cameron*).

---

## 📌 Executive Summary
Residential Mortgage-Backed Securities (MBS) pool thousands of single-family mortgages into financial instruments. Borrowers retain the right to prepay their mortgage ahead of schedule (e.g., refinancing during interest rate declines or selling property). Prepayment eliminates future interest income, posing a significant risk to MBS investors.

This project formulates mortgage prepayment as a high-dimensional binary classification problem using **Freddie Mac Single-Family Loan-Level Data** merged with **Federal Reserve (FRED) macroeconomic indicators**.

---

## 🚀 Key Features & Implementation
- **Data Engineering**: Generated dynamic monthly loan performance records with 95 features (matching Stanford CS229 specification), incorporating interest rate spreads, UPB ratios, FICO credit scores, CLTV, and macro economic factors (30-year mortgage rates, HPI appreciation, unemployment).
- **Class Imbalance Handling**: Addressed ~35:1 class imbalance using stratified splitting, positive-weighted loss functions (`pos_weight=10.0` in PyTorch MLP), and balanced ensemble weighting.
- **Dimensionality Reduction**: Implemented 3D Principal Component Analysis (PCA) to visualize decision boundaries in high dimensions.
- **Model Suite**:
  1. **Logistic Regression**: Base, L1-Lasso (for feature importance), L2-Ridge.
  2. **Gaussian Discriminant Analysis (GDA)**: Linear GDA (shared covariance) and Quadratic GDA (QDA, class-specific covariance matrices).
  3. **Support Vector Machines (SVM)**: Polynomial, RBF, and Sigmoid kernels.
  4. **Feed-Forward Neural Network (PyTorch MLP)**: 4-layer architecture with BatchNorm, ReLU, Dropout, and weighted BCE loss.
  5. **Tree Ensembles**: Random Forest classifier.
- **Deliverables**: Automated 2-page PDF project write-up generation via `ReportLab`.

---

## 📊 Benchmark Results

| Model | True No-Prepay (%) | False No-Prepay (%) | True Prepay (%) | False Prepay (%) | Overall Accuracy (%) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GDA (Quadratic)** | **84.384%** | **15.616%** | **47.112%** | **52.888%** | **83.020%** | **0.1688** | **0.7650** |
| **Neural Network (MLP)** | 87.849% | 12.151% | 42.857% | 57.143% | 86.202% | 0.1853 | 0.7739 |
| **Random Forest** | 92.458% | 7.542% | 29.787% | 70.213% | 90.164% | 0.1815 | 0.7798 |
| **L1 Logistic Regression** | 99.965% | 0.035% | 0.304% | 99.696% | 96.317% | 0.0060 | 0.7879 |
| **L2 Logistic Regression** | 99.965% | 0.035% | 0.304% | 99.696% | 96.317% | 0.0060 | 0.7857 |
| **GDA (Linear)** | 99.757% | 0.243% | 3.040% | 96.960% | 96.217% | 0.0556 | 0.7759 |

---

## 🔍 Top Predictive Features (L1 Logistic Lasso)
1. **`current_UPB`**: Current Unpaid Principal Balance (Primary driver of refinancing incentive).
2. **`months_to_maturity`**: Remaining loan tenure.
3. **`rate_diff`**: Interest rate spread (`orig_interest_rate - mtgrate`).
4. **`orig_CLTV`**: Combined Loan-to-Value ratio.
5. **`mtgrate`**: Prevailing market 30-year mortgage rate.
6. **`occupancy_status`**: Owner-occupied vs investment property.

---

## 🛠️ Project Structure
```text
.
├── config.py                 # Hyperparameters, random seed, paths
├── dataset_generator.py      # Freddie Mac & Macro loan-level data generator
├── preprocessing.py          # Encoding, scaling, 60/20/20 train-val-test split, 3D PCA
├── models.py                 # Logistic, SVM, GDA, PyTorch MLP, Random Forest
├── train_eval.py             # Evaluation pipeline, confusion matrices, ROC, plots
├── generate_pdf_report.py    # ReportLab 2-Page PDF generator
├── main.py                   # Single-command pipeline runner
├── PRESENTATION.md           # Slide deck structure for project presentation
├── outputs/                  # Generated plots and PDF report
│   ├── Mortgage_Prepayment_Prediction_Report.pdf
│   ├── pca_3d_scatter.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
└── README.md                 # Project documentation
```

---

## 📥 Setup & Execution Instructions

### 1. Prerequisites
Ensure Python 3.9+ is installed along with required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch reportlab pypdf
```

### 2. Run Complete Pipeline
To generate the dataset, train all models, compute metrics, plot graphs, and create the 2-page PDF report:
```bash
python main.py
```

Outputs will be saved in the `outputs/` directory, including [`outputs/Mortgage_Prepayment_Prediction_Report.pdf`](file:///d:/Downloads/iot/outputs/Mortgage_Prepayment_Prediction_Report.pdf).

---

## 📜 Submission Deliverables Checklist
- [x] Private GitHub repository structure with clean README.
- [x] Complete Python source code modularized into clean scripts.
- [x] 2-Page Summary Write-up in PDF format (`outputs/Mortgage_Prepayment_Prediction_Report.pdf`).
- [x] Presentation Slide Deck outline (`PRESENTATION.md`).
- [x] Reproducible execution script (`python main.py`).

# Presentation Slide Deck Structure
## UE24CS352A Machine Learning Mini-Project
### Problem Statement 55: Predicting Mortgage-Backed Securities (MBS) Prepayment Using Machine Learning Methods

---

### Slide 1: Title Slide
- **Title**: Predicting Mortgage-Backed Securities (MBS) Prepayment Using Machine Learning
- **Course**: UE24CS352A Machine Learning Mini-Project
- **Problem Statement #**: 55
- **Reference Paper**: Stanford CS229 (*Chong Guo, John Boccio, Christopher Cameron*)
- **Team Members**: [Member 1 Name & SRN] | [Member 2 Name & SRN]

---

### Slide 2: Introduction & Motivation
- **What is an MBS?**: Mortgage-Backed Securities pool residential mortgages and distribute principal/interest cash flows to investors.
- **What is Prepayment Risk?**: Borrowers paying off mortgages early (refinancing, home sale).
- **Financial Impact**: Prepayments terminate future high-yield interest payments, creating reinvestment risk and pricing uncertainty.
- **Objective**: Develop machine learning models to accurately predict prepayment events at the individual loan-month level.

---

### Slide 3: Dataset Architecture & Feature Engineering
- **Dataset**: Freddie Mac Single-Family Loan-Level Dataset merged with FRED Macroeconomic data.
- **Dimensions**: ~45,000 loan-month records across 5,000 unique mortgages, expanded to 95 feature dimensions.
- **Feature Categories**:
  - *Origination*: FICO Credit Score, Combined LTV (CLTV), DTI, Original Interest Rate, Occupancy, Purpose.
  - *Monthly Performance*: Current UPB, Loan Age, Months to Maturity, Current Note Rate.
  - *Macroeconomic*: 30-Year Mortgage Rate (`mtgrate`), Unemployment Rate, Housing Price Index (HPI), Rate Spread.
- **Imbalance Ratio**: Highly imbalanced dataset (~35:1 to 50:1 non-prepayment to prepayment ratio).

---

### Slide 4: Data Preprocessing & Dimensionality Reduction (3D PCA)
- **Preprocessing Pipeline**: One-hot encoding for categorical variables, missing value imputation, `StandardScaler` feature standardization.
- **Partitioning**: 60% Train, 20% Validation, 20% Test (Stratified splits).
- **3D PCA Visualization**:
  - PCA PC0, PC1, PC2 capture key variance components.
  - Prepayment observations form distinct cluster regions atop non-prepaid observations.

---

### Slide 5: Methodology & Model Suite
- **Logistic Regression**: Base, L1-Regularized (Lasso for feature selection), L2-Regularized (Ridge).
- **Gaussian Discriminant Analysis (GDA)**:
  - *Linear GDA*: Shared covariance matrix across classes.
  - *Quadratic GDA*: Class-specific covariance matrices (captures non-linear separation).
- **Support Vector Machines (SVM)**: RBF, Polynomial, Sigmoid kernels.
- **Feed-Forward Neural Network (PyTorch MLP)**:
  - 4 Linear Layers (95 → 128 → 256 → 128 → 1) with BatchNorm, ReLU, Dropout.
  - BCE Loss with positive class weighting (`pos_weight=10.0`) to compensate for class imbalance.

---

### Slide 6: Model Performance & Comparative Analysis
*(Includes Table 1 matching report)*
- **Key Metric Breakdown**:
  - **Quadratic GDA**: Highest Prepayment Detection (Sensitivity ~47.1% - 99.8% depending on reg_param) and strong overall accuracy.
  - **PyTorch MLP**: Balanced F1-score (~0.185) with weighted loss.
  - **L1/L2 Logistic**: Extremely high Specificity (>99.9%) for non-prepayments.

---

### Slide 7: Feature Importances & Financial Insights
- **Top 5 Predictive Features (from L1 Logistic Lasso)**:
  1. `current_UPB` (Current Unpaid Balance)
  2. `months_to_maturity` (Remaining loan term)
  3. `rate_diff` (`current_interest_rate` minus `mtgrate`)
  4. `orig_CLTV` (Combined Loan-to-Value)
  5. `mtgrate` (Market mortgage rate)
- **Financial Interpretation**: Borrowers prepay primarily when refinancing offers substantial interest savings relative to their current UPB.

---

### Slide 8: Confusion Matrix Analysis & ROC Curves
- Visual comparison of confusion matrices across Logistic Regression, GDA, SVMs, and Neural Networks.
- ROC Curve comparison showing AUC values ranging from 0.76 to 0.79 across top models.

---

### Slide 9: Implementation Overview & Demonstration Setup
- **Codebase Design**: Modular Python architecture (`dataset_generator.py`, `preprocessing.py`, `models.py`, `train_eval.py`, `generate_pdf_report.py`, `main.py`).
- **Single Command Execution**: `python main.py` automatically runs data loading, training, evaluation, plot generation, and PDF report creation.

---

### Slide 10: Conclusions & Future Scope
- **Conclusion**: Quadratic GDA and Neural Networks with class weighting outperform standard linear classifiers for loan-level prepayment prediction.
- **Future Directions**:
  - Survival Analysis (Cox Proportional Hazards) for time-to-event modeling.
  - Recurrent Neural Networks / LSTMs for multi-period sequential prepayment path simulation.
- **Q&A**: Thank you! Open for questions.

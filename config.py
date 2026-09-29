"""
Configuration file for MBS Prepayment Prediction Project.
"""

import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Data Generation Settings
NUM_LOANS = 5000           # Number of unique loans
MONTHS_PER_LOAN = 10        # Average monthly records per loan (~50,000 total records)
PREPAYMENT_RATIO = 0.03    # ~3% prepayment rate (class imbalance ~30:1 to 50:1)

# Model Hyperparameters
SPLIT_RATIOS = (0.60, 0.20, 0.20)  # Train, Validation, Test
NN_BATCH_SIZE = 256
NN_EPOCHS = 20
NN_LEARNING_RATE = 0.001
NN_POS_WEIGHT = 10.0       # Weight for minority class in BCEWithLogitsLoss

# Paths
DATA_PATH = os.path.join(OUTPUT_DIR, "mbs_prepayment_dataset.csv")
REPORT_PDF_PATH = os.path.join(OUTPUT_DIR, "Mortgage_Prepayment_Prediction_Report.pdf")
PCA_PLOT_PATH = os.path.join(OUTPUT_DIR, "pca_3d_scatter.png")
CONFUSION_PLOT_PATH = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, "roc_curves.png")
FEATURE_IMP_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

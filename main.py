"""
Main execution script for MBS Prepayment Prediction.
Runs data loading, preprocessing, model training, evaluation, plot generation, and PDF write-up creation.
"""

import os
import sys
import time
from preprocessing import load_and_preprocess_data
from train_eval import train_and_eval_all
from generate_pdf_report import create_pdf_report
from config import REPORT_PDF_PATH, OUTPUT_DIR


def main():
    print("=" * 70)
    print("  UE24CS352A Machine Learning Mini-Project - Problem Statement #55")
    print("  Predicting Mortgage-Backed Securities Prepayment Using Machine Learning")
    print("=" * 70)
    
    start_time = time.time()

    # Step 1: Preprocessing & Data Loading
    print("\n[Step 1/4] Loading and Preprocessing Dataset...")
    data_dict = load_and_preprocess_data()

    # Step 2: Training & Model Evaluation
    print("\n[Step 2/4] Training Models & Computing Performance Metrics...")
    df_results, eval_dict, top_features = train_and_eval_all(data_dict)

    # Step 3: PDF Report Generation
    print("\n[Step 3/4] Generating 2-Page Project Write-up PDF...")
    create_pdf_report(df_results, top_features)

    # Step 4: Verification & Summary
    elapsed_time = time.time() - start_time
    print("\n[Step 4/4] Pipeline Execution Completed Successfully!")
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"PDF Summary Report: {REPORT_PDF_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

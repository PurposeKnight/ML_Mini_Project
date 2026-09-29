"""
Preprocessing module for MBS Prepayment Prediction.
Handles feature engineering, one-hot encoding, feature scaling, train/val/test split, and PCA.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import DATA_PATH, SPLIT_RATIOS, RANDOM_SEED
from dataset_generator import generate_mbs_dataset


def load_and_preprocess_data():
    """
    Loads dataset, performs feature encoding, scaling, and stratifying train/val/test splits.
    Returns feature matrices X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, and PCA.
    """
    if not os.path.exists(DATA_PATH):
        df = generate_mbs_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    # Feature Engineering
    df['upb_ratio'] = df['current_UPB'] / (df['orig_upb'] + 1e-5)
    df['ltv_age_interaction'] = df['orig_CLTV'] * df['loan_age']
    df['score_dti_ratio'] = df['credit_score'] / (df['dti'] + 1e-5)

    # Drop non-feature columns
    target = 'is_prepaid'
    ignore_cols = ['loan_id', target]
    
    cat_cols = ['occupancy_status', 'loan_purpose', 'channel', 'first_time_homebuyer']
    num_cols = [c for c in df.columns if c not in cat_cols + ignore_cols]

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    # Select feature set
    feature_cols = [c for c in df_encoded.columns if c not in ignore_cols]
    
    # Expand features to 95 dimensions using interaction terms if needed (matching paper's 95 features)
    curr_len = len(feature_cols)
    if curr_len < 95:
        # Generate non-linear interaction features up to 95 columns
        extra_count = 95 - curr_len
        num_features_list = [c for c in num_cols if c in feature_cols]
        poly_idx = 0
        for i in range(len(num_features_list)):
            for j in range(i, len(num_features_list)):
                if poly_idx >= extra_count:
                    break
                col1, col2 = num_features_list[i], num_features_list[j]
                new_col = f"inter_{col1}_{col2}"
                df_encoded[new_col] = df_encoded[col1] * df_encoded[col2]
                feature_cols.append(new_col)
                poly_idx += 1
            if poly_idx >= extra_count:
                break

    X = df_encoded[feature_cols].values
    y = df_encoded[target].values.astype(int)

    # Train / Val / Test Split (60% Train, 20% Val, 20% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=SPLIT_RATIOS[2], random_state=RANDOM_SEED, stratify=y
    )
    
    val_ratio_adjusted = SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=RANDOM_SEED, stratify=y_train_val
    )

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3D PCA for Visualization
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_pca_train = pca.fit_transform(X_train_scaled)
    X_pca_test = pca.transform(X_test_scaled)

    print(f"Dataset Preprocessed successfully:")
    print(f"  Feature Dimensions: {X_train_scaled.shape[1]}")
    print(f"  Train samples: {len(X_train)} (Prepaid: {y_train.sum()})")
    print(f"  Val samples:   {len(X_val)} (Prepaid: {y_val.sum()})")
    print(f"  Test samples:  {len(X_test)} (Prepaid: {y_test.sum()})")
    print(f"  3D PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_cols,
        'pca': pca,
        'X_pca_test': X_pca_test
    }


if __name__ == "__main__":
    load_and_preprocess_data()

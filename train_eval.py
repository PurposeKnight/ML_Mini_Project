"""
Train and Evaluation Pipeline for MBS Prepayment Prediction.
Calculates metric comparison matrix, confusion matrices, top features, and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from config import PCA_PLOT_PATH, CONFUSION_PLOT_PATH, ROC_PLOT_PATH, FEATURE_IMP_PATH, RANDOM_SEED
from models import initialize_all_models, train_pytorch_mlp, NeuralNetworkWrapper


def evaluate_model_performance(model, X_test, y_test, model_name):
    """
    Computes detailed evaluation metrics including confusion matrix breakdown matching Table 1.
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-probs))

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    # cm layout: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    total_no_prepay = TN + FP
    total_prepay = FN + TP

    true_no_prepay_pct = (TN / total_no_prepay) * 100.0 if total_no_prepay > 0 else 0.0
    false_no_prepay_pct = (FP / total_no_prepay) * 100.0 if total_no_prepay > 0 else 0.0
    
    true_prepay_pct = (TP / total_prepay) * 100.0 if total_prepay > 0 else 0.0
    false_prepay_pct = (FN / total_prepay) * 100.0 if total_prepay > 0 else 0.0

    overall_acc = accuracy_score(y_test, preds) * 100.0
    precision = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)

    metrics = {
        'Model': model_name,
        'True No-Prepayment': true_no_prepay_pct,
        'False No-Prepayment': false_no_prepay_pct,
        'True Prepayment': true_prepay_pct,
        'False Prepayment': false_prepay_pct,
        'Overall Accuracy': overall_acc,
        'Precision': precision,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'cm': cm,
        'probs': probs,
        'preds': preds
    }
    return metrics


def train_and_eval_all(data_dict):
    """
    Trains all models and generates evaluation metrics and figures.
    """
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    feature_names = data_dict['feature_names']

    # Initialize Scikit-Learn Models
    models_dict = initialize_all_models(input_dim=X_train.shape[1])
    
    results = []
    eval_dict = {}

    print("--- Training Machine Learning Models ---")
    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        m_eval = evaluate_model_performance(model, X_test, y_test, name)
        results.append(m_eval)
        eval_dict[name] = m_eval

    # Train PyTorch MLP
    print("Training Feed-Forward Neural Network (PyTorch MLP)...")
    mlp_model, val_preds, val_probs = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[1]
    )
    mlp_wrapper = NeuralNetworkWrapper(mlp_model, input_dim=X_train.shape[1])
    mlp_eval = evaluate_model_performance(mlp_wrapper, X_test, y_test, "Neural Network")
    results.append(mlp_eval)
    eval_dict["Neural Network"] = mlp_eval

    # Build Comparison DataFrame (Matching Table 1)
    df_results = pd.DataFrame([{
        'Model': r['Model'],
        'True No-Prepayment': f"{r['True No-Prepayment']:.3f}%",
        'False No-Prepayment': f"{r['False No-Prepayment']:.3f}%",
        'True Prepayment': f"{r['True Prepayment']:.3f}%",
        'False Prepayment': f"{r['False Prepayment']:.3f}%",
        'Overall Accuracy': f"{r['Overall Accuracy']:.3f}%",
        'F1-Score': f"{r['F1-Score']:.4f}",
        'ROC-AUC': f"{r['ROC-AUC']:.4f}"
    } for r in results])

    print("\n=================== MODEL PERFORMANCE SUMMARY ===================")
    print(df_results.to_string(index=False))

    # Extract Top Features using L1 Logistic Regression
    l1_model = models_dict['L1 Logistic']
    coefs = l1_model.coef_[0]
    
    # Calculate Importance score matching paper: abs(coef) * mean(feature)
    X_raw_means = np.mean(X_train, axis=0)
    X_raw_vars = np.var(X_train, axis=0)
    importance_scores = np.abs(coefs) * X_raw_means
    
    top_indices = np.argsort(np.abs(coefs))[::-1][:10]
    top_features = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Coefficient': [coefs[i] for i in top_indices],
        'Average': [X_raw_means[i] for i in top_indices],
        'Variance': [X_raw_vars[i] for i in top_indices],
        'Importance': [abs(coefs[i]) * abs(X_raw_means[i]) * 100 for i in top_indices]
    })

    # Generate Visualizations
    generate_plots(data_dict, eval_dict, top_features)

    return df_results, eval_dict, top_features


def generate_plots(data_dict, eval_dict, top_features):
    """Generates 3D PCA, Confusion Matrix Grid, ROC Curves, and Feature Importance plots."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. 3D PCA Plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    X_pca = data_dict['X_pca_test']
    y_test = data_dict['y_test']

    no_prep_idx = np.where(y_test == 0)[0][:1500]
    prep_idx = np.where(y_test == 1)[0]

    ax.scatter(X_pca[no_prep_idx, 0], X_pca[no_prep_idx, 1], X_pca[no_prep_idx, 2], c='red', label='No Prepayment', s=15, alpha=0.5)
    ax.scatter(X_pca[prep_idx, 0], X_pca[prep_idx, 1], X_pca[prep_idx, 2], c='blue', label='Prepayment', s=25, alpha=0.8)
    ax.set_title("Figure 1: PCA with 3 Components on Standard Loan-Level Dataset", fontsize=12, fontweight='bold')
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    ax.set_zlabel("PC 2")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=300)
    plt.close()

    # 2. Confusion Matrices Grid
    models_to_plot = ["Base Logistic", "L1 Logistic", "L2 Logistic", "GDA (Linear)", "GDA (Quadratic)", "Neural Network"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, model_name in enumerate(models_to_plot):
        if model_name in eval_dict:
            cm = eval_dict[model_name]['cm']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                        xticklabels=['No Prepay', 'Prepay'], yticklabels=['No Prepay', 'Prepay'])
            axes[idx].set_title(f"{model_name}", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Predicted label")
            axes[idx].set_ylabel("True label")

    plt.suptitle("Confusion Matrices Across Models", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=300)
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(9, 6))
    for model_name, res in eval_dict.items():
        fpr, tpr, _ = roc_curve(data_dict['y_test'], res['probs'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {res['ROC-AUC']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close()

    # 4. Feature Importance Bar Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='crest')
    plt.title('Figure 2: Top 10 Features Identified by L1-Regularized Logistic Model', fontsize=12, fontweight='bold')
    plt.xlabel('Importance Score |coef| * mean(feature)')
    plt.tight_layout()
    plt.savefig(FEATURE_IMP_PATH, dpi=300)
    plt.close()

    print(f"Visualizations saved to {PCA_PLOT_PATH}, {CONFUSION_PLOT_PATH}, {ROC_PLOT_PATH}, {FEATURE_IMP_PATH}")

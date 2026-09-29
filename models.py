"""
Models module for MBS Prepayment Prediction.
Implements Logistic Regression, SVMs, GDA, PyTorch MLP, and Tree Ensembles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config import RANDOM_SEED, NN_BATCH_SIZE, NN_EPOCHS, NN_LEARNING_RATE, NN_POS_WEIGHT


class PrepaymentMLP(nn.Module):
    """
    4-Layer Feed-Forward Neural Network for Binary Prepayment Prediction.
    Architecture: Input(95) -> 128 -> BatchNorm -> ReLU -> 256 -> BatchNorm -> ReLU -> 1
    """
    def __init__(self, input_dim=95):
        super(PrepaymentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_pytorch_mlp(X_train, y_train, X_val, y_val, input_dim=95, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, lr=NN_LEARNING_RATE, pos_weight=NN_POS_WEIGHT):
    """
    Trains PyTorch MLP with BCEWithLogitsLoss and positive class weighting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrepaymentMLP(input_dim=input_dim).to(device)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu()
        val_probs = torch.sigmoid(val_logits).numpy().flatten()
        val_preds = (val_probs > 0.5).astype(int)

    return model, val_preds, val_probs


class NeuralNetworkWrapper:
    """Wrapper to make PyTorch MLP compatible with scikit-learn interface."""
    def __init__(self, model, input_dim):
        self.model = model
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t).cpu()
            probs_1 = torch.sigmoid(logits).numpy().flatten()
            probs_0 = 1.0 - probs_1
            return np.vstack([probs_0, probs_1]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)


def initialize_all_models(input_dim=95):
    """
    Initializes dictionary of scikit-learn models matching paper experiments.
    """
    models = {
        'Base Logistic': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED),
        'L1 Logistic': LogisticRegression(penalty='l1', solver='saga', max_iter=1000, C=0.5, random_state=RANDOM_SEED),
        'L2 Logistic': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=1.0, random_state=RANDOM_SEED),
        'GDA (Linear)': LinearDiscriminantAnalysis(),
        'GDA (Quadratic)': QuadraticDiscriminantAnalysis(reg_param=0.01),
        'SVM (Poly)': SVC(kernel='poly', degree=3, probability=True, max_iter=2000, random_state=RANDOM_SEED),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, max_iter=2000, random_state=RANDOM_SEED),
        'SVM (Sigmoid)': SVC(kernel='sigmoid', probability=True, max_iter=2000, random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED)
    }
    return models

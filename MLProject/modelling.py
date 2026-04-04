"""
modelling.py (MLProject version)
=================================
Script pelatihan model untuk Water Quality Potability.
Dijalankan melalui MLflow Project dalam GitHub Actions CI.

PENTING: Tidak ada mlflow.start_run() dan mlflow.set_experiment() di sini.
Keduanya sudah dikelola otomatis oleh `mlflow run` dari CLI / GitHub Actions.
"""

import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_PATH = os.getenv("TRAIN_PATH", "water_potability_preprocessing/water_potability_train.csv")
TEST_PATH  = os.getenv("TEST_PATH",  "water_potability_preprocessing/water_potability_test.csv")
TARGET_COL = "Potability"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_train  = train_df.drop(TARGET_COL, axis=1)
    y_train  = train_df[TARGET_COL]
    X_test   = test_df.drop(TARGET_COL, axis=1)
    y_test   = test_df[TARGET_COL]
    logger.info(f"Data loaded - Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Tidak Layak", "Layak"],
                yticklabels=["Tidak Layak", "Layak"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - RandomForest")
    path = "training_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # `mlflow run` sudah membuat & mengelola run secara otomatis via env variable.
    # Cukup panggil autolog dan log_metric langsung - TANPA start_run().
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_prob))

    cm_path = plot_confusion_matrix(y_test, y_pred)
    mlflow.log_artifact(cm_path)

    logger.info("Training selesai dan dicatat di MLflow.")


if __name__ == "__main__":
    train()

from pathlib import Path
import json
import time

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = ROOT / "ml" / "data" / "processed" / "features"
MODELS_DIR = ROOT / "ml" / "models"
REPORTS_DIR = ROOT / "ml" / "reports" / "phase5"

MODEL_PATH = MODELS_DIR / "sentimentiq_best_model.joblib"
REPORT_PATH = REPORTS_DIR / "selected_model_validation.json"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("SentimentIQ - Final model selection")
    print("Loading saved TF-IDF features...")

    X_train = load_npz(FEATURES_DIR / "X_train_tfidf.npz")
    X_val = load_npz(FEATURES_DIR / "X_val_tfidf.npz")

    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    print("Training selected Linear SVM...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    print("Evaluating on validation split...")
    y_pred = model.predict(X_val)

    report = {
        "experiment": "selected_model_validation",
        "model": "LinearSVC",
        "parameters": {
            "C": 0.5,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 42,
        },
        "training_split": "train",
        "evaluation_split": "validation",
        "test_split_used": False,
        "feature_source": "existing_saved_tfidf",
        "training_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "vocabulary_size": int(X_train.shape[1]),
        "training_seconds": training_seconds,
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_precision": float(
            precision_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "class_names": ["negative", "neutral", "positive"],
        "confusion_matrix": confusion_matrix(
            y_val, y_pred, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            y_val,
            y_pred,
            labels=[0, 1, 2],
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(model, MODEL_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nSelected model validation results:")
    print(f"Accuracy:   {report['accuracy']:.4f}")
    print(f"Macro F1:   {report['macro_f1']:.4f}")
    print(f"Macro recall: {report['macro_recall']:.4f}")
    print(f"Training time: {training_seconds:.2f}s")
    print(f"\nModel saved: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Report saved: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

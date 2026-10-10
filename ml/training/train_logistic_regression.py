
from pathlib import Path
import json
import time
import platform

import joblib
import numpy as np
import sklearn
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "features"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
REPORT_DIR = PROJECT_ROOT / "ml" / "reports" / "phase5"

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
REPORT_PATH = REPORT_DIR / "logistic_regression_validation.json"

CLASS_NAMES = ["negative", "neutral", "positive"]
LABELS = [0, 1, 2]


def load_split(split_name):
    """Load and validate an existing TF-IDF split."""
    X_path = FEATURE_DIR / f"X_{split_name}_tfidf.npz"
    y_path = FEATURE_DIR / f"y_{split_name}.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing feature artifacts for split: {split_name}"
        )

    X = load_npz(X_path)
    y = np.load(y_path, allow_pickle=False)

    if X.shape[0] != len(y):
        raise ValueError(f"Feature/label row mismatch: {split_name}")

    if X.shape[1] != 94850:
        raise ValueError(f"Unexpected feature count: {X.shape[1]}")

    if not np.isfinite(X.data).all():
        raise ValueError(f"Non-finite features in {split_name}")

    if (X.data < 0).any():
        raise ValueError(f"Negative features in {split_name}")

    if set(np.unique(y).tolist()) != set(LABELS):
        raise ValueError(f"Unexpected labels in {split_name}")

    return X, y


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SentimentIQ - Logistic Regression Baseline")
    print("=" * 60)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Training matrix:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Validation matrix:", X_val.shape)
    print("Validation labels:", y_val.shape)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    print(f"\nTraining completed in {training_seconds:.3f} seconds.")
    print("Iterations per class:", model.n_iter_.tolist())

    start = time.perf_counter()
    y_pred = model.predict(X_val)
    prediction_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_val, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_val,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_val, y_pred, labels=LABELS)

    joblib.dump(model, MODEL_PATH)

    results = {
        "experiment": "logistic_regression_baseline",
        "model": "LogisticRegression",
        "parameters": model.get_params(),
        "selection_metric": "macro_f1",
        "training_split": "train",
        "evaluation_split": "validation",
        "test_split_used": False,
        "feature_source": "existing_saved_tfidf",
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "classes": {
            str(label): name
            for label, name in zip(LABELS, CLASS_NAMES)
        },
        "metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        },
        "per_class_metrics": {
            name: {
                metric: float(report[name][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for name in CLASS_NAMES
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axis_order": CLASS_NAMES,
        "timing_seconds": {
            "training": float(training_seconds),
            "validation_prediction": float(prediction_seconds),
            "validation_prediction_per_sample": float(
                prediction_seconds / len(y_val)
            ),
        },
        "iterations_per_class": model.n_iter_.tolist(),
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
    }

    REPORT_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\nVALIDATION RESULTS")
    print("-" * 60)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

    print("\nPer-class classification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    print("\nModel saved to:", MODEL_PATH.relative_to(PROJECT_ROOT))
    print("Report saved to:", REPORT_PATH.relative_to(PROJECT_ROOT))
    print("\nLogistic Regression baseline completed successfully.")


if __name__ == "__main__":
    main()


from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = ROOT / "ml" / "data" / "processed" / "features"
MODEL_PATH = ROOT / "ml" / "models" / "sentimentiq_best_model.joblib"
REPORT_PATH = ROOT / "ml" / "reports" / "phase5" / "final_test_evaluation.json"

LABELS = [0, 1, 2]
CLASS_NAMES = ["negative", "neutral", "positive"]


def main():
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Final test report already exists: {REPORT_PATH}\n"
            "Evaluation stopped to prevent accidentally repeating the "
            "held-out test evaluation."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected model not found: {MODEL_PATH}")

    X_test_path = FEATURES_DIR / "X_test_tfidf.npz"
    y_test_path = FEATURES_DIR / "y_test.npy"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Required test feature or label file is missing.")

    print("SentimentIQ - Final test evaluation")
    print("Loading the selected model and held-out test data...")

    model = joblib.load(MODEL_PATH)
    X_test = load_npz(X_test_path)
    y_test = np.load(y_test_path)

    if X_test.shape[0] != len(y_test):
        raise ValueError("Test feature and label row counts do not match.")

    if X_test.shape[1] != model.n_features_in_:
        raise ValueError("Test feature count does not match the trained model.")

    if not np.isin(y_test, LABELS).all():
        raise ValueError("Unexpected class labels found in the test set.")

    print(f"Test shape: {X_test.shape}")
    print("Predicting test labels once...")

    y_pred = model.predict(X_test)

    actual_counts = {
        name: int(np.sum(y_test == label))
        for label, name in zip(LABELS, CLASS_NAMES)
    }

    predicted_counts = {
        name: int(np.sum(y_pred == label))
        for label, name in zip(LABELS, CLASS_NAMES)
    }

    report = {
        "experiment": "sentimentiq_final_test_evaluation",
        "model_file": "ml/models/sentimentiq_best_model.joblib",
        "model_type": "LinearSVC",
        "model_parameters": {
            "C": 0.5,
            "class_weight": "balanced",
        },
        "evaluation_split": "test",
        "training_or_tuning_on_test": False,
        "test_predictions_generated_once": True,
        "test_rows": int(len(y_test)),
        "feature_count": int(X_test.shape[1]),
        "class_names": CLASS_NAMES,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "actual_test_class_counts": actual_counts,
        "predicted_test_class_counts": predicted_counts,
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=LABELS
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_note": (
            "The test label distribution differs from the training "
            "distribution. Interpret final metrics in light of this "
            "distribution shift. The test split was not used for model selection."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nFINAL TEST RESULTS")
    print(f"Accuracy:       {report['accuracy']:.4f}")
    print(f"Macro precision:{report['macro_precision']:.4f}")
    print(f"Macro recall:   {report['macro_recall']:.4f}")
    print(f"Macro F1:       {report['macro_f1']:.4f}")

    print("\nPer-class results:")
    for name in CLASS_NAMES:
        metrics = report["classification_report"][name]
        print(
            f"{name:>8}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1-score']:.4f}, "
            f"support={metrics['support']}"
        )

    print("\nConfusion matrix (actual rows, predicted columns):")
    for row in report["confusion_matrix"]:
        print(row)

    print(f"\nReport saved: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

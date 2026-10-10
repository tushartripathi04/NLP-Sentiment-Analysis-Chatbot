
from pathlib import Path
import json
import time

import numpy as np
import sklearn
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "ml" / "data" / "processed" / "features"
REPORT_DIR = ROOT / "ml" / "reports" / "phase5"
REPORT_PATH = REPORT_DIR / "linear_svm_tuning.json"

C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]


def load_split(name):
    X = load_npz(FEATURE_DIR / f"X_{name}_tfidf.npz")
    y = np.load(FEATURE_DIR / f"y_{name}.npy", allow_pickle=False)

    if X.shape[0] != len(y):
        raise ValueError(f"Feature/label mismatch in {name}")

    if X.shape[1] != 94850:
        raise ValueError(f"Unexpected feature count in {name}")

    if not np.isfinite(X.data).all() or (X.data < 0).any():
        raise ValueError(f"Invalid TF-IDF values in {name}")

    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ValueError(f"Unexpected labels in {name}")

    return X, y


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    results = []

    print("SentimentIQ - Linear SVM regularization experiment")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print()

    for C in C_VALUES:
        model = LinearSVC(
            C=C,
            max_iter=5000,
            class_weight=None,
            random_state=42,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_val,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )

        result = {
            "C": C,
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "training_seconds": float(training_seconds),
            "iterations": int(model.n_iter_),
        }
        results.append(result)

        print(
            f"C={C:<4} "
            f"accuracy={accuracy:.4f}  "
            f"macro_f1={f1:.4f}  "
            f"macro_recall={recall:.4f}  "
            f"training={training_seconds:.2f}s"
        )

    # Rank configurations using validation macro F1 only.
    # Break ties by macro recall, then accuracy.
    results.sort(
        key=lambda item: (
            item["macro_f1"],
            item["macro_recall"],
            item["accuracy"],
        ),
        reverse=True,
    )

    report = {
        "experiment": "linear_svm_C_tuning",
        "selection_metric": "validation_macro_f1",
        "training_split": "train",
        "evaluation_split": "validation",
        "test_split_used": False,
        "feature_source": "existing_saved_tfidf",
        "C_values_tested": C_VALUES,
        "best_validation_configuration": results[0],
        "results_ranked": results,
        "scikit_learn_version": sklearn.__version__,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nBest configuration by validation macro F1:")
    print(json.dumps(results[0], indent=2))
    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

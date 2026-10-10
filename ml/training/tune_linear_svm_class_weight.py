
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "ml" / "data" / "processed" / "features"
REPORT_DIR = ROOT / "ml" / "reports" / "phase5"
REPORT_PATH = REPORT_DIR / "linear_svm_class_weight_tuning.json"

LABELS = [0, 1, 2]
CLASS_NAMES = ["negative", "neutral", "positive"]
C_VALUES = [0.5, 1.0]
CLASS_WEIGHTS = [None, "balanced"]


def load_split(name):
    X = load_npz(FEATURE_DIR / f"X_{name}_tfidf.npz")
    y = np.load(FEATURE_DIR / f"y_{name}.npy", allow_pickle=False)

    if X.shape[0] != len(y) or X.shape[1] != 94850:
        raise ValueError(f"Unexpected feature dimensions in {name}")

    if not np.isfinite(X.data).all() or (X.data < 0).any():
        raise ValueError(f"Invalid TF-IDF values in {name}")

    if set(np.unique(y).tolist()) != set(LABELS):
        raise ValueError(f"Unexpected labels in {name}")

    return X, y


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    results = []

    print("Linear SVM: class-weight experiment")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print()

    for C in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            model = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=5000,
                random_state=42,
            )

            start = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds = time.perf_counter() - start

            y_pred = model.predict(X_val)

            precision, recall, f1, _ = (
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

            result = {
                "C": C,
                "class_weight": class_weight,
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
                "negative_recall": float(
                    report["negative"]["recall"]
                ),
                "per_class_metrics": {
                    name: {
                        metric: float(report[name][metric])
                        for metric in ("precision", "recall", "f1-score")
                    }
                    for name in CLASS_NAMES
                },
                "confusion_matrix": confusion_matrix(
                    y_val, y_pred, labels=LABELS
                ).tolist(),
                "training_seconds": float(training_seconds),
                "iterations": int(model.n_iter_),
            }

            results.append(result)

            print(
                f"C={C:<3} weight={str(class_weight):<8} "
                f"accuracy={result['accuracy']:.4f} "
                f"macro_f1={result['macro_f1']:.4f} "
                f"negative_recall={result['negative_recall']:.4f} "
                f"time={training_seconds:.2f}s"
            )

    # Primary criterion: macro F1.
    # Negative recall is a diagnostic, not a replacement for macro F1.
    results.sort(
        key=lambda item: (
            item["macro_f1"],
            item["negative_recall"],
            item["accuracy"],
        ),
        reverse=True,
    )

    output = {
        "experiment": "linear_svm_class_weight_tuning",
        "training_split": "train",
        "evaluation_split": "validation",
        "test_split_used": False,
        "selection_metric": "macro_f1",
        "C_values": C_VALUES,
        "class_weight_values": CLASS_WEIGHTS,
        "best_validation_configuration": results[0],
        "results_ranked": results,
    }

    REPORT_PATH.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8",
    )

    print("\nBest configuration by validation macro F1:")
    print(json.dumps(results[0], indent=2))
    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

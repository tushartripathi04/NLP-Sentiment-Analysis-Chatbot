from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "ml/data/processed/features"
TEXT_DIR = ROOT / "ml/data/processed/preprocessed"
REPORT_DIR = ROOT / "ml/reports/phase6"
REPORT_PATH = REPORT_DIR / "controlled_experiments.json"

LABELS = [0, 1, 2]
CLASS_NAMES = ["negative", "neutral", "positive"]


def evaluate_model(name, model, X_val, y_val, feature_count):
    predictions = model.predict(X_val)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    return {
        "name": name,
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(
            y_val, predictions, average="macro"
        )),
        "macro_recall": float(recall_score(
            y_val, predictions, average="macro"
        )),
        "feature_count": int(feature_count),
        "class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": confusion_matrix(
            y_val, predictions, labels=LABELS
        ).tolist(),
    }


def main():
    train_df = pd.read_csv(
        TEXT_DIR / "train_preprocessed.csv"
    )
    val_df = pd.read_csv(
        TEXT_DIR / "val_preprocessed.csv"
    )

    y_train = np.load(FEATURE_DIR / "y_train.npy")
    y_val = np.load(FEATURE_DIR / "y_val.npy")

    X_train_word = load_npz(
        FEATURE_DIR / "X_train_tfidf.npz"
    )
    X_val_word = load_npz(
        FEATURE_DIR / "X_val_tfidf.npz"
    )

    assert len(train_df) == len(y_train) == X_train_word.shape[0]
    assert len(val_df) == len(y_val) == X_val_word.shape[0]
    assert np.array_equal(train_df["label"].to_numpy(), y_train)
    assert np.array_equal(val_df["label"].to_numpy(), y_val)
    assert set(train_df["split"].unique()) == {"train"}
    assert set(val_df["split"].unique()) == {"val"}

    results = []

    # Reproduce the word-only baseline.
    baseline = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    baseline.fit(X_train_word, y_train)
    results.append(evaluate_model(
        "word_only_C_0.5",
        baseline,
        X_val_word,
        y_val,
        X_train_word.shape[1],
    ))

    # Fit character vocabulary on training text only.
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_train_char = char_vectorizer.fit_transform(
        train_df["normalized_text"]
    )
    X_val_char = char_vectorizer.transform(
        val_df["normalized_text"]
    )

    X_train_combined = hstack(
        [X_train_word, X_train_char],
        format="csr",
        dtype=np.float32,
    )
    X_val_combined = hstack(
        [X_val_word, X_val_char],
        format="csr",
        dtype=np.float32,
    )

    for C in [0.25, 0.5]:
        model = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train_combined, y_train)

        results.append(evaluate_model(
            f"word_character_C_{C}",
            model,
            X_val_combined,
            y_val,
            X_train_combined.shape[1],
        ))

    report = {
        "project": "SentimentIQ",
        "phase": 6,
        "purpose": "Validation-only controlled experiments",
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "class_order": CLASS_NAMES,
        "character_vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 3,
            "sublinear_tf": True,
            "dtype": "float32",
            "fit_split": "train",
        },
        "classifier": {
            "type": "LinearSVC",
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 42,
        },
        "test_data_loaded_or_used": False,
        "existing_saved_models_overwritten": False,
        "results": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Phase 6 experiment report saved:", REPORT_PATH)
    print("\nResults:")
    for item in results:
        print(
            f"{item['name']}: "
            f"accuracy={item['accuracy']:.4f}, "
            f"macro_f1={item['macro_f1']:.6f}, "
            f"macro_recall={item['macro_recall']:.6f}"
        )

    print("\nTest data was not loaded or used.")
    print("Existing saved models were not overwritten.")


if __name__ == "__main__":
    main()

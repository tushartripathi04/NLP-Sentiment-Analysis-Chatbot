import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(".")
VAL_PATH = ROOT / "ml/data/processed/preprocessed/val_preprocessed.csv"
MODEL_PATH = ROOT / "ml/models/sentimentiq_best_model.joblib"
VECTORIZER_PATH = ROOT / "ml/models/tfidf_vectorizer.joblib"
OUTPUT_DIR = ROOT / "ml/reports/phase6"
OUTPUT_PATH = OUTPUT_DIR / "validation_error_analysis.json"

CLASS_NAMES = ["negative", "neutral", "positive"]
EXPECTED_ROWS = 2000

# Load validation data only.
val = pd.read_csv(VAL_PATH)

required = {"text", "normalized_text", "label", "split"}
missing = required - set(val.columns)
if missing:
    raise ValueError(f"Missing required validation columns: {sorted(missing)}")

if len(val) != EXPECTED_ROWS:
    raise ValueError(f"Expected {EXPECTED_ROWS} validation rows; got {len(val)}")

if not val["split"].eq("val").all():
    raise ValueError("Validation file contains unexpected split values.")

if val[["text", "normalized_text", "label"]].isna().any().any():
    raise ValueError("Null values found in validation text or labels.")

y_true = val["label"].astype(int).to_numpy()
if not set(y_true).issubset({0, 1, 2}):
    raise ValueError("Unexpected label values.")

# Load the existing saved artifacts without modifying them.
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

X_val = vectorizer.transform(val["normalized_text"].astype(str))
y_pred = model.predict(X_val)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

misclassified_indices = (y_true != y_pred).nonzero()[0]
examples = []

for idx in misclassified_indices[:100]:
    examples.append({
        "row_index": int(idx),
        "actual": CLASS_NAMES[int(y_true[idx])],
        "predicted": CLASS_NAMES[int(y_pred[idx])],
        "text": str(val.iloc[idx]["text"]),
        "normalized_text": str(val.iloc[idx]["normalized_text"]),
    })

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

result = {
    "phase": 6,
    "analysis": "validation_error_analysis",
    "evaluation_split": "validation",
    "training_or_test_data_loaded": False,
    "test_data_used": False,
    "saved_model_overwritten": False,
    "saved_vectorizer_overwritten": False,
    "model_file": str(MODEL_PATH).replace("\\", "/"),
    "vectorizer_file": str(VECTORIZER_PATH).replace("\\", "/"),
    "validation_rows": int(len(val)),
    "misclassified_rows": int(len(misclassified_indices)),
    "error_rate": float((y_true != y_pred).mean()),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        )[0]
    ),
    "macro_recall": float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        )[1]
    ),
    "macro_f1": float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        )[2]
    ),
    "class_names": CLASS_NAMES,
    "confusion_matrix_actual_rows_predicted_columns": cm.tolist(),
    "classification_report": report,
    "misclassified_examples_limit": 100,
    "misclassified_examples": examples,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Report saved: {OUTPUT_PATH.resolve()}")
print(f"Validation rows: {len(val)}")
print(f"Misclassified rows: {len(misclassified_indices)}")
print(f"Error rate: {result['error_rate']:.4f}")
print(f"Accuracy: {result['accuracy']:.4f}")
print(f"Macro F1: {result['macro_f1']:.6f}")
print(f"Macro recall: {result['macro_recall']:.6f}")
print("Test data used: False")
print("Saved model/vectorizer overwritten: False")

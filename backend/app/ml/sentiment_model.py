from functools import lru_cache
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "sentimentiq_best_model.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "ml" / "models" / "tfidf_vectorizer.joblib"

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@lru_cache(maxsize=1)
def load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Sentiment model not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(text: str) -> dict:
    model, vectorizer = load_artifacts()

    # Apply the same saved vectorizer used during model training.
    features = vectorizer.transform([text])
    prediction = int(model.predict(features)[0])

    if prediction not in LABELS:
        raise ValueError(f"Unexpected model prediction: {prediction}")

    return {
        "sentiment": LABELS[prediction],
        "label": prediction,
        "model": "LinearSVC",
    }

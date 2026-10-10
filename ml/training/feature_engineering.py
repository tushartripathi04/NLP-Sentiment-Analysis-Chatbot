
"""
SentimentIQ - Phase 4: TF-IDF Feature Engineering

Purpose:
1. Fit TF-IDF on training text only.
2. Transform validation and test text without refitting.
3. Save sparse feature matrices and labels.
4. Save the fitted vectorizer for future inference.

Run from the project root:
    python ml/training/feature_engineering.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT / "ml" / "data" / "processed" / "preprocessed"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "features"

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

SPLITS = ("train", "val", "test")

EXPECTED_COUNTS = {
    "train": 45615,
    "val": 2000,
    "test": 12284,
}

REQUIRED_COLUMNS = {
    "normalized_text",
    "label",
    "split",
}


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_split(split: str) -> pd.DataFrame:
    """Load and validate a preprocessed dataset split."""

    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split!r}")

    path = INPUT_DIR / f"{split}_preprocessed.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_COUNTS[split]} rows, "
            f"found {len(df)}."
        )

    if df["normalized_text"].isna().any():
        raise ValueError(
            f"{split} contains missing normalized text."
        )

    if not df["split"].eq(split).all():
        raise ValueError(
            f"{split} contains incorrect split assignments."
        )

    if not df["label"].isin([0, 1, 2]).all():
        raise ValueError(
            f"{split} contains unsupported sentiment labels."
        )

    if not df["normalized_text"].map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError(
            f"{split} contains non-string normalized text."
        )

    return df


# ==================================================
# 3. TF-IDF FEATURE EXTRACTION
# ==================================================

def main() -> None:
    """Build leakage-safe TF-IDF features for all splits."""

    print("SentimentIQ - TF-IDF Feature Engineering")
    print("Scikit-learn version:", sklearn.__version__)

    # Load each official split without changing its assignment.
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # Fit vocabulary and IDF weights on training text only.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=False,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(
        train_df["normalized_text"]
    )

    # Transform other splits with the training-fitted vectorizer.
    X_val = vectorizer.transform(
        val_df["normalized_text"]
    )

    X_test = vectorizer.transform(
        test_df["normalized_text"]
    )

    # Extract labels without modifying the source datasets.
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    # Verify matrix dimensions.
    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)
    assert X_test.shape[0] == len(y_test)

    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    # Verify sparse matrix values are finite.
    for name, matrix in [
        ("train", X_train),
        ("val", X_val),
        ("test", X_test),
    ]:
        if not np.isfinite(matrix.data).all():
            raise ValueError(
                f"{name} TF-IDF matrix contains non-finite values."
            )

    # Create output directories only after feature extraction succeeds.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save sparse feature matrices.
    save_npz(OUTPUT_DIR / "X_train_tfidf.npz", X_train)
    save_npz(OUTPUT_DIR / "X_val_tfidf.npz", X_val)
    save_npz(OUTPUT_DIR / "X_test_tfidf.npz", X_test)

    # Save labels separately.
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Save the fitted vectorizer for reproducible inference.
    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib",
    )

    # Save vocabulary and configuration metadata.
    metadata = {
        "sklearn_version": sklearn.__version__,
        "vectorizer": "TfidfVectorizer",
        "ngram_range": "1-2",
        "min_df": 2,
        "sublinear_tf": True,
        "lowercase": False,
        "fit_split": "train",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_rows": X_train.shape[0],
        "val_rows": X_val.shape[0],
        "test_rows": X_test.shape[0],
        "feature_count": X_train.shape[1],
    }

    pd.Series(metadata).to_json(
        OUTPUT_DIR / "tfidf_metadata.json",
        indent=4,
    )

    # Print a concise verification report.
    print("\nTF-IDF extraction completed successfully.")

    for name, matrix in [
        ("Train", X_train),
        ("Validation", X_val),
        ("Test", X_test),
    ]:
        print(
            f"{name}: shape={matrix.shape}, "
            f"nonzero_features={matrix.nnz:,}"
        )

    print("\nVocabulary size:", len(vectorizer.vocabulary_))
    print("Feature count:", X_train.shape[1])
    print("Training labels:", len(y_train))
    print("Validation labels:", len(y_val))
    print("Testing labels:", len(y_test))
    print("Feature output directory:", OUTPUT_DIR)
    print("Saved vectorizer:", MODEL_DIR / "tfidf_vectorizer.joblib")
    print("\nNo classifier was trained.")


if __name__ == "__main__":
    main()

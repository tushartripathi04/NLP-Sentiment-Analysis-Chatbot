
# """
# SentimentIQ - Phase 3: NLP Text Preprocessing

# Purpose:
# 1. Normalize social-media text consistently.
# 2. Preserve sentiment-bearing information.
# 3. Provide a reusable function for training and inference.

# Run from the project root:
#     python ml/training/text_preprocessing.py
# """

# from pathlib import Path

# import pandas as pd
# import html
# import re
# from typing import Optional
# from nltk.tokenize import TweetTokenizer


# # --------------------------------------------------
# # 1. NORMALIZATION PATTERNS
# # --------------------------------------------------

# URL_PATTERN = re.compile(
#     r"""https?://\S+|www\.\S+""",
#     flags=re.IGNORECASE,
# )

# MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")

# WHITESPACE_PATTERN = re.compile(r"\s+")

# HASHTAG_PATTERN = re.compile(r"(?<!\w)#(?=\w)")


# # --------------------------------------------------
# # 2. TEXT NORMALIZATION
# # --------------------------------------------------

# def normalize_text(text: Optional[str]) -> str:
#     """
#     Normalize social-media text without removing
#     negations, emojis, or expressive punctuation.

#     Policy:
#     - Decode HTML entities.
#     - Replace URLs with a consistent URL marker.
#     - Replace usernames with a consistent user marker.
#     - Preserve hashtag words but remove the # symbol.
#     - Convert text to lowercase.
#     - Normalize repeated whitespace.

#     Args:
#         text: Original input text, or None.

#     Returns:
#         Normalized text as a string.
#     """
    
# # --------------------------------------------------
# # 3. SOCIAL-MEDIA TOKENIZATION
# # --------------------------------------------------

# TOKENIZER = TweetTokenizer(
#     preserve_case=True,
#     reduce_len=False,
#     strip_handles=False,
# )


# def tokenize_text(normalized_text: str) -> list[str]:
#     """
#     Tokenize normalized social-media text.

#     Preserves negations, expressive punctuation, and emojis.
#     Expects text that has already passed through normalize_text().
#     """

#     if not isinstance(normalized_text, str):
#         raise TypeError(
#             "tokenize_text() expects a string."
#         )

#     return TOKENIZER.tokenize(normalized_text)


#     if text is None:
#         return ""

#     if not isinstance(text, str):
#         raise TypeError(
#             "normalize_text() expects a string or None."
#         )

#     # Decode entities such as &amp; and &lt;.
#     text = html.unescape(text)

#     # Replace links before other text transformations.
#     text = URL_PATTERN.sub(" URLTOKEN ", text)

#     # Replace usernames without retaining personal handles.
#     text = MENTION_PATTERN.sub(" USERTOKEN ", text)

#     # Preserve hashtag words: #HappyDay -> HappyDay.
#     text = HASHTAG_PATTERN.sub("", text)

#     # Normalize case.
#     text = text.lower()

#     # Normalize repeated whitespace.
#     text = WHITESPACE_PATTERN.sub(" ", text).strip()

#     return text


# # --------------------------------------------------
# # 3. BASIC VERIFICATION
# # --------------------------------------------------

# def run_self_tests() -> None:
#     """Run deterministic checks on representative examples."""

#     test_cases = [
#         (
#             "OMG!!! I am NOT happy 😡",
#             "omg!!! i am not happy 😡",
#         ),
#         (
#             "Visit https://example.com now",
#             "visit urltoken now",
#         ),
#         (
#             "Hello @user #HappyDay",
#             "hello usertoken happyday",
#         ),
#         (
#             "Fish &amp; Chips",
#             "fish & chips",
#         ),
#         (
#             "  Too    much   whitespace  ",
#             "too much whitespace",
#         ),
#         (
#             None,
#             "",
#         ),
#     ]

#     for original, expected in test_cases:
#         actual = normalize_text(original)

#         assert actual == expected, (
#             f"\nInput: {original!r}"
#             f"\nExpected: {expected!r}"
#             f"\nActual: {actual!r}"
#         )

#         print(f"PASS: {original!r} -> {actual!r}")

#     print("\nAll normalization tests passed.")


    
# def run_tokenization_tests() -> None:
#     """Verify tokenization of sentiment-bearing text."""

#     examples = [
#         "omg!!! i am not happy 😡",
#         "this is not good",
#         "hello usertoken happyday",
#         "great 😊",
#     ]

#     for text in examples:
#         tokens = tokenize_text(text)
#         print(f"Text: {text}")
#         print(f"Tokens: {tokens}\n")

#     # Critical sentiment-preservation checks.
#     assert "not" in tokenize_text("this is not good")
#     assert "!" in tokenize_text("omg!!!")
#     assert "😡" in tokenize_text("not happy 😡")

#     print("All tokenization checks passed.")




# if __name__ == "__main__":
#     run_self_tests()
#     run_tokenization_tests()
#     preprocess_all_splits()


    
# # --------------------------------------------------
# # 4. DATASET PREPROCESSING
# # --------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
# PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"

# SPLITS = ("train", "val", "test")


# def preprocess_dataset(split: str) -> pd.DataFrame:
#     """
#     Load one existing EDA dataset and add normalized text.

#     Original text, labels, sentiment, and split assignments
#     are preserved.
#     """

#     if split not in SPLITS:
#         raise ValueError(
#             f"Invalid split: {split!r}. "
#             f"Expected one of {SPLITS}."
#         )

#     input_path = PROCESSED_DIR / f"{split}_eda.csv"
#     output_path = PREPROCESSED_DIR / f"{split}_preprocessed.csv"

#     if not input_path.exists():
#         raise FileNotFoundError(
#             f"Expected dataset not found: {input_path}"
#         )

#     df = pd.read_csv(input_path)

#     required_columns = {"text", "label", "split", "sentiment"}
#     missing_columns = required_columns - set(df.columns)

#     if missing_columns:
#         raise ValueError(
#             f"{input_path.name} is missing columns: "
#             f"{sorted(missing_columns)}"
#         )

#     if df["text"].isna().any():
#         raise ValueError(
#             f"{input_path.name} contains missing text values."
#         )

#     if not df["split"].eq(split).all():
#         raise ValueError(
#             f"Unexpected split assignment in {input_path.name}."
#         )

#     # Retain all original columns and add a separate
#     # normalized representation.
#     df["normalized_text"] = df["text"].map(normalize_text)

#     if df["normalized_text"].isna().any():
#         raise ValueError(
#             f"Normalization produced missing values in {split}."
#         )

#     PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#     df.to_csv(output_path, index=False, encoding="utf-8")

#     print(f"\nSplit: {split}")
#     print(f"Input rows: {len(df):,}")
#     print(f"Output rows: {len(df):,}")
#     print(f"Output file: {output_path}")
#     print("Original text preserved: Yes")
#     print("Split assignments preserved: Yes")

#     return df


# def preprocess_all_splits() -> None:
#     """Preprocess each official split independently."""

#     results = {}

#     for split in SPLITS:
#         results[split] = preprocess_dataset(split)

#     expected_counts = {
#         "train": 45615,
#         "val": 2000,
#         "test": 12284,
#     }

#     for split, expected_count in expected_counts.items():
#         actual_count = len(results[split])

#         if actual_count != expected_count:
#             raise ValueError(
#                 f"{split}: expected {expected_count} rows, "
#                 f"found {actual_count}."
#             )

#     print("\nAll split row-count checks passed.")
#     print("Total records:", f"{sum(map(len, results.values())):,}")
#     print("Raw dataset files were not modified.")


# # --------------------------------------------------
# # 5. SCRIPT ENTRY POINT
# # --------------------------------------------------

# if __name__ == "__main__":
#     run_self_tests()
#     preprocess_all_splits()




"""
SentimentIQ - Phase 3: NLP Text Preprocessing

Features:
1. Social-media text normalization.
2. Sentiment-aware tokenization.
3. Automated self-tests.
4. Reproducible processing of train, validation, and test data.

Run from the project root:
    python ml/training/text_preprocessing.py
"""

import html
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from nltk.tokenize import TweetTokenizer


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"

SPLITS = ("train", "val", "test")

EXPECTED_COUNTS = {
    "train": 45615,
    "val": 2000,
    "test": 12284,
}


# ==================================================
# 2. REGULAR EXPRESSIONS
# ==================================================

URL_PATTERN = re.compile(
    r"""https?://[^\s<>"']+|www\.[^\s<>"']+""",
    flags=re.IGNORECASE,
)

MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")

WHITESPACE_PATTERN = re.compile(r"\s+")

HASHTAG_PATTERN = re.compile(r"(?<!\w)#(?=\w)")


# ==================================================
# 3. TEXT NORMALIZATION
# ==================================================

def normalize_text(text: Optional[str]) -> str:
    """
    Normalize social-media text.

    Preserves:
    - Negations
    - Emojis and Unicode characters
    - Expressive punctuation

    Replaces:
    - URLs with urltoken
    - User mentions with usertoken

    Also decodes HTML entities, removes hashtag symbols
    while preserving hashtag words, lowercases text,
    and normalizes whitespace.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        raise TypeError(
            "normalize_text() expects a string or None."
        )

    # Decode HTML entities.
    text = html.unescape(text)

    # Replace URLs consistently.
    text = URL_PATTERN.sub(" urltoken ", text)

    # Replace social-media usernames.
    text = MENTION_PATTERN.sub(" usertoken ", text)

    # Preserve hashtag words while removing the # symbol.
    text = HASHTAG_PATTERN.sub("", text)

    # Apply consistent lowercase normalization.
    text = text.lower()

    # Normalize whitespace.
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# ==================================================
# 4. SOCIAL-MEDIA TOKENIZATION
# ==================================================

TOKENIZER = TweetTokenizer(
    preserve_case=True,
    reduce_len=False,
    strip_handles=False,
)


def tokenize_text(normalized_text: str) -> list[str]:
    """
    Tokenize normalized social-media text.

    This function does not remove stopwords or perform
    lemmatization. It does not require punkt_tab.
    """

    if not isinstance(normalized_text, str):
        raise TypeError(
            "tokenize_text() expects a string."
        )

    return TOKENIZER.tokenize(normalized_text)


# ==================================================
# 5. NORMALIZATION TESTS
# ==================================================

def run_self_tests() -> None:
    """Verify normalization behavior using fixed examples."""

    test_cases = [
        (
            "OMG!!! I am NOT happy 😡",
            "omg!!! i am not happy 😡",
        ),
        (
            "Visit https://example.com now",
            "visit urltoken now",
        ),
        (
            "Hello @user #HappyDay",
            "hello usertoken happyday",
        ),
        (
            "Fish &amp; Chips",
            "fish & chips",
        ),
        (
            "  Too    much   whitespace  ",
            "too much whitespace",
        ),
        (
            None,
            "",
        ),
        (
            "this is not good",
            "this is not good",
        ),
        (
            "",
            "",
        ),
        (
            "   ",
            "",
        ),
        (
            "Great 😊!!!",
            "great 😊!!!",
        ),
    ]

    for original, expected in test_cases:
        actual = normalize_text(original)

        assert actual == expected, (
            f"\nInput: {original!r}"
            f"\nExpected: {expected!r}"
            f"\nActual: {actual!r}"
        )

        print(f"PASS: {original!r} -> {actual!r}")

    print("\nAll normalization tests passed.")


# ==================================================
# 6. TOKENIZATION TESTS
# ==================================================

def run_tokenization_tests() -> None:
    """Verify that important sentiment signals survive."""

    examples = [
        "omg!!! i am not happy 😡",
        "this is not good",
        "hello usertoken happyday",
        "great 😊",
    ]

    for text in examples:
        tokens = tokenize_text(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}\n")

    assert "not" in tokenize_text("this is not good")
    assert "!" in tokenize_text("omg!!!")
    assert "😡" in tokenize_text("not happy 😡")

    print("All tokenization checks passed.")


# ==================================================
# 7. DATASET PREPROCESSING
# ==================================================

def preprocess_dataset(split: str) -> pd.DataFrame:
    """
    Load an existing EDA CSV, normalize its text,
    and save a separate preprocessed CSV.

    The original text and dataset labels are retained.
    """

    if split not in SPLITS:
        raise ValueError(
            f"Invalid split {split!r}. Expected {SPLITS}."
        )

    input_path = PROCESSED_DIR / f"{split}_eda.csv"
    output_path = PREPROCESSED_DIR / f"{split}_preprocessed.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "text",
        "label",
        "split",
        "sentiment",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df["text"].isna().any():
        raise ValueError(
            f"{input_path.name} contains missing text."
        )

    if not df["split"].eq(split).all():
        raise ValueError(
            f"Unexpected split assignment in {input_path.name}."
        )

    # Validate the expected number of records.
    expected_count = EXPECTED_COUNTS[split]

    if len(df) != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} rows, "
            f"found {len(df)}."
        )

    # Keep the original text and add a separate column.
    df["normalized_text"] = df["text"].map(normalize_text)

    if df["normalized_text"].isna().any():
        raise ValueError(
            f"Missing normalized text in {split}."
        )

    PREPROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSplit: {split}")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(df):,}")
    print(f"Output file: {output_path}")
    print("Original text preserved: Yes")
    print("Split assignments preserved: Yes")

    return df


def preprocess_all_splits() -> None:
    """Process each official split independently."""

    total_records = 0

    for split in SPLITS:
        df = preprocess_dataset(split)
        total_records += len(df)

    assert total_records == sum(EXPECTED_COUNTS.values())

    print("\nAll split row-count checks passed.")
    print(f"Total records: {total_records:,}")
    print("Raw dataset files were not modified.")


# ==================================================
# 8. SCRIPT ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_self_tests()
    run_tokenization_tests()
    preprocess_all_splits()


"""
SentimentIQ - Phase 2: Exploratory Data Analysis (EDA)

Purpose:
1. Load TweetEval sentiment train, validation, and test splits.
2. Inspect dataset structure and data quality.
3. Analyze sentiment distribution and tweet lengths.
4. Generate EDA visualizations and summary reports.
5. Save standardized analysis datasets without changing raw text.

Run from the project root:
    python ml/training/eda.py
"""

from pathlib import Path

import matplotlib

# Allows charts to be saved even when no interactive display is available.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
REPORT_DIR = PROCESSED_DIR / "eda_reports"
CHART_DIR = REPORT_DIR / "charts"

SPLITS = ("train", "val", "test")

# TweetEval sentiment label mapping.
LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_split(split_name: str) -> pd.DataFrame:
    """Load one text/label split and validate its structure."""

    text_path = RAW_DIR / f"{split_name}_text.txt"
    label_path = RAW_DIR / f"{split_name}_labels.txt"

    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Keep each raw text record unchanged.
    texts = text_path.read_text(encoding="utf-8").splitlines()

    # Read one integer sentiment label per line.
    labels = pd.read_csv(
        label_path,
        header=None,
        dtype="string",
        encoding="utf-8",
    )[0].str.strip()

    if len(texts) != len(labels):
        raise ValueError(
            f"{split_name}: text count ({len(texts)}) "
            f"does not match label count ({len(labels)})."
        )

    if labels.isna().any() or labels.eq("").any():
        raise ValueError(f"{split_name}: missing or empty labels found.")

    numeric_labels = pd.to_numeric(labels, errors="raise")

    if not numeric_labels.isin(LABEL_MAPPING).all():
        invalid_labels = sorted(
            numeric_labels[~numeric_labels.isin(LABEL_MAPPING)]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{split_name}: unexpected labels found: {invalid_labels}"
        )

    df = pd.DataFrame(
        {
            "text": texts,
            "label": numeric_labels.astype("int64"),
            "split": split_name,
        }
    )

    df["sentiment"] = df["label"].map(LABEL_MAPPING)

    # Basic descriptive measurements, not NLP preprocessing.
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    print(f"\nLoaded {split_name} split successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    return df


# --------------------------------------------------
# 4. LOAD ALL THREE SPLITS
# --------------------------------------------------

datasets = {
    split: load_split(split)
    for split in SPLITS
}

all_data = pd.concat(
    datasets.values(),
    ignore_index=True,
)

print("\n" + "=" * 60)
print("SENTIMENTIQ - DATASET OVERVIEW")
print("=" * 60)

print(f"Total records: {len(all_data):,}")
print(f"Total columns: {len(all_data.columns)}")
print("\nDataset information:")
all_data.info()

print("\nFirst five records:")
print(all_data.head().to_string(index=False))


# --------------------------------------------------
# 5. DATA QUALITY ANALYSIS
# --------------------------------------------------

report_lines = []


def log(message: str = "") -> None:
    """Print a report line and retain it for the text report."""
    print(message)
    report_lines.append(message)


log("\n" + "=" * 60)
log("DATA QUALITY REPORT")
log("=" * 60)

log(f"\nTotal records: {len(all_data):,}")
log(f"Missing values by column:\n{all_data.isna().sum().to_string()}")

# A blank string is not a Pandas NaN, so check it separately.
blank_text_mask = all_data["text"].str.strip().eq("")

log(f"\nBlank or whitespace-only tweets: {blank_text_mask.sum():,}")

for split_name, df in datasets.items():
    log(f"\n--- {split_name.upper()} SPLIT ---")
    log(f"Rows: {len(df):,}")
    log(f"Duplicate text rows: {df['text'].duplicated().sum():,}")
    log(
        "Duplicate complete records: "
        f"{df.duplicated(subset=['text', 'label']).sum():,}"
    )
    log(f"Blank tweets: {df['text'].str.strip().eq('').sum():,}")

# Repeated text across splits can indicate leakage. This is a diagnostic,
# not an instruction to remove records automatically.
text_split_counts = all_data.groupby("text")["split"].nunique()
cross_split_texts = text_split_counts[text_split_counts > 1]

log(
    "\nUnique text values appearing in multiple splits: "
    f"{len(cross_split_texts):,}"
)

log(
    "Note: duplicates and blank records are reported, not automatically "
    "deleted. Investigate them before making cleaning decisions."
)


# --------------------------------------------------
# 6. SENTIMENT LABEL DISTRIBUTION
# --------------------------------------------------

log("\n" + "=" * 60)
log("SENTIMENT DISTRIBUTION")
log("=" * 60)

for split_name, df in datasets.items():
    counts = (
        df["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )

    percentages = counts.div(len(df)).mul(100) if len(df) else counts

    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    log(f"\n{split_name.upper()}:\n{distribution.to_string()}")

    distribution.to_csv(
        REPORT_DIR / f"{split_name}_label_distribution.csv"
    )


# --------------------------------------------------
# 7. TEXT LENGTH ANALYSIS
# --------------------------------------------------

log("\n" + "=" * 60)
log("TEXT LENGTH STATISTICS")
log("=" * 60)

length_stats = (
    all_data.groupby("split")[["char_count", "word_count"]]
    .describe()
    .round(2)
)

log(length_stats.to_string())

length_stats.to_csv(REPORT_DIR / "text_length_statistics.csv")


# --------------------------------------------------
# 8. VISUALIZATION: SENTIMENT DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=all_data,
    x="sentiment",
    hue="split",
    order=SENTIMENT_ORDER,
)

plt.title("Sentiment Distribution Across Dataset Splits")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "sentiment_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 9. VISUALIZATION: TWEET LENGTH DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=all_data,
    x="word_count",
    hue="split",
    bins=40,
    element="step",
    stat="count",
    common_norm=False,
)

plt.title("Tweet Word Count Distribution")
plt.xlabel("Number of Whitespace-Separated Words")
plt.ylabel("Number of Tweets")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "word_count_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 10. VISUALIZATION: WORD COUNT BY SENTIMENT
# --------------------------------------------------

# Exclude extreme lengths from the plot only, using the 99th percentile.
# The underlying data remains unchanged.
plot_limit = all_data["word_count"].quantile(0.99)

plot_data = all_data.loc[
    all_data["word_count"] <= plot_limit
]

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=plot_data,
    x="sentiment",
    y="word_count",
    order=SENTIMENT_ORDER,
)

plt.title("Tweet Length by Sentiment (Up to 99th Percentile)")
plt.xlabel("Sentiment")
plt.ylabel("Whitespace-Separated Word Count")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "word_count_by_sentiment.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 11. SAVE STANDARDIZED EDA DATASETS
# --------------------------------------------------

# Preserve every record and raw text. The additional columns contain
# descriptive statistics only; no tokens are removed or normalized.
for split_name, df in datasets.items():
    output_path = PROCESSED_DIR / f"{split_name}_eda.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    log(f"\nSaved {split_name} EDA data: {output_path}")


# Save a combined copy for convenient exploratory analysis.
combined_path = PROCESSED_DIR / "all_splits_eda.csv"

all_data.to_csv(
    combined_path,
    index=False,
    encoding="utf-8",
)

log(f"Saved combined EDA data: {combined_path}")


# --------------------------------------------------
# 12. SAVE DATA QUALITY REPORT
# --------------------------------------------------

report_path = REPORT_DIR / "data_quality_report.txt"

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print(f"\nSaved report: {report_path}")
print(f"Saved charts in: {CHART_DIR}")

print("\n" + "=" * 60)
print("EDA EXECUTION COMPLETED")
print("=" * 60)
print("Review the generated reports and charts before cleaning data.")

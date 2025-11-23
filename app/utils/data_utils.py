from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJ_ROOT / "output"
PROCESSED_DATA_PATH = OUTPUT_PATH / "processed_id_fan_data.csv"
RESULTS_PATH = OUTPUT_PATH / "results.csv"

# Constants
COLUMNS_TO_DROP = ["Target_Class", "Target_Label", "GENERATOR POWER, (MW)"]


def _load_data():
    start_failure = pd.to_datetime("2024-09-01 10:03:00")
    end_failure = pd.to_datetime("2024-09-02 10:03:00")

    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    # Create masks
    failure_mask = (df.index >= start_failure) & (df.index <= end_failure)
    normal_mask = ~failure_mask

    # Split failure region (50% Train, 50% Test)
    failure_df = df[failure_mask]
    half_fail = len(failure_df) // 2
    failure_train = failure_df.iloc[:half_fail]
    failure_test = failure_df.iloc[half_fail:]

    # Split normal region
    normal_df = df[normal_mask]
    TRAIN_RATIO = 0.8  # (80% Train, 20% Test)
    split_idx = int(len(normal_df) * TRAIN_RATIO)
    normal_train = normal_df.iloc[:split_idx]
    normal_test = normal_df.iloc[split_idx:]

    # Combine to form final datasets, shuffle the data
    train_df = pd.concat([normal_train, failure_train]).sort_index()
    test_df = pd.concat([normal_test, failure_test]).sort_index()

    # Define Features (X) and Target (y)
    X_train = train_df.drop(columns=COLUMNS_TO_DROP)
    y_train = train_df["Target_Class"]
    X_test = test_df.drop(columns=COLUMNS_TO_DROP)
    y_test = test_df["Target_Class"]
    return X_train, y_train, X_test, y_test


def get_columns_to_drop():
    return COLUMNS_TO_DROP.copy()


def get_feature_columns(df):
    return df.drop(columns=COLUMNS_TO_DROP, errors="ignore")


def get_feature_column_names(df):
    feature_df = get_feature_columns(df)
    return feature_df.columns.tolist()


def get_train_data():
    X_train, y_train, _, _ = _load_data()
    return X_train, y_train


def get_test_data():
    _, _, X_test, y_test = _load_data()
    return X_test, y_test


def get_feature_column_names_list():
    try:
        X_train, _ = get_train_data()
        return X_train.columns.tolist()
    except Exception:
        return None


def make_pie(ax, counts, title):
    class_names = {0: "Normal", 1: "Towards Failure"}
    wedges, _ = ax.pie(counts, startangle=90)

    total = counts.sum()
    labels = [
        f"{class_names[cls]}: {cnt} ({cnt/total*100:.1f}%)"
        for cls, cnt in counts.items()
    ]

    ax.legend(
        wedges, labels, title="Classes", loc="upper right", bbox_to_anchor=(1.25, 1)
    )
    ax.set_title(title)

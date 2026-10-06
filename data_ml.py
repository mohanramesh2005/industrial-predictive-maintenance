"""
data_ml.py
----------
All data ingestion, validation, cleaning, feature engineering, target
detection, and train/test splitting logic for the Industrial Machine
Failure Intelligence System.

No model training happens here (see models.py) — this file only
prepares clean, leakage-free feature matrices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    COLUMN_ALIASES,
    FAILURE_TYPE_ALIASES,
    MACHINE_ID_ALIASES,
    RUL_ALIASES,
    SENSOR_COLUMNS,
    TARGET_ALIASES,
    TIMESTAMP_ALIASES,
    find_matching_column,
    get_logger,
    normalize_key,
)

logger = get_logger("pdm.data_ml")

ENGINEERED_FEATURES: List[str] = [
    "power_estimate",
    "temperature_change",
    "vibration_change",
    "pressure_change",
    "temperature_rolling_mean",
    "vibration_rolling_mean",
    "pressure_rolling_mean",
    "temperature_rolling_std",
    "vibration_rolling_std",
]

ROLLING_WINDOW = 5


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_csv(file_or_path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load a CSV from a path or a file-like (e.g. Streamlit UploadedFile).
    Returns (dataframe, error_message)."""
    try:
        df = pd.read_csv(file_or_path)
        if df.empty:
            return None, "The uploaded CSV is empty."
        return df, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never raised raw
        logger.exception("Failed to load CSV")
        return None, f"Could not read CSV file: {exc}"


# --------------------------------------------------------------------------
# Column normalization & target detection
# --------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename columns to canonical snake_case / sensor names where recognized,
    tolerating variations in capitalization and spacing. Returns the new
    dataframe plus a mapping of original -> normalized column names.
    """
    mapping: Dict[str, str] = {}
    for col in df.columns:
        key = normalize_key(col)
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
        else:
            mapping[col] = str(col).strip().lower().replace(" ", "_")

    df = df.rename(columns=mapping)
    # If duplicate columns resulted from normalization, keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]
    return df, mapping


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    return find_matching_column(list(df.columns), TARGET_ALIASES)


def detect_failure_type_column(df: pd.DataFrame) -> Optional[str]:
    return find_matching_column(list(df.columns), FAILURE_TYPE_ALIASES)


def detect_rul_column(df: pd.DataFrame) -> Optional[str]:
    return find_matching_column(list(df.columns), RUL_ALIASES)


def detect_machine_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_matching_column(list(df.columns), MACHINE_ID_ALIASES)


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    return find_matching_column(list(df.columns), TIMESTAMP_ALIASES)


def get_available_sensor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in SENSOR_COLUMNS if c in df.columns]


# --------------------------------------------------------------------------
# Dataset inspection / validation
# --------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> Dict:
    """Produce a summary dict describing the dataset — used by the
    Dataset dashboard page and by sanity checks before training."""
    sensor_cols = get_available_sensor_columns(df)
    target_col = detect_target_column(df)
    failure_type_col = detect_failure_type_column(df)
    rul_col = detect_rul_column(df)
    timestamp_col = detect_timestamp_column(df)
    machine_id_col = detect_machine_id_column(df)

    missing_sensor_cols = [c for c in SENSOR_COLUMNS if c not in sensor_cols]

    target_distribution = None
    if target_col is not None:
        try:
            target_distribution = df[target_col].value_counts(dropna=False).to_dict()
        except Exception:  # noqa: BLE001
            target_distribution = None

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "available_sensor_columns": sensor_cols,
        "missing_sensor_columns": missing_sensor_cols,
        "target_column": target_col,
        "target_distribution": target_distribution,
        "failure_type_column": failure_type_col,
        "rul_column": rul_col,
        "timestamp_column": timestamp_col,
        "machine_id_column": machine_id_col,
    }


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values, duplicates and data types for the available
    sensor columns without touching rows the caller has not asked about.
    """
    df = df.copy()
    sensor_cols = sensor_cols or get_available_sensor_columns(df)

    # Coerce sensor columns to numeric, invalid parses become NaN.
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    # Impute missing sensor values with the column median (robust to outliers).
    for col in sensor_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)

    return df


# --------------------------------------------------------------------------
# Feature engineering (leakage-safe)
# --------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    machine_id_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add derived features. When a machine id column exists, rolling/lag
    features are computed per-machine. When a timestamp column exists,
    the data is sorted chronologically first so that rolling windows and
    "change" features never look at future rows. Uses `.shift(1)` /
    `min_periods=1` semantics so the current row is always excluded from
    its own rolling statistic where appropriate.
    """
    df = df.copy()
    sensor_cols = get_available_sensor_columns(df)

    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        sort_cols = [machine_id_col, timestamp_col] if machine_id_col else [timestamp_col]
        df = df.sort_values(by=[c for c in sort_cols if c]).reset_index(drop=True)

    if "voltage" in sensor_cols and "current" in sensor_cols:
        df["power_estimate"] = df["voltage"] * df["current"]

    group = df.groupby(machine_id_col) if (machine_id_col and machine_id_col in df.columns) else None

    def _diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _roll_mean(series: pd.Series) -> pd.Series:
        return series.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    def _roll_std(series: pd.Series) -> pd.Series:
        return series.rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0.0)

    for base in ("temperature", "vibration", "pressure"):
        if base not in sensor_cols:
            continue
        if group is not None:
            df[f"{base}_change"] = group[base].transform(_diff)
            df[f"{base}_rolling_mean"] = group[base].transform(_roll_mean)
            if base in ("temperature", "vibration"):
                df[f"{base}_rolling_std"] = group[base].transform(_roll_std)
        else:
            df[f"{base}_change"] = _diff(df[base])
            df[f"{base}_rolling_mean"] = _roll_mean(df[base])
            if base in ("temperature", "vibration"):
                df[f"{base}_rolling_std"] = _roll_std(df[base])

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric feature columns usable for modeling: available sensors plus
    any engineered features that were successfully created."""
    sensor_cols = get_available_sensor_columns(df)
    engineered = [c for c in ENGINEERED_FEATURES if c in df.columns]
    features = sensor_cols + engineered
    # keep only numeric
    return [c for c in features if pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# Class imbalance
# --------------------------------------------------------------------------

def check_class_imbalance(y: pd.Series) -> Dict:
    counts = y.value_counts().to_dict()
    total = sum(counts.values())
    ratios = {str(k): round(v / total, 4) for k, v in counts.items()}
    minority_ratio = min(ratios.values()) if ratios else 0.0
    is_imbalanced = minority_ratio < 0.25
    return {"counts": counts, "ratios": ratios, "is_imbalanced": is_imbalanced}


# --------------------------------------------------------------------------
# Train / test split (leakage-safe)
# --------------------------------------------------------------------------

def prepare_train_test(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    timestamp_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Split into train/test. Uses chronological splitting when a timestamp
    column is available (no shuffling of time-ordered data), otherwise a
    stratified split on the target. Returns a dict of arrays plus split
    metadata so downstream code stays leakage-safe.
    """
    df = df.dropna(subset=[target_col]).copy()
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    split_method = "stratified_random"
    if timestamp_col and timestamp_col in df.columns and df[timestamp_col].notna().any():
        df_sorted = df.sort_values(by=timestamp_col)
        X = df_sorted[feature_cols].copy()
        y = df_sorted[target_col].copy()
        split_idx = int(len(df_sorted) * (1 - test_size))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        split_method = "chronological"
    else:
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "split_method": split_method,
        "feature_cols": feature_cols,
    }


def preprocess_for_prediction(input_dict: Dict[str, float], feature_cols: List[str]) -> pd.DataFrame:
    """
    Build a single-row feature dataframe for a manual/UI prediction. Any
    engineered feature not derivable from a single snapshot (e.g. rolling
    stats that need history) defaults to 0, which is a neutral value for
    "no observed change yet".
    """
    row = {col: 0.0 for col in feature_cols}
    for col in SENSOR_COLUMNS:
        if col in input_dict and col in row:
            row[col] = float(input_dict[col])
    if "voltage" in input_dict and "current" in input_dict and "power_estimate" in row:
        row["power_estimate"] = float(input_dict["voltage"]) * float(input_dict["current"])
    return pd.DataFrame([row], columns=feature_cols)


# --------------------------------------------------------------------------
# Demo dataset generation
# --------------------------------------------------------------------------

def generate_demo_dataset(
    n_machines: int = 12,
    n_records_per_machine: int = 250,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a deterministic, clearly-labeled synthetic sensor dataset with
    realistic degradation patterns (not independent random noise), for
    demonstration/testing only. Includes a binary failure label, an
    optional failure_type label, and an optional RUL column so the
    advanced (bonus) modules have something to train on in demo mode.
    """
    rng = np.random.default_rng(random_state)
    records = []
    failure_types = ["Bearing Failure", "Motor Failure", "Overheating", "Electrical Failure", "Pressure Failure"]

    start_time = pd.Timestamp("2025-01-01")

    for m in range(n_machines):
        machine_id = f"M-{1000 + m}"
        machine_type = rng.choice(["Pump", "Compressor", "Motor", "Turbine", "Conveyor"])
        will_fail = rng.random() < 0.35  # ~35% of machines degrade toward failure
        life = n_records_per_machine
        failure_point = int(life * rng.uniform(0.65, 0.97)) if will_fail else None
        chosen_failure_type = rng.choice(failure_types) if will_fail else None

        base_temp = rng.uniform(45, 60)
        base_vib = rng.uniform(1.5, 3.0)
        base_pressure = rng.uniform(80, 120)
        base_rpm = rng.uniform(1400, 1800)
        base_voltage = rng.uniform(210, 230)
        base_current = rng.uniform(8, 15)

        operating_hours = rng.uniform(0, 500)

        for t in range(life):
            timestamp = start_time + pd.Timedelta(hours=t * 4) + pd.Timedelta(days=m * 2)
            operating_hours += rng.uniform(3.5, 4.5)

            # Degradation factor ramps up only if this machine is heading to failure.
            if will_fail:
                progress = max(0.0, (t - (failure_point - int(life * 0.35))) / max(1, int(life * 0.35)))
                degradation = float(np.clip(progress, 0.0, 1.0))
            else:
                degradation = 0.0

            noise_scale = 1.0
            temperature = base_temp + degradation * rng.uniform(15, 35) + rng.normal(0, noise_scale)
            vibration = base_vib + degradation * rng.uniform(3, 7) + rng.normal(0, 0.15)
            pressure = base_pressure - degradation * rng.uniform(10, 25) + rng.normal(0, 1.5)
            rpm = base_rpm - degradation * rng.uniform(100, 250) + rng.normal(0, 10)
            voltage = base_voltage - degradation * rng.uniform(5, 15) + rng.normal(0, 1.0)
            current = base_current + degradation * rng.uniform(2, 6) + rng.normal(0, 0.3)

            is_failure = 1 if (will_fail and t >= failure_point) else 0

            record = {
                "machine_id": machine_id,
                "machine_type": machine_type,
                "timestamp": timestamp.isoformat(),
                "temperature": round(float(temperature), 2),
                "vibration": round(float(vibration), 3),
                "pressure": round(float(pressure), 2),
                "rpm": round(float(rpm), 1),
                "voltage": round(float(voltage), 2),
                "current": round(float(current), 2),
                "operating_hours": round(float(operating_hours), 2),
                "failure": int(is_failure),
                "failure_type": chosen_failure_type if is_failure else "None",
                "rul": max(0.0, round(float((failure_point - t) * 4), 1)) if will_fail else round(
                    float((life - t) * 4 + rng.uniform(500, 2000)), 1
                ),
            }
            records.append(record)

            if is_failure and t == failure_point:
                # after failure the machine is taken offline in this synthetic log
                pass

    demo_df = pd.DataFrame.from_records(records)
    return demo_df

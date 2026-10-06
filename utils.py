"""
utils.py
--------
Shared configuration, scoring engine, risk classification, maintenance
recommendation engine, input validation, and small helper utilities used
across the Industrial Machine Failure Intelligence System.

Nothing in this file trains or loads ML models — it only contains
deterministic, transparent business logic and configuration so that the
same rules are applied consistently everywhere in the application.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Paths / directories
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "predictive_maintenance.db")


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    for d in (DATA_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def get_logger(name: str = "pdm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

# --------------------------------------------------------------------------
# Core sensor schema
# --------------------------------------------------------------------------

# Canonical (normalized) sensor column names required by the pipeline.
SENSOR_COLUMNS: List[str] = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "voltage",
    "current",
    "operating_hours",
]

# Acceptable raw-name variants -> canonical name. Matching is done on a
# lowercased, whitespace/underscore-stripped version of the column name,
# so "Temperature ", "TEMP", "temp_c" etc. are all tolerated.
COLUMN_ALIASES: Dict[str, str] = {
    "temperature": "temperature",
    "temp": "temperature",
    "tempc": "temperature",
    "temperaturec": "temperature",
    "vibration": "vibration",
    "vib": "vibration",
    "vibrationlevel": "vibration",
    "pressure": "pressure",
    "press": "pressure",
    "rpm": "rpm",
    "rotationspeed": "rpm",
    "speed": "rpm",
    "voltage": "voltage",
    "volt": "voltage",
    "volts": "voltage",
    "current": "current",
    "amps": "current",
    "amperage": "current",
    "operatinghours": "operating_hours",
    "operating_hours": "operating_hours",
    "hours": "operating_hours",
    "runtimehours": "operating_hours",
    "usagehours": "operating_hours",
}

TARGET_ALIASES: List[str] = ["failure", "machine_failure", "failure_flag", "failed", "target"]
FAILURE_TYPE_ALIASES: List[str] = ["failure_type", "failure_category", "fault_type"]
RUL_ALIASES: List[str] = [
    "rul",
    "remaining_useful_life",
    "hours_to_failure",
    "cycles_to_failure",
]
MACHINE_ID_ALIASES: List[str] = ["machine_id", "machineid", "machine", "asset_id", "equipment_id"]
TIMESTAMP_ALIASES: List[str] = ["timestamp", "time", "datetime", "date", "reading_time"]


def normalize_key(name: str) -> str:
    """Lowercase and strip everything but letters/digits for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def normalize_column_name(name: str) -> str:
    """
    Map a raw column name to its canonical sensor name when possible.
    Falls back to a lowercase/underscored version of the original name.
    """
    key = normalize_key(name)
    if key in COLUMN_ALIASES:
        return COLUMN_ALIASES[key]
    return re.sub(r"\s+", "_", str(name).strip().lower())


def find_matching_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    """Return the original column name (as given) that matches one of the
    alias keys, or None if no match is found."""
    normalized_lookup = {normalize_key(c): c for c in columns}
    for alias in aliases:
        key = normalize_key(alias)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


# --------------------------------------------------------------------------
# Health score / risk configuration (configurable, not hard safety limits)
# --------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """All thresholds are configurable operational defaults, not
    certified industrial safety limits."""

    failure_weight: float = 0.60
    anomaly_weight: float = 0.40

    health_critical_max: float = 30.0
    health_warning_max: float = 60.0
    # above health_warning_max -> HEALTHY

    risk_low_max: float = 30.0
    risk_medium_max: float = 60.0
    risk_high_max: float = 80.0
    # above risk_high_max -> CRITICAL


SCORING_CONFIG = ScoringConfig()

# Reasonable, user-adjustable input validation ranges. These exist purely to
# catch obviously invalid data entry (typos, sensor glitches) — they are
# NOT certified industrial safety thresholds.
DEFAULT_VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (-50.0, 400.0),
    "vibration": (0.0, 100.0),
    "pressure": (0.0, 500.0),
    "rpm": (0.0, 20000.0),
    "voltage": (0.0, 1000.0),
    "current": (0.0, 500.0),
    "operating_hours": (0.0, 200000.0),
}


def compute_health_risk(
    failure_probability: float,
    anomaly_risk: float,
    config: ScoringConfig = SCORING_CONFIG,
) -> float:
    """
    Combine supervised failure probability (0-100) and normalized anomaly
    risk (0-100) into a single "health risk" number (0-100), using
    configurable weights. Higher = worse.
    """
    failure_probability = clamp(failure_probability, 0.0, 100.0)
    anomaly_risk = clamp(anomaly_risk, 0.0, 100.0)
    risk = (config.failure_weight * failure_probability) + (config.anomaly_weight * anomaly_risk)
    return clamp(risk, 0.0, 100.0)


def compute_health_score(failure_probability: float, anomaly_risk: float,
                          config: ScoringConfig = SCORING_CONFIG) -> float:
    """Health Score = 100 - Health Risk, clamped to [0, 100]."""
    risk = compute_health_risk(failure_probability, anomaly_risk, config)
    return clamp(100.0 - risk, 0.0, 100.0)


def classify_health_status(score: float, config: ScoringConfig = SCORING_CONFIG) -> str:
    if score <= config.health_critical_max:
        return "CRITICAL"
    if score <= config.health_warning_max:
        return "WARNING"
    return "HEALTHY"


def classify_risk_level(failure_probability_pct: float, config: ScoringConfig = SCORING_CONFIG) -> str:
    """failure_probability_pct is expected on a 0-100 scale."""
    p = clamp(failure_probability_pct, 0.0, 100.0)
    if p <= config.risk_low_max:
        return "LOW"
    if p <= config.risk_medium_max:
        return "MEDIUM"
    if p <= config.risk_high_max:
        return "HIGH"
    return "CRITICAL"


def clamp(value: float, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, v))


# --------------------------------------------------------------------------
# Maintenance recommendation engine
# --------------------------------------------------------------------------

def generate_recommendation(
    risk_level: str,
    failure_probability: float,
    anomaly_status: str,
    rul: Optional[float] = None,
) -> str:
    """
    Rule-based maintenance decision support. This system does NOT
    autonomously control machinery — it only produces human-readable
    guidance for maintenance planners.
    """
    base_actions = {
        "LOW": "Continue normal monitoring. No immediate action required.",
        "MEDIUM": "Increase monitoring frequency and schedule a routine inspection.",
        "HIGH": "Schedule a maintenance inspection within 24-48 hours.",
        "CRITICAL": "Immediate engineering/maintenance review recommended.",
    }
    action = base_actions.get(risk_level, "Continue normal monitoring.")

    notes = []
    if anomaly_status == "ANOMALY":
        notes.append("Unsupervised anomaly detector also flagged abnormal sensor behavior.")
    if failure_probability >= 80:
        notes.append("Supervised model indicates a high failure probability.")
    if rul is not None:
        try:
            rul_val = float(rul)
            notes.append(f"Estimated remaining useful life: {rul_val:,.1f} operating hours.")
        except (TypeError, ValueError):
            pass

    if notes:
        return action + " " + " ".join(notes)
    return action


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def validate_sensor_input(
    values: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a dict of {sensor_name: value}. Returns (is_valid, error_messages).
    Rejects NaN, infinity, non-numeric strings, and out-of-range values.
    """
    import math

    ranges = ranges or DEFAULT_VALIDATION_RANGES
    errors: List[str] = []

    for col in SENSOR_COLUMNS:
        if col not in values:
            errors.append(f"Missing value for '{col}'.")
            continue
        raw = values[col]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be numeric (got {raw!r}).")
            continue
        if math.isnan(v):
            errors.append(f"'{col}' cannot be NaN.")
            continue
        if math.isinf(v):
            errors.append(f"'{col}' cannot be infinite.")
            continue
        low, high = ranges.get(col, (-1e12, 1e12))
        if v < low or v > high:
            errors.append(f"'{col}' = {v} is outside the configured range [{low}, {high}].")

    return (len(errors) == 0), errors


def safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf check
            return default
        return f
    except (TypeError, ValueError):
        return default


def format_pct(value: float) -> str:
    return f"{clamp(value, 0.0, 100.0):.1f}%"

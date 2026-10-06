"""
models.py
---------
Model training, evaluation, explainability and persistence for the
Industrial Machine Failure Intelligence System.

Implements:
  * Failure classification: Logistic Regression, Random Forest, XGBoost,
    LightGBM, plus a soft-voting Ensemble built from the individually
    trained models (extra accuracy feature).
  * 5-fold stratified cross-validation alongside the held-out test metrics,
    so reported performance is not a single lucky split (extra accuracy /
    robustness feature).
  * Optional light RandomizedSearchCV tuning for XGBoost/LightGBM (cheap,
    cv=3) — extra accuracy feature, opt-in because it costs more time.
  * Isolation Forest anomaly detection with a normalized anomaly score.
  * Optional multiclass failure-type model and RUL regression model, only
    when the dataset actually supports them.
  * SHAP explainability with a safe fallback to feature importances.
  * Joblib persistence for every trained artifact.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import MODEL_DIR, get_logger

logger = get_logger("pdm.models")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:  # noqa: BLE001
    LIGHTGBM_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001
    SHAP_AVAILABLE = False

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Classification models
# --------------------------------------------------------------------------

def _build_candidate_models(use_class_weight: bool, scale_pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced" if use_class_weight else None,
                ),
            ),
        ]
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight="balanced" if use_class_weight else None,
        n_jobs=-1,
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            eval_metric="logloss",
            n_jobs=-1,
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=-1,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
            class_weight="balanced" if use_class_weight else None,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def _light_tune(model_name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
    """Cheap RandomizedSearchCV tuning (cv=3, few iterations) for boosted
    trees only. Opt-in, since it costs extra compute time."""
    param_distributions = None
    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        param_distributions = {
            "n_estimators": [150, 250, 350],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.12],
            "subsample": [0.7, 0.85, 1.0],
        }
    elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        param_distributions = {
            "n_estimators": [150, 250, 350],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.05, 0.08, 0.12],
        }

    if param_distributions is None:
        return model

    try:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=6,
            cv=3,
            scoring="recall",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        logger.info("%s tuned params: %s", model_name, search.best_params_)
        return search.best_estimator_
    except Exception:  # noqa: BLE001
        logger.exception("Light tuning failed for %s, using default params", model_name)
        return model


def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:  # noqa: BLE001
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:  # noqa: BLE001
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_class_weight: bool = True,
    enable_tuning: bool = False,
    enable_cv: bool = True,
) -> Dict[str, Dict]:
    """
    Train Logistic Regression, Random Forest, XGBoost (if available),
    LightGBM (if available), and a soft-voting Ensemble of all of them.
    Returns a dict keyed by model name with the fitted model, test-set
    metrics, confusion matrix, ROC/PR curve points, and (optionally)
    5-fold stratified cross-validation metrics for robustness.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    candidates = _build_candidate_models(use_class_weight, scale_pos_weight)
    results: Dict[str, Dict] = {}
    fitted_models: Dict[str, object] = {}

    for name, model in candidates.items():
        try:
            if enable_tuning:
                model = _light_tune(name, model, X_train, y_train)
            model.fit(X_train, y_train)
        except Exception:  # noqa: BLE001
            logger.exception("Training failed for %s", name)
            continue

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:  # noqa: BLE001
                y_proba = None

        metrics = _compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        roc_points, pr_points = None, None
        if y_proba is not None and len(np.unique(y_test)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_points = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                pr_points = {"precision": prec.tolist(), "recall": rec.tolist()}
            except Exception:  # noqa: BLE001
                pass

        cv_metrics = None
        if enable_cv:
            cv_metrics = _cross_validate_model(model, X_train, y_train)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "cv_metrics": cv_metrics,
            "confusion_matrix": cm,
            "roc_points": roc_points,
            "pr_points": pr_points,
        }
        fitted_models[name] = model

    # Ensemble (soft-voting) built from the already-fitted candidate models —
    # an extra-accuracy feature that usually smooths out individual model
    # weaknesses. Only added when at least 2 base models trained successfully
    # and all support predict_proba.
    proba_capable = {n: m for n, m in fitted_models.items() if hasattr(m, "predict_proba")}
    if len(proba_capable) >= 2:
        try:
            voting = VotingClassifier(
                estimators=[(n, m) for n, m in proba_capable.items()],
                voting="soft",
            )
            # VotingClassifier requires fitting; refit wrapped estimators are
            # already trained but sklearn's VotingClassifier re-fits clones
            # by default. We instead build a lightweight manual soft-voting
            # wrapper so we reuse the already-trained models directly.
            ensemble = _ManualSoftVotingEnsemble(proba_capable)
            y_pred = ensemble.predict(X_test)
            y_proba = ensemble.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_test, y_pred, y_proba)
            cm = confusion_matrix(y_test, y_pred).tolist()
            roc_points, pr_points = None, None
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_points = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                pr_points = {"precision": prec.tolist(), "recall": rec.tolist()}

            results["Ensemble (Voting)"] = {
                "model": ensemble,
                "metrics": metrics,
                "cv_metrics": None,
                "confusion_matrix": cm,
                "roc_points": roc_points,
                "pr_points": pr_points,
            }
        except Exception:  # noqa: BLE001
            logger.exception("Ensemble construction failed; continuing without it")

    return results


class _ManualSoftVotingEnsemble:
    """Lightweight soft-voting ensemble that reuses already-fitted models
    instead of re-fitting clones (keeps training time low)."""

    def __init__(self, fitted_models: Dict[str, object]):
        self.fitted_models = fitted_models
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.fitted_models.values()]
        return np.mean(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def _cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Optional[Dict]:
    """5-fold stratified CV for a more robust performance estimate than a
    single train/test split. Returns mean +/- std for each metric."""
    try:
        n_splits = min(n_splits, int(y.value_counts().min()))
        if n_splits < 2:
            return None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        summary = {}
        for metric in scoring:
            key = f"test_{metric}"
            if key in scores:
                summary[metric] = {
                    "mean": float(np.mean(scores[key])),
                    "std": float(np.std(scores[key])),
                }
        return summary
    except Exception:  # noqa: BLE001
        logger.exception("Cross-validation failed")
        return None


def select_best_model(results: Dict[str, Dict], primary_metric: str = "recall") -> str:
    """Pick the best model name by a configurable primary metric
    (default recall, since missing a true failure is costly)."""
    best_name, best_score = None, -np.inf
    for name, info in results.items():
        score = info["metrics"].get(primary_metric, np.nan)
        if score is not None and not np.isnan(score) and score > best_score:
            best_score = score
            best_name = name
    return best_name or (next(iter(results)) if results else None)


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def train_anomaly_model(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    contamination: float = 0.1,
) -> IsolationForest:
    """
    Train IsolationForest. When labels are available, train primarily on
    samples labeled "normal" (0) so the model learns what normal operation
    looks like.
    """
    if y_train is not None and (y_train == 0).sum() >= 20:
        X_fit = X_train[y_train == 0]
    else:
        X_fit = X_train

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit)
    return model


def compute_anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (normalized_score in [0,1] where higher = more abnormal,
    is_anomaly boolean array). IsolationForest's raw `decision_function`
    is higher for normal points, so we invert and min-max normalize it.
    This is an unsupervised abnormality signal, NOT a calibrated
    probability.
    """
    raw = model.decision_function(X)  # higher = more normal
    inverted = -raw  # higher = more abnormal
    lo, hi = float(np.min(inverted)), float(np.max(inverted))
    if hi - lo < 1e-9:
        normalized = np.zeros_like(inverted)
    else:
        normalized = (inverted - lo) / (hi - lo)
    predictions = model.predict(X)  # -1 = anomaly, 1 = normal
    is_anomaly = predictions == -1
    return normalized, is_anomaly


# --------------------------------------------------------------------------
# Failure type (multiclass) — only when labels exist
# --------------------------------------------------------------------------

def train_failure_type_model(X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
    valid = y.notna() & (y.astype(str).str.lower() != "none")
    X_valid, y_valid = X[valid], y[valid].astype(str)

    if y_valid.nunique() < 2 or len(y_valid) < 20:
        return None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_valid)

    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    X_train, X_test, y_train, y_test = _stratified_or_random_split(X_valid, y_encoded)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return {"model": model, "label_encoder": encoder, "accuracy": accuracy, "f1_macro": f1}


def _stratified_or_random_split(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split as tts
    from collections import Counter

    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 else None
    return tts(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify)


def predict_failure_type(model_bundle: Dict, X_row: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
    model = model_bundle["model"]
    encoder = model_bundle["label_encoder"]
    proba = model.predict_proba(X_row)[0]
    class_idx = int(np.argmax(proba))
    label = encoder.inverse_transform([class_idx])[0]
    confidence = float(proba[class_idx])
    class_probs = {str(cls): float(p) for cls, p in zip(encoder.classes_, proba)}
    return label, confidence, class_probs


# --------------------------------------------------------------------------
# RUL regression — only when labels exist
# --------------------------------------------------------------------------

def train_rul_model(X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
    valid = y.notna()
    X_valid, y_valid = X[valid], y[valid].astype(float)
    if len(y_valid) < 20:
        return None

    from sklearn.model_selection import train_test_split as tts

    X_train, X_test, y_train, y_test = tts(X_valid, y_valid, test_size=0.2, random_state=RANDOM_STATE)

    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBRegressor"] = xgb.XGBRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE, n_jobs=-1
        )

    best_name, best_model, best_mae = None, None, np.inf
    comparison = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        comparison[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model

    return {"model": best_model, "best_model_name": best_name, "comparison": comparison}


# --------------------------------------------------------------------------
# Explainability (SHAP with safe fallback)
# --------------------------------------------------------------------------

def explain_prediction(model, X_background: pd.DataFrame, X_row: pd.DataFrame, feature_names: List[str]) -> Dict:
    """
    Returns a dict: {"method": "shap"|"feature_importance"|"none",
    "values": {feature: contribution}}. Never raises — always falls back
    gracefully so the app never crashes because of an explainability
    failure.
    """
    inner_model = model
    if hasattr(model, "named_steps"):  # sklearn Pipeline (Logistic Regression)
        try:
            inner_model = model.named_steps.get("clf", model)
        except Exception:  # noqa: BLE001
            inner_model = model

    if SHAP_AVAILABLE:
        try:
            sample_bg = X_background.sample(min(50, len(X_background)), random_state=RANDOM_STATE)
            if isinstance(inner_model, (RandomForestClassifier,)) or "XGB" in type(inner_model).__name__ or \
               "LGBM" in type(inner_model).__name__:
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X_row)
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]
                values = np.array(shap_values).flatten()
            else:
                explainer = shap.Explainer(inner_model.predict, sample_bg)
                shap_values = explainer(X_row)
                values = np.array(shap_values.values).flatten()

            contributions = {f: float(v) for f, v in zip(feature_names, values)}
            return {"method": "shap", "values": contributions}
        except Exception:  # noqa: BLE001
            logger.exception("SHAP explanation failed, falling back to feature importance")

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
        contributions = {f: float(v) for f, v in zip(feature_names, importances)}
        return {"method": "feature_importance", "values": contributions}

    if hasattr(inner_model, "coef_"):
        coefs = np.array(inner_model.coef_).flatten()
        contributions = {f: float(v) for f, v in zip(feature_names, coefs)}
        return {"method": "feature_importance", "values": contributions}

    return {"method": "none", "values": {}}


def global_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    inner_model = model
    if hasattr(model, "named_steps"):
        inner_model = model.named_steps.get("clf", model)
    if hasattr(inner_model, "feature_importances_"):
        return {f: float(v) for f, v in zip(feature_names, inner_model.feature_importances_)}
    if hasattr(inner_model, "coef_"):
        return {f: float(v) for f, v in zip(feature_names, np.abs(np.array(inner_model.coef_).flatten()))}
    return {}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_model(obj, filename: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(obj, path)
    logger.info("Saved model artifact to %s", path)
    return path


def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load model artifact %s", filename)
        return None


def model_artifact_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, filename))


@dataclass
class TrainingMetadata:
    model_name: str
    training_date: str
    features: List[str]
    metrics: Dict
    dataset_size: int
    target_column: str

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "training_date": self.training_date,
            "features": self.features,
            "metrics": self.metrics,
            "dataset_size": self.dataset_size,
            "target_column": self.target_column,
        }


def build_metadata(model_name: str, features: List[str], metrics: Dict, dataset_size: int, target_column: str) -> Dict:
    return TrainingMetadata(
        model_name=model_name,
        training_date=datetime.utcnow().isoformat(),
        features=features,
        metrics=metrics,
        dataset_size=dataset_size,
        target_column=target_column,
    ).to_dict()

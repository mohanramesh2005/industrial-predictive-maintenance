"""
app.py
------
Streamlit entry point for the Industrial Machine Failure Intelligence
System. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import io
import traceback
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_ml
import database
import models
import utils

st.set_page_config(
    page_title="Industrial Machine Failure Intelligence System",
    page_icon="🛠️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Startup / initialization (never crashes if files/db/models are missing)
# --------------------------------------------------------------------------

def initialize_app() -> None:
    try:
        utils.ensure_directories()
        database.init_db()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize application: {exc}")

    defaults = {
        "dataset": None,
        "dataset_source": None,
        "normalized_columns": {},
        "feature_cols": [],
        "target_col": None,
        "failure_type_col": None,
        "rul_col": None,
        "machine_id_col": None,
        "timestamp_col": None,
        "classification_results": None,
        "best_model_name": None,
        "anomaly_model": None,
        "failure_type_bundle": None,
        "rul_bundle": None,
        "train_test_data": None,
        "training_metadata": None,
        "primary_metric": "recall",
        "enable_tuning": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_app()

# --------------------------------------------------------------------------
# Model artifact filenames
# --------------------------------------------------------------------------

CLASSIFIER_FILE = "best_classifier.joblib"
ANOMALY_FILE = "anomaly_model.joblib"
FAILURE_TYPE_FILE = "failure_type_model.joblib"
RUL_FILE = "rul_model.joblib"
METADATA_FILE = "training_metadata.joblib"
FEATURE_COLS_FILE = "feature_columns.joblib"


def try_load_persisted_models() -> None:
    """On startup, silently reload previously trained models if present,
    so the app never retrains unnecessarily."""
    if st.session_state["classification_results"] is not None:
        return
    clf = models.load_model(CLASSIFIER_FILE)
    feature_cols = models.load_model(FEATURE_COLS_FILE)
    metadata = models.load_model(METADATA_FILE)
    anomaly = models.load_model(ANOMALY_FILE)
    if clf is not None and feature_cols is not None:
        st.session_state["classification_results"] = {
            metadata.get("model_name", "Loaded Model"): {
                "model": clf,
                "metrics": metadata.get("metrics", {}),
                "cv_metrics": None,
                "confusion_matrix": None,
                "roc_points": None,
                "pr_points": None,
            }
        } if metadata else None
        st.session_state["best_model_name"] = metadata.get("model_name") if metadata else None
        st.session_state["feature_cols"] = feature_cols
        st.session_state["training_metadata"] = metadata
    if anomaly is not None:
        st.session_state["anomaly_model"] = anomaly
    ft_bundle = models.load_model(FAILURE_TYPE_FILE)
    if ft_bundle is not None:
        st.session_state["failure_type_bundle"] = ft_bundle
    rul_bundle = models.load_model(RUL_FILE)
    if rul_bundle is not None:
        st.session_state["rul_bundle"] = rul_bundle


try_load_persisted_models()

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

PAGES = [
    "Dashboard",
    "Dataset",
    "Model Training",
    "Machine Prediction",
    "Anomaly Detection",
    "Explainability",
    "Maintenance History",
    "Model Performance",
    "About",
]

st.sidebar.title("🛠️ Predictive Maintenance")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Predictive Maintenance & Failure Intelligence System")
if st.session_state["dataset"] is not None:
    st.sidebar.success(f"Dataset loaded: {st.session_state['dataset'].shape[0]} rows")
else:
    st.sidebar.info("No dataset loaded yet.")
if st.session_state["classification_results"]:
    st.sidebar.success(f"Model trained: {st.session_state['best_model_name']}")
else:
    st.sidebar.warning("No trained model yet.")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def get_working_dataframe() -> Optional[pd.DataFrame]:
    return st.session_state.get("dataset")


def run_full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, detect special columns, clean, and engineer
    features. Stores detected column names into session_state."""
    df_norm, _mapping = data_ml.normalize_columns(df)
    st.session_state["target_col"] = data_ml.detect_target_column(df_norm)
    st.session_state["failure_type_col"] = data_ml.detect_failure_type_column(df_norm)
    st.session_state["rul_col"] = data_ml.detect_rul_column(df_norm)
    st.session_state["machine_id_col"] = data_ml.detect_machine_id_column(df_norm)
    st.session_state["timestamp_col"] = data_ml.detect_timestamp_column(df_norm)

    sensor_cols = data_ml.get_available_sensor_columns(df_norm)
    df_clean = data_ml.clean_data(df_norm, sensor_cols)
    df_features = data_ml.engineer_features(
        df_clean,
        machine_id_col=st.session_state["machine_id_col"],
        timestamp_col=st.session_state["timestamp_col"],
    )
    return df_features


def safe_run(fn, *args, error_prefix: str = "An error occurred", **kwargs):
    """Run a function, converting any exception into a user-facing
    st.error instead of a raw traceback."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{error_prefix}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None


# ==========================================================================
# PAGE: Dashboard
# ==========================================================================

if page == "Dashboard":
    st.title("🛠️ Industrial Machine Failure Intelligence — Dashboard")
    st.caption("AI-Based Predictive Maintenance and Machine Failure Prediction System")

    latest_preds = safe_run(database.get_latest_prediction_per_machine, error_prefix="Could not read predictions")
    latest_preds = latest_preds or []

    total_machines = len(latest_preds)
    healthy = sum(1 for p in latest_preds if p.get("health_status") == "HEALTHY")
    warning = sum(1 for p in latest_preds if p.get("health_status") == "WARNING")
    critical = sum(1 for p in latest_preds if p.get("health_status") == "CRITICAL")
    avg_health = np.mean([p.get("health_score", 0) for p in latest_preds]) if latest_preds else 0.0
    avg_fail_prob = np.mean([p.get("failure_probability", 0) for p in latest_preds]) if latest_preds else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Machines", total_machines)
    c2.metric("Healthy", healthy)
    c3.metric("Warning", warning)
    c4.metric("Critical", critical)
    c5.metric("Avg Health Score", f"{avg_health:.1f}")
    c6.metric("Avg Failure Probability", f"{avg_fail_prob:.1f}%")

    if not latest_preds:
        st.info(
            "No predictions logged yet. Go to **Dataset** to load data, then **Model Training** to "
            "train models, then **Machine Prediction** to generate predictions."
        )
    else:
        pred_df = pd.DataFrame(latest_preds)
        col_a, col_b = st.columns(2)
        with col_a:
            status_counts = pred_df["health_status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = px.pie(status_counts, names="status", values="count", title="Machine Health Distribution",
                         color="status",
                         color_discrete_map={"HEALTHY": "#2ecc71", "WARNING": "#f1c40f", "CRITICAL": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            risk_counts = pred_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            fig2 = px.bar(risk_counts, x="risk_level", y="count", title="Failure-Risk Distribution",
                          color="risk_level",
                          color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Health Score by Machine")
        fig3 = px.bar(pred_df.sort_values("health_score"), x="machine_id", y="health_score",
                      color="health_status",
                      color_discrete_map={"HEALTHY": "#2ecc71", "WARNING": "#f1c40f", "CRITICAL": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

    df = get_working_dataframe()
    if df is not None:
        sensor_cols = data_ml.get_available_sensor_columns(df)
        if len(sensor_cols) >= 2:
            st.subheader("Sensor Correlation Heatmap")
            corr = df[sensor_cols].corr()
            fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Sensor Correlation (loaded dataset)")
            st.plotly_chart(fig4, use_container_width=True)


# ==========================================================================
# PAGE: Dataset
# ==========================================================================

elif page == "Dataset":
    st.title("📂 Dataset")

    upload_col, demo_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            df, error = data_ml.load_csv(uploaded)
            if error:
                st.error(error)
            else:
                processed = safe_run(run_full_preprocessing, df, error_prefix="Preprocessing failed")
                if processed is not None:
                    st.session_state["dataset"] = processed
                    st.session_state["dataset_source"] = "uploaded"
                    st.success(f"Loaded {processed.shape[0]} rows and {processed.shape[1]} columns.")

    with demo_col:
        st.write("No dataset? Generate a synthetic demo dataset:")
        n_machines = st.slider("Number of demo machines", 4, 30, 12)
        n_records = st.slider("Records per machine", 50, 500, 250, step=50)
        if st.button("Generate Demo Dataset", type="primary"):
            demo_df = safe_run(
                data_ml.generate_demo_dataset,
                n_machines=n_machines,
                n_records_per_machine=n_records,
                error_prefix="Demo generation failed",
            )
            if demo_df is not None:
                processed = safe_run(run_full_preprocessing, demo_df, error_prefix="Preprocessing failed")
                if processed is not None:
                    st.session_state["dataset"] = processed
                    st.session_state["dataset_source"] = "demo"
                    st.success(f"Generated demo dataset with {processed.shape[0]} rows across {n_machines} machines.")

    if st.session_state["dataset_source"] == "demo":
        st.warning("⚠️ DEMO DATASET — NOT REAL INDUSTRIAL SENSOR DATA. For demonstration/testing only.")

    df = get_working_dataframe()
    if df is None:
        st.info("Upload a CSV or generate a demo dataset to get started.")
    else:
        summary = safe_run(data_ml.validate_dataset, df, error_prefix="Could not summarize dataset")
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", summary["rows"])
            c2.metric("Columns", summary["columns"])
            c3.metric("Missing Values", summary["missing_values"])
            c4.metric("Duplicate Rows", summary["duplicate_rows"])

            st.markdown(f"**Detected target column:** `{summary['target_column'] or 'None'}`")
            st.markdown(f"**Detected failure-type column:** `{summary['failure_type_column'] or 'None'}`")
            st.markdown(f"**Detected RUL column:** `{summary['rul_column'] or 'None'}`")
            st.markdown(f"**Detected machine ID column:** `{summary['machine_id_column'] or 'None'}`")
            st.markdown(f"**Detected timestamp column:** `{summary['timestamp_column'] or 'None'}`")

            missing_sensors = summary["missing_sensor_columns"]
            if missing_sensors:
                st.warning(f"Missing expected sensor columns: {', '.join(missing_sensors)}")

            if summary["target_distribution"]:
                st.subheader("Target Distribution")
                dist_df = pd.DataFrame(
                    list(summary["target_distribution"].items()), columns=["class", "count"]
                )
                st.bar_chart(dist_df.set_index("class"))
            else:
                st.info("No supervised failure target detected — only anomaly detection will be available.")

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)


# ==========================================================================
# PAGE: Model Training
# ==========================================================================

elif page == "Model Training":
    st.title("🧠 Model Training")
    df = get_working_dataframe()

    if df is None:
        st.warning("Load or generate a dataset first (see the Dataset page).")
    else:
        target_col = st.session_state["target_col"]
        feature_cols = data_ml.get_feature_columns(df)

        if not feature_cols:
            st.error("No usable numeric sensor/feature columns were found in the dataset.")
        elif target_col is None:
            st.warning(
                "No binary failure target detected (expected one of: failure, machine_failure, "
                "failure_flag, failed, target). Supervised failure prediction requires a labeled target. "
                "You can still use Anomaly Detection."
            )
        else:
            imbalance = data_ml.check_class_imbalance(df[target_col].dropna())
            st.write("**Class distribution:**", imbalance["counts"])
            if imbalance["is_imbalanced"]:
                st.info("The failure class is imbalanced — class_weight='balanced' will be applied automatically.")

            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)
            with col2:
                primary_metric = st.selectbox(
                    "Primary metric for best-model selection", ["recall", "f1_score", "precision", "roc_auc", "accuracy"]
                )
            with col3:
                enable_tuning = st.checkbox("Enable light hyperparameter tuning (slower, may improve accuracy)", value=False)

            if st.button("Train Models", type="primary"):
                with st.spinner("Training models..."):
                    split = safe_run(
                        data_ml.prepare_train_test,
                        df,
                        target_col,
                        feature_cols,
                        timestamp_col=st.session_state["timestamp_col"],
                        test_size=test_size,
                        error_prefix="Train/test split failed",
                    )
                    if split is not None:
                        st.caption(f"Split method used: **{split['split_method']}**")
                        results = safe_run(
                            models.train_classification_models,
                            split["X_train"], split["y_train"], split["X_test"], split["y_test"],
                            use_class_weight=imbalance["is_imbalanced"],
                            enable_tuning=enable_tuning,
                            error_prefix="Model training failed",
                        )
                        if results:
                            best_name = models.select_best_model(results, primary_metric=primary_metric)
                            st.session_state["classification_results"] = results
                            st.session_state["best_model_name"] = best_name
                            st.session_state["feature_cols"] = feature_cols
                            st.session_state["train_test_data"] = split
                            st.session_state["primary_metric"] = primary_metric

                            best_model = results[best_name]["model"]
                            models.save_model(best_model, CLASSIFIER_FILE)
                            models.save_model(feature_cols, FEATURE_COLS_FILE)
                            metadata = models.build_metadata(
                                best_name, feature_cols, results[best_name]["metrics"], len(df), target_col
                            )
                            models.save_model(metadata, METADATA_FILE)
                            st.session_state["training_metadata"] = metadata

                            # Anomaly model
                            anomaly_model = safe_run(
                                models.train_anomaly_model, split["X_train"], split["y_train"],
                                error_prefix="Anomaly model training failed",
                            )
                            if anomaly_model is not None:
                                st.session_state["anomaly_model"] = anomaly_model
                                models.save_model(anomaly_model, ANOMALY_FILE)

                            # Optional failure-type model
                            ft_col = st.session_state["failure_type_col"]
                            if ft_col and ft_col in df.columns:
                                ft_bundle = safe_run(
                                    models.train_failure_type_model, df[feature_cols], df[ft_col],
                                    error_prefix="Failure-type model training failed",
                                )
                                if ft_bundle is not None:
                                    st.session_state["failure_type_bundle"] = ft_bundle
                                    models.save_model(ft_bundle, FAILURE_TYPE_FILE)

                            # Optional RUL model
                            rul_col = st.session_state["rul_col"]
                            if rul_col and rul_col in df.columns:
                                rul_bundle = safe_run(
                                    models.train_rul_model, df[feature_cols], df[rul_col],
                                    error_prefix="RUL model training failed",
                                )
                                if rul_bundle is not None:
                                    st.session_state["rul_bundle"] = rul_bundle
                                    models.save_model(rul_bundle, RUL_FILE)

                            st.success(f"Training complete. Best model by {primary_metric}: **{best_name}**")

            results = st.session_state.get("classification_results")
            if results:
                st.subheader("Model Comparison (held-out test set)")
                comp_rows = []
                for name, info in results.items():
                    row = {"Model": name}
                    row.update({k.replace("_", " ").title(): round(v, 4) if v == v else None
                                for k, v in info["metrics"].items()})
                    comp_rows.append(row)
                comp_df = pd.DataFrame(comp_rows).set_index("Model")
                st.dataframe(comp_df, use_container_width=True)
                st.markdown(f"**Best model:** `{st.session_state['best_model_name']}`")

                cv_available = any(info.get("cv_metrics") for info in results.values())
                if cv_available:
                    st.subheader("5-Fold Stratified Cross-Validation (robustness check)")
                    for name, info in results.items():
                        if info.get("cv_metrics"):
                            with st.expander(f"{name} — CV metrics"):
                                cv_rows = [
                                    {"Metric": m, "Mean": f"{v['mean']:.4f}", "Std Dev": f"{v['std']:.4f}"}
                                    for m, v in info["cv_metrics"].items()
                                ]
                                st.table(pd.DataFrame(cv_rows))

                selected_model_name = st.selectbox("View details for model", list(results.keys()))
                info = results[selected_model_name]
                col_a, col_b = st.columns(2)
                with col_a:
                    if info.get("confusion_matrix"):
                        cm = np.array(info["confusion_matrix"])
                        fig_cm = px.imshow(cm, text_auto=True, x=["Pred Normal", "Pred Failure"],
                                            y=["Actual Normal", "Actual Failure"], color_continuous_scale="Blues",
                                            title=f"Confusion Matrix — {selected_model_name}")
                        st.plotly_chart(fig_cm, use_container_width=True)
                with col_b:
                    if info.get("roc_points"):
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=info["roc_points"]["fpr"], y=info["roc_points"]["tpr"],
                                                      mode="lines", name="ROC Curve"))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                                      line=dict(dash="dash")))
                        fig_roc.update_layout(title=f"ROC Curve — {selected_model_name}",
                                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                        st.plotly_chart(fig_roc, use_container_width=True)

                if info.get("pr_points"):
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=info["pr_points"]["recall"], y=info["pr_points"]["precision"],
                                                 mode="lines", name="PR Curve"))
                    fig_pr.update_layout(title=f"Precision-Recall Curve — {selected_model_name}",
                                          xaxis_title="Recall", yaxis_title="Precision")
                    st.plotly_chart(fig_pr, use_container_width=True)

            if st.session_state.get("failure_type_bundle"):
                ftb = st.session_state["failure_type_bundle"]
                st.info(f"Failure-type model trained — accuracy: {ftb['accuracy']:.3f}, macro-F1: {ftb['f1_macro']:.3f}")
            if st.session_state.get("rul_bundle"):
                rb = st.session_state["rul_bundle"]
                st.info(f"RUL model trained — best: {rb['best_model_name']}, "
                        f"MAE: {rb['comparison'][rb['best_model_name']]['mae']:.2f}")


# ==========================================================================
# PAGE: Machine Prediction
# ==========================================================================

elif page == "Machine Prediction":
    st.title("🔍 Machine Prediction")

    results = st.session_state.get("classification_results")
    feature_cols = st.session_state.get("feature_cols")

    if not results or not st.session_state.get("best_model_name"):
        st.warning("No trained model available yet. Go to **Model Training** first.")
    else:
        best_model = results[st.session_state["best_model_name"]]["model"]
        st.caption(f"Using best model: **{st.session_state['best_model_name']}**")

        df = get_working_dataframe()
        machine_id_col = st.session_state.get("machine_id_col")

        mode = st.radio("Input mode", ["Manual sensor entry", "Select existing machine"], horizontal=True)

        input_values: Dict[str, float] = {}
        machine_id = "MANUAL-ENTRY"

        if mode == "Select existing machine" and df is not None and machine_id_col and machine_id_col in df.columns:
            machine_ids = sorted(df[machine_id_col].astype(str).unique().tolist())
            selected_id = st.selectbox("Machine", machine_ids)
            machine_id = selected_id
            machine_rows = df[df[machine_id_col].astype(str) == selected_id]
            latest_row = machine_rows.iloc[-1]
            for col in utils.SENSOR_COLUMNS:
                if col in df.columns:
                    input_values[col] = float(latest_row[col])
            st.write("Latest recorded sensor values for this machine:")
            st.dataframe(pd.DataFrame([input_values]))
        else:
            st.write("Enter current sensor readings:")
            c1, c2, c3, c4 = st.columns(4)
            defaults = utils.DEFAULT_VALIDATION_RANGES
            with c1:
                input_values["temperature"] = st.number_input("Temperature (°C)", value=55.0)
                input_values["voltage"] = st.number_input("Voltage (V)", value=220.0)
            with c2:
                input_values["vibration"] = st.number_input("Vibration (mm/s)", value=2.0)
                input_values["current"] = st.number_input("Current (A)", value=10.0)
            with c3:
                input_values["pressure"] = st.number_input("Pressure (psi)", value=100.0)
                input_values["operating_hours"] = st.number_input("Operating Hours", value=1000.0)
            with c4:
                input_values["rpm"] = st.number_input("RPM", value=1500.0)
                machine_id = st.text_input("Machine ID (optional)", value="MANUAL-ENTRY")

        if st.button("Run Prediction", type="primary"):
            is_valid, errors = utils.validate_sensor_input(input_values)
            if not is_valid:
                for e in errors:
                    st.error(e)
            else:
                X_row = data_ml.preprocess_for_prediction(input_values, feature_cols)

                failure_probability = 0.0
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(X_row)
                    failure_probability = float(proba[0][1]) * 100.0
                else:
                    failure_probability = float(best_model.predict(X_row)[0]) * 100.0

                anomaly_status, anomaly_score = "UNKNOWN", None
                anomaly_model = st.session_state.get("anomaly_model")
                if anomaly_model is not None:
                    scores, is_anom = models.compute_anomaly_scores(anomaly_model, X_row)
                    anomaly_score = float(scores[0])
                    anomaly_status = "ANOMALY" if bool(is_anom[0]) else "NORMAL"

                anomaly_risk_pct = (anomaly_score * 100.0) if anomaly_score is not None else 0.0
                health_score = utils.compute_health_score(failure_probability, anomaly_risk_pct)
                health_status = utils.classify_health_status(health_score)
                risk_level = utils.classify_risk_level(failure_probability)

                predicted_failure_type, ft_confidence = None, None
                ft_bundle = st.session_state.get("failure_type_bundle")
                if ft_bundle is not None:
                    predicted_failure_type, ft_confidence, class_probs = models.predict_failure_type(ft_bundle, X_row)

                estimated_rul = None
                rul_bundle = st.session_state.get("rul_bundle")
                if rul_bundle is not None:
                    estimated_rul = float(rul_bundle["model"].predict(X_row)[0])

                recommendation = utils.generate_recommendation(
                    risk_level, failure_probability, anomaly_status, estimated_rul
                )

                st.markdown("---")
                st.subheader(f"Machine Health Report — {machine_id}")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Health Score", f"{health_score:.0f}/100", health_status)
                r2.metric("Failure Probability", f"{failure_probability:.1f}%")
                r3.metric("Anomaly Status", anomaly_status,
                          f"score {anomaly_score:.2f}" if anomaly_score is not None else "n/a")
                r4.metric("Risk Level", risk_level)

                if predicted_failure_type is not None:
                    st.info(f"**Predicted Failure Type:** {predicted_failure_type} (confidence {ft_confidence:.1%})")
                else:
                    st.caption("Predicted Failure Type: Not available — dataset does not contain failure-type labels.")

                if estimated_rul is not None:
                    st.info(f"**Estimated Remaining Useful Life:** {estimated_rul:,.1f} operating hours")
                else:
                    st.caption("Estimated RUL: Not available — dataset does not contain a valid RUL/time-to-failure target.")

                st.success(f"**Recommended Action:** {recommendation}")

                safe_run(
                    database.upsert_machine, machine_id, "Unknown", None,
                    input_values.get("operating_hours", 0.0), health_status,
                    error_prefix="Could not save machine record",
                )
                safe_run(
                    database.insert_prediction,
                    {
                        "machine_id": machine_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "health_score": health_score,
                        "health_status": health_status,
                        "failure_probability": failure_probability,
                        "anomaly_score": anomaly_score,
                        "risk_level": risk_level,
                        "predicted_failure_type": predicted_failure_type,
                        "rul": estimated_rul,
                        "recommendation": recommendation,
                    },
                    error_prefix="Could not save prediction",
                )

                report_text = (
                    "MACHINE HEALTH REPORT\n\n"
                    f"Machine ID: {machine_id}\n"
                    f"Machine Health Score: {health_score:.0f}/100\n"
                    f"Health Status: {health_status}\n"
                    f"Failure Probability: {failure_probability:.1f}%\n"
                    f"Anomaly Status: {anomaly_status}\n"
                    f"Anomaly Score: {anomaly_score if anomaly_score is not None else 'n/a'}\n"
                    f"Risk Level: {risk_level}\n"
                    f"Predicted Failure Type: {predicted_failure_type or 'Not available'}\n"
                    f"Estimated RUL: {estimated_rul if estimated_rul is not None else 'Not available'}\n"
                    f"Recommended Action: {recommendation}\n"
                )
                st.download_button("Download Report (.txt)", data=report_text,
                                    file_name=f"health_report_{machine_id}.txt")


# ==========================================================================
# PAGE: Anomaly Detection
# ==========================================================================

elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    anomaly_model = st.session_state.get("anomaly_model")
    df = get_working_dataframe()
    feature_cols = st.session_state.get("feature_cols")

    if anomaly_model is None or not feature_cols:
        st.warning("Train models first on the **Model Training** page.")
    elif df is None:
        st.warning("Load a dataset first on the **Dataset** page.")
    else:
        X = df[feature_cols].fillna(0.0)
        scores, is_anom = models.compute_anomaly_scores(anomaly_model, X)
        result_df = df.copy()
        result_df["anomaly_score"] = scores
        result_df["anomaly_status"] = np.where(is_anom, "ANOMALY", "NORMAL")

        machine_id_col = st.session_state.get("machine_id_col")
        display_cols = [c for c in [machine_id_col, st.session_state.get("timestamp_col")] if c] + \
                        data_ml.get_available_sensor_columns(df) + ["anomaly_score", "anomaly_status"]
        display_cols = [c for c in display_cols if c in result_df.columns]

        st.metric("Anomalies Detected", int(is_anom.sum()), f"of {len(result_df)} records")
        st.dataframe(result_df[display_cols].sort_values("anomaly_score", ascending=False).head(200),
                     use_container_width=True)

        sensor_cols = data_ml.get_available_sensor_columns(df)
        if sensor_cols:
            x_axis = st.selectbox("X-axis sensor", sensor_cols, index=0)
            y_axis = st.selectbox("Y-axis sensor", sensor_cols, index=min(1, len(sensor_cols) - 1))
            fig = px.scatter(result_df, x=x_axis, y=y_axis, color="anomaly_status",
                              color_discrete_map={"NORMAL": "#2ecc71", "ANOMALY": "#e74c3c"},
                              title=f"{x_axis} vs {y_axis} — anomalies highlighted",
                              opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: Explainability
# ==========================================================================

elif page == "Explainability":
    st.title("🧩 Explainability")

    results = st.session_state.get("classification_results")
    feature_cols = st.session_state.get("feature_cols")
    train_test_data = st.session_state.get("train_test_data")

    if not results or not feature_cols:
        st.warning("Train models first on the **Model Training** page.")
    else:
        model_name = st.selectbox("Model", list(results.keys()))
        model = results[model_name]["model"]

        importance = models.global_feature_importance(model, feature_cols)
        if importance:
            st.subheader("Global Feature Importance")
            imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"]).sort_values(
                "importance", ascending=False
            )
            fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance unavailable for this model type.")

        st.subheader("Individual Prediction Explanation")
        df = get_working_dataframe()
        if df is not None and train_test_data is not None:
            X_background = train_test_data["X_train"]
            row_idx = st.slider("Row index (from loaded dataset)", 0, max(0, len(df) - 1), 0)
            X_row = df.iloc[[row_idx]][feature_cols].fillna(0.0)

            explanation = models.explain_prediction(model, X_background, X_row, feature_cols)
            if explanation["method"] == "shap":
                st.caption("Explanation method: SHAP")
            elif explanation["method"] == "feature_importance":
                st.warning("SHAP explanation unavailable for this model. Showing model feature importance instead.")
            else:
                st.info("No explanation available for this model type.")

            if explanation["values"]:
                exp_df = pd.DataFrame(list(explanation["values"].items()), columns=["feature", "contribution"])
                exp_df = exp_df.reindex(exp_df.contribution.abs().sort_values(ascending=False).index)
                fig2 = px.bar(exp_df.head(10), x="contribution", y="feature", orientation="h",
                              title="Top Contributing Features (this prediction)")
                st.plotly_chart(fig2, use_container_width=True)

                top_features = exp_df.head(4)["feature"].tolist()
                st.markdown("**Main risk contributors (plain language):**")
                for i, f in enumerate(top_features, start=1):
                    st.markdown(f"{i}. {f.replace('_', ' ').title()}")


# ==========================================================================
# PAGE: Maintenance History
# ==========================================================================

elif page == "Maintenance History":
    st.title("🗓️ Maintenance History")

    with st.form("add_maintenance_form"):
        st.subheader("Add Maintenance Record")
        m1, m2, m3 = st.columns(3)
        with m1:
            machine_id = st.text_input("Machine ID")
            maintenance_type = st.selectbox(
                "Maintenance Type", ["Inspection", "Preventive", "Corrective", "Emergency Repair", "Part Replacement"]
            )
        with m2:
            maintenance_date = st.date_input("Maintenance Date", value=date.today())
            cost = st.number_input("Cost", min_value=0.0, value=0.0, step=10.0)
        with m3:
            description = st.text_area("Description", height=100)

        submitted = st.form_submit_button("Add Record")
        if submitted:
            if not machine_id.strip():
                st.error("Machine ID is required.")
            else:
                safe_run(
                    database.insert_maintenance_record,
                    {
                        "machine_id": machine_id.strip(),
                        "maintenance_date": maintenance_date.isoformat(),
                        "maintenance_type": maintenance_type,
                        "description": description,
                        "cost": cost,
                    },
                    error_prefix="Could not save maintenance record",
                )
                st.success("Maintenance record added.")

    st.markdown("---")
    st.subheader("Records")

    f1, f2, f3 = st.columns(3)
    with f1:
        filter_machine = st.text_input("Filter by Machine ID (optional)")
    with f2:
        filter_start = st.date_input("From date", value=None)
    with f3:
        filter_end = st.date_input("To date", value=None)

    records = safe_run(
        database.get_maintenance_records,
        machine_id=filter_machine.strip() if filter_machine.strip() else None,
        start_date=filter_start.isoformat() if filter_start else None,
        end_date=filter_end.isoformat() if filter_end else None,
        error_prefix="Could not load maintenance records",
    )
    records = records or []
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.info("No maintenance records found.")


# ==========================================================================
# PAGE: Model Performance
# ==========================================================================

elif page == "Model Performance":
    st.title("📈 Model Performance")

    results = st.session_state.get("classification_results")
    metadata = st.session_state.get("training_metadata")

    if not results:
        st.warning("No trained models yet. Go to **Model Training**.")
    else:
        st.subheader("Best Model Metadata")
        if metadata:
            st.json(metadata)

        st.subheader("All Models — Test Set Metrics")
        rows = []
        for name, info in results.items():
            row = {"Model": name}
            row.update(info["metrics"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        st.subheader("Cross-Validation Summary")
        cv_rows = []
        for name, info in results.items():
            cv = info.get("cv_metrics")
            if cv:
                row = {"Model": name}
                for metric, stats in cv.items():
                    row[f"{metric}_mean"] = round(stats["mean"], 4)
                    row[f"{metric}_std"] = round(stats["std"], 4)
                cv_rows.append(row)
        if cv_rows:
            st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)
        else:
            st.info("Cross-validation metrics were not computed for the current models.")


# ==========================================================================
# PAGE: About
# ==========================================================================

elif page == "About":
    st.title("ℹ️ About")
    st.markdown(
        """
### Industrial Machine Failure Intelligence System Using Machine Learning

A real, end-to-end **Machine Learning** predictive maintenance system — not a chatbot,
not a GenAI application.

**Capabilities:**
- Supervised failure classification (Logistic Regression, Random Forest, XGBoost, LightGBM,
  plus a soft-voting Ensemble)
- 5-fold stratified cross-validation for more robust performance estimates
- Optional light hyperparameter tuning (RandomizedSearchCV)
- Unsupervised anomaly detection (Isolation Forest) with a clearly separate, normalized
  anomaly score (not a calibrated probability)
- Transparent, configurable Health Score and Risk Level engine
- Optional multiclass failure-type prediction and Remaining Useful Life (RUL) regression,
  only when the dataset actually supports them
- SHAP-based explainability with a safe fallback to model feature importance
- Rule-based maintenance decision support (this system does not autonomously control
  machinery)
- SQLite persistence with parameterized queries only

**Hardware target:** runs comfortably on an 8 GB RAM, CPU-only laptop — no GPU or
deep learning required.

**Important:** this system is a decision-support tool. All health scores, risk levels,
and recommendations are generated dynamically from trained models on the data provided —
nothing is hardcoded.
        """
    )
    st.caption(f"XGBoost available: {models.XGBOOST_AVAILABLE} | "
               f"LightGBM available: {models.LIGHTGBM_AVAILABLE} | "
               f"SHAP available: {models.SHAP_AVAILABLE}")

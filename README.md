# 🏭 Industrial Machine Failure Intelligence System

### AI-Powered Predictive Maintenance & Machine Failure Intelligence

> **A production-oriented, end-to-end Machine Learning system for predicting industrial machine failures, detecting abnormal operating behavior, estimating machine health and risk, explaining model decisions, and supporting maintenance planning.**

Built for **real-world tabular industrial sensor data**, with a lightweight CPU-first architecture using **Python, Streamlit, scikit-learn, XGBoost, LightGBM, SHAP, Plotly, and SQLite**.

> **Important:** This is a classical Machine Learning system—not a chatbot, LLM, or Generative AI application. The core intelligence comes from trained ML models operating on structured industrial sensor data.

---

## 🚀 Why This Project?

Industrial equipment can experience abnormal operating conditions before a complete failure occurs.

Traditional maintenance approaches often depend on:

* Fixed maintenance schedules
* Manual inspection
* Operator experience
* Reactive repairs after failure
* Simple threshold-based alarms

This system introduces a **data-driven predictive maintenance workflow**:

```text
Industrial Sensor Data
        │
        ▼
┌─────────────────────────┐
│ Data Validation         │
│ Cleaning & Normalization│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Feature Engineering     │
│ Temporal / Sensor       │
│ Derived Features        │
└────────────┬────────────┘
             │
             ▼
      ┌──────┴──────┐
      │             │
      ▼             ▼
 Failure ML     Anomaly ML
 Classification Detection
      │             │
      └──────┬──────┘
             ▼
┌─────────────────────────┐
│ Machine Intelligence    │
│                         │
│ • Failure Probability   │
│ • Anomaly Score         │
│ • Health Score          │
│ • Risk Level            │
│ • Failure Type          │
│ • RUL                   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Explainability          │
│ SHAP / Feature Impact   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Maintenance Decision    │
│ Support & History       │
└─────────────────────────┘
```

---

# ⭐ Core Capabilities

| Capability                         | Status     |
| ---------------------------------- | ---------- |
| Binary machine-failure prediction  | ✅          |
| Failure probability estimation     | ✅          |
| Anomaly detection                  | ✅          |
| Machine health scoring             | ✅          |
| Risk classification                | ✅          |
| Maintenance recommendations        | ✅          |
| Failure-type prediction            | ✅ Optional |
| Remaining Useful Life prediction   | ✅ Optional |
| Model comparison                   | ✅          |
| 5-fold stratified cross-validation | ✅          |
| Hyperparameter tuning              | ✅ Optional |
| SHAP explainability                | ✅          |
| Feature-importance fallback        | ✅          |
| Prediction history                 | ✅          |
| Maintenance history                | ✅          |
| SQLite persistence                 | ✅          |
| Interactive Streamlit UI           | ✅          |
| CPU-only operation                 | ✅          |
| Demo dataset generation            | ✅          |

---

# 🧠 Intelligence Architecture

The system separates different forms of machine intelligence instead of incorrectly combining unrelated scores.

### 1. Failure Prediction

A supervised classification model estimates:

```text
P(machine failure | sensor features)
```

The system can train:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Soft-Voting Ensemble

The selected classifier's `predict_proba()` is used for the displayed failure probability.

---

### 2. Anomaly Detection

An **Isolation Forest** independently identifies unusual machine behavior.

```text
Sensor Features
      │
      ▼
Isolation Forest
      │
      ├── Anomaly / Normal
      │
      └── Normalized Anomaly Score
```

The anomaly score is **not treated as a calibrated failure probability**.

This distinction is intentional:

```text
Failure Probability ≠ Anomaly Score
```

The application keeps both signals separate rather than misleadingly merging them into one probability.

---

### 3. Machine Health Intelligence

The application combines model outputs and configured decision logic to produce:

```text
Machine Health Score
        │
        ▼
Health Status
        │
        ▼
Risk Level
        │
        ▼
Maintenance Recommendation
```

This allows the system to move beyond a simple:

> `FAILURE = 1`

prediction and provide an engineering-oriented machine condition report.

---

# 📊 Machine Health Report

For an individual machine, the application can generate:

```text
┌─────────────────────────────────────────┐
│         MACHINE HEALTH REPORT           │
├─────────────────────────────────────────┤
│ Failure Probability : XX.XX%             │
│ Anomaly Status      : NORMAL / ANOMALY   │
│ Anomaly Score       : XX.XX              │
│ Health Score        : XX/100             │
│ Risk Level          : LOW / MEDIUM / HIGH│
│ Failure Type        : Optional           │
│ Remaining Useful Life: Optional           │
│ Recommendation      : Maintenance Action │
└─────────────────────────────────────────┘
```

Predictions are also persisted to SQLite for historical analysis and dashboard reporting.

---

# 🏗️ System Architecture

```text
                         ┌───────────────────┐
                         │   CSV / Dataset   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Data Processing Layer    │
                    │                          │
                    │ • Column normalization   │
                    │ • Validation             │
                    │ • Missing-value handling │
                    │ • Feature engineering    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    ML Training Layer     │
                    │                          │
                    │ Logistic Regression      │
                    │ Random Forest            │
                    │ XGBoost                  │
                    │ LightGBM                 │
                    │ Voting Ensemble          │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │ Failure ML │ │  Anomaly   │ │  Optional  │
          │ Classifier │ │ Detection  │ │ Advanced ML│
          └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                │              │       ┌──────┴──────┐
                │              │       │             │
                │              │       ▼             ▼
                │              │   Failure Type     RUL
                │              │
                └──────────────┼──────────────┘
                               ▼
                   ┌──────────────────────────┐
                   │ Machine Intelligence     │
                   │                          │
                   │ Health Score             │
                   │ Risk Level               │
                   │ Recommendations          │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │ Explainability Layer     │
                   │                          │
                   │ SHAP / Feature Importance│
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │ Streamlit Application     │
                   │                          │
                   │ Dashboard                │
                   │ Prediction               │
                   │ Anomaly Detection        │
                   │ Explainability           │
                   │ Maintenance              │
                   │ Performance              │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │ SQLite Persistence       │
                   │                          │
                   │ Machines                 │
                   │ Sensor Readings          │
                   │ Predictions              │
                   │ Maintenance Records      │
                   └──────────────────────────┘
```

---

# 📁 Project Structure

```text
industrial_predictive_maintenance/
│
├── app.py
├── data_ml.py
├── models.py
├── database.py
├── utils.py
│
├── data/
│   └── machine_data.csv
│
├── models/
│   ├── best_classifier.joblib
│   ├── anomaly_model.joblib
│   ├── failure_type_model.joblib
│   ├── rul_model.joblib
│   ├── training_metadata.joblib
│   └── feature_columns.joblib
│
├── predictive_maintenance.db
├── requirements.txt
└── README.md
```

### Design Philosophy

The application intentionally uses a compact architecture of **five Python files**, avoiding unnecessary modules, notebooks, or infrastructure dependencies.

---

# 🛠️ Technology Stack

| Layer             | Technology     |
| ----------------- | -------------- |
| Language          | Python         |
| UI / Application  | Streamlit      |
| ML Framework      | scikit-learn   |
| Gradient Boosting | XGBoost        |
| Gradient Boosting | LightGBM       |
| Explainability    | SHAP           |
| Visualization     | Plotly         |
| Database          | SQLite         |
| Persistence       | Joblib         |
| Data Processing   | Pandas / NumPy |
| Deployment Model  | CPU-oriented   |

---

# 📦 Installation

## Requirements

The project is designed to operate on:

* Python 3.12
* 8 GB RAM
* CPU-only hardware
* No dedicated GPU
* No PostgreSQL
* No Docker requirement

The documented target environment is CPU-only and does not require deep learning hardware.

---

## 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd industrial_predictive_maintenance
```

> Replace `<YOUR-REPOSITORY-URL>` with your actual GitHub repository URL.

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application normally starts at:

```text
http://localhost:8501
```

On first launch, the application initializes required directories and the SQLite database, attempts to load persisted model artifacts, and provides clear guidance when a dataset or trained model is not yet available.

---

# 📥 Dataset Contract

The system accepts industrial sensor data through CSV.

## Required Sensor Features

| Feature           | Description                |
| ----------------- | -------------------------- |
| `temperature`     | Machine temperature        |
| `vibration`       | Machine vibration          |
| `pressure`        | Operating pressure         |
| `rpm`             | Rotational speed           |
| `voltage`         | Supply voltage             |
| `current`         | Supply current             |
| `operating_hours` | Cumulative operating hours |

Column capitalization and spacing are normalized automatically.

---

## Optional Intelligence Columns

### Binary Failure Target

Supported names include:

```text
failure
machine_failure
failure_flag
failed
target
```

Expected semantic meaning:

```text
0 → Normal
1 → Failure
```

---

### Failure Type

Supported names:

```text
failure_type
failure_category
fault_type
```

Used for optional multiclass failure classification.

---

### Remaining Useful Life

Supported names:

```text
rul
remaining_useful_life
hours_to_failure
cycles_to_failure
```

Used for optional RUL regression.

---

### Machine Identifier

Supported names:

```text
machine_id
machine
asset_id
equipment_id
```

---

### Timestamp

Supported names:

```text
timestamp
time
datetime
date
```

When a timestamp is available, the training workflow can use chronological splitting to reduce temporal leakage risk.

---

# 🧪 Demo Dataset

The application can generate a deterministic synthetic dataset from the **Dataset** page.

The demo dataset contains:

* Multiple machines
* Sensor readings
* Timestamps
* Failure labels
* Failure types
* RUL information
* Degradation trends

The generated data is explicitly labeled:

```text
DEMO DATASET — NOT REAL INDUSTRIAL SENSOR DATA
```

This prevents synthetic demonstration data from being confused with real industrial measurements.

---

# 🤖 Machine Learning Pipeline

## Training Workflow

```text
Dataset
   │
   ▼
Validation
   │
   ▼
Feature Engineering
   │
   ▼
Train/Test Split
   │
   ├───────────────┐
   │               │
   ▼               ▼
Classification   Anomaly Detection
   │               │
   ▼               ▼
Model Evaluation  Isolation Forest
   │
   ▼
Cross Validation
   │
   ▼
Optional Hyperparameter Tuning
   │
   ▼
Best Model Selection
   │
   ▼
Model Persistence
```

---

# 🧮 Classification Models

| Model                | Role                                |
| -------------------- | ----------------------------------- |
| Logistic Regression  | Interpretable linear baseline       |
| Random Forest        | Nonlinear bagging model             |
| XGBoost              | Gradient-boosted tabular model      |
| LightGBM             | Efficient gradient boosting         |
| Soft-Voting Ensemble | Probability-based model combination |

The system can account for class imbalance through model-specific weighting strategies.

---

# 🌲 Anomaly Detection

The anomaly subsystem uses:

**Isolation Forest**

Conceptually:

```text
Normal Operating Behavior
          │
          ▼
   Isolation Forest
          │
     ┌────┴────┐
     ▼         ▼
  Normal     Anomaly
```

The model is trained primarily from normal samples when valid failure labels are available.

The resulting anomaly score is normalized for presentation but should **not be interpreted as a calibrated probability of failure**.

---

# 🔬 Optional Advanced Models

## Failure-Type Prediction

When a valid multiclass failure-type column exists:

```text
Sensor Data
     │
     ▼
Multiclass Classifier
     │
     ├── Failure Type A
     ├── Failure Type B
     ├── Failure Type C
     └── ...
```

The application does not fabricate failure-type labels when the dataset does not contain them.

---

## Remaining Useful Life

When valid RUL data exists:

```text
Sensor History
      │
      ▼
RUL Regression
      │
      ▼
Estimated Remaining Useful Life
```

Supported regression models include:

* Random Forest Regressor
* XGBoost Regressor

The system enables this module only when the dataset provides suitable RUL information.

---

# 📈 Model Evaluation

The system evaluates trained classifiers using multiple complementary metrics.

| Metric        | Purpose                                     |
| ------------- | ------------------------------------------- |
| Accuracy      | Overall classification correctness          |
| Precision     | Reliability of predicted failures           |
| Recall        | Ability to detect actual failures           |
| F1            | Balance between precision and recall        |
| ROC-AUC       | Ranking performance across thresholds       |
| PR-AUC        | Particularly useful with imbalanced classes |
| CV Mean ± Std | Stability across validation folds           |
| MAE           | RUL regression error                        |
| RMSE          | Penalized RUL regression error              |
| R²            | RUL regression goodness of fit              |

---

## 🎯 Why Recall Is Important

The default model-selection metric is **Recall**.

For predictive maintenance:

```text
False Negative
      ↓
Actual failure missed
      ↓
Potential equipment damage
      ↓
Potential downtime
```

Therefore, the system prioritizes the ability to identify actual failures rather than relying solely on overall accuracy.

---

# 🔁 Cross-Validation

The classification models also undergo:

```text
5-Fold Stratified Cross-Validation
```

Conceptually:

```text
Dataset
 │
 ├── Fold 1 → Train / Validate
 ├── Fold 2 → Train / Validate
 ├── Fold 3 → Train / Validate
 ├── Fold 4 → Train / Validate
 └── Fold 5 → Train / Validate
              │
              ▼
       Mean ± Standard Deviation
```

This provides a more robust estimate than depending entirely on one train/test split.

---

# 🔧 Hyperparameter Optimization

An optional lightweight tuning mode can be enabled during model training.

```text
Base Models
     │
     ▼
Candidate Parameters
     │
     ▼
Randomized Search
     │
     ▼
Validation
     │
     ▼
Improved Configuration
```

Tuning is optional because the application is designed to remain practical on CPU-only hardware.

---

# 🧠 Explainability

Machine-learning predictions are more useful when engineers can understand **why** a model produced them.

The application provides:

### Global Explainability

```text
Feature
   │
   ▼
Importance
   │
   ▼
Global Model Behavior
```

### Local Explainability

For an individual prediction:

```text
Machine Sensor Values
        │
        ▼
      Model
        │
        ▼
Prediction
        │
        ▼
SHAP / Feature Importance
        │
        ▼
Top Contributing Features
```

SHAP is used when compatible with the trained model/environment; otherwise the application falls back to model feature importance rather than failing completely.

---

# 🖥️ Application Modules

## 1. Dashboard

Provides:

* Machine health KPIs
* Healthy / warning / critical distribution
* Average health score
* Average failure probability
* Historical prediction information
* Sensor correlation analysis
* Machine health visualization

---

## 2. Dataset

Provides:

* CSV upload
* Demo dataset generation
* Dataset validation
* Missing-value inspection
* Duplicate inspection
* Target detection
* Target distribution
* Dataset preview

---

## 3. Model Training

Provides:

* Class distribution analysis
* Imbalance warnings
* Configurable test-set size
* Model selection metric
* Optional hyperparameter tuning
* Multiple classifier training
* Ensemble training
* Anomaly detection
* Optional failure-type model
* Optional RUL model
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Cross-validation results

---

## 4. Machine Prediction

Supports:

* Manual sensor input
* Existing-machine selection
* Failure probability
* Anomaly detection
* Health score
* Risk level
* Failure-type prediction
* RUL prediction
* Maintenance recommendation
* Prediction persistence
* Downloadable machine health report

---

## 5. Anomaly Detection

Provides:

* Machine-level anomaly analysis
* Anomaly scores
* Normal/anomaly classification
* Visual analysis
* Tabular results

---

## 6. Explainability

Provides:

* Global feature importance
* SHAP-based analysis where supported
* Feature-importance fallback
* Individual prediction interpretation

---

## 7. Maintenance History

Provides:

* Maintenance record creation
* Maintenance history storage
* Filtering
* Machine-specific history
* Maintenance categories

Supported maintenance types include:

```text
Inspection
Preventive
Corrective
Emergency Repair
Part Replacement
```

---

## 8. Model Performance

Provides:

* Test metrics
* Cross-validation summaries
* Model comparison
* Saved training metadata
* Training information
* Feature information

---

# 🗄️ Data Persistence

The system uses **SQLite** for local persistence.

```text
predictive_maintenance.db
```

No external database server is required.

### Database Entities

```text
machines
    │
    ├── Machine identity
    ├── Type
    ├── Installation information
    ├── Operating hours
    └── Current status

sensor_readings
    │
    └── Machine sensor snapshots

predictions
    │
    └── Historical ML predictions

maintenance_records
    │
    └── Maintenance activities
```

The database uses parameterized SQL queries rather than concatenating user-controlled input into SQL statements.

---

# 💾 Model Persistence

Trained artifacts are persisted using Joblib.

Typical artifacts include:

```text
models/
├── best_classifier.joblib
├── anomaly_model.joblib
├── failure_type_model.joblib
├── rul_model.joblib
├── training_metadata.joblib
└── feature_columns.joblib
```

The application reloads existing trained artifacts after restart instead of automatically retraining every time.

---

# 🔐 Reliability & Data-Science Design Principles

The project is designed around several important ML engineering principles.

### No Automatic Label Fabrication

If a failure target does not exist:

```text
No target
   ↓
No supervised failure classifier
   ↓
Anomaly detection remains available
```

The application does not invent failure labels.

---

### Leakage Awareness

The documented design includes:

* Training-only preprocessing
* Chronological splitting when timestamp information exists
* Prevention of future-looking rolling/lag features
* Held-out test data
* Separate validation during model development

---

### No Hardcoded ML Results

Predictions, metrics, health values, and recommendations are generated from the trained models and supplied data rather than hardcoded demonstration numbers.

---

### Graceful Failure Handling

The application explicitly handles missing:

* Datasets
* Models
* Optional ML libraries
* Database state
* SHAP compatibility

Instead of silently failing, the UI reports warnings/errors and provides actionable information.

---

# ⚙️ Application Workflow

A typical user workflow is:

```text
1. Launch Application
        ↓
2. Load CSV / Generate Demo Dataset
        ↓
3. Validate Dataset
        ↓
4. Inspect Data Distribution
        ↓
5. Train ML Models
        ↓
6. Compare Models
        ↓
7. Select Best Model
        ↓
8. Persist Model Artifacts
        ↓
9. Predict Machine Condition
        ↓
10. Analyze Anomalies
        ↓
11. Explain Predictions
        ↓
12. Generate Maintenance Recommendation
        ↓
13. Store Prediction
        ↓
14. Review Maintenance History
```

---

# 🧪 Fresh Installation Behavior

The application is designed so a fresh installation does not require pre-trained artifacts.

On startup it can:

```text
Create required directories
        ↓
Initialize SQLite
        ↓
Load existing model artifacts
        ↓
Check dataset availability
        ↓
Display appropriate UI state
```

A user can therefore install the project first and train models from the application rather than requiring a pre-generated model package.

---

# 🐛 Troubleshooting

| Problem                           | Solution                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| `streamlit: command not found`    | Activate the virtual environment and reinstall requirements  |
| No trained model                  | Open **Model Training** and train the models                 |
| No binary target detected         | Add a supported failure-target column                        |
| Failure type unavailable          | Add `failure_type`, `failure_category`, or `fault_type`      |
| RUL unavailable                   | Add a supported RUL column                                   |
| SHAP fallback appears             | The installed model/library combination may not support SHAP |
| Training is slow                  | Disable lightweight tuning or reduce dataset size            |
| XGBoost/LightGBM/SHAP unavailable | The application can skip unsupported optional components     |
| Port conflict                     | Run `streamlit run app.py --server.port 8502`                |

These behaviors are explicitly handled by the documented application design.

---

# 📊 What Makes This More Than a Basic ML Classifier?

A simple predictive-maintenance project might only implement:

```text
CSV
 ↓
Random Forest
 ↓
Failure / No Failure
```

This project expands the workflow into:

```text
                 ┌─────────────────────┐
                 │ Industrial Dataset  │
                 └──────────┬──────────┘
                            │
                ┌───────────▼───────────┐
                │ Data Intelligence      │
                └───────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    Failure ML        Anomaly ML        Optional ML
          │                 │                 │
          │                 │          ┌──────┴──────┐
          │                 │          ▼             ▼
          │                 │     Failure Type      RUL
          │                 │
          └────────────┬────┴──────────────────────┘
                       ▼
                Machine Health
                       │
                       ▼
                  Risk Level
                       │
                       ▼
              Maintenance Advice
                       │
                       ▼
                Explainability
                       │
                       ▼
               Historical Records
```

The project therefore combines **prediction, anomaly detection, health intelligence, explainability, and maintenance history** into one application rather than exposing a single ML model.

---

# ⚠️ Important Engineering Boundary

This application is a **maintenance decision-support system**.

It is **not** an autonomous industrial control system.

It does not:

* Directly control machinery
* Modify PLC settings
* Trigger physical actuators
* Replace safety interlocks
* Replace certified industrial safety systems
* Guarantee that a machine will not fail

Predictions should be validated against real machine conditions and appropriate engineering procedures before being used in operational environments. The documented design explicitly defines the system as decision support rather than autonomous machinery control.

---

# 💡 Example Intelligence Output

A machine prediction can conceptually produce:

```text
Machine ID
────────────────────────
MACHINE-001

Health Score
────────────────────────
82 / 100

Failure Probability
────────────────────────
14.7%

Anomaly Status
────────────────────────
NORMAL

Risk Level
────────────────────────
LOW

Failure Type
────────────────────────
Not currently indicated

Remaining Useful Life
────────────────────────
Optional — dataset dependent

Recommendation
────────────────────────
Continue monitoring and follow
the configured maintenance strategy.
```

> Values above are illustrative only. Actual results are generated from the trained model and supplied sensor data.

---

# 🎓 Engineering Concepts Demonstrated

This project demonstrates practical knowledge across:

### Machine Learning

* Binary classification
* Multiclass classification
* Regression
* Ensemble learning
* Gradient boosting
* Random forests
* Logistic regression
* Unsupervised anomaly detection

### ML Engineering

* Data preprocessing
* Feature engineering
* Model persistence
* Train/test evaluation
* Cross-validation
* Class imbalance handling
* Hyperparameter optimization
* Model comparison

### Explainable AI

* SHAP
* Feature importance
* Local prediction explanation
* Global model interpretation

### Data Engineering

* CSV ingestion
* Schema normalization
* Data validation
* Missing-data handling
* SQLite persistence

### Application Engineering

* Streamlit UI
* Modular Python architecture
* Error handling
* Persistent state
* Interactive visualization
* Downloadable reports

### Industrial Intelligence

* Predictive maintenance
* Machine health monitoring
* Failure-risk analysis
* Anomaly detection
* Maintenance decision support

---

# 📌 Project Positioning

### Category

**Industrial AI / Machine Learning / Predictive Maintenance**

### Primary Problem

> Predict machine failures early enough to support proactive maintenance decisions.

### Input

Industrial machine sensor measurements.

### Intelligence

Supervised ML + unsupervised anomaly detection + optional failure-type classification + optional RUL regression.

### Output

Machine health, failure probability, anomaly status, risk level, explanations, and maintenance recommendations.

### Interface

Interactive Streamlit application.

### Persistence

SQLite + Joblib model artifacts.

### Hardware Philosophy

CPU-first and lightweight enough for the documented 8 GB RAM environment.

---

# 🔮 Future Expansion Opportunities

The current system can serve as a foundation for more advanced industrial intelligence.

Potential future extensions include:

* Real-time IoT sensor ingestion
* Streaming machine telemetry
* Time-series-specific models
* Automated model monitoring
* Model drift detection
* Feature drift detection
* Alerting systems
* Maintenance cost optimization
* Failure-cost-aware threshold optimization
* Asset fleet management
* Role-based access control
* REST API integration
* Industrial protocol integration
* Edge deployment
* Cloud deployment
* Digital-twin integration
* Automated retraining pipelines

> These are **future extension opportunities**, not claims that they are currently implemented.

---

# 📜 Project Design Philosophy

The project follows four core principles:

### 01 — Data First

```text
No reliable data
      ↓
No reliable model
```

Dataset validation and preprocessing are therefore fundamental.

### 02 — Separate Signals

```text
Failure Probability
        ≠
Anomaly Score
        ≠
Health Score
        ≠
Risk Level
```

Each represents a different concept.

### 03 — Explain the Prediction

A prediction without an explanation can be difficult to trust in engineering workflows.

### 04 — Decision Support, Not Autonomous Control

The system assists maintenance teams rather than attempting to replace industrial safety or control infrastructure.

---

# 📈 Project Maturity

| Area                             | Implementation |
| -------------------------------- | -------------- |
| End-to-end ML workflow           | ✅              |
| Multiple ML algorithms           | ✅              |
| Ensemble learning                | ✅              |
| Cross-validation                 | ✅              |
| Imbalance handling               | ✅              |
| Optional hyperparameter tuning   | ✅              |
| Anomaly detection                | ✅              |
| Explainable AI                   | ✅              |
| Optional RUL                     | ✅              |
| Optional failure-type prediction | ✅              |
| Persistent model artifacts       | ✅              |
| Database persistence             | ✅              |
| Interactive UI                   | ✅              |
| Maintenance history              | ✅              |
| Graceful missing-state handling  | ✅              |
| CPU-first design                 | ✅              |

---

# 👨‍💻 Author

**Mohan R**

Computer Science Engineering
AI / Machine Learning / Agentic AI Developer

---

# ⭐ GitHub Repository

If you find this project useful:

```text
⭐ Star the repository
🍴 Fork the project
🐛 Report issues
💡 Suggest improvements
🔧 Contribute enhancements
```

---

# 📄 License

Add the repository's actual license here once selected.

Example:

```text
MIT License
```

> Do not claim a license in the repository until the corresponding `LICENSE` file has actually been added.

---

# ⚠️ Disclaimer

This project is intended for **educational, research, development, and decision-support purposes**.

Synthetic/demo datasets must not be treated as real industrial measurements. Predictions from machine-learning models should not be considered guarantees of equipment failure or safety.

For real industrial deployment, predictions should be validated against domain-specific engineering requirements, sensor quality, operational constraints, maintenance procedures, and applicable safety standards.

---

## 🏁 Final Summary

**Industrial Machine Failure Intelligence System** transforms structured industrial sensor data into actionable machine intelligence through:

```text
        SENSOR DATA
             │
             ▼
      DATA INTELLIGENCE
             │
             ▼
     ┌───────┴────────┐
     │                │
     ▼                ▼
 FAILURE PREDICTION  ANOMALY DETECTION
     │                │
     └───────┬────────┘
             ▼
       MACHINE HEALTH
             │
             ▼
          RISK LEVEL
             │
             ▼
      FAILURE TYPE / RUL
             │
             ▼
       EXPLAINABILITY
             │
             ▼
   MAINTENANCE DECISION SUPPORT
             │
             ▼
      HISTORICAL RECORDS
```

**The goal is not merely to predict failure.**

> **The goal is to turn machine sensor data into understandable, explainable, and maintenance-oriented intelligence.**

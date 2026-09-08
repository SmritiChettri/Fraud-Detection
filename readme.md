Here is the updated, production-ready `README.md`. It incorporates the **statistical drift detection** pipeline, the **automated GitHub Actions CI/CD** workflow, live deployment indicators, and the exact repository structure you built:

```markdown
# 💳 FraudGuard AI: Real-Time Fraud Detection & Scoring Engine

![CI Pipeline](https://github.com/your-username/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade, low-latency Machine Learning service designed to detect credit card fraud on live transaction streams and high-volume batch submissions. Built with **LightGBM**, served via **FastAPI**, monitored for data drift via **SciPy**, and containerized using **Docker**.

---
🔗 **Live Interactive Swagger API Docs:** [https://fraud-detection-app-ps46.onrender.com/docs](https://fraud-detection-app-ps46.onrender.com/docs)  
🩺 **Service Health Probe:** [https://fraud-detection-app-ps46.onrender.com/health](https://fraud-detection-app-ps46.onrender.com/health)

---

## 📌 Problem & Business Context

Financial fraud causes tens of billions in annual losses across online checkouts and banking rails. Detecting fraudulent transactions presents two engineering challenges:

1. **Extreme Class Imbalance:** Legitimate transactions account for over 99.8% of volume. Naive classification models predict the majority class and miss active threats.
2. **Strict Latency Demands:** Fraud engines must evaluate incoming transactions and return deterministic risk decisions (`APPROVE` vs. `BLOCK`) in under 20ms to avoid degrading checkout conversion.

This project implements an end-to-end MLOps pipeline covering imbalanced feature weighting, low-latency REST serving, statistical data drift auditing, automated CI testing, and containerized deployment.

---

## 🏗️ System Architecture

```text
       [Client CSV Upload]               [Payment Gateway / Real-Time Client]
               │                                            │
               ▼                                            ▼
      ┌─────────────────┐                          ┌─────────────────┐
      │  Streamlit UI   │                          │  External API   │
      │   (Port 8501)   │                          │    Consumer     │
      └────────┬────────┘                          └────────┬────────┘
               │                                            │
               └───────────────────────┬────────────────────┘
                                       │ (HTTP JSON)
                                       ▼
                       ┌──────────────────────────────┐
                       │      FastAPI App Engine      │
                       │         (Port 8000)          │
                       └──────────────┬───────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  [Pydantic Schema]         [In-Memory Model]
                  Strict Validation        LightGBM Classifier
                         │                         │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
            [Instant Decision / Risk]      [Background Audit]
             (APPROVE / REVIEW / BLOCK)            │
                                                   ▼
                                       [Statistical Drift Tests]
                                       (Two-Sample KS Test & Report)

```

---

## ⚡ Core Features

* **Cost-Sensitive Learning Core:** Trained on anonymized PCA signals using `scale_pos_weight` to strongly penalize false negatives (missed fraud).
* **Low-Latency REST API:** Built on **FastAPI** with **Pydantic** schema validation, achieving sub-20ms single-record inferences.
* **Automated Data Drift Monitoring:** Employs non-parametric **Two-Sample Kolmogorov-Smirnov (KS) tests** ($\alpha = 0.05$) to catch distribution shifts and generate audit reports.
* **Continuous Integration (CI):** Automated GitHub Actions pipeline that runs linting and unit test suites on every pull request and push.
* **Vectorized Batch Processing:** Dedicated endpoints optimized to process batched matrix records without per-request HTTP transport overhead.
* **Instant Client Risk Portal:** Interactive **Streamlit** control UI for real-time scenario simulation and file drag-and-drop scoring.
* **Production Docker Container:** Multi-stage, minimal Linux container ready for cloud environments (Render, AWS, GCP).

---

## 🛠️ Tech Stack

* **Language & Runtime:** Python 3.11+
* **Machine Learning:** LightGBM, Scikit-Learn, Pandas, NumPy
* **API Framework:** FastAPI, Uvicorn, Pydantic v2
* **Statistical Monitoring:** SciPy (Two-Sample KS Test)
* **Experiment Tracking:** MLflow
* **Containerization:** Docker
* **Testing & CI:** Pytest, HTTPX TestClient, GitHub Actions

---

## 📁 Repository Structure

```text
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated GitHub Actions CI workflow
├── app/
│   ├── main.py                 # FastAPI production server & prediction routes
│   ├── schemas.py              # Pydantic input/output contracts
│   └── client_app.py           # Optional Streamlit audit portal
├── data/                       # Kaggle creditcard.csv (git-ignored)
├── models/
│   └── fraud_model.joblib      # Serialized LightGBM binary artifact
├── monitoring/
│   ├── drift_monitor_scipy.py  # Statistical KS-test drift engine
│   └── reports/                # Generated drift visual reports (.html)
├── src/
│   ├── train.py                # MLflow training pipeline
│   └── generate_csv_batch.py   # Test fixture synthesis & sampling tool
├── tests/
│   └── test_api.py             # Pytest endpoint verification suite
├── Dockerfile                  # Multi-stage container recipe
├── requirements.txt            # Project dependencies
└── README.md

```

---

## 🚀 Quickstart Guide (Windows / PowerShell)

### 1. Clone & Set Up Environment

```powershell
git clone [https://github.com/your-username/fraud-detection-mlops.git](https://github.com/your-username/fraud-detection-mlops.git)
cd fraud-detection-mlops

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```

### 2. Train the Model Artifact

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into the `data/` folder, then execute:

```powershell
python src/train.py

```

*This balances classes using calculated class frequency weights, logs runs to MLflow, and exports the model to `models/fraud_model.joblib`.*

### 3. Run Automated Tests

Execute the test suite prior to launching services:

```powershell
pytest tests/ -v

```

### 4. Launch the API Service

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

* Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
* Service Health Probe: `http://127.0.0.1:8000/health`

---

## 📈 Statistical Drift Monitoring

Run the offline drift detector to evaluate feature distribution divergence between baseline training distributions and production inference traffic:

```powershell
python monitoring/drift_monitor_scipy.py

```

Open the generated visual HTML report in your browser:

```powershell
Start-Process "monitoring\reports\data_drift_report.html"

```

The report identifies statistically significant shifts ($p < 0.05$) across transaction velocity and amount metrics, flagging models requiring scheduled retraining.

---

## 🐳 Running with Docker

Build and run the self-contained production image:

```powershell
# Build Docker image
docker build -t fraud-detection-api:v1 .

# Run the container mapped to port 8000
docker run -d -p 8000:8000 --name fraud_engine fraud-detection-api:v1

```

Test the containerized endpoint with a payload:

```powershell
$body = @{
    Time = 406.0; Amount = 149.62
    V1 = -2.31; V2 = 1.95; V3 = -1.60; V4 = 3.99; V5 = -0.52
    V6 = -1.42; V7 = -2.53; V8 = 1.39; V9 = -2.77; V10 = -2.77
    V11 = 3.20; V12 = -2.89; V13 = -0.59; V14 = -4.28; V15 = 0.38
    V16 = -1.14; V17 = -2.83; V18 = -0.01; V19 = 0.41; V20 = 0.12
    V21 = 0.51; V22 = -0.03; V23 = -0.46; V24 = 0.32; V25 = 0.04
    V26 = 0.17; V27 = 0.26; V28 = -0.14
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"

```

---

## 📊 Evaluation Metrics

Because transaction data exhibits severe imbalance (approx. 0.17% fraud rate), accuracy is uninformative. The model is evaluated on threshold-agnostic and recall-weighted indicators:

| Metric | Target Value | Business Meaning |
| --- | --- | --- |
| **PR-AUC** | $\ge 0.85$ | Area under the Precision-Recall curve; minimizes false flags on rare events. |
| **ROC-AUC** | $\ge 0.95$ | Diagnostic separation power between classes across varying operational thresholds. |
| **Recall (Fraud)** | $\ge 0.80$ | Proportion of actual fraudulent attempts blocked by the automated engine. |
| **P95 Latency** | $< 20\text{ ms}$ | Real-time response ceiling required for integration into banking gateways. |

```

```
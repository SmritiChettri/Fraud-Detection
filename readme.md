
# 💳 FraudGuard AI: Real-Time Fraud Detection & Scoring Engine

A production-grade, low-latency Machine Learning service designed to detect credit card fraud on live transaction streams and high-volume batch submissions. Built with **LightGBM**, served via **FastAPI**, containerized using **Docker**, and monitored through an interactive **Streamlit** control portal.

---

## 📌 Problem & Business Context

Financial fraud causes tens of billions in annual losses across online checkouts and banking rails. Detecting fraudulent transactions presents two engineering challenges:

1. **Extreme Class Imbalance:** Legitimate transactions account for over 99.8% of volume. Naive classification leads to missed threats.
2. **Strict Latency Demands:** Fraud engines must evaluate transactions and return decisions (`APPROVE` vs. `BLOCK`) in under 20ms to avoid degrading customer checkout conversion.

This project implements an end-to-end MLOps pipeline to handle imbalanced feature learning, low-latency API serving, vectorized batch processing, and user-friendly visualization.

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
                                      ▼
                        ┌───────────────────────────┐
                        │   Instant Risk Scoring    │
                        │   Probability | Decision  │
                        └───────────────────────────┘

```

---

## ⚡ Core Features

* **Gradient Boosted Tree Core:** Trained on anonymized PCA signals using cost-sensitive learning (`scale_pos_weight`) to penalize missed fraud.
* **Low-Latency REST API:** Built on **FastAPI** with **Pydantic** input validation, achieving sub-15ms single-record inferences.
* **Vectorized Batch Processing:** Custom `/predict_batch` route processes thousands of transactions in a single vectorized matrix call without per-record HTTP overhead.
* **Instant Client Risk Portal:** Zero-click **Streamlit** UI that recalculates risk immediately upon file drag-and-drop or parameter adjustment.
* **Reproducible Test Fixture:** Standalone CLI tool to synthesize or sample live non-labeled batches for auditing.
* **Dockerized Packaging:** Standalone Linux containerization ready for cloud deployment (AWS, GCP, Render).

---

## 🛠️ Tech Stack

* **Language & Runtime:** Python 3.11+
* **Machine Learning:** LightGBM, Scikit-Learn, Pandas, NumPy
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Experiment Tracking:** MLflow
* **Client Frontend:** Streamlit
* **Containerization:** Docker & Linux Slim runtime
* **Testing:** Pytest & HTTPX TestClient

---

## 📁 Repository Structure

```text
fraud-detection-mlops/
├── data/                       # Kaggle creditcard.csv (git-ignored)
├── models/                     # Serialized LightGBM binary artifacts (.joblib)
├── src/
│   ├── train.py                # MLflow-wrapped training & evaluation script
│   └── generate_batch.py       # Batch data synthesis and sampling tool
├── app/
│   ├── main.py                 # FastAPI production server & batch routes
│   ├── schemas.py              # Pydantic schema validation contracts
│   └── client_app.py           # Streamlit instant-audit frontend portal
├── tests/
│   └── test_api.py             # Pytest endpoint verification suite
├── Dockerfile                  # Production container recipe
├── requirements.txt            # Locked project dependencies
└── README.md

```

---

## 🚀 Quickstart Guide (Local Windows / PowerShell)

### 1. Clone & Set Up Environment

```powershell
git clone https://github.com/your-username/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```

### 2. Prepare Data & Train the Model

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into the `data/` folder.
2. Run the training script:

```powershell
python src/train.py

```

*This balances classes, logs run metrics to MLflow, and exports the serialized model to `models/fraud_model.joblib`.*

### 3. Launch the Services

Run the API engine in **Terminal 1**:

```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

Run the Client Dashboard in **Terminal 2**:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app/client_app.py

```

* The Client UI is available at: `http://localhost:8501`
* Interactive API Documentation (Swagger) is available at: `http://localhost:8000/docs`

---

## 🧪 Testing with Transaction Batches

You can generate realistic testing files anytime using the included generator:

```powershell
# Extract real sample rows from the dataset
python src/generate_csv_batch.py --count 25 --output test_batch.csv

```

Open `http://localhost:8501`, switch to **"Instant CSV Upload & Score"**, and drop your generated `.csv` into the dashboard. All rows are scored instantly without clicking any buttons, highlighting anomalous transactions in red.

---

## 🐳 Running with Docker

Run the entire prediction backend inside an isolated container:

```powershell
# Build Docker image
docker build -t fraud-detection-api:v1 .

# Run the container on port 8000
docker run -d -p 8000:8000 --name fraud_engine fraud-detection-api:v1

```

Verify service health:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"

```

---

## 📊 Evaluation & Metrics Strategy

Because fraud detection involves severe class imbalance, standard accuracy is misleading. The model is evaluated on:

* **PR-AUC (Precision-Recall Area Under Curve):** Optimizes performance on rare positive instances.
* **F1-Score / Recall Prioritization:** Configured with a probability threshold to catch true fraud attempts while minimizing unnecessary card freezes.
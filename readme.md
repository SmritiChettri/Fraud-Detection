
# 💳 FraudGuard AI: Real-Time Fraud Detection Engine

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Render Deployment](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?logo=render&logoColor=white)

Hi! Welcome to **FraudGuard AI**. This is an end-to-end machine learning project I built to detect credit card fraud on live transaction streams. 

I didn't just want to train a model in a Jupyter Notebook and call it a day. I wanted to build this like a real production system—handling imbalanced data, serving predictions via a low-latency API, monitoring for data drift, and packaging it all up in Docker.

🔗 **Try the Live API:** [Interactive Swagger Docs](https://fraud-detection-app-ps46.onrender.com/docs)  
🩺 **Service Health Status:** [Health Endpoint](https://fraud-detection-app-ps46.onrender.com/health)

---

## 📌 The Problem (and Why It's Hard)

Catching financial fraud in real-time presents two massive engineering headaches:
1. **The Needle in a Haystack:** Legitimate transactions make up over 99.8% of the data. If a model just guesses "not fraud" every time, it gets 99.8% accuracy—but misses every single thief. 
2. **The Need for Speed:** If a fraud check takes too long, the customer abandons their checkout. The engine needs to process the data and return a strict `APPROVE` or `BLOCK` decision in under 20 milliseconds.

This project tackles both problems head-on using cost-sensitive learning and a highly optimized API layer.

---

## 🏗️ How It Works Under the Hood

Here is a high-level look at the system architecture:

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
                                       │ (HTTP JSON payload)
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

### ⚡ What Makes This Production-Ready?

* **Cost-Sensitive Core:** The LightGBM model is tuned specifically to penalize false negatives (missed fraud) heavily using `scale_pos_weight`.
* **FastAPI + Pydantic:** Incoming data is strictly validated. The model itself stays loaded in memory, allowing us to hit single-record inferences in under 15ms.
* **Automated Drift Monitoring:** Data changes over time. I wrote a background script using **SciPy** to run Two-Sample Kolmogorov-Smirnov (KS) tests to automatically detect if live transaction distributions are drifting away from our training baseline.
* **CI/CD Pipeline:** Every push to this repo runs through a GitHub Actions pipeline to verify unit tests and code health.

---

## 🚀 Quickstart: Run It Locally

Want to test it out on your own machine? It's fully containerized, but you can also run it natively on Windows/Linux/Mac.

### 1. Clone & Set Up

```bash
git clone [https://github.com/your-username/fraud-detection-mlops.git](https://github.com/your-username/fraud-detection-mlops.git)
cd fraud-detection-mlops

# Create virtual environment and install dependencies
python -m venv .venv
# On Windows: .venv\Scripts\Activate.ps1
# On Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

```

### 2. Train the Model

Grab the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and drop it in the `data/` folder.

```bash
python src/train.py

```

*This calculates the exact class weights, logs the experiment to MLflow, and saves the optimized model binary to `models/fraud_model.joblib`.*

### 3. Launch the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

You can now open `http://127.0.0.1:8000/docs` in your browser to interact with the API!

---

## 🌐 Test the Live Cloud API

If you don't want to install anything, you can hit my live Render deployment using PowerShell right now:

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

Invoke-RestMethod -Uri "[https://fraud-detection-app-ps46.onrender.com/predict](https://fraud-detection-app-ps46.onrender.com/predict)" -Method Post -Body $body -ContentType "application/json"

```

---

## 📈 Checking for Data Drift

To generate a statistical report comparing production traffic against our baseline data, just run:

```bash
python monitoring/drift_monitor_scipy.py

```

This drops an HTML report in `monitoring/reports/` showing exactly which features are drifting and their corresponding p-values.

---

## 🐳 Running with Docker

Prefer Docker? The `Dockerfile` uses a multi-stage slim build to keep things lightweight.

```bash
docker build -t fraud-detection-api:v1 .
docker run -d -p 8000:8000 --name fraud_engine fraud-detection-api:v1

```

---

## 📊 Evaluation & Metrics Strategy

Because fraud is so rare, looking at standard accuracy is useless. Instead, I measure success via:

* **PR-AUC (Precision-Recall Area Under Curve):** My primary metric. It strictly evaluates how well we find needles in the haystack without flagging everything as a needle.
* **ROC-AUC:** Measures the model's overall diagnostic ability to separate legitimate buyers from fraudsters.
* **Latency:** We hit a P95 latency of `< 20ms` to ensure checkout gateways aren't slowed down.

---

*Built with ❤️ (and a lot of coffee) using Python, LightGBM, and FastAPI.*

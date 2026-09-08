import os
import joblib
import pandas as pd
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import TransactionPayload, PredictionResponse

model = None
MODEL_PATH = os.path.join("models", "fraud_model.joblib")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(
    title="Real-Time Fraud Detection Engine",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Visit /docs to use the API."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is uninitialized.")

    data_dict = payload.model_dump()
    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame([data_dict])[ordered_cols]

    prob = float(model.predict_proba(input_df)[0][1])
    is_fraud = prob >= 0.50

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        decision="BLOCK" if is_fraud else "APPROVE",
        risk_level="HIGH" if prob > 0.75 else "MEDIUM" if prob > 0.35 else "LOW"
    )

@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(payloads: List[TransactionPayload]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is uninitialized.")

    data = [item.model_dump() for item in payloads]
    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame(data)[ordered_cols]

    probabilities = model.predict_proba(input_df)[:, 1]

    responses = []
    for prob in probabilities:
        prob_val = float(prob)
        is_fraud = prob_val >= 0.50
        responses.append(
            PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=round(prob_val, 4),
                decision="BLOCK" if is_fraud else "APPROVE",
                risk_level="HIGH" if prob_val > 0.75 else "MEDIUM" if prob_val > 0.35 else "LOW"
            )
        )
    return responses

@app.get("/")
def root():
    return {
        "service": "FraudGuard AI Scoring Engine",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "prediction": "/predict"
        },
        "model": "LightGBM Classifier (Cost-Sensitive Imbalanced Tuning)"
    }
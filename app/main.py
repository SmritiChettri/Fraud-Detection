# API Server

import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from api.schemas import TransactionPayLoad, PredictionResponse

model = None
MODEL_PATH = os.path.join("models", "fraud_model.joblib")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH} path. Run training first.")
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(
    title="Real-time fraud detection engine",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return{"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionPayLoad):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is ininitialized.")

    data_dict = pay;oad.model_dump()
    ordered_cols = ["Time"] +[f"V{i}" for i in range(1, 29)] + ["Amount"]
    imput_sf = pd.DataFrame([dat_dict])[ordered_cols]

    prob = float(model.predict_proba(input_df)[0][1])
    is_fraud = prob >=0.50

    return predictionResponse(
        is_fraud=is_fraud,
        fraud_probability=rround(prob, 4),
        decision="BLOCK" if is_fraud else "APPROVE",
        risk_level="HIGH" if prob > 0.75 else "MEDIUM" if prob > 0.35 else "LOW"

    )

import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import TelemetryInput, PredictionResponse, BatchTelemetryInput, BatchPredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "battery_rul_model.pkl")
model_artifact = {}

# order: discharge_time, decrement_v, max_v_discharge, min_v_charge, time_415v, time_cc, charging_time.
SCALER_MEAN = np.array([4286.82997338656, 1426.7389480091622, 3.907725049900199, 3.5776343147039253,
                        3686.362761558976, 5317.307093313373, 9945.013322521623])
SCALER_SCALE = np.array([31770.769005050388, 13748.844772725866, 0.08994201069210925, 0.12195047868558291,
                         8844.1084837652, 24277.775779047337, 25727.041894125614])

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model artifact not found.")
    model_artifact["model"] = joblib.load(MODEL_PATH)
    yield
    model_artifact.clear()

app = FastAPI(
    title="Battery RUL Prediction",
    description="Microservice for Battery RUL estimation using ML.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in model_artifact
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_rul(payload: TelemetryInput):
    try:
        features = np.array([[
            payload.discharge_time,
            payload.decrement_v,
            payload.max_v_discharge,
            payload.min_v_charge,
            payload.time_415v,
            payload.time_cc,
            payload.charging_time
        ]])

        model = model_artifact["model"]
        prediction = model.predict((features - SCALER_MEAN) / SCALER_SCALE)

        return PredictionResponse(
            predicted_rul=float(prediction[0])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
    
@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch_rul(payload: BatchTelemetryInput):
    try:
        features = np.array([
            [
                item.discharge_time,
                item.decrement_v,
                item.max_v_discharge,
                item.min_v_charge,
                item.time_415v,
                item.time_cc,
                item.charging_time
            ]
            for item in payload.batch
        ])

        model = model_artifact["model"]
        predictions = model.predict((features - SCALER_MEAN) / SCALER_SCALE)

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            total_samples=len(payload.batch)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch Inference error: {str(e)}"
        )
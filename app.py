from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Deep Core model import
from models.deep_core import DeepCoreModel

# =========================
# App Init
# =========================

app = FastAPI(
    title="PRIZUX Deep Core",
    description="Numerical relationship inference engine",
    version="1.0.0"
)

# Global model instance
model = DeepCoreModel(model_path="weights.json")

# =========================
# Schemas
# =========================

class TrainData(BaseModel):
    x: List[float]
    y: List[float]

class PredictData(BaseModel):
    x: float


# =========================
# Routes
# =========================

@app.get("/")
def root():
    """
    Health check & model status
    """
    return {
        "service": "PRIZUX Deep Core",
        "status": "alive",
        "trained": model.trained
    }


@app.post("/api/train")
def train(data: TrainData):
    """
    Train Deep Core model with (x, y) pairs
    """
    try:
        model.train(data.x, data.y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "trained": True,
        "weight": model.w,
        "bias": model.b,
        "points": len(data.x)
    }


@app.post("/api/predict")
def predict(data: PredictData):
    """
    Predict y value from input x
    """
    try:
        y = model.predict(data.x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "input": data.x,
        "prediction": y
    }


@app.get("/api/model")
def model_info():
    """
    Introspection endpoint (for Education model)
    """
    return model.parameters()
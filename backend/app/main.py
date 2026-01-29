from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from backend.models.deepcore import DeepCoreModel

app = FastAPI(title="PRIZUX Deep Core")

# 간단 버전: 세션별 모델
MODELS: Dict[str, DeepCoreModel] = {}

class TrainRequest(BaseModel):
    session_id: str
    X: List[List[float]]
    Y: List[float]
    epochs: int = 100

class PredictRequest(BaseModel):
    session_id: str
    x: List[float]

@app.post("/api/train")
def train(req: TrainRequest):
    model = MODELS.get(req.session_id)
    if model is None:
        model = DeepCoreModel()
        MODELS[req.session_id] = model

    info = model.train(req.X, req.Y, req.epochs)

    return {
        "status": "trained",
        "loss_history": model.loss_history,
        "trajectory": model.trajectory,
        **info
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    model = MODELS.get(req.session_id)
    if not model or not model.trained:
        raise HTTPException(400, "Model not trained")

    y = model.predict(req.x)

    return {
        "prediction": y,
        "weights": model.w,
        "bias": model.b
    }

@app.get("/api/status/{session_id}")
def status(session_id: str):
    model = MODELS.get(session_id)
    if not model:
        return {"exists": False}
    return {
        "trained": model.trained,
        "loss": model.loss_history
    }

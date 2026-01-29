from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from backend.models.deepcore import DeepCoreModel

app = FastAPI(title="PRIZUX Deep Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = DeepCoreModel()

class TrainRequest(BaseModel):
    X: List[List[float]]
    Y: List[float]

class PredictRequest(BaseModel):
    X: List[float]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/train")
def train(req: TrainRequest):
    if len(req.X) != len(req.Y):
        return {"error": "X row count must match Y count"}

    result = model.train(req.X, req.Y)
    return result

@app.post("/api/predict")
def predict(req: PredictRequest):
    y = model.predict(req.X)
    return {
        "prediction": y,
        "w": model.w,
        "b": model.b
    }

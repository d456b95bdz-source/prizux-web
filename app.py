from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from deep_core_model import DeepCoreService

app = FastAPI()
core = DeepCoreService(degree=2)

# ===== 요청 스키마 =====
class TrainRequest(BaseModel):
    x: List[float]
    y: List[float]
    x_type: str
    x_unit: str
    y_type: str
    y_unit: str


class PredictRequest(BaseModel):
    x_value: float
    x_type: str
    x_unit: str
    y_type: str
    y_unit: str


# ===== Train =====
@app.post("/train")
def train(req: TrainRequest):
    core.train(
        x_vals=req.x,
        y_vals=req.y,
        x_type=req.x_type,
        x_unit=req.x_unit,
        y_type=req.y_type,
        y_unit=req.y_unit
    )
    return {"status": "trained"}


# ===== Predict =====
@app.post("/predict")
def predict(req: PredictRequest):
    result = core.predict(
        x_value=req.x_value,
        x_type=req.x_type,
        x_unit=req.x_unit,
        y_type=req.y_type,
        y_unit=req.y_unit
    )
    return result

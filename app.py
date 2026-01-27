from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import os

app = FastAPI()

# =========================
# Deep Core Model
# =========================

MODEL_PATH = "weights.json"

class DeepCoreModel:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0
        self.trained = False

        if os.path.exists(MODEL_PATH):
            self.load()

    def train(self, x: List[float], y: List[float]):
        n = len(x)
        if n == 0 or n != len(y):
            raise ValueError("Invalid training data")

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((x[i] - mean_x) ** 2 for i in range(n))

        self.w = num / den if den != 0 else 0.0
        self.b = mean_y - self.w * mean_x
        self.trained = True

        self.save()

    def predict(self, x: float) -> float:
        if not self.trained:
            raise RuntimeError("Model not trained yet")
        return self.w * x + self.b

    def save(self):
        with open(MODEL_PATH, "w") as f:
            json.dump(
                {
                    "w": self.w,
                    "b": self.b
                },
                f
            )

    def load(self):
        with open(MODEL_PATH, "r") as f:
            data = json.load(f)
            self.w = data["w"]
            self.b = data["b"]
            self.trained = True


model = DeepCoreModel()

# =========================
# API Schemas
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
    return {
        "service": "PRIZUX Deep Core",
        "status": "alive",
        "trained": model.trained
    }


@app.post("/api/train")
def train(data: TrainData):
    model.train(data.x, data.y)
    return {
        "trained": True,
        "weight": model.w,
        "bias": model.b
    }


@app.post("/api/predict")
def predict(data: PredictData):
    y = model.predict(data.x)
    return {
        "input": data.x,
        "prediction": y
    }
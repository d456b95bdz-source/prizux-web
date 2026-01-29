from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from models.deepcore import DeepCoreModel

app = FastAPI(title="Prizux API")

model = DeepCoreModel()

class TrainRequest(BaseModel):
    x: List[float]
    y: float
    epochs: int = 100
    lr: float = 0.1

@app.get("/")
def root():
    return {"status": "alive"}

@app.post("/train")
def train(req: TrainRequest):
    loss_history = model.train(
        xs=req.x,
        y=req.y,
        epochs=req.epochs,
        lr=req.lr
    )

    surface = model.loss_surface(sum(req.x) / len(req.x))

    return {
        "status": "trained",
        "best_epoch": model.best_epoch,
        "best_loss": model.best_loss,
        "loss_history": loss_history,
        "loss_surface": surface
    }

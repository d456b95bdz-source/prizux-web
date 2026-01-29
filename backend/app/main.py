from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid, time, json
import numpy as np

from backend.models.deepcore import DeepCoreModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

SESSIONS = {}

class TrainRequest(BaseModel):
    session_id: str
    X: list
    y: list
    epochs: int = 50
    lr: float = 0.1

@app.post("/train")
def train(req: TrainRequest):
    X = np.array(req.X, dtype=float)
    y = np.array(req.y, dtype=float)

    model = DeepCoreModel(X.shape[1], req.lr)
    SESSIONS[req.session_id] = {
        "model": model,
        "X": X,
        "y": y,
        "epochs": req.epochs
    }
    return {"status": "started"}

@app.get("/train-stream")
def stream(session_id: str):
    def gen():
        s = SESSIONS[session_id]
        model = s["model"]
        X, y = s["X"], s["y"]

        for epoch in range(s["epochs"]):
            loss = model.train_step(X, y)

            payload = {
                "epoch": epoch + 1,
                "loss": loss,
                "weights": model.w.tolist(),
                "bias": model.b,
                "surface": model.loss_surface(X, y)
            }

            yield f"data:{json.dumps(payload)}\n\n"
            time.sleep(0.1)

        yield f"data:{json.dumps({'done': True})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

from fastapi import APIRouter, WebSocket
from models.deepcore import DeepCoreModel
import asyncio

router = APIRouter()

# 유저별 분리
MODELS = {}
DATASETS = {}

@router.post("/train")
async def start_training(payload: dict):
    user = payload.get("user", "anon")
    data = payload["data"]
    epochs = payload.get("epochs", 100)

    dataset = [(d["x"], d["y"]) for d in data]

    model = DeepCoreModel(len(dataset[0][0]))
    MODELS[user] = model
    DATASETS[user] = {
        "dataset": dataset,
        "epochs": epochs
    }

    return {
        "status": "training_started",
        "user": user,
        "epochs": epochs
    }


@router.websocket("/ws/train/{user}")
async def train_ws(ws: WebSocket, user: str):
    await ws.accept()

    model = MODELS[user]
    dataset = DATASETS[user]["dataset"]
    epochs = DATASETS[user]["epochs"]

    for epoch in range(1, epochs + 1):
        loss = model.train_epoch(dataset)

        await ws.send_json({
            "type": "epoch",
            "epoch": epoch,
            "loss": loss,
            "weights": model.w.tolist(),
            "bias": model.b
        })

        await asyncio.sleep(0.08)

    surface = model.loss_surface(dataset)

    await ws.send_json({
        "type": "complete",
        "trajectory": model.trajectory,
        "surface": surface
    })

    await ws.close()

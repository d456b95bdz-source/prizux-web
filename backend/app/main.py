from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="PRIZUX DeepCore",
    description="Real-time DeepCore training API with 3D loss surface",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "prizux-deepcore",
        "features": [
            "multi-variable",
            "adagrad",
            "epoch-streaming",
            "weight-bias-trajectory",
            "3d-loss-surface"
        ]
    }

import json
import os
from typing import List


class DeepCoreModel:
    """
    PRIZUX Deep Core
    - Single-variable regression model
    - Learns relationship between x and y
    - Stores learned parameters (w, b)
    """

    def __init__(self, model_path: str = "weights.json"):
        self.model_path = model_path
        self.w = 0.0
        self.b = 0.0
        self.trained = False

        if os.path.exists(self.model_path):
            self.load()

    # =========================
    # Training
    # =========================
    def train(self, x: List[float], y: List[float]):
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("x and y must have the same non-zero length")

        n = len(x)

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum(
            (x[i] - mean_x) * (y[i] - mean_y) for i in range(n)
        )
        denominator = sum(
            (x[i] - mean_x) ** 2 for i in range(n)
        )

        if denominator == 0:
            raise ValueError("Cannot train model: zero variance in x")

        self.w = numerator / denominator
        self.b = mean_y - self.w * mean_x
        self.trained = True

        self.save()

    # =========================
    # Prediction
    # =========================
    def predict(self, x: float) -> float:
        if not self.trained:
            raise RuntimeError("Model is not trained")
        return self.w * x + self.b

    # =========================
    # Persistence
    # =========================
    def save(self):
        with open(self.model_path, "w") as f:
            json.dump(
                {
                    "w": self.w,
                    "b": self.b
                },
                f,
                indent=2
            )

    def load(self):
        with open(self.model_path, "r") as f:
            data = json.load(f)
            self.w = float(data.get("w", 0.0))
            self.b = float(data.get("b", 0.0))
            self.trained = True

    # =========================
    # Introspection (Education)
    # =========================
    def parameters(self):
        return {
            "weight": self.w,
            "bias": self.b,
            "trained": self.trained
        }

import math
from typing import List, Dict

class DeepCoreModel:
    def __init__(self):
        self.w: List[float] = []
        self.b: float = 0.0
        self.trained = False

        # education용 기록
        self.loss_history = []
        self.trajectory = []

        # Adagrad
        self.gw = []
        self.gb = 0.0
        self.lr = 0.1
        self.eps = 1e-8

    def _init_params(self, dim: int):
        self.w = [0.0] * dim
        self.gw = [0.0] * dim
        self.b = 0.0
        self.gb = 0.0

    def predict(self, x: List[float]) -> float:
        return sum(self.w[i] * x[i] for i in range(len(x))) + self.b

    def train(
        self,
        X: List[List[float]],
        Y: List[float],
        epochs: int = 100
    ) -> Dict:

        if len(X) == 0 or len(X) != len(Y):
            raise ValueError("X rows and Y length must match")

        dim = len(X[0])
        for row in X:
            if len(row) != dim:
                raise ValueError("All X rows must have same length")

        self._init_params(dim)
        n = len(X)

        for epoch in range(epochs):
            dw = [0.0] * dim
            db = 0.0
            loss = 0.0

            for i in range(n):
                y_pred = self.predict(X[i])
                err = y_pred - Y[i]
                loss += err ** 2

                for j in range(dim):
                    dw[j] += err * X[i][j]
                db += err

            loss /= n
            self.loss_history.append(loss)

            for j in range(dim):
                self.gw[j] += dw[j] ** 2
                self.w[j] -= (self.lr / math.sqrt(self.gw[j] + self.eps)) * dw[j]

            self.gb += db ** 2
            self.b -= (self.lr / math.sqrt(self.gb + self.eps)) * db

            self.trajectory.append({
                "epoch": epoch,
                "w": self.w.copy(),
                "b": self.b,
                "loss": loss
            })

        self.trained = True

        return {
            "epochs": epochs,
            "final_loss": self.loss_history[-1]
        }

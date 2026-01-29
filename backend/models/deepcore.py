import math
from typing import List, Dict

class DeepCoreModel:
    def __init__(self, lr=0.1, epochs=30):
        self.w = 0.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

        self.loss_history = []
        self.trajectory = []

    def predict(self, x_vec: List[float]) -> float:
        return sum(x_vec) * self.w + self.b

    def loss(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

    def train(self, X: List[List[float]], Y: List[float]):
        n = len(X)

        for epoch in range(self.epochs):
            dw, db, total_loss = 0.0, 0.0, 0.0

            for x, y in zip(X, Y):
                y_pred = self.predict(x)
                l = self.loss(y_pred, y)
                total_loss += l

                dw += 2 * (y_pred - y) * sum(x)
                db += 2 * (y_pred - y)

            dw /= n
            db /= n
            total_loss /= n

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.loss_history.append(total_loss)
            self.trajectory.append({
                "epoch": epoch,
                "w": self.w,
                "b": self.b,
                "loss": total_loss
            })

        return {
            "w": self.w,
            "b": self.b,
            "loss_history": self.loss_history,
            "trajectory": self.trajectory
        }

import math
import random

class DeepCoreModel:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.eps = 1e-8

    def train(self, X, y, epochs=50):
        n_samples = len(X)
        n_features = len(X[0])

        w = [0.0] * n_features
        b = 0.0

        gw2 = [0.0] * n_features
        gb2 = 0.0

        history = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                pred = sum(w[j] * X[i][j] for j in range(n_features)) + b
                err = pred - y[i]
                total_loss += err ** 2

                for j in range(n_features):
                    dw[j] += err * X[i][j]
                db += err

            loss = total_loss / n_samples

            for j in range(n_features):
                gw2[j] += dw[j] ** 2
                w[j] -= self.lr * dw[j] / math.sqrt(gw2[j] + self.eps)

            gb2 += db ** 2
            b -= self.lr * db / math.sqrt(gb2 + self.eps)

            history.append({
                "epoch": epoch,
                "loss": loss,
                "weights": w[:],
                "bias": b
            })

        return history

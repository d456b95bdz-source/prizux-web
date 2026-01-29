import numpy as np
import math

class DeepCoreModel:
    def __init__(self, input_dim, lr=0.1):
        self.input_dim = input_dim
        self.lr = lr

        self.w = np.random.uniform(-1, 1, input_dim)
        self.b = np.random.uniform(-1, 1)

        # Adagrad 누적
        self.gw = np.zeros(input_dim)
        self.gb = 0.0

        # 기록
        self.loss_history = []
        self.trajectory = []   # (weights, bias, loss)

    def predict(self, x):
        return float(np.dot(self.w, x) + self.b)

    def loss(self, y_hat, y):
        return (y_hat - y) ** 2

    def train_epoch(self, dataset):
        total_loss = 0.0

        for x, y in dataset:
            x = np.array(x)
            y_hat = self.predict(x)
            err = y_hat - y

            grad_w = 2 * err * x
            grad_b = 2 * err

            self.gw += grad_w ** 2
            self.gb += grad_b ** 2

            self.w -= self.lr * grad_w / (np.sqrt(self.gw) + 1e-8)
            self.b -= self.lr * grad_b / (math.sqrt(self.gb) + 1e-8)

            total_loss += self.loss(y_hat, y)

        avg_loss = total_loss / len(dataset)

        self.loss_history.append(avg_loss)
        self.trajectory.append({
            "weights": self.w.tolist(),
            "bias": self.b,
            "loss": avg_loss
        })

        return avg_loss

    # 🔹 3D Loss Surface (w0, b 기준)
    def loss_surface(self, dataset, steps=40, span=2.0):
        w0 = self.w[0]
        b0 = self.b

        W = np.linspace(w0 - span, w0 + span, steps)
        B = np.linspace(b0 - span, b0 + span, steps)

        surface = []

        for wi in W:
            row = []
            for bi in B:
                l = 0.0
                for x, y in dataset:
                    pred = wi * x[0] + bi
                    l += (pred - y) ** 2
                row.append(l / len(dataset))
            surface.append(row)

        return {
            "w": W.tolist(),
            "b": B.tolist(),
            "loss": surface
        }

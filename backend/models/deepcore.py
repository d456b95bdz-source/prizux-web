import numpy as np

class DeepCoreModel:
    def __init__(self, input_dim, lr=0.1):
        self.w = np.random.randn(input_dim)
        self.b = np.random.randn()
        self.lr = lr

        self.Gw = np.zeros_like(self.w)
        self.Gb = 0.0

        self.history = []

    def predict(self, X):
        return X @ self.w + self.b

    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def train_step(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss(y_pred, y)

        grad_w = 2 * np.mean((y_pred - y)[:, None] * X, axis=0)
        grad_b = 2 * np.mean(y_pred - y)

        # Adagrad
        self.Gw += grad_w ** 2
        self.Gb += grad_b ** 2

        self.w -= self.lr * grad_w / (np.sqrt(self.Gw) + 1e-8)
        self.b -= self.lr * grad_b / (np.sqrt(self.Gb) + 1e-8)

        self.history.append({
            "loss": float(loss),
            "w": self.w.tolist(),
            "b": float(self.b)
        })

        return loss

    def loss_surface(self, X, y, grid=20):
        w_range = np.linspace(self.w[0] - 2, self.w[0] + 2, grid)
        b_range = np.linspace(self.b - 2, self.b + 2, grid)

        Z = []
        for w in w_range:
            row = []
            for b in b_range:
                y_pred = X[:, 0] * w + b
                row.append(float(self.loss(y_pred, y)))
            Z.append(row)

        return {
            "w": w_range.tolist(),
            "b": b_range.tolist(),
            "loss": Z
        }

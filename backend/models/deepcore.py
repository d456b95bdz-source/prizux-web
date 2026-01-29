import numpy as np

class DeepCoreModel:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

        # Adagrad
        self.gw = 0.0
        self.gb = 0.0
        self.eps = 1e-8

        self.best_w = self.w
        self.best_b = self.b
        self.best_loss = float("inf")
        self.best_epoch = 0

    def predict(self, x):
        return self.w * x + self.b

    def loss(self, y_pred, y):
        return (y_pred - y) ** 2

    def train_step(self, x, y, lr):
        y_pred = self.predict(x)
        loss = self.loss(y_pred, y)

        dw = 2 * (y_pred - y) * x
        db = 2 * (y_pred - y)

        self.gw += dw ** 2
        self.gb += db ** 2

        self.w -= lr * dw / (np.sqrt(self.gw) + self.eps)
        self.b -= lr * db / (np.sqrt(self.gb) + self.eps)

        return loss

    def train(self, xs, y, epochs, lr):
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in xs:
                epoch_loss += self.train_step(x, y, lr)

            epoch_loss /= len(xs)
            loss_history.append(epoch_loss)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                self.best_w = self.w
                self.best_b = self.b

        return loss_history

    def loss_surface(self, x_mean):
        surface = []
        w_range = np.linspace(self.best_w - 1, self.best_w + 1, 20)
        b_range = np.linspace(self.best_b - 1, self.best_b + 1, 20)

        for w in w_range:
            for b in b_range:
                y_pred = w * x_mean + b
                loss = (y_pred - self.best_loss) ** 2
                surface.append({
                    "weight": float(w),
                    "bias": float(b),
                    "loss": float(loss)
                })

        return surface

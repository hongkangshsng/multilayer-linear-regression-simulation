"""Multi-layer fully connected network for a linear-regression simulation.

Run with: python main.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 1101
OUTPUT_DIR = Path("outputs")


def make_dataset(samples: int = 240, noise_std: float = 2.2, seed: int = SEED):
    """Create y = 2.8x + 1.2 plus Gaussian measurement noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.0, 10.0, size=(samples, 1))
    y = 2.8 * x + 1.2 + rng.normal(0.0, noise_std, size=(samples, 1))
    order = np.argsort(x[:, 0])
    return x[order], y[order]


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


@dataclass
class LinearRegressionGD:
    """One fully connected layer: y_hat = xW + b."""

    learning_rate: float = 0.02
    seed: int = SEED

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.15, size=(1, 1))
        self.b = np.zeros((1, 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        error = prediction - y
        n = len(x)
        grad_w = (2.0 / n) * x.T @ error
        grad_b = (2.0 / n) * np.sum(error, axis=0, keepdims=True)
        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b
        return mse(prediction, y)


class DeepLinearRegressor:
    """Three linear Dense layers, trained explicitly with back-propagation.

    No activation function is used between layers, so the overall function is
    still linear. Hidden dimensions make the matrix operations visible.
    """

    def __init__(self, hidden_1: int = 8, hidden_2: int = 4, learning_rate: float = 0.0005, seed: int = SEED):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.18, size=(1, hidden_1))
        self.b1 = np.zeros((1, hidden_1))
        self.w2 = rng.normal(0.0, 0.18, size=(hidden_1, hidden_2))
        self.b2 = np.zeros((1, hidden_2))
        self.w3 = rng.normal(0.0, 0.18, size=(hidden_2, 1))
        self.b3 = np.zeros((1, 1))

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        z2 = z1 @ self.w2 + self.b2
        prediction = z2 @ self.w3 + self.b3
        return z1, z2, prediction

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[-1]

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        z1, z2, prediction = self.forward(x)
        n = len(x)
        d_prediction = (2.0 / n) * (prediction - y)

        # Back-propagation through layer 3, then 2, then 1.
        grad_w3 = z2.T @ d_prediction
        grad_b3 = np.sum(d_prediction, axis=0, keepdims=True)
        d_z2 = d_prediction @ self.w3.T
        grad_w2 = z1.T @ d_z2
        grad_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_z1 = d_z2 @ self.w2.T
        grad_w1 = x.T @ d_z1
        grad_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.w3 -= self.learning_rate * grad_w3
        self.b3 -= self.learning_rate * grad_b3
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        return mse(prediction, y)


def fit(model, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        indices = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = indices[start : start + batch_size]
            model.train_step(x[batch], y[batch])
        history.append(mse(model.predict(x), y))
    return history


def save_figures(x_train, y_train, x_test, y_test, baseline, deep, baseline_loss, deep_loss):
    OUTPUT_DIR.mkdir(exist_ok=True)
    x_line = np.linspace(-10, 10, 200).reshape(-1, 1)
    target_line = 2.8 * x_line + 1.2

    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_train, y_train, s=18, alpha=0.65, label="Training samples")
    plt.scatter(x_test, y_test, s=24, alpha=0.85, label="Test samples")
    plt.plot(x_line, target_line, "black", linewidth=2, label="Target line: y = 2.8x + 1.2")
    plt.plot(x_line, baseline.predict(x_line), "--", linewidth=2, label="One-layer regression")
    plt.plot(x_line, deep.predict(x_line), linewidth=2, label="3-layer deep linear network")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression: baseline vs. multi-layer network")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_fit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(baseline_loss, label="One-layer regression")
    plt.semilogy(deep_loss, label="3-layer deep linear network")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE (log scale)")
    plt.title("Training convergence")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=180)
    plt.close()


def main():
    x, y = make_dataset()
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    baseline = LinearRegressionGD()
    deep = DeepLinearRegressor()
    baseline_loss = fit(baseline, x_train, y_train, epochs=320, batch_size=24, seed=SEED)
    deep_loss = fit(deep, x_train, y_train, epochs=1800, batch_size=24, seed=SEED + 1)

    rows = [
        ("Linear regression baseline", mse(baseline.predict(x_train), y_train), mse(baseline.predict(x_test), y_test)),
        ("3-layer deep linear network", mse(deep.predict(x_train), y_train), mse(deep.predict(x_test), y_test)),
    ]
    print("Model                         Train MSE       Test MSE")
    print("-" * 54)
    for name, train_mse, test_mse in rows:
        print(f"{name:<30} {train_mse:>10.4f}      {test_mse:>10.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "train_mse", "test_mse"])
        writer.writerows(rows)

    save_figures(x_train, y_train, x_test, y_test, baseline, deep, baseline_loss, deep_loss)
    print("\nSaved plots and metrics in outputs/.")


if __name__ == "__main__":
    main()

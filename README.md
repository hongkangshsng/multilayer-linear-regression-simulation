# Multi-layer Network Linear Regression Simulation

A small, reproducible Python project that demonstrates **linear regression trained by gradient descent** through two models:

1. `LinearRegressionGD` — a conventional one-layer baseline.
2. `DeepLinearRegressor` — a three-layer, fully connected neural network with **no activation functions**.

The deep model is useful for learning how fully connected layers, weights, biases, mean-squared error (MSE), and back-propagation work.  Because every layer is linear, the composed model is still mathematically a linear function of the input.  This project makes that idea visible instead of hiding it behind a framework.

## What it does

- Generates deterministic noisy samples from `y = 2.8x + 1.2 + noise`.
- Trains both models with mini-batch gradient descent.
- Saves a fit comparison, training-loss curve, and CSV metrics.
- Verifies that the learned predictions approach the ground-truth line.

## Quick start in VS Code

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
pip install -r requirements.txt
python main.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Choose the `.venv` interpreter in VS Code when it prompts you. The terminal will print metrics, and the results will appear in `outputs/`.

## Expected output

```text
Model                         Train MSE       Test MSE
------------------------------------------------------
Linear regression baseline    ...             ...
3-layer deep linear network   ...             ...
```

Files generated after each run:

- `outputs/regression_fit.png` — data, target line, and both fitted lines.
- `outputs/training_loss.png` — loss over training epochs.
- `outputs/metrics.csv` — reproducible numerical evaluation.

## Model equation

For each fully connected layer:

\[
z = XW + b
\]

The deep model composes three such layers:

\[
\hat{y} = ((XW_1 + b_1)W_2 + b_2)W_3 + b_3
\]

With no non-linear activation between layers, this can be reduced to a single affine mapping `ŷ = XW + b`.  The separate layers are nevertheless trained independently using back-propagation, which is the key simulation here.

## Project layout

```text
multilayer-linear-regression/
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── outputs/                 # generated, intentionally not committed
```

## Notes

This is an educational NumPy implementation; it does not require TensorFlow, PyTorch, or MindSpore. It is designed to correspond to the textbook's fully-connected-layer linear-regression example while remaining easy to run locally in VS Code.

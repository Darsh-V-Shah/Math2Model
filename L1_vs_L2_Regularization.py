
import numpy as np


# ===============================================================================================================================
# L1 vs L2 REGULARIZATION FROM SCRATCH
# Linear Regression using NumPy
# This a face-off between L1 and L2 Regularization to see which model performs better at same value of lambda.
# ===============================================================================================================================


# 1. PREDICTION
def predict(X, w, b):
    """Calculate predictions for all training examples."""
    return X @ w + b


# 2. COST FUNCTION
def compute_cost(X, y, w, b, lambda_=0.0, regularization="none"):
    """
    Compute the cost function.

    Ordinary Linear Regression:
        J = MSE / 2

    L1 Regularization:
        J = MSE / 2 + (lambda / m) * sum(abs(w))

    L2 Regularization:
        J = MSE / 2 + (lambda / 2m) * sum(w^2)

    The bias is not regularized.
    """
    m = X.shape[0]

    y_pred = predict(X, w, b)
    errors = y_pred - y

    # Original MSE cost
    mse_cost = np.sum(errors ** 2) / (2 * m)

    # Regularization penalty
    if regularization == "l1":
        penalty = (lambda_ / m) * np.sum(np.abs(w))

    elif regularization == "l2":
        penalty = (lambda_ / (2 * m)) * np.sum(w ** 2)

    elif regularization == "none":
        penalty = 0.0

    else:
        raise ValueError(
            "regularization must be 'none', 'l1', or 'l2'."
        )

    return mse_cost + penalty


# 3. GRADIENT CALCULATION
def compute_gradient(
    X, y, w, b, lambda_=0.0, regularization="none"
):
    """
    Calculate gradients for the selected model.

    Base weight gradient:
        dj_dw = X.T @ (y_pred - y) / m

    L1 contribution:
        (lambda / m) * sign(w)

    L2 contribution:
        (lambda / m) * w

    Bias gradient is unchanged.
    """
    m = X.shape[0]

    y_pred = predict(X, w, b)
    errors = y_pred - y

    # Original MSE gradients
    dj_dw = (X.T @ errors) / m
    dj_db = np.sum(errors) / m

    # Add regularization gradient to weights only
    if regularization == "l1":
        dj_dw += (lambda_ / m) * np.sign(w)

    elif regularization == "l2":
        dj_dw += (lambda_ / m) * w

    elif regularization != "none":
        raise ValueError(
            "regularization must be 'none', 'l1', or 'l2'."
        )

    return dj_dw, dj_db


# 4. GRADIENT DESCENT
def gradient_descent(
    X,
    y,
    w,
    b,
    lambda_=0.0,
    regularization="none",
    alpha=0.01,
    iterations=1000
):
    """Train the selected model using gradient descent."""
    cost_history = []

    for i in range(iterations):

        # Calculate gradients
        dj_dw, dj_db = compute_gradient(
            X, y, w, b, lambda_, regularization
        )

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Record regularized cost
        cost = compute_cost(
            X, y, w, b, lambda_, regularization
        )
        cost_history.append(cost)

    return w, b, cost_history


# 5. TRAINING FUNCTION
def fit(
    X,
    y,
    lambda_=0.0,
    regularization="none",
    alpha=0.01,
    iterations=1000
):
    """Initialize parameters and train the model."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of examples."
        )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain examples and features.")

    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative.")

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    if regularization not in ("none", "l1", "l2"):
        raise ValueError(
            "regularization must be 'none', 'l1', or 'l2'."
        )

    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(y))
    ):
        raise ValueError("X and y must contain finite values.")

    # Initialize weights and bias
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    # Train
    w, b, cost_history = gradient_descent(
        X,
        y,
        w,
        b,
        lambda_=lambda_,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations
    )

    return w, b, cost_history


# 6. EVALUATION METRICS
def compute_mse(y_true, y_pred):
    """Calculate ordinary mean squared prediction error."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have equal shapes.")

    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    """Calculate the coefficient of determination."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have equal shapes.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")

    return 1 - ss_res / ss_tot


# 7. EXPERIMENT: TRAIN ALL THREE MODELS
if __name__ == "__main__":

    # Same dataset used in the L2 implementation
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6]
    ], dtype=float)

    y = np.array([8, 7, 18, 17, 28], dtype=float)

    # Identical training settings
    learning_rate = 0.01
    iterations = 5000
    regularization_strength = 1.0

    # --------------------------------------------------------
    # MODEL A: ORDINARY LINEAR REGRESSION
    # --------------------------------------------------------
    w_plain, b_plain, cost_plain = fit(
        X, y,
        lambda_=0.0,
        regularization="none",
        alpha=learning_rate,
        iterations=iterations
    )

    # --------------------------------------------------------
    # MODEL B: L1 REGULARIZATION
    # --------------------------------------------------------
    w_l1, b_l1, cost_l1 = fit(
        X, y,
        lambda_=regularization_strength,
        regularization="l1",
        alpha=learning_rate,
        iterations=iterations
    )

    # --------------------------------------------------------
    # MODEL C: L2 REGULARIZATION
    # --------------------------------------------------------
    w_l2, b_l2, cost_l2 = fit(
        X, y,
        lambda_=regularization_strength,
        regularization="l2",
        alpha=learning_rate,
        iterations=iterations
    )

    # Predictions
    pred_plain = predict(X, w_plain, b_plain)
    pred_l1 = predict(X, w_l1, b_l1)
    pred_l2 = predict(X, w_l2, b_l2)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    models = [
        ("Ordinary Linear Regression", w_plain, b_plain, pred_plain),
        ("L1 Regularization", w_l1, b_l1, pred_l1),
        ("L2 Regularization", w_l2, b_l2, pred_l2)
    ]

    print("=" * 72)
    print("LINEAR REGRESSION: L1 vs L2 COMPARISON")
    print("=" * 72)

    print("Lambda:", regularization_strength)
    print("Learning rate:", learning_rate)
    print("Iterations:", iterations)
    print("Number of examples:", X.shape[0])
    print("Number of features:", X.shape[1])

    for name, weights, bias, predictions in models:

        print("\n" + "-" * 72)
        print(name)
        print("-" * 72)

        print("Weights:", weights)
        print("Bias:", bias)
        print("Weight L1 norm:", np.sum(np.abs(weights)))
        print("Weight L2 norm:", np.linalg.norm(weights))
        print("Training MSE:", compute_mse(y, predictions))
        print("Training R^2:", compute_r2(y, predictions))

        # Count weights that are exactly zero
        print("Exactly zero weights:", np.sum(weights == 0))

    # --------------------------------------------------------
    # SIDE-BY-SIDE PREDICTIONS
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("PREDICTION COMPARISON")
    print("=" * 72)

    print(
        f"{'Actual':>10}"
        f"{'Ordinary':>14}"
        f"{'L1':>14}"
        f"{'L2':>14}"
    )

    for i in range(len(y)):
        print(
            f"{y[i]:>10.2f}"
            f"{pred_plain[i]:>14.2f}"
            f"{pred_l1[i]:>14.2f}"
            f"{pred_l2[i]:>14.2f}"
        )

    # --------------------------------------------------------
    # TRAINING OBJECTIVE COMPARISON
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL REGULARIZED COSTS")
    print("=" * 72)

    print(
        "Ordinary cost:",
        compute_cost(X, y, w_plain, b_plain, 0.0, "none")
    )

    print(
        "L1 cost:",
        compute_cost(
            X, y, w_l1, b_l1,
            regularization_strength, "l1"
        )
    )

    print(
        "L2 cost:",
        compute_cost(
            X, y, w_l2, b_l2,
            regularization_strength, "l2"
        )
    )

    print("\nNote: Lower training MSE alone does not prove")
    print("better generalization to unseen data.")


import numpy as np


# ============================================================
# L1 REGULARIZATION FROM SCRATCH
# Linear Regression using NumPy
# Same architecture as Linear Regression model.
# ============================================================


# 1. PREDICTION
def predict(X, w, b):
    """
    Calculate predictions for all training examples.

    Parameters:
        X : Feature matrix of shape (m, n)
        w : Weight vector of shape (n,)
        b : Scalar bias

    Returns:
        Predicted values of shape (m,)
    """
    return X @ w + b


# 2. COST FUNCTION WITH L1 REGULARIZATION
def compute_cost(X, y, w, b, lambda_):
    """
    Compute the regularized cost:

        J(w, b) = (1 / 2m) * sum((y_pred - y)^2)
                  + (lambda / m) * sum(abs(w))

    The bias term b is not regularized.
    """
    m = X.shape[0]

    # Predictions
    y_pred = predict(X, w, b)
    errors = y_pred - y

    # Mean squared error cost
    mse_cost = np.sum(errors ** 2) / (2 * m)

    # L1 regularization penalty
    l1_penalty = (lambda_ / m) * np.sum(np.abs(w))

    return mse_cost + l1_penalty


# 3. GRADIENT CALCULATION
def compute_gradient(X, y, w, b, lambda_):
    """
    Calculate gradients of the L1-regularized cost.

    dj_dw = (X.T @ errors) / m
            + (lambda / m) * sign(w)

    dj_db = sum(errors) / m

    At w = 0, np.sign(w) returns 0, which is a valid
    choice of subgradient for the absolute value function.
    """
    m = X.shape[0]

    # Predictions and errors
    y_pred = predict(X, w, b)
    errors = y_pred - y

    # Gradient of the original MSE cost
    dj_dw = (X.T @ errors) / m
    dj_db = np.sum(errors) / m

    # Add L1 regularization gradient
    dj_dw += (lambda_ / m) * np.sign(w)

    return dj_dw, dj_db


# 4. GRADIENT DESCENT
def gradient_descent(
    X,
    y,
    w,
    b,
    lambda_=0.0,
    alpha=0.01,
    iterations=1000
):
    """
    Train linear regression using gradient descent
    with L1 regularization.

    Returns:
        w            : Trained weights
        b            : Trained bias
        cost_history : Cost recorded at each iteration
    """
    cost_history = []

    for i in range(iterations):

        # Calculate gradients
        dj_dw, dj_db = compute_gradient(
            X, y, w, b, lambda_
        )

        # Update weights and bias
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Record cost
        cost = compute_cost(X, y, w, b, lambda_)
        cost_history.append(cost)

    return w, b, cost_history


# 5. TRAINING FUNCTION
def fit(X, y, lambda_=0.0, alpha=0.01, iterations=1000):
    """
    Initialize parameters and train the L1-regularized model.

    Supports any number of input features.

    Returns:
        w, b, cost_history
    """
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

    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(y))
    ):
        raise ValueError("X and y must contain finite values.")

    # Initialize weights and bias
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    # Train the model
    w, b, cost_history = gradient_descent(
        X,
        y,
        w,
        b,
        lambda_=lambda_,
        alpha=alpha,
        iterations=iterations
    )

    return w, b, cost_history


# 6. EVALUATION METRICS
def compute_mse(y_true, y_pred):
    """Calculate mean squared prediction error."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes.")

    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    """
    Calculate the coefficient of determination (R^2).

    R^2 = 1 - SS_res / SS_tot
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")

    return 1 - (ss_res / ss_tot)


# 7. EXPERIMENT
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

    # Training settings
    learning_rate = 0.01
    iterations = 5000
    regularization_strength = 1.0

    # Train the L1-regularized model
    w, b, cost_history = fit(
        X,
        y,
        lambda_=regularization_strength,
        alpha=learning_rate,
        iterations=iterations
    )

    # Predictions
    predictions = predict(X, w, b)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    print("=" * 55)
    print("LINEAR REGRESSION WITH L1 REGULARIZATION")
    print("=" * 55)

    print("Lambda:", regularization_strength)
    print("Weights:", w)
    print("Bias:", b)
    print("Weight L1 norm:", np.sum(np.abs(w)))
    print("Weight L2 norm:", np.linalg.norm(w))
    print("Training MSE:", compute_mse(y, predictions))
    print("Training R^2:", compute_r2(y, predictions))
    print("Exactly zero weights:", np.sum(w == 0))

    print("\nPredictions:")
    for i in range(len(y)):
        print(
            f"Actual: {y[i]:.2f} | "
            f"Predicted: {predictions[i]:.2f}"
        )

    print("\nInitial regularized cost:", cost_history[0])
    print("Final regularized cost:", cost_history[-1])

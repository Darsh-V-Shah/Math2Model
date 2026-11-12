import numpy as np

# L2-regularized linear regression, same architecture as the Linear Regression model,
# except the penalty (λ/2m)·Σw² is added to cost and (λ/m)·w to the gradient.


def predict(X, w, b):
    return X @ w + b


def compute_cost(X, y, w, b, lambda_):
    m = X.shape[0]
    errors = predict(X, w, b) - y
    mse = np.sum(errors ** 2) / (2 * m)
    l2 = (lambda_ / (2 * m)) * np.sum(w ** 2)  # bias not regularized
    return mse + l2


def compute_gradient(X, y, w, b, lambda_):
    m = X.shape[0]
    errors = predict(X, w, b) - y
    dj_dw = (X.T @ errors) / m + (lambda_ / m) * w  # L2 only hits w
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, lambda_=0.0, alpha=0.01, iterations=1000):
    cost_history = []
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b, lambda_))
    return w, b, cost_history


def fit(X, y, lambda_=0.0, alpha=0.01, iterations=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of examples.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain examples and features.")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X and y must contain finite values.")

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    return gradient_descent(X, y, w, b, lambda_=lambda_, alpha=alpha, iterations=iterations)


def compute_mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes.")
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
    ], dtype=float)
    y = np.array([8, 7, 18, 17, 28], dtype=float)

    learning_rate = 0.01
    iterations = 5000
    regularization_strength = 1.0

    w_plain, b_plain, _ = fit(X, y, lambda_=0.0,
                              alpha=learning_rate, iterations=iterations)
    predictions_plain = predict(X, w_plain, b_plain)

    w_l2, b_l2, _ = fit(X, y, lambda_=regularization_strength,
                        alpha=learning_rate, iterations=iterations)
    predictions_l2 = predict(X, w_l2, b_l2)

    print("Ordinary linear regression")
    print("  Weights:         ", w_plain)
    print("  Bias:            ", b_plain)
    print("  Weight magnitude:", np.linalg.norm(w_plain))
    print("  Training MSE:    ", compute_mse(y, predictions_plain))
    print("  Training R^2:    ", compute_r2(y, predictions_plain))

    print("\nL2-regularized linear regression (lambda = {})".format(regularization_strength))
    print("  Weights:         ", w_l2)
    print("  Bias:            ", b_l2)
    print("  Weight magnitude:", np.linalg.norm(w_l2))
    print("  Training MSE:    ", compute_mse(y, predictions_l2))
    print("  Training R^2:    ", compute_r2(y, predictions_l2))

    print("\nWeight magnitude comparison")
    print("  Ordinary:", np.linalg.norm(w_plain))
    print("  L2      :", np.linalg.norm(w_l2))

    print("\nFirst five predictions")
    for i in range(min(5, len(y))):
        print(f"  Actual: {y[i]:.2f} | Ordinary: {predictions_plain[i]:.2f} | L2: {predictions_l2[i]:.2f}")

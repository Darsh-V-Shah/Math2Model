import numpy as np

# L1 vs L2 face-off: three models trained on identical settings,
# same lambda for the regularized pair, so the penalties are the only variable.


def predict(X, w, b):
    return X @ w + b


def compute_cost(X, y, w, b, lambda_=0.0, regularization="none"):
    m = X.shape[0]
    errors = predict(X, w, b) - y
    cost = np.sum(errors ** 2) / (2 * m)  # plain MSE term, bias never penalized

    if regularization == "l1":
        cost += (lambda_ / m) * np.sum(np.abs(w))
    elif regularization == "l2":
        cost += (lambda_ / (2 * m)) * np.sum(w ** 2)
    elif regularization != "none":
        raise ValueError("regularization must be 'none', 'l1', or 'l2'.")

    return cost


def compute_gradient(X, y, w, b, lambda_=0.0, regularization="none"):
    m = X.shape[0]
    errors = predict(X, w, b) - y
    dj_dw = (X.T @ errors) / m
    dj_db = np.sum(errors) / m

    if regularization == "l1":
        # sign(0) = 0 is a valid subgradient of |w| at the origin.
        dj_dw += (lambda_ / m) * np.sign(w)
    elif regularization == "l2":
        dj_dw += (lambda_ / m) * w
    elif regularization != "none":
        raise ValueError("regularization must be 'none', 'l1', or 'l2'.")

    return dj_dw, dj_db


def gradient_descent(X, y, w, b, lambda_=0.0, regularization="none",
                     alpha=0.01, iterations=1000):
    cost_history = []
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_, regularization)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b, lambda_, regularization))
    return w, b, cost_history


def fit(X, y, lambda_=0.0, regularization="none", alpha=0.01, iterations=1000):
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
    if regularization not in ("none", "l1", "l2"):
        raise ValueError("regularization must be 'none', 'l1', or 'l2'.")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X and y must contain finite values.")

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    return gradient_descent(X, y, w, b, lambda_=lambda_,
                            regularization=regularization,
                            alpha=alpha, iterations=iterations)


def compute_mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have equal shapes.")
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have equal shapes.")

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

    w_plain, b_plain, _ = fit(X, y, lambda_=0.0, regularization="none",
                              alpha=learning_rate, iterations=iterations)
    w_l1, b_l1, _ = fit(X, y, lambda_=regularization_strength, regularization="l1",
                        alpha=learning_rate, iterations=iterations)
    w_l2, b_l2, _ = fit(X, y, lambda_=regularization_strength, regularization="l2",
                        alpha=learning_rate, iterations=iterations)

    pred_plain = predict(X, w_plain, b_plain)
    pred_l1 = predict(X, w_l1, b_l1)
    pred_l2 = predict(X, w_l2, b_l2)

    print("L1 vs L2 comparison")
    print("  Lambda:            ", regularization_strength)
    print("  Learning rate:     ", learning_rate)
    print("  Iterations:        ", iterations)
    print("  Number of examples:", X.shape[0])
    print("  Number of features:", X.shape[1])

    models = [
        ("Ordinary", w_plain, b_plain, pred_plain),
        ("L1",       w_l1,    b_l1,    pred_l1),
        ("L2",       w_l2,    b_l2,    pred_l2),
    ]

    for name, weights, bias, predictions in models:
        print(f"\n{name}")
        print("  Weights:             ", weights)
        print("  Bias:                ", bias)
        print("  Weight L1 norm:      ", np.sum(np.abs(weights)))
        print("  Weight L2 norm:      ", np.linalg.norm(weights))
        print("  Training MSE:        ", compute_mse(y, predictions))
        print("  Training R^2:        ", compute_r2(y, predictions))
        print("  Exactly zero weights:", np.sum(weights == 0))

    print("\nPredictions")
    print(f"{'Actual':>10}{'Ordinary':>14}{'L1':>14}{'L2':>14}")
    for i in range(len(y)):
        print(f"{y[i]:>10.2f}{pred_plain[i]:>14.2f}{pred_l1[i]:>14.2f}{pred_l2[i]:>14.2f}")

    print("\nFinal regularized costs")
    print("  Ordinary:", compute_cost(X, y, w_plain, b_plain, 0.0, "none"))
    print("  L1:      ", compute_cost(X, y, w_l1, b_l1, regularization_strength, "l1"))
    print("  L2:      ", compute_cost(X, y, w_l2, b_l2, regularization_strength, "l2"))

    print("\nLower training MSE alone doesn't prove better generalization.")

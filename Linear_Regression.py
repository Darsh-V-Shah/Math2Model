"""Multiple linear regression, from scratch with numpy/matplotlib.

Model:  f(x) = w^T x + b
Cost:   J(w,b) = (1/2m) * sum((f(x) - y)^2)
Grads:  dJ/dw = (1/m) X^T (f(X) - y)
        dJ/db = (1/m) sum(f(X) - y)
"""

import numpy as np
import matplotlib.pyplot as plt


def standardize_features(X):
    # Zero-mean, unit-variance per column. Without this, area (thousands)
    # and bedrooms (single digits) fight over the same learning rate.
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # avoid div-by-zero on constant columns
    return (X - mean) / std, mean, std


def apply_standardization(X, mean, std):
    # Same transform, applied to new data at inference time.
    std = np.where(std == 0, 1, std)
    return (X - mean) / std


def predict(X, w, b):
    return X @ w + b


def compute_cost(X, y, w, b):
    m = len(y)
    errors = predict(X, w, b) - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


def compute_gradient(X, y, w, b):
    m = len(y)
    errors = predict(X, w, b) - y
    dw = (1 / m) * (X.T @ errors)
    db = (1 / m) * np.sum(errors)
    return dw, db


def gradient_descent(X, y, learning_rate=0.01, iterations=2000):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    cost_history = []

    for i in range(iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 200 == 0:
            print(f"Iteration {i:5d} | Cost = {cost:.6f}")

    return w, b, cost_history


def r_squared(y, predictions):
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def plot_cost(cost_history):
    plt.figure(figsize=(8, 5))
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.title("Multiple Linear Regression - Gradient Descent")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    print("Multiple Linear Regression - gradient descent from scratch")
    print("-" * 60)

    # Features per row:
    #   area, bedrooms, bathrooms, age, distance_to_center, parking
    # Target: price in thousands of dollars.
    X = np.array([
        [1200, 2, 1, 15, 12, 1],
        [1500, 3, 2, 10, 10, 1],
        [1800, 3, 2,  8,  8, 2],
        [2000, 4, 2,  5,  7, 2],
        [2200, 4, 3,  4,  6, 2],
        [2500, 4, 3,  3,  5, 2],
        [2800, 5, 3,  2,  4, 3],
        [3000, 5, 4,  1,  3, 3],
        [1600, 3, 2, 12, 11, 1],
        [1900, 3, 2,  7,  9, 2],
        [2300, 4, 3,  6,  6, 2],
        [2700, 5, 3,  3,  5, 3],
        [3200, 5, 4,  1,  2, 3],
        [1400, 2, 1, 18, 14, 1],
        [2100, 4, 2,  9,  7, 2],
        [2600, 4, 3,  5,  5, 2],
        [2900, 5, 3,  2,  4, 3],
        [3400, 5, 4,  1,  2, 3],
        [1750, 3, 2, 11, 10, 1],
        [2400, 4, 3,  4,  6, 2],
    ], dtype=float)

    y = np.array([
        185, 240, 295, 340, 390, 440, 500, 570, 255, 315,
        405, 480, 610, 195, 330, 425, 515, 630, 275, 410,
    ], dtype=float)

    print(f"Training examples: {X.shape[0]}  |  Features: {X.shape[1]}")

    X_scaled, feature_mean, feature_std = standardize_features(X)

    # Training
    print("\nTraining...")
    w, b, cost_history = gradient_descent(
        X_scaled, y, learning_rate=0.05, iterations=2000
    )

    # Learned parameters
    feature_names = [
        "Area", "Bedrooms", "Bathrooms",
        "House Age", "Distance from City Center", "Parking Spaces",
    ]
    print("\nLearned parameters:")
    for name, weight in zip(feature_names, w):
        print(f"  {name:30s}: {weight:.6f}")
    print(f"  {'Bias':30s}: {b:.6f}")

    # Training fit
    predictions = predict(X_scaled, w, b)
    print(f"\nR² on training set: {r_squared(y, predictions):.4f}")

    # Sanity check: predict the price of a hypothetical house.
    new_house = np.array([[2600, 4, 3, 4, 6, 2]], dtype=float)
    new_house_scaled = apply_standardization(new_house, feature_mean, feature_std)
    price = predict(new_house_scaled, w, b)[0]
    print(f"\nPredicted price for a 2600 sqft, 4bd/3ba, 4yr-old house "
          f"6 km from center with 2 parking spots: ${price:.2f}k")

    plot_cost(cost_history)

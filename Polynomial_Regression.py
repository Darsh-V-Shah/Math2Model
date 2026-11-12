import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# y has linear, quadratic, interaction, and cubic terms in x. Model doesn't see that.
np.random.seed(42)
m, n = 800, 5
X = np.random.randn(m, n)
noise = np.random.randn(m) * 1.5
y = (
    3.0 * X[:, 0]
    + 2.0 * X[:, 1] ** 2
    - 1.5 * X[:, 2] * X[:, 3]
    + 0.8 * X[:, 4] ** 3
    + noise
)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)


# 70/15/15 split
indices = np.random.permutation(m)
train_end = int(0.70 * m)
val_end = int(0.85 * m)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

print("\nTraining examples:", len(y_train))
print("Validation examples:", len(y_val))
print("Test examples:", len(y_test))


def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return mean, std


def transform_standardizer(X, mean, std):
    return (X - mean) / std


original_mean, original_std = fit_standardizer(X_train)

X_train_scaled = transform_standardizer(X_train, original_mean, original_std)
X_val_scaled = transform_standardizer(X_val, original_mean, original_std)
X_test_scaled = transform_standardizer(X_test, original_mean, original_std)


def polynomial_features(X, degree):
    n_rows, n_features = X.shape
    columns = []

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            col = np.ones(n_rows)
            for j in combo:
                col *= X[:, j]
            columns.append(col)

    return np.column_stack(columns)


def prepare_polynomial_data(X_train, X_other, degree):
    train_poly = polynomial_features(X_train, degree)
    other_poly = polynomial_features(X_other, degree)

    poly_mean, poly_std = fit_standardizer(train_poly)

    train_poly = transform_standardizer(train_poly, poly_mean, poly_std)
    other_poly = transform_standardizer(other_poly, poly_mean, poly_std)

    return train_poly, other_poly, poly_mean, poly_std


def predict(X, w, b):
    return X @ w + b


def compute_cost(X, y, w, b):
    m = len(y)
    errors = predict(X, w, b) - y
    return np.sum(errors ** 2) / (2 * m)


def gradient_descent(X, y, learning_rate=0.03, iterations=2500):
    m, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    cost_history = []

    for i in range(iterations):
        errors = predict(X, w, b) - y
        dj_dw = (X.T @ errors) / m
        dj_db = np.sum(errors) / m
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        if i % 100 == 0:
            cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


def calculate_mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def calculate_rmse(y_actual, y_predicted):
    return np.sqrt(calculate_mse(y_actual, y_predicted))


# Sweep degrees 1..5
degrees = [1, 2, 3, 4, 5]
training_errors = []
validation_errors = []
models = {}

for degree in degrees:
    Xtr, Xv, poly_mean, poly_std = prepare_polynomial_data(
        X_train_scaled, X_val_scaled, degree
    )
    w, b, cost_history = gradient_descent(
        Xtr, y_train, learning_rate=0.03, iterations=2500
    )

    train_mse = calculate_mse(y_train, predict(Xtr, w, b))
    val_mse = calculate_mse(y_val, predict(Xv, w, b))

    training_errors.append(train_mse)
    validation_errors.append(val_mse)

    models[degree] = {
        "w": w, "b": b,
        "poly_mean": poly_mean, "poly_std": poly_std,
        "cost_history": cost_history,
    }

    print(f"\nDegree {degree} ({Xtr.shape[1]} features)")
    print(f"  Training MSE:   {train_mse:.4f}")
    print(f"  Validation MSE: {val_mse:.4f}")


best_index = int(np.argmin(validation_errors))
best_degree = degrees[best_index]
print("\nSelected degree:", best_degree)


# Refit on train+val, test once
X_train_val = np.vstack((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))

final_original_mean, final_original_std = fit_standardizer(X_train_val)

X_train_val_scaled = transform_standardizer(
    X_train_val, final_original_mean, final_original_std)
X_test_final_scaled = transform_standardizer(
    X_test, final_original_mean, final_original_std)

X_train_val_poly = polynomial_features(X_train_val_scaled, best_degree)
X_test_poly = polynomial_features(X_test_final_scaled, best_degree)

final_poly_mean, final_poly_std = fit_standardizer(X_train_val_poly)

X_train_val_poly = transform_standardizer(
    X_train_val_poly, final_poly_mean, final_poly_std)
X_test_poly = transform_standardizer(
    X_test_poly, final_poly_mean, final_poly_std)

final_w, final_b, final_cost_history = gradient_descent(
    X_train_val_poly, y_train_val, learning_rate=0.03, iterations=2500
)

test_predictions = predict(X_test_poly, final_w, final_b)
test_mse = calculate_mse(y_test, test_predictions)
test_rmse = calculate_rmse(y_test, test_predictions)

print(f"\nTest MSE:  {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


plt.figure(figsize=(9, 5))
plt.plot(degrees, training_errors, marker="o", label="Training MSE")
plt.plot(degrees, validation_errors, marker="o", label="Validation MSE")
plt.axvline(best_degree, linestyle="--", label=f"Selected degree = {best_degree}")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Regression: Degree Selection")
plt.xticks(degrees)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(np.arange(len(final_cost_history)) * 100, final_cost_history)
plt.xlabel("Gradient Descent Iteration")
plt.ylabel("Cost J(w, b)")
plt.title("Final Model: Cost During Gradient Descent")
plt.grid(True)
plt.tight_layout()
plt.show()


new_example = np.array([[0.5, -0.2, 1.0, 0.3, -0.7]])

new_example_scaled = transform_standardizer(
    new_example, final_original_mean, final_original_std)
new_example_poly = polynomial_features(new_example_scaled, best_degree)
new_example_poly = transform_standardizer(
    new_example_poly, final_poly_mean, final_poly_std)

new_prediction = predict(new_example_poly, final_w, final_b)

print("\nInput features:", new_example)
print(f"Predicted target: {new_prediction[0]:.4f}")

"""
Polynomial regression on the California housing dataset.

Everything is from scratch — NumPy for the math, pandas for I/O,
matplotlib for plots. No scikit-learn.

Pipeline: standardize → expand to polynomial features → standardize again →
L2-regularized linear regression via gradient descent.

A grid of (degree, lambda) pairs is scored on a validation split; the winner
gets retrained on train+val and evaluated once on test.
"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


# ----- Config -----

DATASET_PATH = "housing.csv"

FEATURES = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "households",
    "latitude",
    "longitude",
]

TARGET = "median_house_value"

POLYNOMIAL_DEGREES = [1, 2, 3]
L2_STRENGTHS = [0.0, 100.0, 1000.0]

LEARNING_RATE = 0.03
ITERATIONS = 600
RANDOM_SEED = 42


# ----- Data loading -----

def load_dataset(path):
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    print(f"\nLoaded {len(X)} usable rows, {X.shape[1]} features.")
    return X, y


def split_dataset(X, y, seed=42):
    # 70/15/15 train/val/test.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    train_end = int(0.70 * len(X))
    val_end = int(0.85 * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


# ----- Preprocessing -----

class Standardizer:
    """Per-column zero-mean unit-variance scaling."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PolynomialFeatures:
    """Expand X into all monomials of total degree 1..degree."""

    def __init__(self, degree):
        self.degree = degree
        self.combinations = None
        self.n_features = None
        self.feature_names = None

    def fit(self, X):
        self.n_features = X.shape[1]
        self.combinations = []
        self.feature_names = []

        for power in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(self.n_features), power):
                self.combinations.append(combo)

                # Build a human-readable name, e.g. median_income*latitude^2.
                counts = {}
                for idx in combo:
                    counts[idx] = counts.get(idx, 0) + 1

                parts = []
                for idx in sorted(counts):
                    feature = FEATURES[idx]
                    exp = counts[idx]
                    parts.append(feature if exp == 1 else f"{feature}^{exp}")
                self.feature_names.append("*".join(parts))

        return self

    def transform(self, X):
        columns = [np.prod(X[:, combo], axis=1) for combo in self.combinations]
        return np.column_stack(columns)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----- Model -----

class L2LinearRegression:
    """Linear regression with L2 penalty, trained by gradient descent."""

    def __init__(self, learning_rate=0.03, iterations=600, lambda_=100.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_

        self.w = None
        self.b = 0.0
        self.cost_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def compute_cost(self, X, y):
        m = X.shape[0]
        errors = self.predict(X) - y
        data_cost = np.sum(errors ** 2) / (2 * m)
        l2_penalty = self.lambda_ * np.sum(self.w ** 2) / (2 * m)
        return data_cost + l2_penalty

    def compute_gradient(self, X, y):
        m = X.shape[0]
        errors = self.predict(X) - y
        dj_dw = (X.T @ errors) / m + (self.lambda_ / m) * self.w
        dj_db = np.sum(errors) / m  # bias is never regularized
        return dj_dw, dj_db

    def fit(self, X, y, verbose=False):
        _, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for iteration in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.w -= self.learning_rate * dj_dw
            self.b -= self.learning_rate * dj_db

            if iteration % 50 == 0:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                if verbose and iteration % 100 == 0:
                    print(f"  iter {iteration:4d}  cost {cost:.6f}")

        return self


class HousingPipeline:
    """Standardize → expand → standardize → L2 regression → unscale target."""

    def __init__(self, degree, lambda_, learning_rate=0.03, iterations=600):
        self.degree = degree
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.input_scaler = Standardizer()
        self.polynomial = PolynomialFeatures(degree)
        self.poly_scaler = Standardizer()

        self.target_mean = None
        self.target_std = None
        self.model = None

    def fit(self, X, y, verbose=False):
        X_scaled = self.input_scaler.fit_transform(X)
        X_poly = self.polynomial.fit_transform(X_scaled)
        X_poly_scaled = self.poly_scaler.fit_transform(X_poly)

        # Scaling the target keeps the learning rate usable across datasets.
        self.target_mean = np.mean(y)
        self.target_std = np.std(y) or 1.0
        y_scaled = (y - self.target_mean) / self.target_std

        self.model = L2LinearRegression(
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            lambda_=self.lambda_,
        )
        self.model.fit(X_poly_scaled, y_scaled, verbose=verbose)
        return self

    def predict(self, X):
        X_scaled = self.input_scaler.transform(X)
        X_poly = self.polynomial.transform(X_scaled)
        X_poly_scaled = self.poly_scaler.transform(X_poly)
        pred_scaled = self.model.predict(X_poly_scaled)
        return pred_scaled * self.target_std + self.target_mean


# ----- Metrics -----

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


# ----- Grid search -----

def select_best_model(X_train, y_train, X_val, y_val):
    results = []
    best_config = None
    best_rmse = float("inf")

    for degree in POLYNOMIAL_DEGREES:
        for lambda_ in L2_STRENGTHS:
            print(f"\nTrying degree={degree}, lambda={lambda_}")

            model = HousingPipeline(
                degree=degree,
                lambda_=lambda_,
                learning_rate=LEARNING_RATE,
                iterations=ITERATIONS,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            val_rmse = rmse(y_val, preds)
            val_mae = mae(y_val, preds)
            val_r2 = r2_score(y_val, preds)

            print(f"  val RMSE ${val_rmse:,.2f}   MAE ${val_mae:,.2f}   R² {val_r2:.4f}")

            results.append({
                "degree": degree,
                "lambda": lambda_,
                "features": len(model.polynomial.feature_names),
                "validation_RMSE": val_rmse,
                "validation_MAE": val_mae,
                "validation_R2": val_r2,
            })

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_config = {"degree": degree, "lambda": lambda_}

    results_df = pd.DataFrame(results).sort_values("validation_RMSE")
    print("\nValidation results, sorted by RMSE:")
    print(results_df.to_string(index=False))

    return best_config, results_df


# ----- Plots -----

def show_evaluation_plots(y_true, predictions):
    residuals = y_true - predictions

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true, predictions, alpha=0.25, s=10)
    lo = min(np.min(y_true), np.min(predictions))
    hi = max(np.max(y_true), np.max(predictions))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--")
    axes[0].set_xlabel("Actual house value ($)")
    axes[0].set_ylabel("Predicted house value ($)")
    axes[0].set_title("Actual vs Predicted")

    axes[1].scatter(predictions, residuals, alpha=0.25, s=10)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Predicted house value ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residuals")

    plt.tight_layout()
    plt.show()


# ----- Interactive UI -----

def interactive_prediction(model):
    print("\nOpening interactive predictor. Close the window when done.")

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("California Housing Price Predictor", fontsize=16)

    defaults = {
        "median_income": "5.0",
        "housing_median_age": "25",
        "total_rooms": "2000",
        "households": "500",
        "latitude": "34.0",
        "longitude": "-118.0",
    }

    input_boxes = {}
    for i, feature in enumerate(FEATURES):
        row, col = i // 2, i % 2
        left = 0.08 + col * 0.48
        bottom = 0.77 - row * 0.19

        fig.text(left, bottom + 0.07, feature.replace("_", " ").title())
        ax = fig.add_axes([left, bottom, 0.36, 0.055])
        input_boxes[feature] = TextBox(ax, "", initial=defaults[feature])

    result_text = fig.text(
        0.08, 0.14, "Enter values and click Predict.", fontsize=12
    )

    button_ax = fig.add_axes([0.68, 0.055, 0.22, 0.07])
    predict_button = Button(button_ax, "Predict")

    def on_predict(_event):
        try:
            values = [float(input_boxes[f].text) for f in FEATURES]
            X_input = np.array([values], dtype=float)
            pred = model.predict(X_input)[0]
            result_text.set_text(f"Predicted median house value: ${pred:,.0f}")
        except ValueError:
            result_text.set_text("Error: enter a valid number in every field.")
        fig.canvas.draw_idle()

    predict_button.on_clicked(on_predict)
    plt.show()


# ----- Main -----

def main():
    X, y = load_dataset(DATASET_PATH)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, RANDOM_SEED
    )
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    # Pick degree and lambda using validation data only.
    best_config, results = select_best_model(X_train, y_train, X_val, y_val)
    print(f"\nSelected: degree={best_config['degree']}, lambda={best_config['lambda']}")

    # Refit on train + val with the chosen config.
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = HousingPipeline(
        degree=best_config["degree"],
        lambda_=best_config["lambda"],
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
    )
    print("\nTraining final model on train + val...")
    final_model.fit(X_trainval, y_trainval, verbose=True)

    # Touch the test set exactly once.
    test_predictions = final_model.predict(X_test)

    print("\nTest results")
    print(f"  RMSE ${rmse(y_test, test_predictions):,.2f}")
    print(f"  MAE  ${mae(y_test, test_predictions):,.2f}")
    print(f"  R²   {r2_score(y_test, test_predictions):.4f}")
    print(f"  Features after expansion: {len(final_model.polynomial.feature_names)}")

    print("\nFirst five predictions:")
    for actual, predicted in zip(y_test[:5], test_predictions[:5]):
        print(f"  actual ${actual:,.0f} | predicted ${predicted:,.0f}")

    results.to_csv("model_comparison.csv", index=False)
    print("\nSaved model comparison to model_comparison.csv")

    show_evaluation_plots(y_test, test_predictions)
    interactive_prediction(final_model)


if __name__ == "__main__":
    main()

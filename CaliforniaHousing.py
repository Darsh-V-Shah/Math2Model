
"""
POLYNOMIAL HOUSING PRICE PREDICTOR
==================================
Stage A:
- Load and clean a real-world housing dataset
- Train/validation/test split
- Standardization
- Polynomial feature expansion from scratch
- Linear regression with L2 regularization
- Gradient descent from scratch
- Hyperparameter selection
- RMSE, MAE, R² evaluation

Stage B:
- Actual vs predicted visualization
- Residual visualization

No scikit-learn is used.
Uses Polynomial Regression with L2 Regularization. Loop to find the ideal hyperparameters.

DATASET: California Housing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, TextBox


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "housing.csv"

FEATURES = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "households",
    "latitude",
    "longitude"
]

TARGET = "median_house_value"

POLYNOMIAL_DEGREES = [1, 2, 3]
L2_STRENGTHS = [0.0, 100.0, 1000.0]

LEARNING_RATE = 0.03
ITERATIONS = 600
RANDOM_SEED = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(path):
    """
    Load the CSV and retain the numerical features needed
    for this project.
    """

    df = pd.read_csv(path)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    df = df[required_columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    print("\n========== DATASET ==========")
    print("Usable samples:", len(X))
    print("Original features:", X.shape[1])
    print("Target:", TARGET)

    return X, y


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(X, y, seed=42):
    """
    70% training
    15% validation
    15% testing

    Training:
        Learns model parameters.

    Validation:
        Selects polynomial degree and lambda.

    Testing:
        Evaluates the final selected model.
    """

    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))

    train_end = int(0.70 * len(X))
    validation_end = int(0.85 * len(X))

    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]

    return (
        X[train_idx], y[train_idx],
        X[validation_idx], y[validation_idx],
        X[test_idx], y[test_idx]
    )


# ============================================================
# 4. STANDARDIZATION
# ============================================================

class Standardizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Prevent division by zero.
        self.std[self.std < 1e-12] = 1.0

        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# 5. POLYNOMIAL FEATURE EXPANSION FROM SCRATCH
# ============================================================

class PolynomialFeatures:

    def __init__(self, degree):
        self.degree = degree
        self.combinations = None
        self.n_features = None
        self.feature_names = None

    def fit(self, X):
        """
        Generate feature-index combinations.

        For two variables x1 and x2, degree 2 generates:

        x1, x2, x1², x1*x2, x2²

        Degree 3 adds cubic terms and their interactions.
        """

        from itertools import combinations_with_replacement

        self.n_features = X.shape[1]
        self.combinations = []
        self.feature_names = []

        for power in range(1, self.degree + 1):

            for combo in combinations_with_replacement(
                range(self.n_features), power
            ):

                self.combinations.append(combo)

                counts = {}

                for index in combo:
                    counts[index] = counts.get(index, 0) + 1

                name_parts = []

                for index in sorted(counts):

                    feature = FEATURES[index]
                    exponent = counts[index]

                    if exponent == 1:
                        name_parts.append(feature)
                    else:
                        name_parts.append(
                            f"{feature}^{exponent}"
                        )

                self.feature_names.append(
                    "*".join(name_parts)
                )

        return self

    def transform(self, X):
        """
        Calculate each polynomial term using NumPy.
        """

        columns = []

        for combo in self.combinations:

            column = np.prod(X[:, combo], axis=1)

            columns.append(column)

        return np.column_stack(columns)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# 6. LINEAR REGRESSION WITH L2 REGULARIZATION
# ============================================================

class L2LinearRegression:

    def __init__(
        self,
        learning_rate=0.03,
        iterations=600,
        lambda_=100.0
    ):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_

        self.w = None
        self.b = 0.0
        self.cost_history = []

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(self, X):

        return X @ self.w + self.b

    # --------------------------------------------------------
    # Cost function
    # --------------------------------------------------------

    def compute_cost(self, X, y):

        m = X.shape[0]

        predictions = self.predict(X)
        errors = predictions - y

        data_cost = np.sum(errors ** 2) / (2 * m)

        l2_penalty = (
            self.lambda_
            * np.sum(self.w ** 2)
            / (2 * m)
        )

        return data_cost + l2_penalty

    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    def compute_gradient(self, X, y):

        m = X.shape[0]

        predictions = self.predict(X)
        errors = predictions - y

        # Gradient of ordinary squared-error cost.
        dj_dw = (X.T @ errors) / m

        # Gradient of L2 penalty.
        dj_dw += (self.lambda_ / m) * self.w

        # Bias is NOT regularized.
        dj_db = np.sum(errors) / m

        return dj_dw, dj_db

    # --------------------------------------------------------
    # Gradient descent
    # --------------------------------------------------------

    def fit(self, X, y, verbose=False):

        m, n = X.shape

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

                    print(
                        f"Iteration {iteration:4d} | "
                        f"Cost: {cost:.6f}"
                    )

        return self


# ============================================================
# 7. COMPLETE MODEL PIPELINE
# ============================================================

class HousingPipeline:

    def __init__(
        self,
        degree,
        lambda_,
        learning_rate=0.03,
        iterations=600
    ):

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

        # Step 1: Scale original features.
        X_scaled = self.input_scaler.fit_transform(X)

        # Step 2: Generate polynomial terms.
        X_poly = self.polynomial.fit_transform(X_scaled)

        # Step 3: Scale expanded polynomial features.
        X_poly_scaled = self.poly_scaler.fit_transform(X_poly)

        # Step 4: Scale target for numerical stability.
        self.target_mean = np.mean(y)
        self.target_std = np.std(y)

        if self.target_std < 1e-12:
            self.target_std = 1.0

        y_scaled = (
            y - self.target_mean
        ) / self.target_std

        # Step 5: Train regularized regression.
        self.model = L2LinearRegression(
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            lambda_=self.lambda_
        )

        self.model.fit(
            X_poly_scaled,
            y_scaled,
            verbose=verbose
        )

        return self

    def predict(self, X):

        # Apply exactly the same transformations learned
        # during training.

        X_scaled = self.input_scaler.transform(X)

        X_poly = self.polynomial.transform(X_scaled)

        X_poly_scaled = self.poly_scaler.transform(X_poly)

        predictions_scaled = self.model.predict(X_poly_scaled)

        # Return predictions in original dollar units.
        return (
            predictions_scaled * self.target_std
            + self.target_mean
        )


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def mse(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):

    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_total = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_total == 0:
        return 0.0

    return 1 - ss_res / ss_total


# ============================================================
# 9. MODEL SELECTION
# ============================================================

def select_best_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    results = []
    best_model = None
    best_rmse = float("inf")

    print("\n========== MODEL SELECTION ==========")

    for degree in POLYNOMIAL_DEGREES:

        for lambda_ in L2_STRENGTHS:

            print(
                f"\nTraining degree={degree}, "
                f"lambda={lambda_}"
            )

            model = HousingPipeline(
                degree=degree,
                lambda_=lambda_,
                learning_rate=LEARNING_RATE,
                iterations=ITERATIONS
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_validation)

            validation_rmse = rmse(
                y_validation,
                predictions
            )

            validation_r2 = r2_score(
                y_validation,
                predictions
            )

            validation_mae = mae(
                y_validation,
                predictions
            )

            feature_count = len(
                model.polynomial.feature_names
            )

            print(
                f"Validation RMSE: ${validation_rmse:,.2f}"
            )

            print(
                f"Validation MAE:  ${validation_mae:,.2f}"
            )

            print(
                f"Validation R²:   {validation_r2:.4f}"
            )

            results.append({
                "degree": degree,
                "lambda": lambda_,
                "features": feature_count,
                "validation_RMSE": validation_rmse,
                "validation_MAE": validation_mae,
                "validation_R2": validation_r2
            })

            if validation_rmse < best_rmse:

                best_rmse = validation_rmse

                best_model = {
                    "degree": degree,
                    "lambda": lambda_
                }

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "validation_RMSE"
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(results_df.to_string(index=False))

    return best_model, results_df


# ============================================================
# 10. VISUALIZATIONS
# ============================================================

def show_evaluation_plots(y_true, predictions):

    residuals = y_true - predictions

    # Actual versus predicted.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(
        y_true,
        predictions,
        alpha=0.25,
        s=10
    )

    low = min(np.min(y_true), np.min(predictions))
    high = max(np.max(y_true), np.max(predictions))

    axes[0].plot(
        [low, high],
        [low, high],
        linestyle="--"
    )

    axes[0].set_xlabel("Actual house value ($)")
    axes[0].set_ylabel("Predicted house value ($)")
    axes[0].set_title("Actual vs Predicted")

    # Residual plot.
    axes[1].scatter(
        predictions,
        residuals,
        alpha=0.25,
        s=10
    )

    axes[1].axhline(0, linestyle="--")

    axes[1].set_xlabel("Predicted house value ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residual Analysis")

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. INTERACTIVE PREDICTION INTERFACE
# ============================================================

def interactive_prediction(model):

    print("\n========== INTERACTIVE PREDICTION ==========")
    print("Enter housing features in their original units.")
    print("Close the prediction window to finish.")

    fig = plt.figure(figsize=(10, 7))

    fig.suptitle(
        "California Housing Price Predictor",
        fontsize=16
    )

    input_boxes = {}

    default_values = {
        "median_income": "5.0",
        "housing_median_age": "25",
        "total_rooms": "2000",
        "households": "500",
        "latitude": "34.0",
        "longitude": "-118.0"
    }

    for index, feature in enumerate(FEATURES):

        row = index // 2
        col = index % 2

        left = 0.08 + col * 0.48
        bottom = 0.77 - row * 0.19

        fig.text(
            left,
            bottom + 0.07,
            feature.replace("_", " ").title()
        )

        ax_box = fig.add_axes([
            left,
            bottom,
            0.36,
            0.055
        ])

        input_boxes[feature] = TextBox(
            ax_box,
            "",
            initial=default_values[feature]
        )

    result_text = fig.text(
        0.08,
        0.14,
        "Enter values and click Predict.",
        fontsize=12
    )

    button_ax = fig.add_axes([
        0.68,
        0.055,
        0.22,
        0.07
    ])

    predict_button = Button(
        button_ax,
        "Predict"
    )

    def on_predict(event):

        try:

            values = []

            for feature in FEATURES:

                value = float(
                    input_boxes[feature].text
                )

                values.append(value)

            X_input = np.array(
                [values],
                dtype=float
            )

            prediction = model.predict(X_input)[0]

            result_text.set_text(
                f"Predicted median house value: "
                f"${prediction:,.0f}"
            )

            fig.canvas.draw_idle()

        except ValueError:

            result_text.set_text(
                "Error: enter a valid number in every field."
            )

            fig.canvas.draw_idle()

    predict_button.on_clicked(on_predict)

    plt.show()


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    # Load data.
    X, y = load_dataset(DATASET_PATH)

    # Split data.
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = split_dataset(X, y, RANDOM_SEED)

    print("\n========== DATA SPLIT ==========")
    print("Training samples:   ", len(X_train))
    print("Validation samples: ", len(X_validation))
    print("Test samples:       ", len(X_test))

    # Select degree and lambda using validation data only.
    best_config, results = select_best_model(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print("\n========== SELECTED MODEL ==========")
    print("Polynomial degree:", best_config["degree"])
    print("L2 strength:", best_config["lambda"])

    # Refit the selected configuration using training
    # and validation samples together.
    X_trainval = np.vstack([
        X_train,
        X_validation
    ])

    y_trainval = np.concatenate([
        y_train,
        y_validation
    ])

    final_model = HousingPipeline(
        degree=best_config["degree"],
        lambda_=best_config["lambda"],
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS
    )

    print("\nTraining final model...")

    final_model.fit(
        X_trainval,
        y_trainval,
        verbose=True
    )

    # Evaluate the test set after model selection.
    test_predictions = final_model.predict(X_test)

    print("\n========== FINAL TEST RESULTS ==========")

    print(
        f"Test RMSE: ${rmse(y_test, test_predictions):,.2f}"
    )

    print(
        f"Test MAE:  ${mae(y_test, test_predictions):,.2f}"
    )

    print(
        f"Test R²:   {r2_score(y_test, test_predictions):.4f}"
    )

    print(
        "Expanded feature count:",
        len(final_model.polynomial.feature_names)
    )

    print("\nFirst five test predictions:")

    for actual, predicted in zip(
        y_test[:5],
        test_predictions[:5]
    ):

        print(
            f"Actual: ${actual:,.0f} | "
            f"Predicted: ${predicted:,.0f}"
        )

    # Show model-selection comparison.
    results.to_csv(
        "model_comparison.csv",
        index=False
    )

    print("\nSaved model comparison to model_comparison.csv")

    # Stage B: visualizations.
    show_evaluation_plots(
        y_test,
        test_predictions
    )

    # Stage B: interactive prediction.
    interactive_prediction(final_model)


if __name__ == "__main__":
    main()


"""
Math2Model - Multiple Linear Regression

Linear Regression implemented from mathematics using:

    NumPy
    Matplotlib

No scikit-learn or other machine-learning libraries are used.

Model:

    f_w,b(x) = w^T x + b

Cost:

    J(w,b) =
        (1 / 2m) * sum((f_w,b(x) - y)^2)

Gradient:

    dJ/dw =
        (1/m) * X^T (f_w,b(X) - y)

    dJ/db =
        (1/m) * sum(f_w,b(X) - y)

Gradient descent:

    w := w - alpha * dJ/dw

    b := b - alpha * dJ/db
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. FEATURE STANDARDIZATION
# ============================================================

def standardize_features(X):
    """
    Standardize every feature:

        x_j := (x_j - mean_j) / std_j

    This helps gradient descent converge efficiently when
    features have very different numerical scales.
    """

    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    # Prevent division by zero
    std[std == 0] = 1

    X_scaled = (X - mean) / std

    return X_scaled, mean, std


def apply_standardization(X, mean, std):
    """
    Apply the same scaling parameters used during training.
    """

    std = np.where(std == 0, 1, std)

    return (X - mean) / std


# ============================================================
# 2. LINEAR REGRESSION PREDICTION
# ============================================================

def predict(X, w, b):
    """
    Multiple linear regression:

        f_w,b(x) = w^T x + b

    For n features:

        f(x) =
            w1*x1 +
            w2*x2 +
            ...
            wn*xn +
            b
    """

    return X @ w + b


# ============================================================
# 3. COST FUNCTION
# ============================================================

def compute_cost(X, y, w, b):
    """
    Mean squared error cost:

        J(w,b) =
            (1 / 2m)
            * sum((f_w,b(x) - y)^2)
    """

    m = len(y)

    predictions = predict(
        X,
        w,
        b
    )

    errors = predictions - y

    cost = (
        1 / (2 * m)
    ) * np.sum(errors ** 2)

    return cost


# ============================================================
# 4. COMPUTE GRADIENT
# ============================================================

def compute_gradient(X, y, w, b):
    """
    Calculate gradients for ALL features simultaneously.

    dJ/dw =
        (1/m) * X^T (predictions - y)

    dJ/db =
        (1/m) * sum(predictions - y)
    """

    m = len(y)

    predictions = predict(
        X,
        w,
        b
    )

    errors = predictions - y

    # Gradient for every feature
    dw = (
        1 / m
    ) * (
        X.T @ errors
    )

    # Gradient for bias
    db = (
        1 / m
    ) * np.sum(errors)

    return dw, db


# ============================================================
# 5. GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=2000
):
    """
    Train the model using gradient descent.

    Andrew Ng update equations:

        w := w - alpha * dJ/dw

        b := b - alpha * dJ/db
    """

    number_of_features = X.shape[1]

    # Initialize all weights to zero
    w = np.zeros(
        number_of_features
    )

    # Initialize bias
    b = 0.0

    cost_history = []


    for iteration in range(iterations):

        # Calculate gradients
        dw, db = compute_gradient(
            X,
            y,
            w,
            b
        )

        # Update weights
        w = (
            w -
            learning_rate * dw
        )

        # Update bias
        b = (
            b -
            learning_rate * db
        )

        # Calculate cost
        cost = compute_cost(
            X,
            y,
            w,
            b
        )

        cost_history.append(cost)


        # Display progress
        if iteration % 200 == 0:

            print(
                f"Iteration {iteration:5d} | "
                f"Cost = {cost:.6f}"
            )


    return w, b, cost_history


# ============================================================
# 6. R-SQUARED
# ============================================================

def r_squared(y, predictions):
    """
    R² measures how much of the variance in y is explained
    by the model.

        R² = 1 - SS_res / SS_tot
    """

    ss_res = np.sum(
        (y - predictions) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    return 1 - (
        ss_res / ss_tot
    )


# ============================================================
# 7. PLOT COST
# ============================================================

def plot_cost(cost_history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        cost_history
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Cost J(w,b)"
    )

    plt.title(
        "Multiple Linear Regression - Gradient Descent"
    )

    plt.grid()

    plt.show()


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("Math2Model - Multiple Linear Regression")
    print("=" * 65)


    # ========================================================
    # DATASET
    # ========================================================

    """
    We use SIX features:

        x1 = Area (square feet)
        x2 = Bedrooms
        x3 = Bathrooms
        x4 = House Age (years)
        x5 = Distance from city center (km)
        x6 = Parking Spaces

    Target:

        y = House Price (thousands of dollars)
    """


    X = np.array([

        [1200, 2, 1, 15, 12, 1],
        [1500, 3, 2, 10, 10, 1],
        [1800, 3, 2, 8, 8, 2],
        [2000, 4, 2, 5, 7, 2],
        [2200, 4, 3, 4, 6, 2],
        [2500, 4, 3, 3, 5, 2],
        [2800, 5, 3, 2, 4, 3],
        [3000, 5, 4, 1, 3, 3],
        [1600, 3, 2, 12, 11, 1],
        [1900, 3, 2, 7, 9, 2],
        [2300, 4, 3, 6, 6, 2],
        [2700, 5, 3, 3, 5, 3],
        [3200, 5, 4, 1, 2, 3],
        [1400, 2, 1, 18, 14, 1],
        [2100, 4, 2, 9, 7, 2],
        [2600, 4, 3, 5, 5, 2],
        [2900, 5, 3, 2, 4, 3],
        [3400, 5, 4, 1, 2, 3],
        [1750, 3, 2, 11, 10, 1],
        [2400, 4, 3, 4, 6, 2]

    ], dtype=float)


    # Target:
    #
    # House price in thousands of dollars

    y = np.array([

        185,
        240,
        295,
        340,
        390,
        440,
        500,
        570,
        255,
        315,
        405,
        480,
        610,
        195,
        330,
        425,
        515,
        630,
        275,
        410

    ], dtype=float)


    print("\nDataset")
    print("-" * 65)

    print(
        "Number of training examples:",
        X.shape[0]
    )

    print(
        "Number of features:",
        X.shape[1]
    )

    print(
        "\nFeatures:"
    )

    print(
        "1. Area"
    )

    print(
        "2. Bedrooms"
    )

    print(
        "3. Bathrooms"
    )

    print(
        "4. House Age"
    )

    print(
        "5. Distance from City Center"
    )

    print(
        "6. Parking Spaces"
    )


    # ========================================================
    # FEATURE SCALING
    # ========================================================

    X_scaled, feature_mean, feature_std = (
        standardize_features(X)
    )


    # ========================================================
    # INITIAL PARAMETERS
    # ========================================================

    print("\nInitial parameters")

    initial_w = np.zeros(
        X.shape[1]
    )

    initial_b = 0

    print(
        "w =",
        initial_w
    )

    print(
        "b =",
        initial_b
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\n" + "=" * 65)
    print("TRAINING")
    print("=" * 65)

    w, b, cost_history = gradient_descent(
        X_scaled,
        y,
        learning_rate=0.05,
        iterations=2000
    )


    # ========================================================
    # LEARNED PARAMETERS
    # ========================================================

    print("\n" + "=" * 65)
    print("LEARNED PARAMETERS")
    print("=" * 65)

    feature_names = [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "House Age",
        "Distance from City Center",
        "Parking Spaces"
    ]

    for name, weight in zip(
        feature_names,
        w
    ):

        print(
            f"{name:30s}: {weight:.6f}"
        )

    print(
        f"{'Bias':30s}: {b:.6f}"
    )


    # ========================================================
    # TRAINING PREDICTIONS
    # ========================================================

    predictions = predict(
        X_scaled,
        w,
        b
    )


    # ========================================================
    # R²
    # ========================================================

    score = r_squared(
        y,
        predictions
    )

    print(
        "\nR² =",
        score
    )


    # ========================================================
    # EXAMPLE PREDICTION
    # ========================================================

    """
    Predict the price of a new house:

        Area                  = 2600 sq ft
        Bedrooms              = 4
        Bathrooms             = 3
        House Age             = 4 years
        Distance from center  = 6 km
        Parking Spaces        = 2
    """

    new_house = np.array([[
        2600,
        4,
        3,
        4,
        6,
        2
    ]], dtype=float)


    new_house_scaled = (
        apply_standardization(
            new_house,
            feature_mean,
            feature_std
        )
    )


    predicted_price = predict(
        new_house_scaled,
        w,
        b
    )


    print(
        "\nPredicted price for new house:"
    )

    print(
        f"${predicted_price[0]:.2f} thousand"
    )


    # ========================================================
    # COST PLOT
    # ========================================================

    plot_cost(
        cost_history
    )


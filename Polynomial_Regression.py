
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


# ============================================================
# 1. CREATE A SYNTHETIC DATASET
# ============================================================

np.random.seed(42)

m = 800  # Number of examples
n = 5    # Number of original features

# Five features, each drawn from a standard normal distribution.
X = np.random.randn(m, n)

# Our underlying relationship contains nonlinear terms.
# The model does not get to see this equation.
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


# ============================================================
# 2. SPLIT DATA INTO TRAINING, VALIDATION AND TEST SETS
# ============================================================

# Shuffle the indices so the sets contain randomly selected examples.
indices = np.random.permutation(m)

train_end = int(0.70 * m)
val_end = int(0.85 * m)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("\nTraining examples:", len(y_train))
print("Validation examples:", len(y_val))
print("Test examples:", len(y_test))


# ============================================================
# 3. STANDARDIZE THE ORIGINAL FEATURES
# ============================================================

def fit_standardizer(X):
    """
    Calculate the mean and standard deviation of each feature.
    These values must be learned from training data only.
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant features.
    std[std == 0] = 1.0

    return mean, std


def transform_standardizer(X, mean, std):
    """Apply previously calculated standardization values."""

    return (X - mean) / std


# Fit on training data, then reuse on validation and test data.
original_mean, original_std = fit_standardizer(X_train)

X_train_scaled = transform_standardizer(
    X_train, original_mean, original_std
)

X_val_scaled = transform_standardizer(
    X_val, original_mean, original_std
)

X_test_scaled = transform_standardizer(
    X_test, original_mean, original_std
)


# ============================================================
# 4. POLYNOMIAL FEATURE TRANSFORMATION FROM SCRATCH
# ============================================================

def polynomial_features(X, degree):
    """
    Generate every non-constant polynomial feature
    whose total degree is between 1 and 'degree'.

    For two features and degree 2:

        [x1, x2] becomes
        [x1, x2, x1^2, x1*x2, x2^2]

    The constant/intercept term is handled separately by b.
    """

    number_of_examples, number_of_features = X.shape

    columns = []

    # Generate degree 1, then degree 2, and so on.
    for current_degree in range(1, degree + 1):

        # Repeated indices generate powers.
        # Different indices generate interaction terms.
        for combination in combinations_with_replacement(
            range(number_of_features),
            current_degree
        ):

            # Start with 1 for each example.
            column = np.ones(number_of_examples)

            # Multiply the selected original features together.
            for feature_index in combination:
                column *= X[:, feature_index]

            columns.append(column)

    # Convert the list of columns into a 2D matrix.
    return np.column_stack(columns)


# ============================================================
# 5. STANDARDIZE THE GENERATED POLYNOMIAL FEATURES
# ============================================================

def prepare_polynomial_data(X_train, X_other, degree):
    """
    Transform polynomial features and standardize them.

    The polynomial-feature scaler is fitted on training data
    only, avoiding information leakage from validation/test data.
    """

    train_poly = polynomial_features(X_train, degree)
    other_poly = polynomial_features(X_other, degree)

    poly_mean, poly_std = fit_standardizer(train_poly)

    train_poly = transform_standardizer(
        train_poly, poly_mean, poly_std
    )

    other_poly = transform_standardizer(
        other_poly, poly_mean, poly_std
    )

    return train_poly, other_poly, poly_mean, poly_std


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict(X, w, b):
    """
    Linear model operating on the transformed features.

    y_hat = Xw + b
    """

    return X @ w + b


# ============================================================
# 7. COST FUNCTION
# ============================================================

def compute_cost(X, y, w, b):
    """
    J(w, b) = (1 / 2m) * sum((prediction - actual)^2)
    """

    m = len(y)

    predictions = predict(X, w, b)
    errors = predictions - y

    cost = np.sum(errors ** 2) / (2 * m)

    return cost


# ============================================================
# 8. GRADIENT DESCENT FROM SCRATCH
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.03,
    iterations=2500
):
    """
    Learn weights and bias using gradient descent.

    dJ/dw = X.T @ (prediction - y) / m
    dJ/db = sum(prediction - y) / m
    """

    m, number_of_features = X.shape

    # Initialize every weight and the bias to zero.
    w = np.zeros(number_of_features)
    b = 0.0

    cost_history = []

    for iteration in range(iterations):

        # ----------------------------------------------------
        # A. Forward propagation: calculate predictions
        # ----------------------------------------------------

        predictions = predict(X, w, b)

        # ----------------------------------------------------
        # B. Calculate errors
        # ----------------------------------------------------

        errors = predictions - y

        # ----------------------------------------------------
        # C. Calculate gradients
        # ----------------------------------------------------

        dj_dw = (X.T @ errors) / m
        dj_db = np.sum(errors) / m

        # ----------------------------------------------------
        # D. Update weights and bias
        # ----------------------------------------------------

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        # Save cost occasionally for inspection.
        if iteration % 100 == 0:
            cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def calculate_mse(y_actual, y_predicted):
    """Mean Squared Error."""

    return np.mean((y_actual - y_predicted) ** 2)


def calculate_rmse(y_actual, y_predicted):
    """Root Mean Squared Error."""

    return np.sqrt(calculate_mse(y_actual, y_predicted))


# ============================================================
# 10. COMPARE POLYNOMIAL DEGREES
# ============================================================

degrees = [1, 2, 3, 4, 5]

training_errors = []
validation_errors = []

models = {}

print("\n" + "=" * 65)
print("POLYNOMIAL DEGREE EXPERIMENT")
print("=" * 65)

for degree in degrees:

    print(f"\nTraining polynomial degree {degree}...")

    # Create and standardize polynomial features.
    Xtr, Xv, poly_mean, poly_std = prepare_polynomial_data(
        X_train_scaled,
        X_val_scaled,
        degree
    )

    # Train the model.
    w, b, cost_history = gradient_descent(
        Xtr,
        y_train,
        learning_rate=0.03,
        iterations=2500
    )

    # Predict on training and validation sets.
    train_predictions = predict(Xtr, w, b)
    val_predictions = predict(Xv, w, b)

    # Calculate errors.
    train_mse = calculate_mse(y_train, train_predictions)
    val_mse = calculate_mse(y_val, val_predictions)

    training_errors.append(train_mse)
    validation_errors.append(val_mse)

    # Save everything needed to reuse this model.
    models[degree] = {
        "w": w,
        "b": b,
        "poly_mean": poly_mean,
        "poly_std": poly_std,
        "cost_history": cost_history
    }

    print("Number of polynomial features:", Xtr.shape[1])
    print(f"Training MSE:   {train_mse:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")


# ============================================================
# 11. SELECT THE BEST DEGREE USING VALIDATION ERROR
# ============================================================

best_index = int(np.argmin(validation_errors))
best_degree = degrees[best_index]

print("\n" + "=" * 65)
print("MODEL SELECTION")
print("=" * 65)

for degree, train_error, val_error in zip(
    degrees, training_errors, validation_errors
):
    print(
        f"Degree {degree}: "
        f"Training MSE = {train_error:.4f}, "
        f"Validation MSE = {val_error:.4f}"
    )

print("\nSelected polynomial degree:", best_degree)


# ============================================================
# 12. RETRAIN ON TRAINING + VALIDATION DATA
# ============================================================

# Once the degree is selected, combine training and validation
# data for the final fit. The test set remains untouched.

X_train_val = np.vstack((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))

# Refit the original-feature scaler on the combined data.
final_original_mean, final_original_std = fit_standardizer(
    X_train_val
)

X_train_val_scaled = transform_standardizer(
    X_train_val,
    final_original_mean,
    final_original_std
)

X_test_final_scaled = transform_standardizer(
    X_test,
    final_original_mean,
    final_original_std
)

# Generate polynomial features for the chosen degree.
X_train_val_poly = polynomial_features(
    X_train_val_scaled, best_degree
)

X_test_poly = polynomial_features(
    X_test_final_scaled, best_degree
)

# Fit the polynomial-feature scaler on combined training data.
final_poly_mean, final_poly_std = fit_standardizer(
    X_train_val_poly
)

X_train_val_poly = transform_standardizer(
    X_train_val_poly,
    final_poly_mean,
    final_poly_std
)

X_test_poly = transform_standardizer(
    X_test_poly,
    final_poly_mean,
    final_poly_std
)

# Train the final model.
final_w, final_b, final_cost_history = gradient_descent(
    X_train_val_poly,
    y_train_val,
    learning_rate=0.03,
    iterations=2500
)


# ============================================================
# 13. FINAL TEST-SET EVALUATION
# ============================================================

test_predictions = predict(
    X_test_poly,
    final_w,
    final_b
)

test_mse = calculate_mse(y_test, test_predictions)
test_rmse = calculate_rmse(y_test, test_predictions)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print("Chosen degree:", best_degree)
print(f"Test MSE:  {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


# ============================================================
# 14. PLOT TRAINING AND VALIDATION ERROR
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    degrees,
    training_errors,
    marker="o",
    label="Training MSE"
)

plt.plot(
    degrees,
    validation_errors,
    marker="o",
    label="Validation MSE"
)

plt.axvline(
    best_degree,
    linestyle="--",
    label=f"Selected degree = {best_degree}"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Regression: Degree Selection")
plt.xticks(degrees)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 15. PLOT THE FINAL MODEL'S COST HISTORY
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    np.arange(len(final_cost_history)) * 100,
    final_cost_history
)

plt.xlabel("Gradient Descent Iteration")
plt.ylabel("Cost J(w, b)")
plt.title("Final Model: Cost During Gradient Descent")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 16. TRY A PREDICTION ON A NEW EXAMPLE
# ============================================================

# Five new feature values.
new_example = np.array([[0.5, -0.2, 1.0, 0.3, -0.7]])

# Apply the original-feature standardization.
new_example_scaled = transform_standardizer(
    new_example,
    final_original_mean,
    final_original_std
)

# Apply the selected polynomial transformation.
new_example_poly = polynomial_features(
    new_example_scaled,
    best_degree
)

# Apply the polynomial-feature standardization.
new_example_poly = transform_standardizer(
    new_example_poly,
    final_poly_mean,
    final_poly_std
)

# Predict the target.
new_prediction = predict(
    new_example_poly,
    final_w,
    final_b
)

print("\n" + "=" * 65)
print("NEW PREDICTION")
print("=" * 65)

print("Input features:", new_example)
print(f"Predicted target: {new_prediction[0]:.4f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"

print("Training shape:", train.shape)
print("Testing shape :", test.shape)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

numeric_features = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level"
]

categorical_features = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]


# ============================================================
# 3. CONVERT TARGET TO 0 / 1
# ============================================================

# No = 0
# Yes = 1

y = (train[TARGET].values == "Yes").astype(float)


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

# We keep 20% of the training data aside to evaluate
# how well the model generalizes.

random_generator = np.random.default_rng(42)

class_0 = np.where(y == 0)[0]
class_1 = np.where(y == 1)[0]

random_generator.shuffle(class_0)
random_generator.shuffle(class_1)

validation_0 = int(0.20 * len(class_0))
validation_1 = int(0.20 * len(class_1))

val_indices = np.concatenate([
    class_0[:validation_0],
    class_1[:validation_1]
])

train_indices = np.concatenate([
    class_0[validation_0:],
    class_1[validation_1:]
])

random_generator.shuffle(train_indices)
random_generator.shuffle(val_indices)


train_data = train.iloc[train_indices].copy()
val_data = train.iloc[val_indices].copy()

y_train = y[train_indices]
y_val = y[val_indices]


print("\nTraining examples :", len(train_data))
print("Validation examples:", len(val_data))


# ============================================================
# 5. FEATURE SCALING
# ============================================================

# feature scaling:
#
# x_j := (x_j - mean_j) / standard_deviation_j
#
# IMPORTANT:
# Mean and standard deviation are calculated ONLY
# from the training data.

means = train_data[numeric_features].mean().values
stds = train_data[numeric_features].std(ddof=0).values

# Avoid division by zero
stds[stds == 0] = 1


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

# Logistic regression requires numerical input.
#
# Example:
#
# Gender:
#
# Male   -> [1, 0, 0]
# Female -> [0, 1, 0]
# Other  -> [0, 0, 1]
#
# We create the encoding ourselves instead of using
# sklearn.preprocessing.


category_values = {}

for column in categorical_features:
    # Convert the column to string type before getting unique values
    # to handle potential NaN values which are floats and would cause TypeError in sorted()
    category_values[column] = sorted(
        train_data[column].astype(str).unique()
    )


def one_hot_encode(values, categories):

    category_to_index = {
        category: i
        for i, category in enumerate(categories)
    }

    result = np.zeros(
        (len(values), len(categories))
    )

    for row in range(len(values)):
        category = str(values[row]) # Ensure category is string, handles NaN converted to 'nan'
        # Handle cases where a category in 'values' might not be in 'categories'
        # This could happen if new categories appear in validation/test data that weren't in training data
        if category in category_to_index:
            column_index = category_to_index[category]
            result[row, column_index] = 1

    return result


# ============================================================
# 7. CREATE FEATURE MATRIX
# ============================================================

def create_features(data):

    # ----------------------------
    # Numerical features
    # ----------------------------

    numerical = data[numeric_features].values.astype(float)

    numerical = (
        numerical - means
    ) / stds


    # ----------------------------
    # Categorical features
    # ----------------------------

    categorical_matrices = []

    for column in categorical_features:

        encoded = one_hot_encode(
            data[column].values, # Pass original values, 'one_hot_encode' now handles string conversion
            category_values[column]
        )

        categorical_matrices.append(encoded)


    # ----------------------------
    # Combine everything
    # ----------------------------

    X = numerical

    for matrix in categorical_matrices:
        X = np.hstack((X, matrix))


    return X


X_train = create_features(train_data)
X_val = create_features(val_data)


# ============================================================
# 8. ADD INTERCEPT TERM
# ============================================================

# Our mathematical equation is:
#
# z = w^T x + b
#
# We can implement the bias/intercept separately,
# but adding a column of ones makes the mathematics
# convenient:
#
# z = Xw
#
# where the first weight represents b.

X_train = np.column_stack(
    (np.ones(X_train.shape[0]), X_train)
)

X_val = np.column_stack(
    (np.ones(X_val.shape[0]), X_val)
)


print("\nNumber of features:", X_train.shape[1])


# ============================================================
# 9. SIGMOID FUNCTION
# ============================================================

# 
#
# f_w,b(x) = 1 / (1 + e^(-z))
#
# where:d
#
# z = w^T x + b


def sigmoid(z):

    # Prevent numerical overflow in exp()
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# ============================================================
# 10. LOGISTIC REGRESSION PREDICTION
# ============================================================

def predict_probability(X, w):

    z = X @ w

    return sigmoid(z)


# ============================================================
# 11. COST FUNCTION
# ============================================================

# Logistic regression cost:
#
# J(w) =
#
# -1/m * sum[
#
# y log(f(x))
#
# + (1-y) log(1-f(x))
#
# ]
#
#
# We also add L2 regularization:
#
# lambda/(2m) * sum(w_j^2)
#
# We DO NOT regularize the bias.
#
# This is the regularized logistic regression
# 


def compute_cost(X, y, w, lambda_reg):

    m = len(y)

    probabilities = predict_probability(X, w)

    # Avoid log(0)
    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15
    )

    # Logistic loss
    cost = -(1 / m) * np.sum(
        y * np.log(probabilities)
        +
        (1 - y) * np.log(1 - probabilities)
    )

    # Regularization
    #
    # Do not regularize w[0]
    # because w[0] is the bias.

    regularization = (
        lambda_reg / (2 * m)
    ) * np.sum(w[1:] ** 2)

    return cost + regularization


# ============================================================
# 12. COMPUTE GRADIENTS
# ============================================================

#logistic regression gradient:
#
# dJ/dw_j =
#
# 1/m * sum[
#     (f(x) - y) * x_j
# ]
#
# with regularization:
#
# dJ/dw_j =
#
# 1/m * sum[
#     (f(x) - y) * x_j
# ]
#
# + lambda/m * w_j
#
# for j >= 1.
#
# The bias is NOT regularized.


def compute_gradient(X, y, w, lambda_reg):

    m = len(y)

    probabilities = predict_probability(X, w)

    error = probabilities - y

    # Vectorized gradient
    gradient = (1 / m) * (X.T @ error)

    # Regularization
    regularization = (
        lambda_reg / m
    ) * w

    # Don't regularize bias
    regularization[0] = 0

    gradient = gradient + regularization

    return gradient


# ============================================================
# 13. GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.05,
    iterations=500,
    lambda_reg=1.0
):

    number_of_features = X.shape[1]

    # Initialize parameters to zero
    w = np.zeros(number_of_features)

    cost_history = []


    for iteration in range(iterations):

        # Calculate gradient
        gradient = compute_gradient(
            X,
            y,
            w,
            lambda_reg
        )

        #update rule:
        #
        # w := w - alpha * gradient

        w = w - learning_rate * gradient


        # Store cost
        cost = compute_cost(
            X,
            y,
            w,
            lambda_reg
        )

        cost_history.append(cost)


        # Progress display
        if iteration % 50 == 0:

            print(
                "Iteration:",
                iteration,
                "Cost:",
                cost
            )


    return w, cost_history


# ============================================================
# 14. TRAIN THE MODEL
# ============================================================

print("\n==============================")
print("TRAINING LOGISTIC REGRESSION")
print("==============================")

learning_rate = 0.05
iterations = 1000 # Increased iterations to 1000

# Lambda controls regularization.
#
# Larger lambda:
# stronger regularization
#
# Smaller lambda:
# weaker regularization

lambda_reg = 1.0


weights, cost_history = gradient_descent(
    X_train,
    y_train,
    learning_rate=learning_rate,
    iterations=iterations,
    lambda_reg=lambda_reg
)


# ============================================================
# 15. PLOT COST
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost J(w)")
plt.title("Logistic Regression - Gradient Descent")

plt.grid()

plt.show()


# ============================================================
# 16. PREDICTION
# ============================================================

def predict(X, w, threshold=0.5):

    probabilities = predict_probability(
        X,
        w
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions


y_pred = predict(
    X_val,
    weights
)


# ============================================================
# 17. ACCURACY
# ============================================================

accuracy = np.mean(
    y_pred == y_val
)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

print(
    "Accuracy:",
    accuracy
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

true_positive = np.sum(
    (y_pred == 1) &
    (y_val == 1)
)

true_negative = np.sum(
    (y_pred == 0) &
    (y_val == 0)
)

false_positive = np.sum(
    (y_pred == 1) &
    (y_val == 0)
) # Removed extraneous text

false_negative = np.sum(
    (y_pred == 0) &
    (y_val == 1)
)


print("\nConfusion Matrix")
print("----------------")

print(
    "True Positive :",
    true_positive
)

print(
    "True Negative :",
    true_negative
)

print(
    "False Positive:",
    false_positive
)

print(
    "False Negative:",
    false_negative
)


# ============================================================
# 19. PRECISION
# ============================================================

# Precision =
#
# TP / (TP + FP)

precision = (
    true_positive /
    (true_positive + false_positive)
)


# ============================================================
# 20. RECALL
# ============================================================

# Recall =
#
# TP / (TP + FN)

recall = (
    true_positive /
    (true_positive + false_negative)
)


# ============================================================
# 21. F1 SCORE
# ============================================================

# F1 =
#
# 2 * Precision * Recall /
# (Precision + Recall)

f1 = (
    2 * precision * recall /
    (precision + recall)
)


print(
    "\nPrecision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1 Score:",
    f1
)


# ============================================================
# 22. TRAIN MODEL ON ALL TRAINING DATA
# ============================================================

# Once we are satisfied with the model,
# we can retrain using ALL available training examples.
#
# This gives the final model more data to learn from.

print("\n==============================")
print("TRAINING FINAL MODEL")
print("==============================")


# Recalculate preprocessing parameters
# using ALL training data.

final_means = train[numeric_features].mean().values
final_stds = train[numeric_features].std(ddof=0).values

final_stds[final_stds == 0] = 1


# Save old preprocessing values
old_means = means
old_stds = stds
old_category_values = category_values


# Use all training data for category definitions
final_category_values = {}

for column in categorical_features:

    final_category_values[column] = sorted(
        train[column].astype(str).unique()
    )


def create_final_features(data):

    numerical = data[numeric_features].values.astype(float)

    numerical = (
        numerical - final_means
    ) / final_stds

    categorical_matrices = []

    for column in categorical_features:

        encoded = one_hot_encode(
            data[column].values,
            final_category_values[column]
        )

        categorical_matrices.append(encoded)

    X = numerical

    for matrix in categorical_matrices:
        X = np.hstack((X, matrix))

    return np.column_stack(
        (np.ones(X.shape[0]), X)
    )


X_final = create_final_features(train)
X_test = create_final_features(test)

y_final = (
    train[TARGET].values == "Yes"
).astype(float)


# Train final model

final_weights, final_cost_history = gradient_descent(
    X_final,
    y_final,
    learning_rate=learning_rate,
    iterations=iterations,
    lambda_reg=lambda_reg
)


# ============================================================
# 23. PREDICT TEST DATA
# ============================================================

test_probabilities = predict_probability(
    X_test,
    final_weights
)

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


# Convert back to Yes / No

test_labels = np.where(
    test_predictions == 1,
    "Yes",
    "No"
)


# ============================================================
# 24. CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({

    "id": test["id"],

    "Will_Buy_EV": test_labels

})


submission.to_csv(
    "predictions.csv",
    index=False
)


print("\n==============================")
print("DONE")
print("==============================")

print(
    "Predictions saved to: predictions.csv"
)

print("\nFirst predictions:")

print(
    submission.head(10)
)

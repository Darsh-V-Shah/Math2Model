import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"

print("Training shape:", train.shape)
print("Testing shape :", test.shape)


numeric_features = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

categorical_features = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

y = (train[TARGET].values == "Yes").astype(float)


# ------------------------------------------------------------
# Stratified 80/20 train/val split
# ------------------------------------------------------------

rng = np.random.default_rng(42)

class_0 = np.where(y == 0)[0]
class_1 = np.where(y == 1)[0]
rng.shuffle(class_0)
rng.shuffle(class_1)

val_0 = int(0.20 * len(class_0))
val_1 = int(0.20 * len(class_1))

val_indices = np.concatenate([class_0[:val_0], class_1[:val_1]])
train_indices = np.concatenate([class_0[val_0:], class_1[val_1:]])

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_data = train.iloc[train_indices].copy()
val_data = train.iloc[val_indices].copy()

y_train = y[train_indices]
y_val = y[val_indices]

print("\nTraining examples :", len(train_data))
print("Validation examples:", len(val_data))


# ------------------------------------------------------------
# Preprocessing (fit on training data only)
# ------------------------------------------------------------

# Standardize numerics; one-hot the categoricals by hand (no sklearn).
means = train_data[numeric_features].mean().values
stds = train_data[numeric_features].std(ddof=0).values
stds[stds == 0] = 1

# Cast to str before building categories so NaN doesn't break sorted().
category_values = {
    col: sorted(train_data[col].astype(str).unique())
    for col in categorical_features
}


def one_hot_encode(values, categories):
    index = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(values), len(categories)))
    for row, v in enumerate(values):
        s = str(v)
        if s in index:
            out[row, index[s]] = 1
    return out


def create_features(data):
    numeric = (data[numeric_features].values.astype(float) - means) / stds

    parts = [numeric]
    for col in categorical_features:
        parts.append(one_hot_encode(data[col].values, category_values[col]))

    X = np.hstack(parts)

    # Prepend a column of ones so the bias folds into w.
    return np.column_stack((np.ones(X.shape[0]), X))


X_train = create_features(train_data)
X_val = create_features(val_data)

print("\nNumber of features:", X_train.shape[1])


# ------------------------------------------------------------
# Logistic regression
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)  # guard against exp overflow
    return 1 / (1 + np.exp(-z))


def predict_probability(X, w):
    return sigmoid(X @ w)


def compute_cost(X, y, w, lambda_reg):
    m = len(y)
    p = np.clip(predict_probability(X, w), 1e-15, 1 - 1e-15)

    cost = -(1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    reg = (lambda_reg / (2 * m)) * np.sum(w[1:] ** 2)  # don't penalize bias
    return cost + reg


def compute_gradient(X, y, w, lambda_reg):
    m = len(y)
    error = predict_probability(X, w) - y

    grad = (1 / m) * (X.T @ error)

    reg = (lambda_reg / m) * w
    reg[0] = 0  # bias not regularized

    return grad + reg


def gradient_descent(X, y, learning_rate=0.05, iterations=500, lambda_reg=1.0):
    w = np.zeros(X.shape[1])
    cost_history = []

    for i in range(iterations):
        w -= learning_rate * compute_gradient(X, y, w, lambda_reg)
        cost = compute_cost(X, y, w, lambda_reg)
        cost_history.append(cost)

        if i % 50 == 0:
            print("Iteration:", i, "Cost:", cost)

    return w, cost_history


# ------------------------------------------------------------
# Train on the split, evaluate on validation
# ------------------------------------------------------------

print("\nTraining logistic regression")
learning_rate = 0.05
iterations = 1000
lambda_reg = 1.0

weights, cost_history = gradient_descent(
    X_train, y_train,
    learning_rate=learning_rate,
    iterations=iterations,
    lambda_reg=lambda_reg,
)

plt.figure(figsize=(8, 5))
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(w)")
plt.title("Logistic Regression - Gradient Descent")
plt.grid()
plt.show()


def predict(X, w, threshold=0.5):
    return (predict_probability(X, w) >= threshold).astype(int)


y_pred = predict(X_val, weights)

accuracy = np.mean(y_pred == y_val)

tp = np.sum((y_pred == 1) & (y_val == 1))
tn = np.sum((y_pred == 0) & (y_val == 0))
fp = np.sum((y_pred == 1) & (y_val == 0))
fn = np.sum((y_pred == 0) & (y_val == 1))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("\nValidation")
print("  Accuracy :", accuracy)
print("  Confusion: TP={} TN={} FP={} FN={}".format(tp, tn, fp, fn))
print("  Precision:", precision)
print("  Recall   :", recall)
print("  F1       :", f1)


# ------------------------------------------------------------
# Retrain on all training data, predict test, write submission
# ------------------------------------------------------------

print("\nRetraining on the full training set")

final_means = train[numeric_features].mean().values
final_stds = train[numeric_features].std(ddof=0).values
final_stds[final_stds == 0] = 1

final_category_values = {
    col: sorted(train[col].astype(str).unique())
    for col in categorical_features
}


def create_final_features(data):
    numeric = (data[numeric_features].values.astype(float) - final_means) / final_stds

    parts = [numeric]
    for col in categorical_features:
        parts.append(one_hot_encode(data[col].values, final_category_values[col]))

    X = np.hstack(parts)
    return np.column_stack((np.ones(X.shape[0]), X))


X_final = create_final_features(train)
X_test = create_final_features(test)

y_final = (train[TARGET].values == "Yes").astype(float)

final_weights, _ = gradient_descent(
    X_final, y_final,
    learning_rate=learning_rate,
    iterations=iterations,
    lambda_reg=lambda_reg,
)

test_probs = predict_probability(X_test, final_weights)
test_labels = np.where(test_probs >= 0.5, "Yes", "No")

submission = pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_labels})
submission.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")
print(submission.head(10))

"""
COMP 2200 Final Project - starter file
Dataset: Documenting Hate data

This is just the first working version.
It loads the data, makes some simple features, splits train/test,
and trains logistic regression with numpy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# PART 0: Setup
# -------------------------------------------------------------------
np.random.seed(42)

DATA_FILE = "20170816_Documenting_Hate - Data.csv"


# -------------------------------------------------------------------
# PART 1: Load and clean data
# -------------------------------------------------------------------
def load_hate_data():
    """
    Loads the CSV file and fixes the column names.

    The file header is a little weird, so I rename the columns
    to match what the rows actually contain.
    """
    if not os.path.exists(DATA_FILE):
        print("Could not find the dataset file.")
        print("Put this python file in the same folder as:")
        print(DATA_FILE)
        return None

    data = pd.read_csv(DATA_FILE)

    data.columns = [
        "date_added",
        "article_date",
        "title",
        "organization",
        "city",
        "state",
        "url",
        "keywords",
        "summary"
    ]

    text_cols = ["title", "organization", "city", "state", "url", "keywords", "summary"]

    for col in text_cols:
        data[col] = data[col].fillna("").astype(str)

    data["article_date"] = pd.to_datetime(data["article_date"], errors="coerce")

    return data


def print_basic_stats(data):
    """
    Prints simple dataset info so I can understand what I am working with.
    """
    print("\n--- Dataset Info ---")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])
    print("Column names:")
    print(list(data.columns))

    print("\nMissing city:", (data["city"] == "").sum())
    print("Missing state:", (data["state"] == "").sum())
    print("Missing summary:", (data["summary"] == "").sum())

    print("\nTop 10 states:")
    print(data["state"].replace("", "Missing").value_counts().head(10))

    print("\nTop 10 organizations:")
    print(data["organization"].replace("", "Missing").value_counts().head(10))


# -------------------------------------------------------------------
# PART 2: Make target label and features
# -------------------------------------------------------------------
def make_violent_label(data):
    """
    Creates a starter target for logistic regression.

    y = 1 means the article seems to talk about a violent incident.
    y = 0 means it does not match those words.

    This is not perfect. It is only a first target so the model can run.
    Later we can improve this part.
    """
    violent_words = [
        "attack", "attacked", "assault", "shooting", "shot",
        "killed", "murder", "stabbed", "gun", "threat",
        "threatened", "bomb", "beat", "beating", "violent",
        "violence"
    ]

    all_text = (
        data["title"] + " " +
        data["keywords"] + " " +
        data["summary"]
    ).str.lower()

    y = []

    for text in all_text:
        found_word = 0

        for word in violent_words:
            if word in text:
                found_word = 1

        y.append(found_word)

    return np.array(y)


def count_words(text):
    """
    Counts words in a simple way.
    """
    if text == "":
        return 0

    return len(text.split())


def make_features(data):
    """
    Makes basic numeric features.

    I am avoiding scikit-learn here because the project says the model
    should be from scratch using numpy.
    """
    title_words = data["title"].apply(count_words).to_numpy()
    keyword_words = data["keywords"].apply(count_words).to_numpy()
    summary_words = data["summary"].apply(count_words).to_numpy()

    has_city = (data["city"] != "").astype(int).to_numpy()
    has_state = (data["state"] != "").astype(int).to_numpy()
    has_summary = (data["summary"] != "").astype(int).to_numpy()

    date_min = data["article_date"].min()
    days_after_start = (data["article_date"] - date_min).dt.days
    days_after_start = days_after_start.fillna(0).to_numpy()

    # Just a few common states as simple category features
    state_text = data["state"].str.upper()
    is_ny = (state_text == "NY").astype(int).to_numpy()
    is_ca = (state_text == "CA").astype(int).to_numpy()
    is_dc = (state_text == "DC").astype(int).to_numpy()

    x = np.column_stack([
        title_words,
        keyword_words,
        summary_words,
        has_city,
        has_state,
        has_summary,
        days_after_start,
        is_ny,
        is_ca,
        is_dc
    ])

    feature_names = [
        "title_words",
        "keyword_words",
        "summary_words",
        "has_city",
        "has_state",
        "has_summary",
        "days_after_start",
        "is_ny",
        "is_ca",
        "is_dc"
    ]

    return x.astype(float), feature_names


# -------------------------------------------------------------------
# PART 3: Train-test split and scaling
# -------------------------------------------------------------------
def train_test_split(x, y, test_size=0.2):
    """
    Shuffles and splits the rows into train/test.
    """
    indices = np.random.permutation(len(x))
    test_set_size = int(len(x) * test_size)

    test_idx = indices[:test_set_size]
    train_idx = indices[test_set_size:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def scale_train_test(x_train, x_test):
    """
    Scales using only the training data.
    This helps avoid data leakage.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    std[std == 0] = 1

    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std

    return x_train_scaled, x_test_scaled


# -------------------------------------------------------------------
# PART 4: Logistic Regression from scratch
# -------------------------------------------------------------------
def sigmoid(z):
    """
    Keeps sigmoid stable.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(x, y, lr=0.05, lmbda=0.0, epochs=1500):
    """
    Trains logistic regression with gradient descent.

    lmbda = 0 means baseline logistic regression.
    Later, lmbda can be changed to use L2 regularization.
    """
    weights = np.zeros(x.shape[1])
    bias = 0.0
    n = len(x)
    losses = []

    for epoch in range(epochs):
        z = np.dot(x, weights) + bias
        y_prob = sigmoid(z)

        error = y_prob - y

        dw = (1/n) * np.dot(x.T, error) + (lmbda * weights)
        db = (1/n) * np.sum(error)

        weights = weights - lr * dw
        bias = bias - lr * db

        if epoch % 100 == 0:
            loss = -np.mean(y * np.log(y_prob + 0.000001) +
                            (1 - y) * np.log(1 - y_prob + 0.000001))
            losses.append(loss)

    return weights, bias, losses


def get_metrics(y_true, y_prob):
    """
    Computes accuracy, precision, recall, and F1.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, (tp, tn, fp, fn)


# -------------------------------------------------------------------
# PART 5: Simple plots
# -------------------------------------------------------------------
def make_plots(data, y, losses):
    """
    Makes a couple quick plots for the project.
    These can be improved later.
    """
    plt.figure(figsize=(7, 4))
    data["state"].replace("", "Missing").value_counts().head(10).plot(kind="bar")
    plt.title("Top States in the Hate Crime Article Dataset")
    plt.xlabel("State")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    labels = ["Not violent label", "Violent label"]
    counts = [np.sum(y == 0), np.sum(y == 1)]
    plt.bar(labels, counts)
    plt.title("Starter Target Label Counts")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("Training Loss Every 100 Epochs")
    plt.xlabel("Checkpoint")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# PART 6: Run everything
# -------------------------------------------------------------------
data = load_hate_data()

if data is not None:
    print_basic_stats(data)

    x, feature_names = make_features(data)
    y = make_violent_label(data)

    print("\n--- Target Info ---")
    print("Not violent label:", np.sum(y == 0))
    print("Violent label:", np.sum(y == 1))

    print("\nFeatures used:")
    for name in feature_names:
        print("-", name)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_test = scale_train_test(x_train, x_test)

    print("\nTrain/Test sizes:")
    print("x_train:", x_train.shape, "x_test:", x_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)

    weights, bias, losses = train_logistic_regression(
        x_train,
        y_train,
        lr=0.05,
        lmbda=0.0,
        epochs=1500
    )

    test_probs = sigmoid(np.dot(x_test, weights) + bias)

    acc, prec, rec, f1, counts = get_metrics(y_test, test_probs)
    tp, tn, fp, fn = counts

    print("\n--- Logistic Regression Results ---")
    print("Accuracy: ", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:   ", round(rec, 4))
    print("F1 Score: ", round(f1, 4))

    print("\nConfusion matrix counts:")
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

    print("\nWeights:")
    for i in range(len(feature_names)):
        print(feature_names[i], ":", round(weights[i], 4))

    make_plots(data, y, losses)

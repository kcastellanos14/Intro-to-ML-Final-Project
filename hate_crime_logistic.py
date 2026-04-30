"""
COMP 2200 Final Project - starter file
Dataset: Documenting Hate data
 
This is still an early version.
It loads the hate crime data, makes some small features,
then trains a basic logistic regression model using numpy.
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
PREVIEW_ROWS = 9
 
 
# -------------------------------------------------------------------
# PART 1: Load and clean data
# -------------------------------------------------------------------
def load_hate_data():
    """
    Loads the CSV file and fixes the column names.
 
    The file header is kind of shifted, so these names match
    what the columns actually mean in the rows.
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
 
 
def get_keyword_counts(data):
    """
    Gets the keyword word counts.
    I made this separate so the print part and the graph use the same numbers.
    """
    keyword_text = " ".join(data["keywords"].tolist()).lower()

    for ch in [",", ".", ";", ":", "/", "\\", "(", ")", "[", "]", '"', "'"]:
        keyword_text = keyword_text.replace(ch, " ")

    skip_words = [
        "the", "and", "for", "with", "from", "that",
        "this", "are", "was", "were", "into", "after",
        "about", "over"
    ]

    words = []

    for word in keyword_text.split():
        if len(word) > 2 and word not in skip_words:
            words.append(word)

    return pd.Series(words).value_counts()


def print_basic_stats(data):
    """
    Prints simple dataset info so I can understand what I am working with.
    """
    n = data.shape[0]

    print("\n--- Dataset Info ---")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])
    print("Column names:")
    print(list(data.columns))

    missing_city    = (data["city"] == "").sum()
    missing_state   = (data["state"] == "").sum()
    missing_summary = (data["summary"] == "").sum()

    print("\nMissing city:   ", missing_city,   f"({round(100 * missing_city / n, 1)}%)")
    print("Missing state:  ", missing_state,   f"({round(100 * missing_state / n, 1)}%)")
    print("Missing summary:", missing_summary, f"({round(100 * missing_summary / n, 1)}%)")

    print("\nTop 10 states:")
    print(data["state"].replace("", "Missing").value_counts().head(10))

    print("\nTop 10 organizations:")
    print(data["organization"].replace("", "Missing").value_counts().head(10))

    valid_dates = data["article_date"].dropna()
    print("\nFirst article date:", valid_dates.min().date())
    print("Last article date: ", valid_dates.max().date())

    print("\nArticles per year:")
    year_counts = data["article_date"].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {int(year)}: {count} articles")

    print("\nTop 10 cities:")
    print(data["city"].replace("", "Missing").value_counts().head(10))

    print("\nTop 15 most common keyword words:")
    print(get_keyword_counts(data).head(15))


def print_data_preview(data):
    """
    Shows a few rows so we can check the dataset before modeling.
    This is mainly for us while we are still building the project.
    """
    print("\n--- Small Data Preview ---")
 
    cols = ["article_date", "title", "organization", "city", "state"]
    preview = data[cols].head(PREVIEW_ROWS)
 
    for i in range(len(preview)):
        print("\nRow", i + 1)
        print("Date:", preview.iloc[i]["article_date"])
        print("Title:", preview.iloc[i]["title"])
        print("Source:", preview.iloc[i]["organization"])
        print("Place:", preview.iloc[i]["city"], preview.iloc[i]["state"])
 
 
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
 
 
def print_label_examples(data, y):
    """
    Prints a few examples from each class.
    This helps us see if the starter label makes sense or not.
    """
    print("\n--- Example Rows By Label ---")
 
    for label in [0, 1]:
        if label == 0:
            print("\nExamples labeled 0 (not violent by our simple rule):")
        else:
            print("\nExamples labeled 1 (violent by our simple rule):")
 
        rows = np.where(y == label)[0][:3]
 
        for row in rows:
            print("-", data.iloc[row]["title"])
 
 
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
 
 
def print_feature_summary(x, feature_names):
    """
    Prints quick min, max, and average for each feature.
    It gives the teammate something easy to check next.
    """
    print("\n--- Feature Summary ---")
 
    for i in range(len(feature_names)):
        print(feature_names[i])
        print("  min:", round(np.min(x[:, i]), 3))
        print("  max:", round(np.max(x[:, i]), 3))
        print("  avg:", round(np.mean(x[:, i]), 3))
 
 
def print_violent_by_state(data, y):
    """
    Shows how many violent vs non-violent articles exist per state.
    Also shows the violent percentage for each state.
    Only shows states with at least 5 articles so the percentages are meaningful.
    """
    print("\n--- Violent Label Counts by State ---")
 
    states = data["state"].replace("", "Missing")
    unique_states = states.value_counts()
    qualifying_states = unique_states[unique_states >= 5].index
 
    print(f"{'State':<12} {'Total':>7} {'Violent':>9} {'Not Violent':>12} {'% Violent':>10}")
    print("-" * 55)
 
    for state in qualifying_states[:15]:
        mask = states == state
        total   = np.sum(mask)
        violent = np.sum(y[mask.to_numpy()] == 1)
        not_vio = total - violent
        pct     = round(100 * violent / total, 1)
        print(f"{state:<12} {total:>7} {violent:>9} {not_vio:>12} {pct:>9}%")
 
 
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
 
 
def print_split_info(x_train, x_test, y_train, y_test):
    """
    Shows the split size and the class counts inside each split.
    """
    total_rows = len(x_train) + len(x_test)
 
    print("\nTrain/Test sizes:")
    print("x_train:", x_train.shape, "x_test:", x_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
 
    print("\nSplit percentages:")
    print("Train:", round(len(x_train) / total_rows, 3))
    print("Test: ", round(len(x_test) / total_rows, 3))
 
    print("\nClass counts in train:")
    print("0:", np.sum(y_train == 0), "1:", np.sum(y_train == 1))
 
    print("\nClass counts in test:")
    print("0:", np.sum(y_test == 0), "1:", np.sum(y_test == 1))
 
 
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
 
    lmbda = 0.0 means baseline logistic regression (no regularization).
    lmbda > 0.0 adds L2 regularization, which penalizes large weights
    and helps prevent overfitting.
 
    The L2 term modifies the weight gradient:
        dw = (1/n) * X^T * error + lmbda * w
 
    Note: the bias is not regularized, which is standard practice.
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
 
 
def predict_probabilities(x, weights, bias):
    """
    Small helper so the prediction line is easier to read.
    """
    return sigmoid(np.dot(x, weights) + bias)
 
 
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
def plot_top_organizations(data, top_n=10):
    """
    Graphs the sources that show up the most in the dataset.
    This is the same info we print, but easier to use in the report.
    """
    counts = data["organization"].replace("", "Missing").value_counts().head(top_n)
    counts = counts.sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values)
    plt.title("Top 10 Organizations in the Dataset")
    plt.xlabel("Number of Articles")
    plt.ylabel("Organization")
    plt.tight_layout()
    plt.savefig("plot_top_organizations.png")
    plt.show()


def plot_top_keyword_words(data, top_n=15):
    """
    Graphs the most common keyword words.
    These words help show what topics show up a lot in the dataset.
    """
    counts = get_keyword_counts(data).head(top_n)
    counts = counts.sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values)
    plt.title("Top 15 Keyword Words")
    plt.xlabel("Number of Times Word Appears")
    plt.ylabel("Keyword Word")
    plt.tight_layout()
    plt.savefig("plot_top_keyword_words.png")
    plt.show()


def make_plots(data, y, losses_b, losses_e):
    """
    Makes plots for the project.
    """
    plt.figure(figsize=(7, 4))
    data["state"].replace("", "Missing").value_counts().head(10).plot(kind="bar")
    plt.title("Top States in the Hate Crime Article Dataset")
    plt.xlabel("State")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("plot_top_states.png")
    plt.show()

    plot_top_organizations(data, top_n=10)
    plot_top_keyword_words(data, top_n=15)

 
    plt.figure(figsize=(7, 4))
    labels = ["Not violent label", "Violent label"]
    counts = [np.sum(y == 0), np.sum(y == 1)]
    plt.bar(labels, counts, color=["steelblue", "tomato"])
    plt.title("Starter Target Label Counts")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("plot_label_counts.png")
    plt.show()
 
    plt.figure(figsize=(7, 4))
    plt.plot(losses_b, label="Baseline (λ=0.0)", marker="o", markersize=3)
    plt.plot(losses_e, label="Extended (λ=0.1)", marker="s", markersize=3)
    plt.title("Training Loss: Baseline vs L2 Regularization")
    plt.xlabel("Checkpoint (every 100 epochs)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_loss_curves.png")
    plt.show()
 
 
def plot_confusion_matrices(counts_base, counts_ext):
    """
    Side-by-side confusion matrices for baseline and extended model.
    Rows = true label, columns = predicted label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
    titles = ["Baseline (No L2)", "Extended (L2 Regularization)"]
    all_counts = [counts_base, counts_ext]
 
    for i, ax in enumerate(axes):
        tp, tn, fp, fn = all_counts[i]
        matrix = np.array([[tn, fp], [fn, tp]])
 
        ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.6)
 
        for (row, col), val in np.ndenumerate(matrix):
            ax.text(x=col, y=row, s=val, va='center', ha='center', size=14)
 
        ax.set_title(titles[i], pad=20)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0 (Not Violent)', '1 (Violent)'])
        ax.set_yticklabels(['0', '1'])
 
    plt.tight_layout()
    plt.savefig("plot_confusion_matrices.png")
    plt.show()
 
 
def plot_weight_comparison(weights_b, weights_e, feature_names):
    """
    Bar chart comparing feature weights for baseline vs L2 model.
    L2 regularization should shrink the weights toward zero.
    """
    x_pos = np.arange(len(feature_names))
    width = 0.35
 
    plt.figure(figsize=(10, 4))
    plt.bar(x_pos - width/2, weights_b, width, label="Baseline",      color="steelblue")
    plt.bar(x_pos + width/2, weights_e, width, label="Extended (L2)", color="tomato")
    plt.xticks(x_pos, feature_names, rotation=45, ha="right")
    plt.title("Feature Weights: Baseline vs L2 Regularization")
    plt.ylabel("Weight Value")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plot_weights.png")
    plt.show()
 
 
# -------------------------------------------------------------------
# PART 6: Model comparison and analysis
# -------------------------------------------------------------------
def compare_lambda_values(x_train, y_train, x_test, y_test):
    """
    Trains the model with several lambda values and prints a comparison table.
    This helps us pick the best lambda and understand how regularization
    strength affects the results.
    """
    lambdas = [0.0, 0.01, 0.1, 0.5, 1.0]
 
    print("\n--- Lambda Comparison ---")
    print(f"{'Lambda':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)
 
    for lmbda in lambdas:
        w, b, _ = train_logistic_regression(x_train, y_train, lr=0.05, lmbda=lmbda, epochs=1500)
        probs = predict_probabilities(x_test, w, b)
        acc, prec, rec, f1, _ = get_metrics(y_test, probs)
        print(f"{lmbda:<10} {acc:>10.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")
 
 
def print_weight_report(weights_b, weights_e, feature_names):
    """
    Prints the learned weights for both models side by side, sorted by
    the absolute value of the baseline weight from highest to lowest.
    This makes it easy to see which features matter most and how much
    L2 regularization shrank each weight.
    """
    print("\n--- Feature Weight Report (sorted by |baseline weight|) ---")
    print(f"{'Feature':<18} {'Baseline':>10} {'Extended (L2)':>14} {'Shrinkage':>10}")
    print("-" * 56)
 
    order = np.argsort(np.abs(weights_b))[::-1]
 
    for i in order:
        shrinkage = round(abs(weights_b[i]) - abs(weights_e[i]), 4)
        print(
            f"{feature_names[i]:<18} "
            f"{weights_b[i]:>10.4f} "
            f"{weights_e[i]:>14.4f} "
            f"{shrinkage:>10.4f}"
        )
 
 
# -------------------------------------------------------------------
# PART 7: Run everything
# -------------------------------------------------------------------
data = load_hate_data()
 
if data is not None:
    print_basic_stats(data)
    print_data_preview(data)
 
    x, feature_names = make_features(data)
    y = make_violent_label(data)
 
    print("\n--- Target Info ---")
    print("Not violent label:", np.sum(y == 0))
    print("Violent label:", np.sum(y == 1))
 
    print_label_examples(data, y)
 
    print("\nFeatures used:")
    for name in feature_names:
        print("-", name)
 
    print_feature_summary(x, feature_names)
    print_violent_by_state(data, y)
 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_test = scale_train_test(x_train, x_test)
 
    print_split_info(x_train, x_test, y_train, y_test)
 
    compare_lambda_values(x_train, y_train, x_test, y_test)
 
    print("\nTraining Baseline Model (λ=0.0)...")
    weights_b, bias_b, losses_b = train_logistic_regression(
        x_train,
        y_train,
        lr=0.05,
        lmbda=0.0,
        epochs=1500
    )
 
    print("Training Extended Model (λ=0.1)...")
    weights_e, bias_e, losses_e = train_logistic_regression(
        x_train,
        y_train,
        lr=0.05,
        lmbda=0.1,
        epochs=1500
    )
 
    probs_b = predict_probabilities(x_test, weights_b, bias_b)
    probs_e = predict_probabilities(x_test, weights_e, bias_e)
 
    acc_b, prec_b, rec_b, f1_b, counts_b = get_metrics(y_test, probs_b)
    acc_e, prec_e, rec_e, f1_e, counts_e = get_metrics(y_test, probs_e)
 
    print("\n--- Model Comparison ---")
    print(f"{'Metric':<12} | {'Baseline':<10} | {'Extended (L2)':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<12} | {acc_b:<10.4f} | {acc_e:<10.4f}")
    print(f"{'Precision':<12} | {prec_b:<10.4f} | {prec_e:<10.4f}")
    print(f"{'Recall':<12} | {rec_b:<10.4f} | {rec_e:<10.4f}")
    print(f"{'F1 Score':<12} | {f1_b:<10.4f} | {f1_e:<10.4f}")
 
    print("\nConfusion matrix counts (Baseline):")
    tp, tn, fp, fn = counts_b
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
 
    print("\nConfusion matrix counts (Extended):")
    tp, tn, fp, fn = counts_e
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
 
    print_weight_report(weights_b, weights_e, feature_names)
 
    make_plots(data, y, losses_b, losses_e)
    plot_confusion_matrices(counts_b, counts_e)
    plot_weight_comparison(weights_b, weights_e, feature_names)
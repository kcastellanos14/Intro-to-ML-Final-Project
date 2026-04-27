"""
COMP-2200: Decision Trees Homework
==================================

In this assignment, you will:
1. Implement Gini impurity and entropy calculations from scratch
2. Implement information gain to evaluate splits
3. Build a simple decision stump (single split)
4. Use sklearn's DecisionTreeClassifier to explore hyperparameters
5. Analyze feature importance and tree structure

Dataset: Iris (classic ML dataset for classification)

You may use NumPy for numerical operations.
You may use sklearn ONLY for DecisionTreeClassifier and data loading.
You must implement the impurity and split functions FROM SCRATCH.

NAME: Pablo Velasquez
DATE: 3/12/26
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


# =============================================================================
# PART 1: IMPURITY MEASURES
# =============================================================================

def gini_impurity(y):
    """
    Compute the Gini impurity of a set of labels.

    Gini(S) = 1 - sum(p_c^2) for each class c

    where p_c is the proportion of class c in the set.

    Parameters
    ----------
    y : np.ndarray
        Array of class labels (integers)

    Returns
    -------
    float
        Gini impurity (between 0 and 1 - 1/n_classes)
        0 means pure (all same class), higher means more mixed

    Examples
    --------
    >>> gini_impurity(np.array([0, 0, 0, 0]))  # Pure
    0.0
    >>> gini_impurity(np.array([0, 0, 1, 1]))  # Perfectly mixed (binary)
    0.5
    >>> gini_impurity(np.array([0, 0, 0, 1]))  # Mostly one class
    0.375
    """
    # handle an empty array
    if len(y) == 0:
        return 0.0

    # get class counts
    _, counts = np.unique(y, return_counts=True)

    # compute proportions
    proportions = counts / len(y)

    # Gini = 1 - sum(p^2)
    return 1.0 - np.sum(proportions ** 2)


def entropy(y):
    """
    Compute the entropy of a set of labels.

    Entropy(S) = -sum(p_c * log2(p_c)) for each class c

    where p_c is the proportion of class c in the set.

    Parameters
    ----------
    y : np.ndarray
        Array of class labels (integers)

    Returns
    -------
    float
        Entropy (between 0 and log2(n_classes))
        0 means pure, higher means more mixed

    Examples
    --------
    >>> entropy(np.array([0, 0, 0, 0]))  # Pure
    0.0
    >>> entropy(np.array([0, 0, 1, 1]))  # Perfectly mixed (binary)
    1.0
    >>> entropy(np.array([0, 0, 0, 1]))  # Mostly one class
    0.811...
    """
    # handle empty array
    if len(y) == 0:
        return 0.0
    
    # get class counts
    _, counts = np.unique(y, return_counts = True)

    # compute proportions
    proportions = counts / len(y)

    # entropy = -sum(p * log_2(p))
    return -np.sum(proportions * np.log2(proportions))


# =============================================================================
# PART 2: INFORMATION GAIN
# =============================================================================

def information_gain(y, left_mask, impurity_func=gini_impurity):
    """
    Compute the information gain from a split.

    Information Gain = Impurity(parent) - Weighted Average Impurity(children)

    Parameters
    ----------
    y : np.ndarray
        Array of class labels for the parent node
    left_mask : np.ndarray
        Boolean array indicating which samples go to the left child
    impurity_func : callable
        Function to compute impurity (gini_impurity or entropy)

    Returns
    -------
    float
        Information gain (higher is better)

    Examples
    --------
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> left_mask = np.array([True, True, True, False, False, False])  # Perfect split
    >>> information_gain(y, left_mask)  # Should be 0.5 (parent Gini) - 0 = 0.5
    0.5
    """
    # compute our parent impurity
    parent_impurity = impurity_func(y)

    # split into left and right
    y_left = y[left_mask]
    y_right = y[~left_mask]

    # handle edge cases
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    # calculate weighted average impurity of children
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    weighted_child_impurity = (
        (n_left / n) * impurity_func(y_left) +
        (n_right / n) * impurity_func(y_right)
    )

    # IG = impurity parent - impurity child
    return parent_impurity - weighted_child_impurity

# =============================================================================
# PART 3: FINDING THE BEST SPLIT
# =============================================================================

def find_best_split(X, y, feature_idx, impurity_func=gini_impurity):
    """
    Find the best split threshold for a single feature.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Class labels (n_samples,)
    feature_idx : int
        Index of the feature to split on
    impurity_func : callable
        Function to compute impurity

    Returns
    -------
    tuple (best_threshold, best_gain)
        best_threshold : float or None
            The threshold value that gives the best split
        best_gain : float
            The information gain from that split

    Notes
    -----
    For each unique value in the feature, try the midpoint between consecutive
    values as a potential threshold. Return the threshold with highest info gain.
    """
    # Get feature values
    feature_values = X[:, feature_idx]

    # get sorted unique values
    unique_values = np.unique(feature_values)

    # if only one unique value..... no split is possible
    if len(unique_values) <= 1:
        return None, 0.0
    
    # establish best thresh and gain vars
    best_threshold = None
    best_gain = 0.0

    # try midpoints b/w consecutive unique vals
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + 
                     unique_values[i + 1]) / 2
        
        # create split mask
        left_mask = feature_values <= threshold

        # skip if split results in an empty child
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            continue

        # compute IG
        gain = information_gain(y, left_mask, impurity_func)

        # update best if this is better
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    return best_threshold, best_gain

def find_best_feature_and_split(X, y, impurity_func=gini_impurity):
    """
    Find the best feature and threshold to split on.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Class labels (n_samples,)
    impurity_func : callable
        Function to compute impurity

    Returns
    -------
    tuple (best_feature, best_threshold, best_gain)
        best_feature : int
            Index of the best feature to split on
        best_threshold : float
            The threshold value for that feature
        best_gain : float
            The information gain from that split
    """
    # how many features?
    n_features = X.shape[1]

    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for feature_idx in range(n_features):
        threshold, gain = find_best_split(X, y,
                                          feature_idx,
                                          impurity_func)
        
        if threshold is not None and gain > best_gain:
            best_gain = gain
            best_feature = feature_idx
            best_threshold = threshold
    
    return best_feature, best_threshold, best_gain


# =============================================================================
# PART 4: DECISION STUMP
# =============================================================================

class DecisionStump:
    """
    A decision stump is a decision tree with only one split (depth=1).

    This is a simple classifier that makes a single yes/no decision.
    """

    def __init__(self, impurity_func=gini_impurity):
        """
        Initialize the decision stump.

        Parameters
        ----------
        impurity_func : callable
            Function to compute impurity (gini_impurity or entropy)
        """
        self.impurity_func = impurity_func
        self.feature_idx = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        """
        Fit the decision stump to the data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Class labels (n_samples,)

        Returns
        -------
        self
        """
        # find the best split
        self.feature_idx, self.threshold, _ = find_best_feature_and_split(X, y, 
                                                                          self.impurity_func)

        # get feature values
        feature_values = X[:, self.feature_idx]

        # split the data
        left_mask = feature_values <= self.threshold
        y_left = y[left_mask]
        y_right = y[~left_mask]

        # find majority class for each side
        self.left_class = np.argmax(
            np.bincount(y_left)
            )
        
        self.right_class = np.argmax(
            np.bincount(y_right)
        )

        return self

    def predict(self, X):
        """
        Make predictions using the fitted stump.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted class labels (n_samples,)
        """
        # grab feature values
        feature_values = X[:, self.feature_idx]

        # initialize our preds
        predictions = np.empty(
            len(X),
            dtype = int
        )

        # assign left or right class based on
        # threshold
        predictions[feature_values <= self.threshold] = self.left_class
        predictions[feature_values > self.threshold] = self.right_class

        return predictions


# =============================================================================
# PART 5: SKLEARN EXPLORATION
# =============================================================================

def explore_tree_depth(X_train, X_val, y_train, y_val):
    """
    Explore how tree depth affects training and validation accuracy.

    Train decision trees with max_depth from 1 to 10.
    Plot training accuracy and validation accuracy vs. depth.

    NOTE: We use a VALIDATION set here (not the test set) so we can
    pick the best depth without biasing our final test evaluation.
    The test set stays untouched until our very last evaluation.

    Returns
    -------
    tuple (depths, train_accs, val_accs)
    """
    depths = range(1, 11)
    train_accs = []
    val_accs = []

    for depth in depths:
        # train our tree at the current depth
        clf = DecisionTreeClassifier(max_depth = depth,
                                     random_state=42)
        
        clf.fit(X_train, y_train)

        # compute accuracies
        train_accs.append(clf.score(X_train, y_train))
        val_accs.append(clf.score(X_val, y_val))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs, 'b-o', label='Training Accuracy')
    plt.plot(depths, val_accs, 'r-o', label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Effect of Tree Depth on Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return depths, train_accs, val_accs


def visualize_tree(clf, feature_names, class_names, filename='tree_visualization.png'):
    """
    Visualize a fitted decision tree.

    Parameters
    ----------
    clf : DecisionTreeClassifier
        Fitted decision tree
    feature_names : list
        Names of the features
    class_names : list
        Names of the classes
    filename : str
        Output filename for the visualization
    """
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree Visualization')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_feature_importance(clf, feature_names):
    """
    Analyze and display feature importance from a fitted tree.

    Parameters
    ----------
    clf : DecisionTreeClassifier
        Fitted decision tree
    feature_names : list
        Names of the features
    """
    importances = clf.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("Feature Importance Ranking:")
    print("-" * 40)
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run all parts of the homework."""

    print("=" * 60)
    print("DECISION TREES HOMEWORK")
    print("=" * 60)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = list(iris.target_names)

    print(f"\nDataset: Iris")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]} - {feature_names}")
    print(f"Classes: {len(class_names)} - {class_names}")

    # -------------------------------------------------------------------------
    # Split data into train, validation, and test sets
    # -------------------------------------------------------------------------
    # WHY THREE SPLITS?
    # - Training set: used to fit the model
    # - Validation set: used to choose hyperparameters (like tree depth)
    # - Test set: used ONCE at the end for a fair, unbiased accuracy estimate
    #
    # If we used the test set to pick the best depth AND to report accuracy,
    # we'd be "peeking" at the test data during model selection, giving us
    # an overly optimistic estimate of how well we'd do on truly unseen data.
    # -------------------------------------------------------------------------

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y 
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25,
        random_state=42, stratify=y_trainval
    )    

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # -------------------------------------------------------------------------
    # PART 1: Test impurity functions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 1: IMPURITY MEASURES")
    print("=" * 60)

    # Test cases
    pure = np.array([0, 0, 0, 0])
    mixed_binary = np.array([0, 0, 1, 1])
    mixed_ternary = np.array([0, 0, 1, 1, 2, 2])

    print("\nGini Impurity Tests:")
    print(f"  Pure [0,0,0,0]: {gini_impurity(pure):.4f} (expected: 0.0)")
    print(f"  Mixed binary [0,0,1,1]: {gini_impurity(mixed_binary):.4f} (expected: 0.5)")
    print(f"  Mixed ternary [0,0,1,1,2,2]: {gini_impurity(mixed_ternary):.4f} (expected: 0.6667)")

    print("\nEntropy Tests:")
    print(f"  Pure [0,0,0,0]: {entropy(pure):.4f} (expected: 0.0)")
    print(f"  Mixed binary [0,0,1,1]: {entropy(mixed_binary):.4f} (expected: 1.0)")
    print(f"  Mixed ternary [0,0,1,1,2,2]: {entropy(mixed_ternary):.4f} (expected: 1.585)")

    # -------------------------------------------------------------------------
    # PART 2: Test information gain
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 2: INFORMATION GAIN")
    print("=" * 60)

    # perfect split test
    y_test_split = np.array([0,0,0,1,1,1])
    perfect_mask = np.array([True, True, True, 
                             False, False, False])
    
    bad_mask = np.array([True, False, True, False,
                         True, False])

    print("\nInformation Gain Tests (using Gini):")
    print(f"  Perfect split: {information_gain(y_test_split, perfect_mask):.4f} (expected: 0.5)")
    print(f"  Bad split: {information_gain(y_test_split, bad_mask):.4f} (expected: 0.0556)")

    # -------------------------------------------------------------------------
    # PART 3: Test finding best split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 3: FINDING BEST SPLIT")
    print("=" * 60)

    for i, name in enumerate(feature_names):
        threshold, gain = find_best_split(
            X_train, y_train, i   
        )
        print(f'{name}: Threshold={threshold:.2f}, gain = {gain:.4f}')
    
    best_feat, best_thresh, best_gain = find_best_feature_and_split(
        X_train, y_train
    )

    print(f'Best feature: {feature_names[best_feat]}')
    print(f'Best threshold: {best_thresh:.2f}')
    print(f'Best Information gain: {best_gain:.4f}')

    # -------------------------------------------------------------------------
    # PART 4: Test decision stump
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 4: DECISION STUMP")
    print("=" * 60)

    stump = DecisionStump()
    stump.fit(X_train, y_train)

    print(f"\nDecision Stump trained:")
    print(f"  Split feature: {feature_names[stump.feature_idx]}")
    print(f"  Split threshold: {stump.threshold:.2f}")
    print(f"  Left class: {class_names[stump.left_class]}")
    print(f"  Right class: {class_names[stump.right_class]}")

    y_pred_train = stump.predict(X_train)
    y_pred_test = stump.predict(X_test)
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)

    print(f"\n  Training accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # PART 5: Sklearn exploration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 5: SKLEARN EXPLORATION")
    print("=" * 60)

    # Explore depth effect (using validation set to pick best depth)
    print("\nAnalyzing effect of tree depth...")
    depths, train_accs, val_accs = explore_tree_depth(X_train, X_val, y_train, y_val)

    # Find optimal depth using VALIDATION accuracy (not test!)
    best_depth = depths[np.argmax(val_accs)]
    print(f"\nBest depth based on validation accuracy: {best_depth}")

    # Train final tree with optimal depth
    print(f"\nTraining tree with max_depth={best_depth}...")
    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    clf.fit(X_train, y_train)

    print(f"  Training accuracy:   {clf.score(X_train, y_train):.4f}")
    print(f"  Validation accuracy: {clf.score(X_val, y_val):.4f}")
    print(f"  Test accuracy:       {clf.score(X_test, y_test):.4f}")
    print("  (Test accuracy is our unbiased estimate of real-world performance)")

    # Visualize tree
    print("\nVisualizing tree...")
    visualize_tree(clf, feature_names, class_names)

    # Feature importance
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(clf, feature_names)

    # -------------------------------------------------------------------------
    # ANALYSIS QUESTIONS (Answer in comments or separate document)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS QUESTIONS")
    print("=" * 60)
    print("""
    Answer the following questions based on your results:

1. Impurity Measures (1 point)
Compare Gini impurity and Entropy. When might you prefer one over the other? Do they
give similar results in practice?
          *I saw that Gini and entropy give almost the same result in this homework. 
          *Both of them measure how mixed the classes are, 
          *but Gini was easier for me to work with because the formula is more simple. 
          *If you want something faster and more direct, I think Gini is better

2. Overfitting (1 point)
Based on your depth analysis plot, at what depth does overfitting begin to appear? How can
you tell from the training vs. validation accuracy curves?
          *I noticed overfitting starts around depth 4.
          *At this point the training accuracy became 1.0000, 
          *but the validation accuracy stayed around 0.9333 and 
          *did not really improve more. 
          That showed me the tree was fitting the training data too much

3. Feature Importance (1 point)
Which features are most important for classifying Iris species according to your tree? Does
this make intuitive sense given what you know about the dataset?
          *I saw that petal length and petal width were the most 
          *important features. The values were about 0.5156 for petal length 
          *and 0.4677 for petal width, while sepal width was 0.0000 
          *and sepal length was very small. 
          *To me this makes sense because the petal measurements separate the Iris flowers better

4. Stump vs. Full Tree (1 point)
Compare the accuracy of your decision stump to the full sklearn tree. What does this tell
you about how separable the Iris classes are?
          *I got about 0.6667 accuracy with the decision stump, 
          *but the full tree got about 0.9333 on the test set. 
          *That is a big difference, so one split was not enough 
          *for this dataset

5. Interpretability (1 point)
Look at your tree visualization. Could you explain the tree’s decision-making process to
someone without ML knowledge? Write a 2–3 sentence explanation of how the tree classifies
an Iris flower.
          *I would say the tree looks at one flower measurement, 
          *like petal length or petal width, 
          *and then it makes a small decision based on a cutoff value. 
          *After doing this a few times, it puts the flower into one 
          *of the three classes

6. Data Splitting Strategy (1 point)
Why did we use THREE data splits (train/validation/test) instead of just two (train/test)?
What could go wrong if we used the test set to choose our tree depth?
          *We used three splits because each split had its own job. 
          *The training set was for fitting the model, 
          *the validation set was for choosing the best depth, 
          *and the test set was for the final evaluation. 
          *If I used the test set to choose the depth too, 
          *then the test score would not be a fair score anymore. 
          *It would look better than it really is, 
          *because I already used that data

    """)

    print("\n" + "=" * 60)
    print("HOMEWORK COMPLETE")
    print("=" * 60)
    print("\nFiles generated:")
    print("  - depth_analysis.png")
    print("  - tree_visualization.png")
    print("  - feature_importance.png")


if __name__ == "__main__":
    main()
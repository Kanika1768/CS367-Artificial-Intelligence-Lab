
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Decision Tree
class MyDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _best_split(self, X, y):
        """Find the feature and threshold with the smallest weighted Gini impurity."""
        n_samples, n_features = X.shape
        count = n_features if self.max_features is None else min(self.max_features, n_features)
        selected_features = self.rng.choice(n_features, size=count, replace=False)
        best_score = float("inf")
        best_feature, best_threshold = None, None

        for feature in selected_features:
            # Sort one feature
            order = np.argsort(X[:, feature])
            values = X[order, feature]
            labels = y[order]  # Classes are 0 and 1
            left_size = np.arange(1, n_samples)
            right_size = n_samples - left_size
            left_ones = np.cumsum(labels)[:-1]
            right_ones = labels.sum() - left_ones

            # For binary classification, Gini = 2 * p * (1 - p)
            p_left = left_ones / left_size
            p_right = right_ones / right_size
            gini_left = 2 * p_left * (1 - p_left)
            gini_right = 2 * p_right * (1 - p_right)
            scores = (left_size * gini_left + right_size * gini_right) / n_samples

            # Cannot split between two identical values
            scores[values[:-1] == values[1:]] = np.inf
            index = np.argmin(scores)
            if scores[index] < best_score:
                best_score = scores[index]
                best_feature = feature
                best_threshold = (values[index] + values[index + 1]) / 2

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        # Leaf stores the majority class (0 or 1)
        majority_class = int(np.bincount(y).argmax())
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return majority_class

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return majority_class

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return (feature, threshold, left, right)

    def _predict_one(self, row, node):
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if row[feature] <= threshold else right
        return node

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])


# Random Forest
class MyRandomForest:
    def __init__(self, n_estimators=15, max_depth=5, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        max_features = max(1, int(np.sqrt(X.shape[1])))

        for i in range(self.n_estimators):
            # Bootstrap: sample training rows randomly with replacement.
            indices = rng.choice(len(X), size=len(X), replace=True)
            tree = MyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        # Majority vote across all trees (binary class labels 0/1)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return (predictions.mean(axis=0) > 0.5).astype(int)


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("DATASET: Breast Cancer Wisconsin")
print("Samples:", len(X), "| Features:", X.shape[1], "| Classes:", len(np.unique(y)))
print("Training samples:", len(X_train), "| Testing samples:", len(X_test))


# Custom vs library models
models = [
    ("Decision Tree (Scratch)", MyDecisionTree(max_depth=5, min_samples_split=5)),
    ("Decision Tree (Library)", DecisionTreeClassifier(
        max_depth=5, min_samples_split=5, random_state=42)),
    ("Random Forest (Scratch)", MyRandomForest(
        n_estimators=15, max_depth=5, min_samples_split=5)),
    ("Random Forest (Library)", RandomForestClassifier(
        n_estimators=15, max_depth=5, min_samples_split=5,
        max_features="sqrt", random_state=42)),
]

print("\n{:<26} {:>10} {:>10} {:>10} {:>10}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"))
print("-" * 74)

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        name,
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        recall_score(y_test, predictions),
        f1_score(y_test, predictions),
    ))
    print("Confusion matrix:", confusion_matrix(y_test, predictions).tolist())

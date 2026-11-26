"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    _, counts = np.unique(labels, return_counts=True)

    if len(counts) == 0:
        return 0.0

    probs = counts / len(labels)

    return 1 - np.sum(probs ** 2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    left_mask = features[:, feature_index] <= threshold

    left_features = features[left_mask]
    right_features = features[~left_mask]
    left_labels = labels[left_mask]
    right_labels = labels[~left_mask]

    return (
        left_features,
        left_labels,
        right_features,
        right_labels
    )

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    P, L, R = len(parent_labels), len(left_labels), len(right_labels)

    gain = impurity(parent_labels) - ((L / P) * impurity(left_labels) + (R / P) * impurity(right_labels))

    return gain

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    feature_index = None
    threshold = None 
    score = 0.0

    for idx in feature_indices:
        values = np.unique(features[:, idx])
        for val in values:
            left_features, left_labels, right_features, right_labels = split_dataset(
                features, labels, idx, val
            )

            curr_gain = split_score(labels, left_labels, right_labels)
            if curr_gain > score:
                score = curr_gain
                threshold = val
                feature_index = idx 

    return {
        'feature_index' : feature_index,
        'threshold' : threshold,
        'score' : score
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    unq_vals = np.unique(labels)

    if len(unq_vals) == 1:
        return True
    if depth >= max_depth:
        return True 
    if len(labels) <= min_samples_split:
        return True 

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    unq_labels, cnt = np.unique(labels, return_counts=True)

    return int(unq_labels[-1])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    if should_stop(labels, depth, max_depth, min_samples_split):
        prediction = leaf_prediction(labels)

        return {
            'leaf' : True,
            'prediction' : prediction
        }

    feature_subset = np.arange(features.shape[1]) if feature_subset is None else feature_subset

    feature_index, threshold, _ = best_split(features, labels, feature_subset).values()

    left_features, left_labels, right_features, right_labels = split_dataset(
        features, labels, feature_index, threshold
    )

    return {
        'leaf' : False,
        'feature_index' : feature_index,
        'threshold' : threshold,
        'left' : build_tree(left_features, left_labels, max_depth, min_samples_split, feature_subset, depth + 1),
        'right' : build_tree(right_features, right_labels, max_depth, min_samples_split, feature_subset, depth + 1)
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    if tree['leaf']:
        return tree['prediction']

    if example[tree['feature_index']] <= tree['threshold']:
        return predict_example_tree(tree['left'], example)

    return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    preds = []
    for idx in range(features.shape[0]):
        pred = predict_example_tree(tree, features[idx, :])
        preds.append(pred)

    return np.asarray(preds)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    bootstrap_indices = rng.integers(low=0, high=features.shape[0], size=features.shape[0])

    return (
        features[bootstrap_indices],
        labels[bootstrap_indices]
    )

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(np.arange(num_features), num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    n_samples, n_features = features.shape
    rng = np.random.default_rng(random_state)

    forest = []
    for _ in range(num_trees):
        curr_features, curr_labels = bootstrap_sample(features, labels, rng)

        num_features_per_split = max(1, int(np.sqrt(n_features))) if num_features_per_split is None else num_features_per_split

        curr_feature_subset = feature_subset(n_features, num_features_per_split, rng)

        tree = build_tree(
            curr_features,
            curr_labels,
            max_depth,
            min_samples_split,
            curr_feature_subset,
        )

        forest.append(
            {
                'tree' : tree,
                'feature_indices' : curr_feature_subset
            }
        )

    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    return np.array([np.bincount(tree_predictions[:, j]).argmax() for j in range(tree_predictions.shape[1])])

# Step 14 - predict_forest
def predict_forest(forest, features):
    predict = []
    for tree in forest:
        fitted_tree = tree['tree']
        feature_indices = tree['feature_indices']

        pred = predict_tree(fitted_tree, features)
        predict.append(pred)

    predict = np.asarray(predict)

    return combine_predictions(predict)

# Step 15 - accuracy
def accuracy(predictions, labels):
    return np.mean(predictions == labels)


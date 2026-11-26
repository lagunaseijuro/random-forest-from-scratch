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

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement


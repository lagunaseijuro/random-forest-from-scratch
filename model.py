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

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

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


"""
data/data_splitter.py

Splits the Data Matrix D (400 × 10304) and label vector y into
training and test sets following the assignment rule:

    • Odd  rows (1-indexed) → training   (200 samples, 5 per subject)
    • Even rows (1-indexed) → test       (200 samples, 5 per subject)

Because Python uses 0-based indexing, "odd rows (1-indexed)" correspond
to indices 0, 2, 4, … (even Python indices), and "even rows (1-indexed)"
correspond to indices 1, 3, 5, … (odd Python indices).
"""

import numpy as np


def split_train_test(
    D: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    D : np.ndarray, shape (400, 10304)
        Full data matrix — every row is one flattened face image.
    y : np.ndarray, shape (400,)
        Integer subject labels 1..40.

    Returns
    -------
    X_train : np.ndarray, shape (200, 10304)
    X_test  : np.ndarray, shape (200, 10304)
    y_train : np.ndarray, shape (200,)
    y_test  : np.ndarray, shape (200,)
    """
    # 1-indexed odd  → 0-indexed even  → [0, 2, 4, …, 398]
    train_idx = np.arange(0, len(D), 2)   # rows 1,3,5,…  (1-indexed)
    # 1-indexed even → 0-indexed odd   → [1, 3, 5, …, 399]
    test_idx  = np.arange(1, len(D), 2)   # rows 2,4,6,…  (1-indexed)

    X_train, y_train = D[train_idx], y[train_idx]
    X_test,  y_test  = D[test_idx],  y[test_idx]

    # Sanity checks
    assert X_train.shape == (200, D.shape[1]), \
        f"Unexpected training set shape: {X_train.shape}"
    assert X_test.shape  == (200, D.shape[1]), \
        f"Unexpected test set shape: {X_test.shape}"
    assert len(np.unique(y_train)) == 40, \
        "Training set does not contain all 40 subjects."
    assert len(np.unique(y_test))  == 40, \
        "Test set does not contain all 40 subjects."

    return X_train, X_test, y_train, y_test

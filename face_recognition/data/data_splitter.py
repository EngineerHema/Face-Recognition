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
    train_idx = np.arange(0, len(D), 2)
    test_idx  = np.arange(1, len(D), 2)

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

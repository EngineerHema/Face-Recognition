"""
pca/pca_model.py

From-scratch PCA implementation that:
    1. Mean-centres the training data.
    2. Computes the covariance matrix (using the compact trick for high-dim data).
    3. Finds eigenvalues / eigenvectors via np.linalg.eigh.
    4. Selects the minimum number of components that retain ≥ α of total variance.
    5. Caches eigenvalues to disk (as suggested in the assignment hint).
    6. Projects new data into the reduced subspace.
"""

import os
import numpy as np

from utils.config import PCA_CACHE, OUTPUT_DIR


class PCAModel:
    """
    Parameters
    ----------
    variance_threshold : float
        α ∈ {0.80, 0.85, 0.90, 0.95} — fraction of variance to retain.
    random_seed : int
        Kept for API consistency (eigh is deterministic, but stored for reproducibility).
    """

    def __init__(self, variance_threshold: float = 0.95, random_seed: int = 42):
        self.variance_threshold = variance_threshold
        self.random_seed        = random_seed

        # Set after fit()
        self.mean_         : np.ndarray | None = None   # (n_features,)
        self.components_   : np.ndarray | None = None   # (n_components, n_features)
        self.eigenvalues_  : np.ndarray | None = None   # (n_components,)
        self.explained_var_ratio_: np.ndarray | None = None
        self.n_components_ : int | None = None

    # ── fit ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "PCAModel":
        """
        Fit PCA on training data X of shape (n_samples, n_features).

        Strategy (compact-covariance trick)
        ------------------------------------
        When n_features >> n_samples (here 10304 >> 200):
            • Form L = X_c @ X_c.T  of shape (n_samples × n_samples).
            • Compute eigenvectors of L  →  map back to the n_features space.
        This avoids an expensive 10304×10304 covariance matrix.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.random.seed(self.random_seed)

        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)                          # (n_features,)
        X_c = X - self.mean_                                 # centred

        # ── Eigendecomposition ────────────────────────────────────────
        if os.path.isfile(PCA_CACHE):
            # Load cached eigenvalues; recompute eigenvectors
            all_eigenvalues = np.load(PCA_CACHE)
            L = (X_c @ X_c.T) / (n_samples - 1)
            _, V_small = np.linalg.eigh(L)
        else:
            L = (X_c @ X_c.T) / (n_samples - 1)             # (n_samples, n_samples)
            all_eigenvalues, V_small = np.linalg.eigh(L)     # ascending order
            np.save(PCA_CACHE, all_eigenvalues)

        # eigh returns ascending → reverse to descending
        idx            = np.argsort(all_eigenvalues)[::-1]
        all_eigenvalues = all_eigenvalues[idx]
        V_small         = V_small[:, idx]

        # ── Select components that retain ≥ α variance ────────────────
        total_var      = all_eigenvalues.sum()
        cumulative_var = np.cumsum(all_eigenvalues) / total_var
        self.n_components_ = int(np.searchsorted(cumulative_var, self.variance_threshold) + 1)

        # Map small eigenvectors back to original feature space
        # u_i = X_c.T @ v_i  (then normalise)
        V_reduced     = V_small[:, : self.n_components_]      # (n_samples, k)
        U             = X_c.T @ V_reduced                     # (n_features, k)
        norms         = np.linalg.norm(U, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        U            /= norms                                  # unit vectors

        self.components_          = U.T                        # (k, n_features)
        self.eigenvalues_         = all_eigenvalues[: self.n_components_]
        self.explained_var_ratio_ = self.eigenvalues_ / total_var

        return self

    # ── transform ─────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X (n_samples, n_features) → (n_samples, n_components)."""
        assert self.mean_ is not None, "Call fit() before transform()."
        X_c = X - self.mean_
        return X_c @ self.components_.T          # (n_samples, k)

    # ── inverse_transform ────────────────────────────────────────────

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Reconstruct approximate faces from PCA coefficients."""
        assert self.mean_ is not None, "Call fit() before inverse_transform()."
        return X_pca @ self.components_ + self.mean_

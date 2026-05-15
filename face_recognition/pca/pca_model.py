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

        self.mean_         : np.ndarray | None = None
        self.components_   : np.ndarray | None = None
        self.eigenvalues_  : np.ndarray | None = None
        self.explained_var_ratio_: np.ndarray | None = None
        self.n_components_ : int | None = None


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
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        if os.path.isfile(PCA_CACHE):
            # Load cached eigenvalues; recompute eigenvectors
            all_eigenvalues = np.load(PCA_CACHE)
            L = (X_c @ X_c.T) / (n_samples - 1)
            _, V_small = np.linalg.eigh(L)
        else:
            L = (X_c @ X_c.T) / (n_samples - 1)
            all_eigenvalues, V_small = np.linalg.eigh(L)     # ascending order
            np.save(PCA_CACHE, all_eigenvalues)

        idx            = np.argsort(all_eigenvalues)[::-1]
        all_eigenvalues = all_eigenvalues[idx]
        V_small         = V_small[:, idx]

        total_var      = all_eigenvalues.sum()
        cumulative_var = np.cumsum(all_eigenvalues) / total_var
        self.n_components_ = int(np.searchsorted(cumulative_var, self.variance_threshold) + 1)

        V_reduced     = V_small[:, : self.n_components_]
        U             = X_c.T @ V_reduced
        norms         = np.linalg.norm(U, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        U            /= norms

        self.components_          = U.T
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

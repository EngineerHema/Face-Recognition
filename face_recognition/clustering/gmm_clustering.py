import numpy as np


class GMMClustering:
    """
    Gaussian Mixture Model via EM algorithm.

    Uses a single tied covariance matrix (all components share one covariance)
    with K-Means warm-start initialisation.

    Parameters
    ----------
    k           : number of mixture components
    max_iters   : maximum EM iterations
    tolerance   : convergence threshold on log-likelihood change
    random_seed : for reproducibility
    reg_covar   : ridge added to covariance for numerical stability
    """

    def __init__(self, k, max_iters=200, tolerance=1e-3, random_seed=42, reg_covar=1e-3):
        self.k           = k
        self.max_iters   = max_iters
        self.tolerance   = tolerance
        self.random_seed = random_seed
        self.reg_covar   = reg_covar

        self.means_      = None   # (k, d)
        self.covariance_ = None   # (d, d)  — tied
        self.pi_         = None   # (k,)

    # ── K-Means warm start ────────────────────────────────────────────

    def _kmeans_init(self, X: np.ndarray) -> np.ndarray:
        """Run plain K-Means and return final labels."""
        rng       = np.random.default_rng(self.random_seed)
        centroids = X[rng.choice(X.shape[0], self.k, replace=False)].copy()

        for _ in range(100):
            dists  = np.linalg.norm(X[:, None] - centroids[None], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c  = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j)
                else X[rng.integers(X.shape[0])]
                for j in range(self.k)
            ])
            if np.linalg.norm(new_c - centroids) < 1e-6:
                break
            centroids = new_c

        self.means_ = centroids
        return labels

    # ── EM steps ──────────────────────────────────────────────────────

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Return responsibilities (n, k)."""
        n, d         = X.shape
        _, logdet    = np.linalg.slogdet(self.covariance_)
        inv_cov      = np.linalg.inv(self.covariance_)
        diff         = X[:, None, :] - self.means_[None, :, :]   # (n, k, d)
        maha         = np.einsum('nkd,de,nke->nk', diff, inv_cov, diff)
        log_prob     = np.log(self.pi_ + 1e-12) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)

        # Numerically stable softmax
        log_prob    -= log_prob.max(axis=1, keepdims=True)
        resp         = np.exp(log_prob)
        resp        /= resp.sum(axis=1, keepdims=True) + 1e-12
        return resp

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Update means, tied covariance, and mixing weights in-place."""
        n, d = X.shape
        Nk   = resp.sum(axis=0)                          # (k,)

        self.means_ = (resp.T @ X) / Nk[:, None]        # (k, d)

        # Tied covariance: pool residuals across all components
        cov = np.zeros((d, d))
        for j in range(self.k):
            diff  = X - self.means_[j]                  # (n, d)
            cov  += (resp[:, j:j+1] * diff).T @ diff
        self.covariance_ = cov / n + self.reg_covar * np.eye(d)

        self.pi_ = np.clip(Nk / n, 1e-6, None)
        self.pi_ /= self.pi_.sum()

    def _log_likelihood(self, X: np.ndarray) -> float:
        n, d         = X.shape
        _, logdet    = np.linalg.slogdet(self.covariance_)
        inv_cov      = np.linalg.inv(self.covariance_)
        diff         = X[:, None, :] - self.means_[None, :, :]
        maha         = np.einsum('nkd,de,nke->nk', diff, inv_cov, diff)
        log_prob     = np.log(self.pi_ + 1e-12) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
        max_lp       = log_prob.max(axis=1, keepdims=True)
        ll           = (max_lp.squeeze() + np.log(np.exp(log_prob - max_lp).sum(axis=1) + 1e-12))
        return float(ll.sum())

    # ── Public API ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GMMClustering":
        n, d = X.shape

        # Initialise from K-Means hard assignments
        labels           = self._kmeans_init(X)
        resp             = np.zeros((n, self.k))
        resp[np.arange(n), labels] = 1.0

        # Bootstrap covariance and pi from hard assignments
        self._m_step(X, resp)

        prev_ll = -np.inf
        for _ in range(self.max_iters):
            resp    = self._e_step(X)
            self._m_step(X, resp)
            ll      = self._log_likelihood(X)
            if abs(ll - prev_ll) < self.tolerance:
                break
            prev_ll = ll

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._e_step(X), axis=1)
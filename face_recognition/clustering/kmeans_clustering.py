import numpy as np


class KMeansClustering:
    def __init__(self, k, max_iters=100, random_seed=None, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.tolerance = tolerance
        self.centroids = None

    def fit(self, X):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # select random k data points from X as initial centroids
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices].copy()

        for i in range(self.max_iters):
            labels = self.predict(X)

            new_centroids = np.zeros_like(self.centroids)

            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    # empty clusters are initialized to a random point
                    new_centroids[j] = X[np.random.choice(X.shape[0])]

            centroid_shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tolerance:  # check convergence
                break

        return self

    def predict(self, X):
        # Resulting distances shape: (N, K)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Return the index of the closest centroid for each data point
        return np.argmin(distances, axis=1)

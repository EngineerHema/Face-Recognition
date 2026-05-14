"""
utils/visualizer.py

All plotting helpers for the Face Recognition pipeline:
    • Eigenfaces grid
    • Cumulative explained-variance curve
    • Accuracy vs K  (one line per alpha)
    • Accuracy vs alpha  (one line per K)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import OUTPUT_DIR, IMAGE_HEIGHT, IMAGE_WIDTH


class Visualizer:

    # ── PCA ───────────────────────────────────────────────────────────

    @staticmethod
    def plot_eigenfaces(pca_model, n_faces: int = 10) -> None:
        """Show the first n_faces principal components as face images."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, n_faces, figsize=(2 * n_faces, 2.5))
        for i, ax in enumerate(axes):
            eigenface = pca_model.components_[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            ax.imshow(eigenface, cmap="gray")
            ax.set_title(f"PC {i+1}", fontsize=8)
            ax.axis("off")
        fig.suptitle("Eigenfaces (top principal components)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "eigenfaces.png"), dpi=120)
        plt.close()

    @staticmethod
    def plot_variance_explained(pca_model) -> None:
        """Plot cumulative variance explained vs number of components."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cumvar = np.cumsum(pca_model.explained_var_ratio_)
        plt.figure(figsize=(7, 4))
        plt.plot(cumvar, linewidth=2)
        plt.axhline(pca_model.variance_threshold, color="red", linestyle="--",
                    label=f"α = {pca_model.variance_threshold}")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative variance explained")
        plt.title("PCA – Cumulative Explained Variance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance_explained.png"), dpi=120)
        plt.close()

    # ── Clustering ────────────────────────────────────────────────────

    @staticmethod
    def plot_accuracy_vs_k(
        results   : dict,
        model_name: str,
        alpha_values: list[float],
        k_values    : list[int],
    ) -> None:
        """One curve per alpha, x-axis = K."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.figure(figsize=(7, 4))
        for alpha in alpha_values:
            accs = [results[(alpha, k)].train_accuracy_ for k in k_values]
            plt.plot(k_values, accs, marker="o", label=f"α={alpha}")
        plt.xlabel("K (number of clusters)")
        plt.ylabel("Clustering accuracy (train)")
        plt.title(f"{model_name} – Accuracy vs K")
        plt.legend()
        plt.tight_layout()
        safe = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe}_accuracy_vs_k.png"), dpi=120)
        plt.close()

    @staticmethod
    def plot_accuracy_vs_alpha(
        results   : dict,
        model_name: str,
        alpha_values: list[float],
        k_values    : list[int],
    ) -> None:
        """One curve per K, x-axis = alpha."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.figure(figsize=(7, 4))
        for k in k_values:
            accs = [results[(alpha, k)].train_accuracy_ for alpha in alpha_values]
            plt.plot(alpha_values, accs, marker="s", label=f"K={k}")
        plt.xlabel("α (variance threshold)")
        plt.ylabel("Clustering accuracy (train)")
        plt.title(f"{model_name} – Accuracy vs α")
        plt.legend()
        plt.tight_layout()
        safe = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe}_accuracy_vs_alpha.png"), dpi=120)
        plt.close()

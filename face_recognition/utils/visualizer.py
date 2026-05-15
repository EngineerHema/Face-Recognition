import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import OUTPUT_DIR, IMAGE_HEIGHT, IMAGE_WIDTH


class Visualizer:

    # ── PCA ───────────────────────────────────────────────────────────

    @staticmethod
    def plot_eigenfaces(pca_model, n_faces: int = 10, alpha_value: int = 0) -> None:
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"eigenfaces_{alpha_value}.png"), dpi=120)
        plt.close()

    @staticmethod
    def plot_variance_explained(pca_model, alpha_value: int = 0) -> None:
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"pca_variance_explained_{alpha_value}.png"), dpi=120)
        plt.close()

    @staticmethod
    def plot_transformed_faces(faces, alpha_value: int = 0) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        n_faces = len(faces)
        fig, axes = plt.subplots(1, n_faces, figsize=(2 * n_faces, 2.5))
        for i, ax in enumerate(axes):
            face = faces[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            ax.imshow(face, cmap="gray")
            ax.set_title(f"PC {i + 1}", fontsize=8)
            ax.axis("off")
        fig.suptitle("Transformed faces")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"transformed_faces_{alpha_value}.png"), dpi=120)
        plt.close()
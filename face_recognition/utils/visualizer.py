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
        plt.axhline(
            pca_model.variance_threshold,
            color="red",
            linestyle="--",
            label=f"α = {pca_model.variance_threshold}",
        )
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative variance explained")
        plt.title("PCA – Cumulative Explained Variance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"pca_variance_explained_{alpha_value}.png"),
            dpi=120,
        )
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
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"transformed_faces_{alpha_value}.png"), dpi=120
        )
        plt.close()

    # ── Autoencoder ───────────────────────────────────────────────────

    @staticmethod
    def plot_ae_reconstructions(
        X_original: np.ndarray,
        X_reconstructed: np.ndarray,
        n_faces: int = 8,
    ) -> None:
        """
        Show original faces (top row) vs autoencoder reconstructions (bottom row).

        Parameters
        ----------
        X_original      : (n_samples, 10304)  raw pixel values
        X_reconstructed : (n_samples, 10304)  decoded pixel values
        n_faces         : how many samples to display
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.random.seed(42)
        indices = np.random.choice(len(X_original), size=n_faces, replace=False)

        fig, axes = plt.subplots(2, n_faces, figsize=(2 * n_faces, 5))
        for col, idx in enumerate(indices):
            # original
            axes[0, col].imshow(
                X_original[idx].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap="gray"
            )
            axes[0, col].set_title(f"#{idx}", fontsize=8)
            axes[0, col].axis("off")

            # reconstruction
            axes[1, col].imshow(
                np.clip(X_reconstructed[idx], 0, 255).reshape(
                    IMAGE_HEIGHT, IMAGE_WIDTH
                ),
                cmap="gray",
            )
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=9)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
        fig.suptitle("Autoencoder — original vs reconstruction")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ae_reconstructions.png"), dpi=120)
        plt.close()
        print("Saved ae_reconstructions.png")

    # ── K-Means ─────────────────────────────────────────────────────

    @staticmethod
    def plot_kmeans_accuracy_vs_k(
        kmeans_results, alpha_values, k_values, save_filename="kmeans_acc_vs_k.png"
    ):
        """Plots Accuracy against K value, multiple lines for alpha."""
        plt.figure(figsize=(8, 6))
        for alpha in alpha_values:
            accuracies = [
                kmeans_results[(alpha, k)]["train_accuracy"] for k in k_values
            ]
            plt.plot(k_values, accuracies, marker="o", label=f"α={alpha}")

        plt.title("K-Means Training Accuracy vs K")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Training Accuracy")
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"      Saved: {save_filename}")

    @staticmethod
    def plot_kmeans_accuracy_vs_alpha(
        kmeans_results, alpha_values, k_values, save_filename="kmeans_acc_vs_alpha.png"
    ):
        """Plots Accuracy against Alpha value, multiple lines for K."""
        plt.figure(figsize=(8, 6))
        for k in k_values:
            accuracies = [
                kmeans_results[(alpha, k)]["train_accuracy"] for alpha in alpha_values
            ]
            plt.plot(alpha_values, accuracies, marker="s", label=f"K={k}")

        plt.title("K-Means Training Accuracy vs α")
        plt.xlabel("Alpha (Variance Retained)")
        plt.ylabel("Training Accuracy")
        plt.xticks(alpha_values)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"      Saved: {save_filename}")

    @staticmethod
    def plot_confusion_matrix(
        cm, title="Confusion Matrix", save_filename="confusion_matrix.png"
    ):
        """Plots and saves the confusion matrix."""
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"      Saved: {save_filename}")

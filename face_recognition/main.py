import os.path

from autoencoder.autoencoder_model import AutoencoderModel
from data.dataset_loader import load_orl_dataset, generate_data_matrix
from data.data_splitter import split_train_test
from pca.pca_model import PCAModel
from utils.visualizer import Visualizer
from utils.config import *
from clustering.kmeans_clustering import KMeansClustering
from evaluation.evaluator import Evaluator
import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    print("[1] Loading ORL dataset...")
    images, labels = load_orl_dataset()

    print("[2] Generating data matrix D and label vector y...")
    D, y = generate_data_matrix(images, labels)
    print(f"      D shape : {D.shape}")  # (400, 10304)
    print(f"      y shape : {y.shape}")  # (400,)

    print("[3] Splitting into training and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(D, y)
    print(f"      Train : {X_train.shape}  |  Test : {X_test.shape}")

    print("[4] Running PCA for all alpha values...")
    pca_results = {}
    for alpha in ALPHA_VALUES:
        pca = PCAModel(variance_threshold=alpha, random_seed=RANDOM_SEED)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        pca_results[alpha] = {
            "pca": pca,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "n_components": pca.n_components_,
        }
        print(f"      α={alpha} → {pca.n_components_} components")

    print("[5] Running K-Means Clustering on PCA Data...")

    kmeans_results = {}
    for alpha in ALPHA_VALUES:
        X_train_pca = pca_results[alpha]["X_train_pca"]

        for k in K_VALUES:
            print(f"      Running K-Means (alpha={alpha}, K={k})...")
            kmeans = KMeansClustering(
                k=k, max_iters=MAX_ITER_KM, random_seed=RANDOM_SEED, tolerance=TOL_KM
            )
            kmeans.fit(X_train_pca)

            train_preds = kmeans.predict(X_train_pca)

            # Choose mapping strategy: Hungarian for K=40, Majority Vote for others
            if k == N_SUBJECTS:
                mapped_preds = Evaluator.map_clusters_to_labels_hungarian(
                    train_preds, y_train
                )
            else:
                mapped_preds = Evaluator.map_clusters_to_labels_majority_vote(
                    train_preds, y_train
                )

            train_acc = Evaluator.calculate_accuracy(mapped_preds, y_train)

            kmeans_results[(alpha, k)] = {"model": kmeans, "train_accuracy": train_acc}
            print(f"        Train Accuracy: {train_acc:.4f}")

    print("[6] Generating K-Means evaluation plots...")
    from utils.visualizer import Visualizer

    Visualizer.plot_kmeans_accuracy_vs_k(kmeans_results, ALPHA_VALUES, K_VALUES)
    Visualizer.plot_kmeans_accuracy_vs_alpha(kmeans_results, ALPHA_VALUES, K_VALUES)

    print("[7] Evaluating Best K-Means Model on Test Set...")
    # 1. Find the best model based on training accuracy
    best_combo = max(kmeans_results, key=lambda k: kmeans_results[k]["train_accuracy"])
    best_alpha, best_k = best_combo
    best_kmeans = kmeans_results[best_combo]["model"]
    print(
        f"      Best Model: alpha={best_alpha}, K={best_k} (Train Acc: {kmeans_results[best_combo]['train_accuracy']:.4f})"
    )

    # 2. Extract cluster->label mapping from the training set for the best model
    X_train_pca_best = pca_results[best_alpha]["X_train_pca"]
    train_preds = best_kmeans.predict(X_train_pca_best)
    mapping = {}
    if best_k == N_SUBJECTS:
        max_c = int(np.max(train_preds)) + 1
        max_t = int(np.max(y_train)) + 1
        overlap = np.zeros((max_c, max_t), dtype=np.int64)
        for c, t in zip(train_preds, y_train):
            overlap[c, t] += 1
        row_ind, col_ind = linear_sum_assignment(overlap.max() - overlap)
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        for c in np.unique(train_preds):
            mask = train_preds == c
            if np.sum(mask) > 0:
                mapping[int(c)] = np.bincount(y_train[mask]).argmax()

    # 3. Apply the model AND mapping to the Test set
    X_test_pca = pca_results[best_alpha]["X_test_pca"]
    test_clusters = best_kmeans.predict(X_test_pca)
    # Ensure test predictions are also checked using Python int
    test_mapped_preds = np.array([mapping.get(int(c), -1) for c in test_clusters])

    # 4. Extract and print metrics
    test_metrics = Evaluator.evaluate_test_set(test_mapped_preds, y_test)
    print(f"      Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"      Test Macro F1: {test_metrics['f1_score']:.4f}")

    # Plot confusion matrix
    Visualizer.plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"K-Means (alpha={best_alpha}, K={best_k})",
    )

    print("[8] Running Autoencoder")
    autoencoder = AutoencoderModel()
    autoencoder.load_checkpoint()
    autoencoder.fit(X_train)
    X_train_auto = autoencoder.transform(X_train)
    X_test_auto = autoencoder.transform(X_test)
    reconstructed_image = autoencoder.inverse_transform(X_train_auto)

    ## For visualize purposes only, they take time to render image, so no need to run it every time.
    # Visualizer.plot_ae_reconstructions(X_train, reconstructed_image, n_faces=8)
    # for i, alpha in enumerate(ALPHA_VALUES):
    #     Visualizer.plot_eigenfaces(
    #         pca_results[alpha]["pca"],
    #         n_faces = pca_results[alpha]["pca"].n_components_,
    #         alpha_value = alpha
    #     )
    #     Visualizer.plot_variance_explained(
    #         pca_results[alpha]["pca"],
    #         alpha_value=alpha
    #     )
    #     Visualizer.plot_transformed_faces(
    #         faces = pca_results[alpha]["pca"].inverse_transform(pca_results[alpha]["X_train_pca"]),
    #     alpha_value= alpha)

    """
    compressed values to work on:
        X_train_pca
        X_test_pca
        X_train_auto
        X_test_auto
    """


if __name__ == "__main__":
    main()

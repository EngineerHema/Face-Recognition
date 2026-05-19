import os.path

from autoencoder.autoencoder_model import AutoencoderModel
from data.dataset_loader import load_orl_dataset, generate_data_matrix
from data.data_splitter import split_train_test
from pca.pca_model import PCAModel
from utils.visualizer import Visualizer
from utils.config import *
from clustering.kmeans_clustering import KMeansClustering
from clustering.gmm_clustering import GMMClustering
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
    print(f"      K-Means Test Results: Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_score']:.4f}")

    print("[8] Running Gaussian Mixture Model (GMM) Clustering on PCA Data...")

    gmm_results = {}
    for alpha in ALPHA_VALUES:
        X_train_pca = pca_results[alpha]["X_train_pca"]

        for k in K_VALUES:
            print(f"      Running GMM (alpha={alpha}, K={k})...")
            gmm = GMMClustering(
                k=k, max_iters=MAX_ITER_GMM, random_seed=RANDOM_SEED, tolerance=TOL_GMM
            )
            gmm.fit(X_train_pca)

            train_preds = gmm.predict(X_train_pca)

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

            gmm_results[(alpha, k)] = {"model": gmm, "train_accuracy": train_acc}
            print(f"        Train Accuracy: {train_acc:.4f}")

    print("[9] Generating GMM evaluation plots...")
    Visualizer.plot_gmm_accuracy_vs_k(gmm_results, ALPHA_VALUES, K_VALUES)
    Visualizer.plot_gmm_accuracy_vs_alpha(gmm_results, ALPHA_VALUES, K_VALUES)

    print("[10] Evaluating Best GMM Model on Test Set...")
    # 1. Find the best model based on training accuracy
    best_combo_gmm = max(gmm_results, key=lambda k: gmm_results[k]["train_accuracy"])
    best_alpha_gmm, best_k_gmm = best_combo_gmm
    best_gmm = gmm_results[best_combo_gmm]["model"]
    print(
        f"      Best Model: alpha={best_alpha_gmm}, K={best_k_gmm} (Train Acc: {gmm_results[best_combo_gmm]['train_accuracy']:.4f})"
    )

    # 2. Extract cluster->label mapping from the training set for the best model
    X_train_pca_best_gmm = pca_results[best_alpha_gmm]["X_train_pca"]
    train_preds_gmm = best_gmm.predict(X_train_pca_best_gmm)
    mapping_gmm = {}
    if best_k_gmm == N_SUBJECTS:
        max_c = int(np.max(train_preds_gmm)) + 1
        max_t = int(np.max(y_train)) + 1
        overlap = np.zeros((max_c, max_t), dtype=np.int64)
        for c, t in zip(train_preds_gmm, y_train):
            overlap[c, t] += 1
        row_ind, col_ind = linear_sum_assignment(overlap.max() - overlap)
        mapping_gmm = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        for c in np.unique(train_preds_gmm):
            mask = train_preds_gmm == c
            if np.sum(mask) > 0:
                mapping_gmm[int(c)] = np.bincount(y_train[mask]).argmax()

    # 3. Apply the model AND mapping to the Test set
    X_test_pca_gmm = pca_results[best_alpha_gmm]["X_test_pca"]
    test_clusters_gmm = best_gmm.predict(X_test_pca_gmm)
    test_mapped_preds_gmm = np.array([mapping_gmm.get(int(c), -1) for c in test_clusters_gmm])

    # 4. Extract and print metrics
    test_metrics_gmm = Evaluator.evaluate_test_set(test_mapped_preds_gmm, y_test)
    print(f"      Test Accuracy: {test_metrics_gmm['accuracy']:.4f}")
    print(f"      Test Macro F1: {test_metrics_gmm['f1_score']:.4f}")

    # Plot confusion matrix
    Visualizer.plot_confusion_matrix(
        test_metrics_gmm["confusion_matrix"],
        title=f"GMM (alpha={best_alpha_gmm}, K={best_k_gmm})",
        save_filename="confusion_matrix_gmm.png"
    )
    print(f"      GMM Test Results: Accuracy={test_metrics_gmm['accuracy']:.4f}, F1={test_metrics_gmm['f1_score']:.4f}")

    print("[11] Comparing K-Means vs GMM Performance...")
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: K-Means vs GMM")
    print("="*70)
    print(f"\n{'Metric':<25} {'K-Means':<20} {'GMM':<20}")
    print("-"*70)
    print(f"{'Train Accuracy':<25} {kmeans_results[best_combo]['train_accuracy']:<20.4f} {gmm_results[best_combo_gmm]['train_accuracy']:<20.4f}")
    print(f"{'Test Accuracy':<25} {test_metrics['accuracy']:<20.4f} {test_metrics_gmm['accuracy']:<20.4f}")
    print(f"{'Test F1-Score':<25} {test_metrics['f1_score']:<20.4f} {test_metrics_gmm['f1_score']:<20.4f}")
    print(f"{'Best Alpha':<25} {best_alpha:<20} {best_alpha_gmm:<20}")
    print(f"{'Best K':<25} {best_k:<20} {best_k_gmm:<20}")
    print("="*70 + "\n")

    # Plot comparison
    Visualizer.plot_kmeans_vs_gmm_comparison(
        kmeans_results, gmm_results, ALPHA_VALUES, K_VALUES
    )

    print("[12] Running Autoencoder")
    autoencoder = AutoencoderModel()
    autoencoder.load_checkpoint()
    autoencoder.fit(X_train)
    X_train_auto = autoencoder.transform(X_train)
    X_test_auto = autoencoder.transform(X_test)
    reconstructed_image = autoencoder.inverse_transform(X_train_auto)

    print("[13] Running K-Means on Autoencoder Features...")
    kmeans_auto_results = {}
    for k in K_VALUES:
        print(f"      Running K-Means (autoencoder, K={k})...")
        kmeans = KMeansClustering(k=k, max_iters=MAX_ITER_KM, random_seed=RANDOM_SEED, tolerance=TOL_KM)
        kmeans.fit(X_train_auto)
        train_preds = kmeans.predict(X_train_auto)

        if k == N_SUBJECTS:
            mapped_preds = Evaluator.map_clusters_to_labels_hungarian(train_preds, y_train)
        else:
            mapped_preds = Evaluator.map_clusters_to_labels_majority_vote(train_preds, y_train)

        train_acc = Evaluator.calculate_accuracy(mapped_preds, y_train)
        kmeans_auto_results[k] = {"model": kmeans, "train_accuracy": train_acc}
        print(f"        Train Accuracy: {train_acc:.4f}")

    # Evaluate best on test set
    best_k_auto = max(kmeans_auto_results, key=lambda k: kmeans_auto_results[k]["train_accuracy"])
    best_kmeans_auto = kmeans_auto_results[best_k_auto]["model"]

    train_preds = best_kmeans_auto.predict(X_train_auto)
    mapping = {}
    if best_k_auto == N_SUBJECTS:
        overlap = np.zeros((best_k_auto, int(np.max(y_train))+1), dtype=np.int64)
        for c, t in zip(train_preds, y_train):
            overlap[c, t] += 1
        row_ind, col_ind = linear_sum_assignment(overlap.max() - overlap)
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        for c in np.unique(train_preds):
            mask = train_preds == c
            mapping[int(c)] = np.bincount(y_train[mask]).argmax()

    test_clusters = best_kmeans_auto.predict(X_test_auto)
    test_preds = np.array([mapping.get(int(c), -1) for c in test_clusters])
    test_metrics_auto_km = Evaluator.evaluate_test_set(test_preds, y_test)
    print(f"      [AE + K-Means] Best K={best_k_auto} → Test Acc: {test_metrics_auto_km['accuracy']:.4f}, F1: {test_metrics_auto_km['f1_score']:.4f}")
    Visualizer.plot_confusion_matrix(test_metrics_auto_km["confusion_matrix"],
        title=f"K-Means on Autoencoder (K={best_k_auto})",
        save_filename="confusion_matrix_ae_kmeans.png")


    print("[14] Running GMM on Autoencoder Features...")
    gmm_auto_results = {}
    for k in K_VALUES:
        print(f"      Running GMM (autoencoder, K={k})...")
        gmm = GMMClustering(k=k, max_iters=MAX_ITER_GMM, random_seed=RANDOM_SEED, tolerance=TOL_GMM)
        gmm.fit(X_train_auto)
        train_preds = gmm.predict(X_train_auto)

        if k == N_SUBJECTS:
            mapped_preds = Evaluator.map_clusters_to_labels_hungarian(train_preds, y_train)
        else:
            mapped_preds = Evaluator.map_clusters_to_labels_majority_vote(train_preds, y_train)

        train_acc = Evaluator.calculate_accuracy(mapped_preds, y_train)
        gmm_auto_results[k] = {"model": gmm, "train_accuracy": train_acc}
        print(f"        Train Accuracy: {train_acc:.4f}")

    # Evaluate best on test set
    best_k_auto_gmm = max(gmm_auto_results, key=lambda k: gmm_auto_results[k]["train_accuracy"])
    best_gmm_auto = gmm_auto_results[best_k_auto_gmm]["model"]

    train_preds_gmm = best_gmm_auto.predict(X_train_auto)
    mapping_gmm = {}
    if best_k_auto_gmm == N_SUBJECTS:
        overlap = np.zeros((best_k_auto_gmm, int(np.max(y_train))+1), dtype=np.int64)
        for c, t in zip(train_preds_gmm, y_train):
            overlap[c, t] += 1
        row_ind, col_ind = linear_sum_assignment(overlap.max() - overlap)
        mapping_gmm = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        for c in np.unique(train_preds_gmm):
            mask = train_preds_gmm == c
            mapping_gmm[int(c)] = np.bincount(y_train[mask]).argmax()

    test_clusters_gmm = best_gmm_auto.predict(X_test_auto)
    test_preds_gmm = np.array([mapping_gmm.get(int(c), -1) for c in test_clusters_gmm])
    test_metrics_auto_gmm = Evaluator.evaluate_test_set(test_preds_gmm, y_test)
    print(f"      [AE + GMM] Best K={best_k_auto_gmm} → Test Acc: {test_metrics_auto_gmm['accuracy']:.4f}, F1: {test_metrics_auto_gmm['f1_score']:.4f}")
    Visualizer.plot_confusion_matrix(test_metrics_auto_gmm["confusion_matrix"],
        title=f"GMM on Autoencoder (K={best_k_auto_gmm})",
        save_filename="confusion_matrix_ae_gmm.png")


    print("[15] Final Comparison: PCA vs Autoencoder")
    print("\n" + "="*80)
    print(f"{'Method':<30} {'Test Accuracy':<20} {'Test F1':<20}")
    print("-"*80)
    print(f"{'K-Means + PCA':<30} {test_metrics['accuracy']:<20.4f} {test_metrics['f1_score']:<20.4f}")
    print(f"{'GMM + PCA':<30} {test_metrics_gmm['accuracy']:<20.4f} {test_metrics_gmm['f1_score']:<20.4f}")
    print(f"{'K-Means + Autoencoder':<30} {test_metrics_auto_km['accuracy']:<20.4f} {test_metrics_auto_km['f1_score']:<20.4f}")
    print(f"{'GMM + Autoencoder':<30} {test_metrics_auto_gmm['accuracy']:<20.4f} {test_metrics_auto_gmm['f1_score']:<20.4f}")
    print("="*80)

    # # For visualize purposes only, they take time to render image, so no need to run it every time.
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


if __name__ == "__main__":
    main()

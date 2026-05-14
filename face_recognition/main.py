"""
Assignment #3: Face Recognition - Main Pipeline
CSE: Pattern Recognition - Alexandria University
"""

from data.dataset_loader import load_orl_dataset, generate_data_matrix
from data.data_splitter import split_train_test
from pca.pca_model import PCAModel
from clustering.kmeans_clustering import KMeansClustering
from clustering.gmm_clustering import GMMClustering
from evaluation.evaluator import Evaluator
from utils.visualizer import Visualizer
from utils.config import ALPHA_VALUES, K_VALUES, RANDOM_SEED

import numpy as np


def main():
    # ── 1. Load Dataset ──────────────────────────────────────────────
    print("[1/6] Loading ORL dataset...")
    images, labels = load_orl_dataset()

    # ── 2. Build Data Matrix D (400 x 10304) & label vector y ────────
    print("[2/6] Generating data matrix D and label vector y...")
    D, y = generate_data_matrix(images, labels)
    print(f"      D shape : {D.shape}")   # (400, 10304)
    print(f"      y shape : {y.shape}")   # (400,)

    # ── 3. Train / Test Split ────────────────────────────────────────
    print("[3/6] Splitting into training and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(D, y)
    print(f"      Train : {X_train.shape}  |  Test : {X_test.shape}")

    # ── 4. PCA ───────────────────────────────────────────────────────
    print("[4/6] Running PCA for all alpha values...")
    pca_results = {}
    for alpha in ALPHA_VALUES:
        pca = PCAModel(variance_threshold=alpha, random_seed=RANDOM_SEED)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca  = pca.transform(X_test)
        pca_results[alpha] = {
            "pca"         : pca,
            "X_train_pca" : X_train_pca,
            "X_test_pca"  : X_test_pca,
            "n_components": pca.n_components_,
        }
        print(f"      α={alpha} → {pca.n_components_} components")

    Visualizer.plot_eigenfaces(pca_results[0.95]["pca"])
    Visualizer.plot_variance_explained(pca_results[0.95]["pca"])

    # ── 5. Clustering ─────────────────────────────────────────────────
    print("[5/6] Running K-Means and GMM clustering...")
    km_results  = {}
    gmm_results = {}

    for alpha in ALPHA_VALUES:
        X_tr = pca_results[alpha]["X_train_pca"]
        for k in K_VALUES:
            # K-Means
            km = KMeansClustering(n_clusters=k, random_seed=RANDOM_SEED)
            km.fit(X_tr, y_train)
            km_results[(alpha, k)] = km

            # GMM
            gmm = GMMClustering(n_components=k, random_seed=RANDOM_SEED)
            gmm.fit(X_tr, y_train)
            gmm_results[(alpha, k)] = gmm

    Visualizer.plot_accuracy_vs_k(km_results,  "K-Means", ALPHA_VALUES, K_VALUES)
    Visualizer.plot_accuracy_vs_k(gmm_results, "GMM",     ALPHA_VALUES, K_VALUES)
    Visualizer.plot_accuracy_vs_alpha(km_results,  "K-Means", ALPHA_VALUES, K_VALUES)
    Visualizer.plot_accuracy_vs_alpha(gmm_results, "GMM",     ALPHA_VALUES, K_VALUES)

    # ── 6. Evaluation ─────────────────────────────────────────────────
    print("[6/6] Evaluating best models on test set...")
    evaluator = Evaluator()

    best_km_key  = max(km_results,  key=lambda k: km_results[k].train_accuracy_)
    best_gmm_key = max(gmm_results, key=lambda k: gmm_results[k].train_accuracy_)

    alpha_km,  k_km  = best_km_key
    alpha_gmm, k_gmm = best_gmm_key

    X_test_km  = pca_results[alpha_km]["X_test_pca"]
    X_test_gmm = pca_results[alpha_gmm]["X_test_pca"]

    evaluator.evaluate(km_results[best_km_key],   X_test_km,  y_test, model_name="Best K-Means")
    evaluator.evaluate(gmm_results[best_gmm_key], X_test_gmm, y_test, model_name="Best GMM")
    evaluator.compare_models()

    print("\nDone! All results saved to ./outputs/")


if __name__ == "__main__":
    main()

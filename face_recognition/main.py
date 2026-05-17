import os.path

from autoencoder.autoencoder_model import AutoencoderModel
from data.dataset_loader import load_orl_dataset, generate_data_matrix
from data.data_splitter import split_train_test
from pca.pca_model import PCAModel
from utils.visualizer import Visualizer
from utils.config import *
import numpy as np


def main():
    print("[1] Loading ORL dataset...")
    images, labels = load_orl_dataset()

    print("[2] Generating data matrix D and label vector y...")
    D, y = generate_data_matrix(images, labels)
    print(f"      D shape : {D.shape}")   # (400, 10304)
    print(f"      y shape : {y.shape}")   # (400,)

    print("[3] Splitting into training and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(D, y)
    print(f"      Train : {X_train.shape}  |  Test : {X_test.shape}")

    print("[4] Running PCA for all alpha values...")
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

    print("[5] Running Autoencoder")
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

    '''
    compressed values to work on:
        X_train_pca
        X_test_pca
        X_train_auto
        X_test_auto
    '''
if __name__ == "__main__":
    main()
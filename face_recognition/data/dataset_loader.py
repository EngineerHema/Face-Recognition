import os
import numpy as np
from PIL import Image

from utils.config import *


# ── Public API ────────────────────────────────────────────────────────────────

def load_orl_dataset() -> tuple[list[np.ndarray], list[int]]:
    """
    Walk the dataset directory and return two parallel lists:
        images : list of (112, 92) uint8 arrays, length = 400
        labels : list of ints in 1..40,           length = 400

    Images are ordered subject-first, then by image index, so
    subject 1 occupies positions 0-9, subject 2 positions 10-19, etc.
    """
    images: list[np.ndarray] = []
    labels: list[int]        = []

    for subject_id in range(1, N_SUBJECTS + 1):
        subject_dir = os.path.join(DATASET_ROOT, f"s{subject_id}")
        if not os.path.isdir(subject_dir):
            raise FileNotFoundError(
                f"Subject directory not found: {subject_dir}\n"
                f"Please set DATASET_ROOT correctly in utils/config.py"
            )

        for img_idx in range(1, N_IMAGES_PER_SUB + 1):
            img_path = os.path.join(subject_dir, f"{img_idx}.pgm")
            img      = Image.open(img_path).convert("L")
            arr      = np.array(img, dtype=np.uint8)

            # Sanity-check dimensions
            assert arr.shape == (IMAGE_HEIGHT, IMAGE_WIDTH), (
                f"Unexpected image size {arr.shape} in {img_path}. "
                f"Expected ({IMAGE_HEIGHT}, {IMAGE_WIDTH})."
            )

            images.append(arr)
            labels.append(subject_id)

    return images, labels


def generate_data_matrix(
    images: list[np.ndarray],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the raw image list produced by load_orl_dataset() into:

        D : np.ndarray, shape (400, 10304), dtype float64
            Each row is one flattened image vector.
            Pixel values are kept in [0, 255] (un-normalised) so that
            the caller / PCA step can choose its own normalisation.

        y : np.ndarray, shape (400,), dtype int32
            Integer subject labels 1..40.
    """
    n_samples = len(images)

    # Pre-allocate for speed
    D = np.empty((n_samples, IMAGE_VECTOR_SIZE), dtype=np.float64)

    for i, img_arr in enumerate(images):
        D[i] = img_arr.flatten()

    y = np.array(labels, dtype=np.int32)

    assert D.shape == (n_samples, IMAGE_VECTOR_SIZE), \
        f"Data matrix shape mismatch: got {D.shape}"
    assert y.shape == (n_samples,), \
        f"Label vector shape mismatch: got {y.shape}"
    assert y.min() == 1 and y.max() == N_SUBJECTS, \
        f"Label range mismatch: got {y.min()}..{y.max()}"

    return D, y

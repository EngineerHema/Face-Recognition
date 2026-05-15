import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "att_faces")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ── Dataset constants ─────────────────────────────────────────────────────────
N_SUBJECTS        = 40          # number of subjects
N_IMAGES_PER_SUB  = 10          # images per subject
IMAGE_HEIGHT      = 112         # pixels
IMAGE_WIDTH       = 92          # pixels
IMAGE_VECTOR_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH   # 10304

# ── PCA ───────────────────────────────────────────────────────────────────────
ALPHA_VALUES = [0.80, 0.85, 0.90, 0.95]   # variance thresholds
PCA_CACHE    = os.path.join(OUTPUT_DIR, "pca_eigenvalues.npy")

# ── Clustering ────────────────────────────────────────────────────────────────
K_VALUES     = [20, 40, 60]   # number of clusters / mixture components
MAX_ITER_KM  = 300
MAX_ITER_GMM = 200
TOL_KM       = 1e-4
TOL_GMM      = 1e-3

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.config import *


# ── Architecture ─────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(),
            nn.Linear(512, 128),    nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 512),        nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024),       nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim),   nn.Sigmoid(),          # pixels in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _AE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = _Encoder(in_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, in_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Public interface ──────────────────────────────────────────────────────────

class AutoencoderModel:
    """
    Autoencoder for dimensionality reduction on face images.

    Usage
    -----
    ae = AutoencoderModel()
    ae.fit(X_train)                   # train
    Z       = ae.transform(X_test)    # encode  → latent codes
    X_recon = ae.inverse_transform(Z) # decode  → pixel space
    """

    # ── constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        in_dim:     int   = AUTOENCODER_INPUT_SIZE,
        latent_dim: int   = AUTOENCODER_LATENT_SIZE,
        epochs:     int   = EPOCHS,
        batch_size: int   = BATCH_SIZE,
        lr:         float = LEARNING_RATE,
        random_state: int = RANDOM_SEED,
    ):
        self.in_dim       = in_dim
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self._model: _AE | None = None
        self.train_losses: list[float] = []

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        """
        Train the autoencoder on X.

        Parameters
        ----------
        X : (n_samples, 10304)  raw pixel values in [0, 255]

        Returns
        -------
        self
        """
        if self._model is not None:
            print("Checkpoint already loaded — skipping training.")
            return self

        X_norm = (X / 255.0).astype(np.float32)
        tensor = torch.tensor(X_norm)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = _AE(self.in_dim, self.latent_dim).to(self.device)
        optimizer   = optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion   = nn.MSELoss()

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, _ = self._model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(X_norm)
            self.train_losses.append(epoch_loss)
            scheduler.step()

            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{self.epochs}  loss={epoch_loss:.6f}")

        self._save_checkpoint()
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X into the latent space.

        Parameters
        ----------
        X : (n_samples, 10304)  raw pixel values in [0, 255]

        Returns
        -------
        Z : (n_samples, latent_dim)
        """
        self._check_fitted()
        X_norm = torch.tensor((X / 255.0).astype(np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            Z = self._model.encoder(X_norm)
        return Z.cpu().numpy()

    # ── inverse_transform ────────────────────────────────────────────────────

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent codes back to pixel space.

        Parameters
        ----------
        Z : (n_samples, latent_dim)

        Returns
        -------
        X_reconstructed : (n_samples, 10304)  values in [0, 255]
        """
        self._check_fitted()
        Z_tensor = torch.tensor(Z.astype(np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            X_recon = self._model.decoder(Z_tensor)
        return (X_recon.cpu().numpy() * 255.0).astype(np.float32)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit() before transform() or inverse_transform().")

    def _save_checkpoint(self, save_dir: str = CHECKPOINT_DIR):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "autoencoder.pt")
        torch.save(self._model.state_dict(), path)
        print(f"Saved checkpoint → {path}")

    def load_checkpoint(self, path: str | None = None):
        """Load a previously saved checkpoint and handle if does not exist."""
        path = path or os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        if not os.path.exists(path):
            print(f"No checkpoint found at {path} — will train from scratch.")
            self._model = None
            return self
        try:
            self._model = _AE(self.in_dim, self.latent_dim).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            print(f"Loaded checkpoint ← {path}")
        except RuntimeError:
            print(f"Probably size mismatch Error happened. Retraining")
            self._model = None
            return self
        return self
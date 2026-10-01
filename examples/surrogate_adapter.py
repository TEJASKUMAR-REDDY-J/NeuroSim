"""Example Python adapter for AI Mode (Models -> Python adapter).

NeuroSim calls:
  Surrogate(manifest)   manifest = dataset/manifest.json: channel names, field groups, grid, normalization stats
  fit(train)            optional; train = list of (states [T, C, *S], solid mask [*S], condition list) per training sequence
  predict(x)            x: float32 [1, C + 1 + K, *S] = state channels (physical units), solid mask, K condition values
                        (K = 1 for Reynolds-number or velocity conditions, else 0); returns the next state [1, C, *S]
*S is (Y, X) for 2D datasets and (Z, Y, X) for 3D ones. Solid cells are reset by NeuroSim after each step.

This example is a per-channel linear relaxation towards the training mean, fitted by least squares: a
deliberately simple reference that shows the contract. Replace it with your own model (any framework).
"""
import numpy as np


class Surrogate:
    def __init__(self, manifest):
        self.C = len(manifest["channels"])
        self.mean = np.asarray(manifest["stats"]["mean"], np.float32)
        self.a = np.zeros(self.C, np.float32)

    def fit(self, train):
        num, den = np.zeros(self.C), np.zeros(self.C)
        for X, mask, _ in train:
            fluid = ~mask
            d = (X[:-1] - self.mean.reshape((-1,) + (1,) * mask.ndim))[:, :, fluid]
            step = np.diff(X, axis=0)[:, :, fluid]
            num += (d * step).sum(axis=(0, 2)); den += (d * d).sum(axis=(0, 2))
        self.a = (num / np.maximum(den, 1e-30)).astype(np.float32)

    def predict(self, x):
        s = x[:, :self.C]
        sh = (1, -1) + (1,) * (x.ndim - 2)
        return s + self.a.reshape(sh) * (s - self.mean.reshape(sh))

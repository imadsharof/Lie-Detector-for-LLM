"""The four truth probes used in the Phi-2 reproduction.

Each probe maps a hidden-state vector to a scalar score. Higher scores are
oriented to mean "more likely true" using the training labels. Evaluation uses
relative ranking inside each question group, so no global threshold is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


ALL_PROBE_METHODS: list[str] = ["dim", "lat", "lr", "pca-g"]
PROBE_DISPLAY_NAMES: dict[str, str] = {
    "dim": "DIM",
    "lat": "LAT",
    "lr": "LR",
    "pca-g": "PCA-G",
}
SUPERVISED_PROBES: list[str] = ["dim", "lr"]
UNSUPERVISED_PROBES: list[str] = ["lat", "pca-g"]
GROUPED_PROBES: list[str] = ["pca-g"]


class BaseProbe:
    """Minimal scoring interface shared by all probes."""

    def score(self, activations: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DotProductProbe(BaseProbe):
    """Linear direction probe with an optional centering vector."""

    direction: np.ndarray
    center: np.ndarray | None = None

    def score(self, activations: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        x = activations if self.center is None else activations - self.center
        return np.asarray(x @ self.direction).ravel()


@dataclass
class GroupCenteredDotProductProbe(BaseProbe):
    """Direction probe that centers activations within each question group."""

    direction: np.ndarray

    def score(self, activations: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        if groups is None:
            raise ValueError("This probe requires group identifiers at scoring time.")
        return np.asarray(_center_by_group(activations, groups) @ self.direction).ravel()


@dataclass
class SklearnLinearProbe(BaseProbe):
    """Wrapper for scikit-learn linear classifiers."""

    model: object

    def score(self, activations: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model.decision_function(activations)).ravel()


def _as_bool(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if labels.sum() == 0 or (~labels).sum() == 0:
        raise ValueError("A probe needs at least one true and one false example.")
    return labels


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalise a zero-norm direction.")
    return vector / norm


def _orient(direction: np.ndarray, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Flip a direction so true examples receive larger scores."""
    scores = activations @ direction
    return direction if scores[labels].mean() >= scores[~labels].mean() else -direction


def train_dim_probe(activations: np.ndarray, labels: np.ndarray) -> DotProductProbe:
    """Difference-in-Means: ``mean(true) - mean(false)``."""
    labels = _as_bool(labels)
    activations = np.asarray(activations, dtype=np.float64)
    direction = activations[labels].mean(axis=0) - activations[~labels].mean(axis=0)
    return DotProductProbe(direction=_unit(direction))


def train_lat_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
) -> GroupCenteredDotProductProbe:
    """Linear Artificial Tomography: PCA on random pairwise differences."""
    labels = _as_bool(labels)
    activations = np.asarray(activations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(activations))
    order = order[: len(order) // 2 * 2]
    if len(order) < 2:
        raise ValueError("LAT needs at least two activation rows.")
    first, second = order.reshape(2, -1)
    diffs = activations[first] - activations[second]
    center = diffs.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(diffs - center)
    direction = _unit(pca.components_[0])
    direction = _orient(direction, activations - center, labels)
    return DotProductProbe(direction=direction, center=center)


def train_lr_probe(activations: np.ndarray, labels: np.ndarray) -> SklearnLinearProbe:
    """Supervised logistic regression on raw Phi-2 hidden states."""
    labels = _as_bool(labels)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=10_000)
    model.fit(np.asarray(activations, dtype=np.float64), labels)
    return SklearnLinearProbe(model=model)


def _center_by_group(activations: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each question group's mean activation from its candidates."""
    centered = np.asarray(activations, dtype=np.float64).copy()
    groups = np.asarray(groups)
    for group in np.unique(groups):
        mask = groups == group
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def train_grouped_pca_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> DotProductProbe:
    """Grouped PCA (PCA-G): per-question centering, then first PC."""
    labels = _as_bool(labels)
    centered = _center_by_group(activations, groups)
    pca = PCA(n_components=1)
    pca.fit(centered)
    direction = _unit(pca.components_[0])
    direction = _orient(direction, centered, labels)
    return GroupCenteredDotProductProbe(direction=direction)


def train_probe(
    method: str,
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
) -> BaseProbe:
    """Train one of ``dim``, ``lat``, ``lr`` or ``pca-g``."""
    if method not in ALL_PROBE_METHODS:
        raise ValueError(f"Unknown probe method {method!r}. Use one of {ALL_PROBE_METHODS}.")
    if method == "dim":
        return train_dim_probe(activations, labels)
    if method == "lat":
        return train_lat_probe(activations, labels)
    if method == "lr":
        return train_lr_probe(activations, labels)
    if groups is None:
        raise ValueError("PCA-G requires group identifiers.")
    return train_grouped_pca_probe(activations, labels, groups)


def grouped_accuracy(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Group-level accuracy: the top-scoring candidate must be true."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels, dtype=bool)
    groups = np.asarray(groups)
    if not (len(scores) == len(labels) == len(groups)):
        raise ValueError("scores, labels and groups must have the same length.")

    correct = 0
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        predicted = int(np.argmax(scores[mask]))
        correct += int(labels[mask][predicted])
    return correct / len(unique_groups) if len(unique_groups) else 0.0

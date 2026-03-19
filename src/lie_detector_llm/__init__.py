"""Lightweight RepEng-inspired lie detector package."""

from .datasets import build_dataset_collection
from .experiment import run_probe_experiment, run_transfer_experiment, run_layer_sweep, run_full_transfer_matrix

__all__ = [
    "build_dataset_collection",
    "run_probe_experiment",
    "run_transfer_experiment",
    "run_layer_sweep",
    "run_full_transfer_matrix",
]

"""Lightweight RepEng-inspired lie detector package."""

from .datasets import build_dataset_collection
from .experiment import run_probe_experiment, run_transfer_experiment

__all__ = [
    "build_dataset_collection",
    "run_probe_experiment",
    "run_transfer_experiment",
]
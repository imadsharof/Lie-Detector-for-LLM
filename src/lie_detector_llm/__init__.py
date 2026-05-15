"""Phi-2 truth-probe generalisation package."""

from .datasets import (
    ALL_DATASET_NAMES,
    DEFAULT_DATASET_NAMES,
    PHI2_PROMPT_TEMPLATE,
    build_dataset_collection,
    split_groups,
)
from .experiment import (
    DEFAULT_MODEL,
    PROBE_METHODS,
    extract_activations_for_collection,
    run_all_probe_transfer_matrices,
    run_full_transfer_matrix,
    run_layer_method_sweep,
    run_layer_sweep,
    run_probe_comparison,
    run_probe_experiment,
    run_transfer_experiment,
    transfer_matrix_summary,
)
from .models import (
    clear_model_cache,
    extract_last_token_activations,
    extract_or_load_last_token_activations,
)
from .probes import ALL_PROBE_METHODS, grouped_accuracy, train_probe

__all__ = [
    "ALL_DATASET_NAMES",
    "ALL_PROBE_METHODS",
    "DEFAULT_DATASET_NAMES",
    "DEFAULT_MODEL",
    "PHI2_PROMPT_TEMPLATE",
    "PROBE_METHODS",
    "build_dataset_collection",
    "split_groups",
    "extract_last_token_activations",
    "extract_or_load_last_token_activations",
    "extract_activations_for_collection",
    "clear_model_cache",
    "train_probe",
    "grouped_accuracy",
    "run_probe_experiment",
    "run_probe_comparison",
    "run_transfer_experiment",
    "run_layer_sweep",
    "run_layer_method_sweep",
    "run_full_transfer_matrix",
    "run_all_probe_transfer_matrices",
    "transfer_matrix_summary",
]

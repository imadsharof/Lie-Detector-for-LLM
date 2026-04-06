"""Model loading and hidden-state extraction utilities.

This module provides the core functionality for extracting internal
representations (hidden states) from HuggingFace causal language models.
The key idea is that a transformer's hidden states at the last token of
a prompt encode information about the truthfulness of the statement —
this is what the linear probes in ``probes.py`` are trained to read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None


load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
transformers_logging.set_verbosity_error()
if disable_progress_bars is not None:
    disable_progress_bars()


@dataclass
class ActivationCache:
    """Container for extracted hidden-state activations.

    Attributes:
        activations: array of shape (num_prompts, num_layers, hidden_dim).
            Each entry is the hidden-state vector at the *last token*
            of the corresponding prompt, at the given transformer layer.
        layers: list of integer layer indices (0-based).
    """
    activations: np.ndarray
    layers: list[int]


def _get_huggingface_token() -> str | None:
    for variable_name in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ):
        token = os.getenv(variable_name)
        if token:
            return token
    return None


def get_device() -> torch.device:
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=4)
def _load_cached_model_and_tokenizer(model_name: str, device_type: str):
    """Load and cache a HuggingFace causal LM and its tokenizer.

    For large models (>1B parameters), float16 is used automatically
    to fit within memory constraints.  The model is moved to the
    specified device (CUDA, MPS, or CPU) and set to eval mode.
    """
    token = _get_huggingface_token()
    load_kwargs = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float16 for large models to save memory
    device = torch.device(device_type)
    if device_type in ("mps", "cuda"):
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = device_type
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer


def load_model_and_tokenizer(model_name: str, device: torch.device | None = None):
    device = device or get_device()
    model, tokenizer = _load_cached_model_and_tokenizer(model_name, device.type)
    return model, tokenizer, device


@torch.inference_mode()
def extract_last_token_activations(
    prompts: list[str],
    model_name: str = "distilgpt2",
    device: torch.device | None = None,
    batch_size: int = 2,
    show_progress: bool = False,
) -> ActivationCache:
    """Run each prompt through the model and capture hidden-state vectors.

    For every prompt the model performs a single forward pass (no text
    generation).  At each transformer layer we record the hidden-state
    vector of the **last token** — this is the point where the model has
    read the entire prompt and is about to predict the next token.

    Returns an ActivationCache of shape (len(prompts), num_layers, hidden_dim).
    """
    model, tokenizer, device = load_model_and_tokenizer(model_name, device=device)
    collected_batches: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Extracting activations",
        disable=not show_progress,
    ):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("The model did not return hidden states.")

        layers = hidden_states[1:]
        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        attention_mask = encoded["attention_mask"]
        last_token_positions = attention_mask.sum(dim=1) - 1

        batch_activations = []
        for batch_index in range(len(batch_prompts)):
            layer_vectors = []
            token_position = int(last_token_positions[batch_index].item())
            for layer_tensor in layers:
                vector = layer_tensor[batch_index, token_position].detach().cpu().numpy()
                layer_vectors.append(vector)
            batch_activations.append(np.stack(layer_vectors, axis=0))

        collected_batches.append(np.stack(batch_activations, axis=0))

    activations = np.concatenate(collected_batches, axis=0)
    return ActivationCache(activations=activations, layers=layer_indices or [])
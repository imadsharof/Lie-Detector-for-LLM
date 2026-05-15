"""Phi-2 model loading and last-token hidden-state extraction.

This project intentionally targets one model only: ``microsoft/phi-2``.
The experiment is a white-box probing study, so no text generation is used.
Each prompt is passed through the model once with ``output_hidden_states=True``,
and the saved representation is the hidden state at the final real token of the
sequence for every transformer layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:  # pragma: no cover - optional dependency path
    disable_progress_bars = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency path
    BitsAndBytesConfig = None


DEFAULT_MODEL = "microsoft/phi-2"
DEFAULT_MAX_LENGTH = 512

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
transformers_logging.set_verbosity_error()
if disable_progress_bars is not None:
    disable_progress_bars()


@dataclass
class ActivationCache:
    """Hidden-state activations for a fixed list of prompts.

    Attributes:
        activations: Array with shape ``(num_prompts, num_layers, hidden_dim)``.
            The layer axis excludes the embedding output, so Phi-2 layers are
            indexed ``0..31``.
        layers: Layer indices matching ``activations[:, layer, :]``.
        model_name: Always ``microsoft/phi-2`` in this project.
        prompt_hash: SHA1 hash of the prompt list, used for disk caching.
    """

    activations: np.ndarray
    layers: list[int]
    model_name: str = DEFAULT_MODEL
    prompt_hash: str | None = None


_ACTIVATION_CACHE: dict[tuple, ActivationCache] = {}
_ACTIVATION_CACHE_MAXSIZE = 3


def _validate_phi2_model(model_name: str) -> None:
    if model_name != DEFAULT_MODEL:
        raise ValueError(
            f"This reproduction is locked to {DEFAULT_MODEL!r}; got {model_name!r}."
        )


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
    """Auto-detect the best available local device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prompt_hash(prompts: list[str]) -> str:
    """Stable SHA1 hash for an ordered prompt list."""
    digest = hashlib.sha1()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def activation_cache_path(
    cache_dir: str | Path,
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Path:
    """Return the deterministic disk-cache path for a prompt list."""
    _validate_phi2_model(model_name)
    model_slug = model_name.replace("/", "__")
    return Path(cache_dir) / f"{model_slug}_len{max_length}_{prompt_hash(prompts)}.npz"


def save_activation_cache(cache: ActivationCache, path: str | Path) -> Path:
    """Save activations as a compressed NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activations=cache.activations.astype(np.float32, copy=False),
        layers=np.asarray(cache.layers, dtype=np.int64),
        model_name=np.asarray(cache.model_name),
        prompt_hash=np.asarray(cache.prompt_hash or ""),
    )
    return path


def load_activation_cache(path: str | Path) -> ActivationCache:
    """Load an activation cache written by :func:`save_activation_cache`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return ActivationCache(
            activations=data["activations"],
            layers=data["layers"].astype(int).tolist(),
            model_name=str(data["model_name"].item()),
            prompt_hash=str(data["prompt_hash"].item()) or None,
        )


@lru_cache(maxsize=1)
def _load_cached_model_and_tokenizer(
    model_name: str,
    device_type: str,
    load_in_4bit: bool,
):
    """Load and cache Phi-2 plus its tokenizer."""
    _validate_phi2_model(model_name)

    token = _get_huggingface_token()
    load_kwargs = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(device_type)
    if load_in_4bit:
        if device_type != "cuda":
            raise ValueError("4-bit loading requires a CUDA GPU.")
        if BitsAndBytesConfig is None:
            raise ImportError("4-bit loading requires bitsandbytes.")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    elif device_type == "cuda":
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    elif device_type == "mps":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer


def clear_model_cache() -> None:
    """Release cached Hugging Face models and in-memory activations."""
    _load_cached_model_and_tokenizer.cache_clear()
    _ACTIVATION_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
    load_in_4bit: bool = False,
):
    """Load the locked Phi-2 model/tokenizer pair."""
    _validate_phi2_model(model_name)
    device = device or get_device()
    model, tokenizer = _load_cached_model_and_tokenizer(
        model_name,
        device.type,
        load_in_4bit,
    )
    return model, tokenizer, device


@torch.inference_mode()
def extract_last_token_activations(
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
    batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    use_memory_cache: bool = True,
) -> ActivationCache:
    """Extract Phi-2 hidden states at the last real token of each prompt.

    Right padding is forced when the tokenizer is loaded, so
    ``attention_mask.sum(dim=1) - 1`` is exactly the index of the final
    non-padding token in every sequence.
    """
    _validate_phi2_model(model_name)
    if not prompts:
        raise ValueError("At least one prompt is required.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if max_length < 8:
        raise ValueError("max_length is too small for the Phi-2 prompt template.")

    prompts_key = prompt_hash(prompts)
    cache_key = (model_name, load_in_4bit, batch_size, max_length, len(prompts), prompts_key)
    if use_memory_cache and cache_key in _ACTIVATION_CACHE:
        return _ACTIVATION_CACHE[cache_key]

    model, tokenizer, device = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
    )

    measured = tokenizer(prompts, truncation=True, max_length=max_length)
    fixed_length = max((len(ids) for ids in measured["input_ids"]), default=1)
    fixed_length = min(max_length, fixed_length)

    collected_batches: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Extracting Phi-2 activations",
        disable=not show_progress,
    ):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=fixed_length,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Phi-2 did not return hidden states.")

        layers = hidden_states[1:]
        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        attention_mask = encoded["attention_mask"]
        last_token_positions = attention_mask.sum(dim=1) - 1
        batch_positions = torch.arange(len(batch_prompts), device=attention_mask.device)
        batch_layers = torch.stack(
            [layer[batch_positions, last_token_positions] for layer in layers],
            dim=1,
        )
        collected_batches.append(batch_layers.float().cpu().numpy())

    activations = np.concatenate(collected_batches, axis=0).astype(np.float32, copy=False)
    result = ActivationCache(
        activations=activations,
        layers=layer_indices or [],
        model_name=model_name,
        prompt_hash=prompts_key,
    )

    if use_memory_cache:
        _ACTIVATION_CACHE[cache_key] = result
        while len(_ACTIVATION_CACHE) > _ACTIVATION_CACHE_MAXSIZE:
            _ACTIVATION_CACHE.pop(next(iter(_ACTIVATION_CACHE)))
    return result


def extract_or_load_last_token_activations(
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    force_recompute: bool = False,
    **kwargs,
) -> ActivationCache:
    """Load activations from disk when possible, otherwise extract and save."""
    _validate_phi2_model(model_name)
    max_length = int(kwargs.get("max_length", DEFAULT_MAX_LENGTH))
    if cache_dir is None:
        return extract_last_token_activations(prompts, model_name=model_name, **kwargs)

    path = activation_cache_path(
        cache_dir=cache_dir,
        prompts=prompts,
        model_name=model_name,
        max_length=max_length,
    )
    if path.exists() and not force_recompute:
        return load_activation_cache(path)

    cache = extract_last_token_activations(prompts, model_name=model_name, **kwargs)
    save_activation_cache(cache, path)
    return cache

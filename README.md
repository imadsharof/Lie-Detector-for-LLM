# Lie Detector for LLMs

This repository is a course-project reproduction of the main experimental idea behind
[mishajw/repeng](https://github.com/mishajw/repeng): use representation engineering to
read a truthfulness signal from a language model's hidden states.

The project does **not** claim to detect human-like intent. It studies whether true and
false candidate answers are linearly separable inside an LLM's internal activations.

## Research Question

The central question is:

> If a language model is given a question and a candidate answer, can a simple linear
> probe detect whether the answer is true by looking only at the model's hidden states?

This follows the RepEng/RepE methodology:

1. build grouped true/false candidate-answer datasets,
2. run a causal language model in white-box mode,
3. extract hidden states at the final prompt token,
4. train a linear truth probe,
5. evaluate whether the probe generalizes across datasets, layers, probe methods, and
   model families.

## Current Default Model

The local notebooks and `run_experiment.py` are currently configured to run on:

```text
microsoft/phi-2
```

Phi-2 has roughly 2.7B parameters and 32 transformer layers. It is the current local
baseline, but it is still much smaller than the original RepEng comparison model
(`Llama-2-13b-chat-hf`) and much smaller than the 70B models requested for the final
course comparison.

Any older tiny-model result should be treated only as a debugging sanity check, not as
the final project result.

## Relation to `mishajw/repeng`

The original `repeng` repository contains:

- a reusable `repeng` package,
- ELK/RepE/Geometry-of-Truth datasets,
- activation extraction scripts,
- probe implementations,
- a comparison workflow split into:
  - `experiments/comparison_dataset.py` for GPU activation extraction,
  - `experiments/comparison.py` for probe training, evaluation, and plotting.

The original comparison workflow uses `Llama-2-13b-chat-hf` and evaluates transfer
across many datasets and layers. This repository is a compact, course-friendly version
of that logic, plus a Colab notebook for comparing several modern LLMs.

## Repository Structure

- [src/lie_detector_llm/datasets.py](src/lie_detector_llm/datasets.py): grouped datasets and split helpers
- [src/lie_detector_llm/models.py](src/lie_detector_llm/models.py): Hugging Face model loading, optional 4-bit loading, activation extraction
- [src/lie_detector_llm/probes.py](src/lie_detector_llm/probes.py): DIM, LAT, LR, and grouped PCA probes
- [src/lie_detector_llm/experiment.py](src/lie_detector_llm/experiment.py): probe, transfer, layer-sweep, and matrix experiments
- [src/lie_detector_llm/plotting.py](src/lie_detector_llm/plotting.py): transfer and layer plots
- [run_experiment.py](run_experiment.py): script-style end-to-end run
- [data/raw/repeng/truthful.jsonl](data/raw/repeng/truthful.jsonl): local RepEng truthfulness data
- [notebooks](notebooks): explained English notebooks

## Notebook Workflow

Run the notebooks in this order:

1. [notebooks/01_method_overview.ipynb](notebooks/01_method_overview.ipynb):
   conceptual explanation of representation engineering and truth probes.
2. [notebooks/02_train_a_probe.ipynb](notebooks/02_train_a_probe.ipynb):
   trains one truth probe on one dataset.
3. [notebooks/03_generalization_study.ipynb](notebooks/03_generalization_study.ipynb):
   tests whether a probe trained on one dataset transfers to other datasets.
4. [notebooks/04_layer_sweep.ipynb](notebooks/04_layer_sweep.ipynb):
   measures which transformer layers encode the truth signal most clearly.
5. [notebooks/05_full_transfer_matrix.ipynb](notebooks/05_full_transfer_matrix.ipynb):
   builds the train-dataset by eval-dataset transfer matrix.
6. [notebooks/06_large_llm_comparison_colab.ipynb](notebooks/06_large_llm_comparison_colab.ipynb):
   Colab-oriented notebook for comparing Phi-2, Mistral, Llama 3 8B, and optional
   Llama 3 70B/Llama 3.3 70B models.

## Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For gated Hugging Face models such as Llama, create `.env` from `.env.example` and set:

```text
HF_TOKEN=your_huggingface_token
```

You must also accept the model license on Hugging Face before loading gated Meta models.

## Running the Script

```bash
python3 run_experiment.py
```

The script saves figures to `results/`.

## Example API Use

```python
from lie_detector_llm.datasets import build_dataset_collection
from lie_detector_llm.experiment import run_transfer_experiment

collection = build_dataset_collection(include_hf_datasets=False)
results = run_transfer_experiment(
    collection=collection,
    train_dataset_name="repeng_truthful",
    eval_dataset_names=collection.dataset_names(),
    model_name="microsoft/phi-2",
    probe_method="lr",
    layer_index=-1,
    split_evaluation=True,
)
print(results.summary_table())
```

## What Is Already Implemented

- White-box activation extraction from Hugging Face causal LMs.
- Grouped true/false candidate-answer evaluation.
- Four linear probe families:
  - DIM: difference in means,
  - LAT: PCA on pairwise differences,
  - LR: logistic regression,
  - PCA-G: grouped PCA.
- RepEng-style transfer experiments.
- Layer sweeps.
- Full transfer matrices.
- Optional 4-bit model loading for large models on CUDA.
- A Colab workflow for comparing multiple LLMs.

## What Still Limits the Project

The project is now a solid reproduction scaffold, but it is not a full paper-level clone
of `mishajw/repeng` yet. The main gaps are:

- fewer datasets than the full original comparison,
- fewer probe algorithms than the full `repeng` package,
- smaller local default model,
- no precomputed large activation cache,
- no final reported table yet for several 7B/8B/70B LLMs.

The most important next result for the professor is therefore the output of
`06_large_llm_comparison_colab.ipynb`: a table comparing grouped accuracy across several
LLMs on the same datasets and probe settings.

## Suggested Report Positioning

> This project reproduces the representation-engineering approach to truth probing:
> hidden states are extracted from language models, linear probes are trained to rank
> true answers above false answers, and the resulting truth signal is compared across
> datasets, layers, and LLM families.

## Acknowledgment

This work is inspired by [mishajw/repeng](https://github.com/mishajw/repeng) and the
Representation Engineering line of work on reading and controlling high-level concepts
inside transformer activations.

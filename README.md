# Lie Detector for LLM

This repository is a course project whose goal is to reproduce the main idea and, as far as possible, the experimental logic of the `repeng` project by Misha Johnson.

- Initial repository: <https://github.com/mishajw/repeng>
- Reference article: <https://www.lesswrong.com/posts/cmicXAAEuPGqcs9jw/how-well-do-truth-probes-generalise>

## Project Goal

The purpose of this project is to build a lie detector for AI systems by using **representation engineering**.

The central hypothesis is the following:

> a language model may encode information related to truthfulness inside its hidden activations, and a simple linear probe may be able to recover that signal.

In practical terms, the project tries to answer this question:

> if a model is given true and false candidate answers, can we detect which answer is true by looking only at the model's internal activations?

## Target Reproduction

The real target is not just to make a toy classifier, but to reproduce the style of result reported in the `repeng` ecosystem and in the LessWrong post:

1. build grouped datasets containing true and false candidate answers,
2. extract hidden activations from a language model,
3. train linear probes on those activations,
4. evaluate them in-distribution,
5. test whether they generalize to different datasets.

The LessWrong post studies whether **truth probes generalize out of distribution**. That is the scientific benchmark this project is moving toward.

## Current Status

This repository is currently a **compact reproduction**, not yet a full clone of the original `repeng` pipeline.

What is already implemented:

- a clean Python package under [src/lie_detector_llm](src/lie_detector_llm)
- grouped datasets inspired by RepE and Geometry of Truth
- activation extraction from a Hugging Face causal language model
- several probe families:
  - difference in means
  - logistic regression
  - grouped PCA
  - LAT
- three English notebooks for explanation and experimentation
- a first end-to-end validation on `distilgpt2`

What is **not finished yet** if the goal is to match the original paper-level reproduction more closely:

- using the same large benchmark collection as `repeng`
- running on stronger chat models comparable to those used in the original work
- sweeping many layers and many datasets systematically
- reproducing the exact reported accuracy and recovered-accuracy plots
- reproducing the full comparison pipeline from the original repository

So the honest answer is: **no, the project is not finished if the requirement is to reproduce the original repo's results closely**. It is now a solid starting point and a clean educational scaffold, but not yet the final reproduction.

## Repository Structure

- [src/lie_detector_llm/datasets.py](src/lie_detector_llm/datasets.py): grouped truth/falsehood datasets and split helpers
- [src/lie_detector_llm/models.py](src/lie_detector_llm/models.py): model loading and last-token activation extraction
- [src/lie_detector_llm/probes.py](src/lie_detector_llm/probes.py): linear probe implementations
- [src/lie_detector_llm/experiment.py](src/lie_detector_llm/experiment.py): end-to-end experiment runners
- [src/lie_detector_llm/plotting.py](src/lie_detector_llm/plotting.py): simple plotting utilities
- [notebooks/01_method_overview.ipynb](notebooks/01_method_overview.ipynb): conceptual explanation of the method
- [notebooks/02_train_a_probe.ipynb](notebooks/02_train_a_probe.ipynb): one full probe-training experiment
- [notebooks/03_generalization_study.ipynb](notebooks/03_generalization_study.ipynb): compact transfer/generalization study

## Method Overview

For each candidate answer, we create a prompt of the form:

```text
Consider the correctness of the answer to the following question:

Question: What is the capital of France?
Answer: Paris
The probability of the answer being correct is
```

We then:

1. tokenize the prompt,
2. run the model with hidden states enabled,
3. extract the hidden state of the last token,
4. treat that hidden vector as a feature representation,
5. train a linear probe that assigns higher scores to true answers than to false ones.

Because answers are grouped by question, evaluation is performed at the group level:

- score all answers for the same question,
- pick the highest-scoring one,
- check whether it is the true answer.

This gives a grouped accuracy metric, which is much closer to the evaluation logic used in the original RepEng-style work than a naive row-wise threshold.

## Why This Can Be Called a Lie Detector

Strictly speaking, this project currently detects a **truthfulness-related representation** rather than intent in a psychological sense.

That still matches the core research idea behind the project:

- if the model internally distinguishes correct from incorrect statements,
- and if that distinction can be read linearly from activations,
- then we can use that signal as a basis for detecting lies, falsehoods, or implausible claims.

In later extensions, this can be pushed closer to a literal lie detector by generating deliberate lies from the model and probing those activations directly.

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

If you want to use a gated or private Hugging Face model, create a local `.env` file from `.env.example` and set `HF_TOKEN`. The project loads that token automatically and `.env` is ignored by Git.

## Running the Project

### Notebook workflow

Run the notebooks in this order:

1. [notebooks/01_method_overview.ipynb](notebooks/01_method_overview.ipynb)
2. [notebooks/02_train_a_probe.ipynb](notebooks/02_train_a_probe.ipynb)
3. [notebooks/03_generalization_study.ipynb](notebooks/03_generalization_study.ipynb)

### Script-style example

```python
from lie_detector_llm.datasets import build_dataset_collection
from lie_detector_llm.experiment import run_transfer_experiment

collection = build_dataset_collection()
results = run_transfer_experiment(
    collection=collection,
    train_dataset_name="cities",
    eval_dataset_names=["cities", "larger_than", "qa"],
    model_name="distilgpt2",
    probe_method="lr",
    layer_index=-1,
)
print(results.summary_table())
```

## Current Validation Result

The current compact implementation was run end to end in the project environment with `distilgpt2` and a logistic regression probe trained on `cities`.

Observed grouped accuracies:

- `cities -> cities`: `0.80`
- `cities -> larger_than`: `0.70`
- `cities -> qa`: `0.125`

These numbers show that the pipeline works technically, but they are **not yet the same level of result as the original `repeng` reproduction**. That is expected: the current project uses a much smaller model and a much smaller dataset suite.

## What Must Be Done To Reproduce The Original Results More Faithfully

To get closer to the professor's stated objective, the next steps are:

1. replace the toy datasets with benchmark datasets closer to those used in `repeng`
2. add systematic train/validation/test splits by dataset
3. sweep multiple layers rather than keeping only one layer
4. run the full set of probe algorithms used in the comparison
5. use a stronger instruction-tuned model closer to the original experimental setup
6. reproduce the transfer matrix and summary plots from the LessWrong post

## Course Report Positioning

If you need one sentence to explain the project in a report, this is a good version:

> This project studies whether truthfulness is linearly represented inside a language model's hidden states, and whether that signal can be used as the basis of a lie detector that generalizes across datasets.

## Acknowledgment

This project is inspired by and attempts to reproduce the methodology introduced in [mishajw/repeng](https://github.com/mishajw/repeng) and discussed in the LessWrong article [How well do truth probes generalise?](https://www.lesswrong.com/posts/cmicXAAEuPGqcs9jw/how-well-do-truth-probes-generalise).
"""Dataset construction for Phi-2 truth-probe generalisation.

All rows are converted to the same contrastive format:

* one ``group_id`` is one question or claim context,
* each group contains candidate answers/statements,
* at least one candidate is true and at least one is false,
* every candidate is wrapped in the same non-chat Phi-2 prompt template.

The default collection mixes dataset families used in the original
generalisation study: DLK-style text classification, RepE-style multiple
choice QA, TruthfulQA, the local repeng truthful JSONL file, and a small
GoT-style ``facts`` dataset of simple factual statements.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from random import Random
from typing import Callable

import pandas as pd


PHI2_PROMPT_TEMPLATE = (
    "Task: Decide whether the candidate answer is correct.\n\n"
    "Question:\n{question}\n\n"
    "Candidate answer:\n{answer}\n\n"
    "Correctness:"
)

# Backwards-compatible name used by earlier notebooks.
REPE_TEMPLATE = PHI2_PROMPT_TEMPLATE

ALL_DATASET_NAMES: list[str] = [
    "facts",
    "amazon_polarity",
    "dbpedia_14",
    "ag_news",
    "imdb",
    "rte",
    "boolq",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "commonsense_qa",
    "piqa",
    "truthful_qa",
    "repeng_truthful",
]

DEFAULT_DATASET_NAMES: list[str] = [
    "facts",
    "dbpedia_14",
    "amazon_polarity",
    "boolq",
    "arc_easy",
    "truthful_qa",
    "repeng_truthful",
]

DATASET_ALIASES: dict[str, str] = {
    "dbpedia": "dbpedia_14",
    "open_book_qa": "openbookqa",
    "common_sense_qa": "commonsense_qa",
    "commonsenseqa": "commonsense_qa",
}

_MAX_CONTEXT_CHARS = 1_200


@dataclass(frozen=True)
class CandidateExample:
    dataset_name: str
    group_id: str
    question: str
    answer: str
    prompt: str
    label: bool


@dataclass(frozen=True)
class DatasetCollection:
    frame: pd.DataFrame

    def dataset_names(self) -> list[str]:
        return sorted(self.frame["dataset_name"].unique().tolist())

    def subset(self, dataset_name: str) -> pd.DataFrame:
        dataset_name = normalise_dataset_name(dataset_name)
        subset = self.frame[self.frame["dataset_name"] == dataset_name].copy()
        if subset.empty:
            raise ValueError(f"Unknown or empty dataset: {dataset_name}")
        return subset.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        return (
            self.frame.groupby("dataset_name", as_index=False)
            .agg(groups=("group_id", "nunique"), prompts=("prompt", "size"))
            .sort_values("dataset_name")
        )


def normalise_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = os.getenv(name)
        if token:
            return token
    return None


def _load_hf(path: str, name: str | None, split: str):
    from datasets import load_dataset

    kwargs = {"split": split}
    token = _hf_token()
    if token:
        kwargs["token"] = token
    if name is None:
        return load_dataset(path, **kwargs)
    return load_dataset(path, name, **kwargs)


def _clip(text: object) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX_CONTEXT_CHARS else text[:_MAX_CONTEXT_CHARS] + " ..."


def _make_prompt(question: str, answer: str) -> str:
    return PHI2_PROMPT_TEMPLATE.format(question=_clip(question), answer=_clip(answer))


def _emit(
    rows: list[CandidateExample],
    dataset_name: str,
    group_id: str,
    question: str,
    candidates: list[tuple[str, bool]],
) -> None:
    if len(candidates) < 2:
        return
    labels = [bool(label) for _, label in candidates]
    if not any(labels) or all(labels):
        return
    for answer, label in candidates:
        answer = _clip(answer)
        question = _clip(question)
        rows.append(
            CandidateExample(
                dataset_name=dataset_name,
                group_id=group_id,
                question=question,
                answer=answer,
                prompt=_make_prompt(question, answer),
                label=bool(label),
            )
        )


def _shuffled_indices(n: int, seed: int) -> list[int]:
    indices = list(range(n))
    Random(seed).shuffle(indices)
    return indices


def _build_binary_text_dataset(
    dataset_name: str,
    hf_path: str,
    hf_name: str | None,
    split: str,
    text_field: str,
    label_field: str,
    question_text: str,
    class_names: list[str],
    max_groups: int,
    seed: int,
) -> list[CandidateExample]:
    ds = _load_hf(hf_path, hf_name, split)
    rows: list[CandidateExample] = []
    groups = 0
    for idx in _shuffled_indices(len(ds), seed):
        if groups >= max_groups:
            break
        item = ds[idx]
        true_label = int(item[label_field])
        if true_label not in (0, 1):
            continue
        question = f"{question_text}\n\n{_clip(item[text_field])}"
        candidates = [
            (class_names[0], true_label == 0),
            (class_names[1], true_label == 1),
        ]
        _emit(rows, dataset_name, f"{dataset_name}::{idx}", question, candidates)
        groups += 1
    return rows


def _build_multiclass_text_dataset(
    dataset_name: str,
    hf_path: str,
    hf_name: str | None,
    split: str,
    text_field: str,
    label_field: str,
    question_text: str,
    class_names: list[str],
    max_groups: int,
    seed: int,
    num_distractors: int = 3,
) -> list[CandidateExample]:
    ds = _load_hf(hf_path, hf_name, split)
    rng = Random(seed)
    rows: list[CandidateExample] = []
    groups = 0
    for idx in _shuffled_indices(len(ds), seed):
        if groups >= max_groups:
            break
        item = ds[idx]
        true_label = int(item[label_field])
        if not 0 <= true_label < len(class_names):
            continue
        wrong = [name for i, name in enumerate(class_names) if i != true_label]
        rng.shuffle(wrong)
        candidates = [(class_names[true_label], True)]
        candidates.extend((name, False) for name in wrong[:num_distractors])
        rng.shuffle(candidates)
        question = f"{question_text}\n\n{_clip(item[text_field])}"
        _emit(rows, dataset_name, f"{dataset_name}::{idx}", question, candidates)
        groups += 1
    return rows


def _build_multiple_choice_dataset(
    dataset_name: str,
    hf_path: str,
    hf_name: str | None,
    split: str,
    max_groups: int,
    seed: int,
    extract: Callable[[dict], tuple[str, list[tuple[str, bool]]] | None],
) -> list[CandidateExample]:
    ds = _load_hf(hf_path, hf_name, split)
    rows: list[CandidateExample] = []
    groups = 0
    for idx in _shuffled_indices(len(ds), seed):
        if groups >= max_groups:
            break
        parsed = extract(ds[idx])
        if parsed is None:
            continue
        question, candidates = parsed
        _emit(rows, dataset_name, f"{dataset_name}::{idx}", question, candidates)
        if any(label for _, label in candidates) and not all(label for _, label in candidates):
            groups += 1
    return rows


def _facts(max_groups: int, seed: int) -> list[CandidateExample]:
    """Small GoT-style factual contrast dataset built locally."""
    city_country = [
        ("Paris", "France"),
        ("Tokyo", "Japan"),
        ("Ottawa", "Canada"),
        ("Canberra", "Australia"),
        ("Cairo", "Egypt"),
        ("Nairobi", "Kenya"),
        ("Madrid", "Spain"),
        ("Rome", "Italy"),
        ("Brasilia", "Brazil"),
        ("Seoul", "South Korea"),
        ("Dublin", "Ireland"),
        ("Lisbon", "Portugal"),
        ("Warsaw", "Poland"),
        ("Athens", "Greece"),
        ("Hanoi", "Vietnam"),
        ("Lima", "Peru"),
    ]
    countries = [country for _, country in city_country]
    rng = Random(seed)
    groups: list[list[tuple[str, bool]]] = []

    for city, country in city_country:
        wrong_country = rng.choice([candidate for candidate in countries if candidate != country])
        groups.append(
            [
                (f"The city of {city} is in {country}.", True),
                (f"The city of {city} is in {wrong_country}.", False),
            ]
        )

    number_pairs = [(2, 9), (4, 17), (8, 3), (12, 19), (21, 13), (31, 44), (50, 7)]
    for left, right in number_pairs:
        if left < right:
            groups.append(
                [
                    (f"{left} is less than {right}.", True),
                    (f"{left} is greater than {right}.", False),
                ]
            )
        else:
            groups.append(
                [
                    (f"{left} is greater than {right}.", True),
                    (f"{left} is less than {right}.", False),
                ]
            )

    rng.shuffle(groups)
    rows: list[CandidateExample] = []
    question = "Which statement is factually correct?"
    for idx, candidates in enumerate(groups[:max_groups]):
        rng.shuffle(candidates)
        _emit(rows, "facts", f"facts::{idx}", question, candidates)
    return rows


def _imdb(max_groups: int, seed: int) -> list[CandidateExample]:
    return _build_binary_text_dataset(
        "imdb",
        "stanfordnlp/imdb",
        None,
        "test",
        "text",
        "label",
        "What is the sentiment of the following movie review?",
        ["negative", "positive"],
        max_groups,
        seed,
    )


def _amazon_polarity(max_groups: int, seed: int) -> list[CandidateExample]:
    return _build_binary_text_dataset(
        "amazon_polarity",
        "fancyzhx/amazon_polarity",
        None,
        "test",
        "content",
        "label",
        "What is the sentiment of the following product review?",
        ["negative", "positive"],
        max_groups,
        seed,
    )


def _ag_news(max_groups: int, seed: int) -> list[CandidateExample]:
    return _build_multiclass_text_dataset(
        "ag_news",
        "fancyzhx/ag_news",
        None,
        "test",
        "text",
        "label",
        "What is the topic of the following news article?",
        ["World", "Sports", "Business", "Science and technology"],
        max_groups,
        seed,
    )


def _dbpedia_14(max_groups: int, seed: int) -> list[CandidateExample]:
    return _build_multiclass_text_dataset(
        "dbpedia_14",
        "fancyzhx/dbpedia_14",
        None,
        "test",
        "content",
        "label",
        "What category does the following text belong to?",
        [
            "Company",
            "Educational institution",
            "Artist",
            "Athlete",
            "Office holder",
            "Mean of transportation",
            "Building",
            "Natural place",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "Written work",
        ],
        max_groups,
        seed,
    )


def _rte(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        question = (
            f"Premise: {_clip(item['sentence1'])}\n"
            f"Hypothesis: {_clip(item['sentence2'])}\n\n"
            "Does the premise entail the hypothesis?"
        )
        entailment = int(item["label"]) == 0
        return question, [("Yes", entailment), ("No", not entailment)]

    return _build_multiple_choice_dataset("rte", "nyu-mll/glue", "rte", "validation", max_groups, seed, extract)


def _boolq(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        question = f"Passage: {_clip(item['passage'])}\n\nQuestion: {item['question']}?"
        answer = bool(item["answer"])
        return question, [("Yes", answer), ("No", not answer)]

    return _build_multiple_choice_dataset("boolq", "google/boolq", None, "validation", max_groups, seed, extract)


def _arc(dataset_name: str, hf_name: str, max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        choices = item["choices"]
        texts, labels = choices["text"], choices["label"]
        answer_key = item["answerKey"]
        candidates = [(text, label == answer_key) for text, label in zip(texts, labels)]
        return item["question"], candidates

    return _build_multiple_choice_dataset(
        dataset_name,
        "allenai/ai2_arc",
        hf_name,
        "validation",
        max_groups,
        seed,
        extract,
    )


def _openbookqa(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        choices = item["choices"]
        texts, labels = choices["text"], choices["label"]
        answer_key = item["answerKey"]
        candidates = [(text, label == answer_key) for text, label in zip(texts, labels)]
        return item["question_stem"], candidates

    return _build_multiple_choice_dataset(
        "openbookqa",
        "allenai/openbookqa",
        "main",
        "validation",
        max_groups,
        seed,
        extract,
    )


def _commonsense_qa(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        choices = item["choices"]
        texts, labels = choices["text"], choices["label"]
        answer_key = item["answerKey"]
        if not answer_key:
            return None
        candidates = [(text, label == answer_key) for text, label in zip(texts, labels)]
        return item["question"], candidates

    return _build_multiple_choice_dataset(
        "commonsense_qa",
        "tau/commonsense_qa",
        None,
        "validation",
        max_groups,
        seed,
        extract,
    )


def _piqa(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        question = f"Goal: {_clip(item['goal'])}\n\nWhich solution is more appropriate?"
        correct = int(item["label"])
        return question, [
            (_clip(item["sol1"]), correct == 0),
            (_clip(item["sol2"]), correct == 1),
        ]

    return _build_multiple_choice_dataset("piqa", "ybisk/piqa", None, "validation", max_groups, seed, extract)


def _truthful_qa(max_groups: int, seed: int) -> list[CandidateExample]:
    def extract(item: dict):
        targets = item["mc1_targets"]
        choices, labels = targets["choices"], targets["labels"]
        candidates = [(choice, bool(label)) for choice, label in zip(choices, labels)]
        return item["question"], candidates

    return _build_multiple_choice_dataset(
        "truthful_qa",
        "truthfulqa/truthful_qa",
        "multiple_choice",
        "validation",
        max_groups,
        seed,
        extract,
    )


def _repeng_truthful(dataset_path: Path, max_groups: int, seed: int) -> list[CandidateExample]:
    if not dataset_path.exists():
        return []
    honest, dishonest = [], []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line).get("value", {})
            statement = str(value.get("statement", "")).strip()
            if not statement:
                continue
            if bool(value.get("honest", False)):
                honest.append(statement)
            else:
                dishonest.append(statement)

    rng = Random(seed)
    rng.shuffle(honest)
    rng.shuffle(dishonest)
    rows: list[CandidateExample] = []
    question = "Is this self-report statement truthful?"
    for idx in range(min(len(honest), len(dishonest), max_groups)):
        _emit(
            rows,
            "repeng_truthful",
            f"repeng_truthful::{idx}",
            question,
            [(honest[idx], True), (dishonest[idx], False)],
        )
    return rows


def build_dataset_collection(
    dataset_names: list[str] | None = None,
    max_groups: int = 80,
    seed: int = 0,
    shuffle: bool = True,
    repeng_truthful_path: str | Path | None = None,
    verbose: bool = True,
) -> DatasetCollection:
    """Build a collection of grouped true/false candidate prompts."""
    requested = [normalise_dataset_name(name) for name in (dataset_names or DEFAULT_DATASET_NAMES)]
    if repeng_truthful_path is not None:
        truthful_path = Path(repeng_truthful_path)
    else:
        truthful_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "repeng" / "truthful.jsonl"

    loaders: dict[str, Callable[[], list[CandidateExample]]] = {
        "facts": lambda: _facts(max_groups, seed),
        "imdb": lambda: _imdb(max_groups, seed),
        "amazon_polarity": lambda: _amazon_polarity(max_groups, seed),
        "ag_news": lambda: _ag_news(max_groups, seed),
        "dbpedia_14": lambda: _dbpedia_14(max_groups, seed),
        "rte": lambda: _rte(max_groups, seed),
        "boolq": lambda: _boolq(max_groups, seed),
        "arc_easy": lambda: _arc("arc_easy", "ARC-Easy", max_groups, seed),
        "arc_challenge": lambda: _arc("arc_challenge", "ARC-Challenge", max_groups, seed),
        "openbookqa": lambda: _openbookqa(max_groups, seed),
        "commonsense_qa": lambda: _commonsense_qa(max_groups, seed),
        "piqa": lambda: _piqa(max_groups, seed),
        "truthful_qa": lambda: _truthful_qa(max_groups, seed),
        "repeng_truthful": lambda: _repeng_truthful(truthful_path, max_groups, seed),
    }

    rows: list[CandidateExample] = []
    for name in requested:
        if name not in loaders:
            if verbose:
                print(f"[skip]  {name}: unknown dataset name")
            continue
        try:
            dataset_rows = loaders[name]()
        except Exception as error:  # noqa: BLE001 - one failed dataset should not kill the run
            if verbose:
                print(f"[skip]  {name}: {type(error).__name__}: {error}")
            continue
        if not dataset_rows:
            if verbose:
                print(f"[skip]  {name}: produced no rows")
            continue
        rows.extend(dataset_rows)
        if verbose:
            n_groups = len({row.group_id for row in dataset_rows})
            print(f"[ok]    {name}: {n_groups} groups, {len(dataset_rows)} candidate prompts")

    if not rows:
        raise RuntimeError("No dataset could be loaded. Check the dataset names and network access.")

    frame = pd.DataFrame([row.__dict__ for row in rows])
    if shuffle:
        order = list(frame.index)
        Random(seed).shuffle(order)
        frame = frame.loc[order].reset_index(drop=True)
    return DatasetCollection(frame=frame)


def split_groups(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Split at group level to prevent candidates from leaking across splits."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    group_ids = frame["group_id"].drop_duplicates().tolist()
    Random(seed).shuffle(group_ids)
    n_groups = len(group_ids)
    if n_groups < 3:
        raise ValueError("At least three groups are needed for train/validation/test splits.")

    train_end = max(1, int(n_groups * train_fraction))
    validation_end = int(n_groups * (train_fraction + validation_fraction))
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, n_groups - 1)

    train_groups = set(group_ids[:train_end])
    validation_groups = set(group_ids[train_end:validation_end])
    test_groups = set(group_ids[validation_end:])

    return {
        "train": frame[frame["group_id"].isin(train_groups)].reset_index(drop=True),
        "validation": frame[frame["group_id"].isin(validation_groups)].reset_index(drop=True),
        "test": frame[frame["group_id"].isin(test_groups)].reset_index(drop=True),
    }

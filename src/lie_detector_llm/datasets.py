from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

import pandas as pd


REPE_TEMPLATE = (
    "Consider the correctness of the answer to the following question:\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being correct is"
)


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
        subset = self.frame[self.frame["dataset_name"] == dataset_name].copy()
        if subset.empty:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return subset.reset_index(drop=True)


def _make_prompt(question: str, answer: str) -> str:
    return REPE_TEMPLATE.format(question=question, answer=answer)


def _build_cities_dataset() -> list[CandidateExample]:
    rows: list[CandidateExample] = []
    data = [
        ("Paris", "France", ["Italy", "Germany", "Spain"]),
        ("Tokyo", "Japan", ["China", "South Korea", "Thailand"]),
        ("Nairobi", "Kenya", ["Nigeria", "South Africa", "Ghana"]),
        ("Lima", "Peru", ["Chile", "Argentina", "Colombia"]),
        ("Ottawa", "Canada", ["United States", "Australia", "Ireland"]),
        ("Cairo", "Egypt", ["Morocco", "Jordan", "Turkey"]),
        ("Bern", "Switzerland", ["Austria", "Belgium", "Sweden"]),
        ("Bangkok", "Thailand", ["Vietnam", "Malaysia", "Indonesia"]),
        ("Canberra", "Australia", ["New Zealand", "United Kingdom", "Canada"]),
        ("Bogota", "Colombia", ["Venezuela", "Ecuador", "Peru"]),
    ]
    for city, correct_country, wrong_countries in data:
        question = f"Which country contains the city {city}?"
        group_id = f"cities::{city}"
        all_answers = [correct_country, *wrong_countries]
        for answer in all_answers:
            rows.append(
                CandidateExample(
                    dataset_name="cities",
                    group_id=group_id,
                    question=question,
                    answer=answer,
                    prompt=_make_prompt(question, answer),
                    label=answer == correct_country,
                )
            )
    return rows


def _build_larger_than_dataset() -> list[CandidateExample]:
    rows: list[CandidateExample] = []
    pairs = [
        (12, 7),
        (31, 5),
        (18, 42),
        (90, 12),
        (4, 15),
        (77, 56),
        (101, 99),
        (23, 24),
        (300, 10),
        (8, 2),
    ]
    for left, right in pairs:
        question = f"Is {left} larger than {right}?"
        group_id = f"larger_than::{left}::{right}"
        truth_value = "Yes" if left > right else "No"
        false_value = "No" if truth_value == "Yes" else "Yes"
        for answer, label in [(truth_value, True), (false_value, False)]:
            rows.append(
                CandidateExample(
                    dataset_name="larger_than",
                    group_id=group_id,
                    question=question,
                    answer=answer,
                    prompt=_make_prompt(question, answer),
                    label=label,
                )
            )
    return rows


def _build_qa_dataset() -> list[CandidateExample]:
    rows: list[CandidateExample] = []
    data = [
        (
            "What gas do plants absorb from the atmosphere?",
            "Carbon dioxide",
            ["Oxygen", "Nitrogen", "Helium"],
        ),
        (
            "Who wrote Hamlet?",
            "William Shakespeare",
            ["Charles Dickens", "Jane Austen", "Homer"],
        ),
        (
            "What is the largest planet in the Solar System?",
            "Jupiter",
            ["Mars", "Venus", "Saturn"],
        ),
        (
            "What is the boiling point of water at sea level in degrees Celsius?",
            "100",
            ["0", "50", "212"],
        ),
        (
            "Which organ pumps blood through the human body?",
            "Heart",
            ["Liver", "Lung", "Kidney"],
        ),
        (
            "What is the capital of Portugal?",
            "Lisbon",
            ["Madrid", "Porto", "Rome"],
        ),
        (
            "Which element has the chemical symbol Au?",
            "Gold",
            ["Silver", "Argon", "Copper"],
        ),
        (
            "What is the smallest prime number?",
            "2",
            ["1", "3", "5"],
        ),
    ]
    for index, (question, correct_answer, wrong_answers) in enumerate(data):
        group_id = f"qa::{index}"
        for answer in [correct_answer, *wrong_answers]:
            rows.append(
                CandidateExample(
                    dataset_name="qa",
                    group_id=group_id,
                    question=question,
                    answer=answer,
                    prompt=_make_prompt(question, answer),
                    label=answer == correct_answer,
                )
            )
    return rows


def _build_repeng_truthful_dataset(
    dataset_path: Path,
    max_pairs: int = 40,
    seed: int = 0,
) -> list[CandidateExample]:
    rows: list[CandidateExample] = []
    if not dataset_path.exists():
        return rows

    honest_statements: list[str] = []
    dishonest_statements: list[str] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            value = parsed.get("value", {})
            statement = str(value.get("statement", "")).strip()
            is_honest = bool(value.get("honest", False))
            if not statement:
                continue
            if is_honest:
                honest_statements.append(statement)
            else:
                dishonest_statements.append(statement)

    random = Random(seed)
    random.shuffle(honest_statements)
    random.shuffle(dishonest_statements)

    pair_count = min(len(honest_statements), len(dishonest_statements), max_pairs)
    question = "Is this self-report statement truthful?"

    for index in range(pair_count):
        group_id = f"repeng_truthful::{index}"
        true_answer = honest_statements[index]
        false_answer = dishonest_statements[index]

        rows.append(
            CandidateExample(
                dataset_name="repeng_truthful",
                group_id=group_id,
                question=question,
                answer=true_answer,
                prompt=_make_prompt(question, true_answer),
                label=True,
            )
        )
        rows.append(
            CandidateExample(
                dataset_name="repeng_truthful",
                group_id=group_id,
                question=question,
                answer=false_answer,
                prompt=_make_prompt(question, false_answer),
                label=False,
            )
        )

    return rows


def build_dataset_collection(
    shuffle: bool = True,
    seed: int = 0,
    include_repeng_truthful: bool = True,
    repeng_truthful_path: str | Path | None = None,
) -> DatasetCollection:
    rows = [
        *_build_cities_dataset(),
        *_build_larger_than_dataset(),
        *_build_qa_dataset(),
    ]

    if include_repeng_truthful:
        if repeng_truthful_path is not None:
            dataset_path = Path(repeng_truthful_path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            dataset_path = project_root / "data" / "raw" / "repeng" / "truthful.jsonl"
        rows.extend(_build_repeng_truthful_dataset(dataset_path=dataset_path, seed=seed))

    frame = pd.DataFrame([row.__dict__ for row in rows])
    if shuffle:
        random = Random(seed)
        order = list(frame.index)
        random.shuffle(order)
        frame = frame.loc[order].reset_index(drop=True)
    return DatasetCollection(frame=frame)


def split_groups(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Fractions must be strictly positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be smaller than 1.")

    group_ids = frame["group_id"].drop_duplicates().tolist()
    random = Random(seed)
    random.shuffle(group_ids)

    n_groups = len(group_ids)
    train_end = max(1, int(n_groups * train_fraction))
    validation_end = max(train_end + 1, int(n_groups * (train_fraction + validation_fraction)))

    train_groups = set(group_ids[:train_end])
    validation_groups = set(group_ids[train_end:validation_end])
    test_groups = set(group_ids[validation_end:])

    return {
        "train": frame[frame["group_id"].isin(train_groups)].reset_index(drop=True),
        "validation": frame[frame["group_id"].isin(validation_groups)].reset_index(drop=True),
        "test": frame[frame["group_id"].isin(test_groups)].reset_index(drop=True),
    }
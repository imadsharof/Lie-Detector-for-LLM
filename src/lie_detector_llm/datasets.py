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
    """Build a factual geography dataset: 'Which country contains city X?'

    Each question has 1 correct country and 3 plausible distractors,
    yielding 4 candidate prompts per group.  30 cities across all
    continents are included to give the probe enough variance.
    """
    rows: list[CandidateExample] = []
    data = [
        # Europe
        ("Paris", "France", ["Italy", "Germany", "Spain"]),
        ("Bern", "Switzerland", ["Austria", "Belgium", "Sweden"]),
        ("Lisbon", "Portugal", ["Spain", "Brazil", "Italy"]),
        ("Prague", "Czech Republic", ["Poland", "Austria", "Hungary"]),
        ("Athens", "Greece", ["Turkey", "Italy", "Cyprus"]),
        ("Dublin", "Ireland", ["United Kingdom", "Iceland", "Scotland"]),
        ("Helsinki", "Finland", ["Sweden", "Norway", "Estonia"]),
        ("Warsaw", "Poland", ["Germany", "Czech Republic", "Ukraine"]),
        # Asia
        ("Tokyo", "Japan", ["China", "South Korea", "Thailand"]),
        ("Bangkok", "Thailand", ["Vietnam", "Malaysia", "Indonesia"]),
        ("Seoul", "South Korea", ["Japan", "China", "North Korea"]),
        ("Hanoi", "Vietnam", ["Thailand", "Cambodia", "Laos"]),
        ("Delhi", "India", ["Pakistan", "Bangladesh", "Nepal"]),
        ("Ankara", "Turkey", ["Greece", "Syria", "Iran"]),
        # Americas
        ("Lima", "Peru", ["Chile", "Argentina", "Colombia"]),
        ("Ottawa", "Canada", ["United States", "Australia", "Ireland"]),
        ("Bogota", "Colombia", ["Venezuela", "Ecuador", "Peru"]),
        ("Havana", "Cuba", ["Mexico", "Jamaica", "Dominican Republic"]),
        ("Santiago", "Chile", ["Argentina", "Peru", "Bolivia"]),
        ("Brasilia", "Brazil", ["Argentina", "Portugal", "Colombia"]),
        ("Quito", "Ecuador", ["Colombia", "Peru", "Bolivia"]),
        # Africa
        ("Nairobi", "Kenya", ["Nigeria", "South Africa", "Ghana"]),
        ("Cairo", "Egypt", ["Morocco", "Jordan", "Turkey"]),
        ("Accra", "Ghana", ["Nigeria", "Ivory Coast", "Togo"]),
        ("Dakar", "Senegal", ["Mali", "Gambia", "Guinea"]),
        ("Addis Ababa", "Ethiopia", ["Kenya", "Sudan", "Somalia"]),
        # Oceania & Middle East
        ("Canberra", "Australia", ["New Zealand", "United Kingdom", "Canada"]),
        ("Wellington", "New Zealand", ["Australia", "Fiji", "Samoa"]),
        ("Riyadh", "Saudi Arabia", ["United Arab Emirates", "Iraq", "Qatar"]),
        ("Tehran", "Iran", ["Iraq", "Afghanistan", "Turkey"]),
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
    """Build a logical/numerical dataset: 'Is A larger than B?'

    Each question has exactly 2 candidates (Yes / No), one correct.
    30 pairs are included with a mix of easy and tricky comparisons.
    """
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
        # additional pairs for larger dataset
        (55, 53),
        (1000, 999),
        (3, 30),
        (67, 89),
        (250, 125),
        (14, 41),
        (99, 100),
        (500, 50),
        (7, 70),
        (33, 33),
        (48, 47),
        (201, 200),
        (9, 11),
        (150, 151),
        (88, 44),
        (6, 60),
        (444, 443),
        (19, 20),
        (75, 57),
        (1, 1000),
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
    """Build a general-knowledge QA dataset spanning science, history, and more.

    Each question has 1 correct answer and 3 plausible distractors
    (4 candidates per group).  30 questions are included.
    """
    rows: list[CandidateExample] = []
    data = [
        # Science
        (
            "What gas do plants absorb from the atmosphere?",
            "Carbon dioxide",
            ["Oxygen", "Nitrogen", "Helium"],
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
            "Which element has the chemical symbol Au?",
            "Gold",
            ["Silver", "Argon", "Copper"],
        ),
        (
            "What is the smallest prime number?",
            "2",
            ["1", "3", "5"],
        ),
        (
            "What is the speed of light in vacuum approximately in km/s?",
            "300000",
            ["150000", "1000000", "30000"],
        ),
        (
            "What is the chemical formula for water?",
            "H2O",
            ["CO2", "NaCl", "O2"],
        ),
        (
            "How many chromosomes do humans have?",
            "46",
            ["23", "48", "44"],
        ),
        (
            "Which planet is closest to the Sun?",
            "Mercury",
            ["Venus", "Mars", "Earth"],
        ),
        # History & Literature
        (
            "Who wrote Hamlet?",
            "William Shakespeare",
            ["Charles Dickens", "Jane Austen", "Homer"],
        ),
        (
            "In which year did World War II end?",
            "1945",
            ["1939", "1944", "1950"],
        ),
        (
            "Who painted the Mona Lisa?",
            "Leonardo da Vinci",
            ["Michelangelo", "Raphael", "Picasso"],
        ),
        (
            "Which ancient civilization built the pyramids of Giza?",
            "Egyptians",
            ["Romans", "Greeks", "Mesopotamians"],
        ),
        (
            "Who discovered penicillin?",
            "Alexander Fleming",
            ["Louis Pasteur", "Marie Curie", "Joseph Lister"],
        ),
        (
            "What year did the French Revolution begin?",
            "1789",
            ["1776", "1804", "1815"],
        ),
        (
            "Who wrote The Origin of Species?",
            "Charles Darwin",
            ["Gregor Mendel", "Alfred Wallace", "Thomas Huxley"],
        ),
        (
            "Which empire was ruled by Julius Caesar?",
            "Roman Empire",
            ["Greek Empire", "Persian Empire", "Ottoman Empire"],
        ),
        # Geography & General Knowledge
        (
            "What is the capital of Portugal?",
            "Lisbon",
            ["Madrid", "Porto", "Rome"],
        ),
        (
            "What is the longest river in the world?",
            "Nile",
            ["Amazon", "Mississippi", "Yangtze"],
        ),
        (
            "Which continent has the most countries?",
            "Africa",
            ["Asia", "Europe", "South America"],
        ),
        (
            "What is the tallest mountain in the world?",
            "Mount Everest",
            ["K2", "Kangchenjunga", "Mont Blanc"],
        ),
        (
            "Which ocean is the largest?",
            "Pacific Ocean",
            ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean"],
        ),
        (
            "What is the most spoken language in the world by native speakers?",
            "Mandarin Chinese",
            ["English", "Spanish", "Hindi"],
        ),
        (
            "How many continents are there?",
            "7",
            ["5", "6", "8"],
        ),
        # Math & Computing
        (
            "What is the square root of 144?",
            "12",
            ["14", "11", "13"],
        ),
        (
            "What does CPU stand for?",
            "Central Processing Unit",
            ["Central Power Unit", "Computer Processing Unit", "Central Program Unit"],
        ),
        (
            "What is the value of pi rounded to two decimal places?",
            "3.14",
            ["3.41", "3.12", "2.14"],
        ),
        (
            "Who is considered the father of computer science?",
            "Alan Turing",
            ["Charles Babbage", "John von Neumann", "Ada Lovelace"],
        ),
        (
            "What does HTML stand for?",
            "HyperText Markup Language",
            ["HyperText Machine Language", "HighText Markup Language", "HyperTool Markup Language"],
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
    max_pairs: int = 100,
    seed: int = 0,
) -> list[CandidateExample]:
    """Build the RepEng self-report truthfulness dataset from a JSONL file.

    Each line in the JSONL contains a statement and an 'honest' flag.
    We pair honest and dishonest statements into contrast groups of 2.
    This dataset tests meta-cognitive truthfulness (the model judging
    its own honesty) rather than factual correctness.

    Source: https://github.com/mishajw/repeng — truthful.jsonl
    """
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
    """Assemble all sub-datasets into a single DatasetCollection.

    Returns a DatasetCollection containing cities, larger_than, qa,
    and (optionally) repeng_truthful datasets.  Each row has columns:
    dataset_name, group_id, question, answer, prompt, label.
    """
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
    """Split question groups into train / validation / test sets.

    Splitting is done at the *group* level (not the row level) so that
    all candidate answers for the same question stay in the same split.
    This prevents data leakage between splits.
    """
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
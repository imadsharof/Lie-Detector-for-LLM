"""
questions.py — Chargement et préparation des datasets benchmark
================================================================
Ce module charge les fichiers JSON du dossier data/raw/pacchiardi_2024_benchmarks/
et les transforme en un format unifié pour le pipeline de lie detection.

Datasets disponibles (du paper Pacchiardi et al., 2023):
- General knowledge: known_1000.json, questions_1000_all.json, wikidata_refined.json, sciq
- Common sense reasoning: commonsense_QA_v2_dev.json
- Mathematics: svamp (addition, subtraction, multiplication, division)
- Synthetic facts: questions_made_up_all.json
"""

import json
import os
import random
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Configuration des chemins
# ──────────────────────────────────────────────

DEFAULT_RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "pacchiardi_2024_benchmarks"


# ──────────────────────────────────────────────
# Fonctions de chargement
# ──────────────────────────────────────────────

def load_json(filepath: str | Path) -> dict | list:
    """Charge un fichier JSON et retourne son contenu."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filepath: str | Path) -> list[dict]:
    """Charge un fichier JSON Lines (1 objet JSON par ligne)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def list_available_datasets(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[str]:
    """Liste tous les fichiers JSON disponibles dans le dossier raw."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")
    return sorted([f.name for f in raw_dir.glob("*.json")])


# ──────────────────────────────────────────────
# Parsers pour chaque dataset
# ──────────────────────────────────────────────

def _parse_known_1000(data: list[dict]) -> list[dict]:
    """
    Parse known_1000.json
    Format réel : [{"known_id": 0, "subject": "Vinson Massif", "attribute": "Antarctica",
                     "template": "{} is located in the continent", "prompt": "Vinson Massif is located in the continent of"}, ...]
    """
    questions = []
    for item in data:
        subject = item.get("subject", "")
        attribute = item.get("attribute", "")
        template = item.get("template", "")

        if subject and attribute and template:
            statement = template.replace("{}", subject)
            question_text = f"What is: {statement}?"

            questions.append({
                "question": question_text,
                "answer": attribute,
                "category": "general_knowledge",
                "source": "known_1000",
                "metadata": item,
            })
    return questions


def _parse_questions_1000(data: dict) -> list[dict]:
    """
    Parse questions_1000_all.json
    Format réel : {"0": {"question": "...", "answer": "...", "category": "...", "false_answer": "..."}, ...}
    """
    questions = []
    for key, item in data.items():
        q = item.get("question", "").strip()
        a = item.get("answer", "")
        false_a = item.get("false_answer", "")

        if q and a:
            questions.append({
                "question": q,
                "answer": a,
                "false_answer": false_a,
                "category": "general_knowledge",
                "source": "questions_1000",
                "metadata": item,
            })
    return questions


def _parse_wikidata(data: dict) -> list[dict]:
    """
    Parse wikidata_refined.json
    Format réel : {"question": {"1": "The language of X is?", ...}, "answer": {"1": "English", ...}}
    """
    questions = []
    q_dict = data.get("question", {})
    a_dict = data.get("answer", {})

    for key in q_dict:
        q = q_dict[key]
        a = a_dict.get(key, "")
        if q and a:
            questions.append({
                "question": q,
                "answer": a,
                "category": "general_knowledge",
                "source": "wikidata",
                "metadata": {"id": key, "question": q, "answer": a},
            })
    return questions


def _parse_sciq(data: dict) -> list[dict]:
    """
    Parse task591_sciq_answer_generation.json (format NaturalInstructions).
    Format : {"Instances": [{"input": "...", "output": ["..."]}, ...]}
    """
    questions = []
    instances = data.get("Instances", data.get("instances", []))

    if isinstance(data, list):
        instances = data

    for item in instances:
        q = item.get("input", "")
        outputs = item.get("output", [])
        a = outputs[0] if outputs else ""
        if q and a:
            questions.append({
                "question": q,
                "answer": a,
                "category": "general_knowledge",
                "source": "sciq",
                "metadata": item,
            })
    return questions


def _parse_commonsense_qa(filepath: str | Path) -> list[dict]:
    """
    Parse commonsense_QA_v2_dev.json — format JSON Lines (1 objet par ligne).
    Format réel : {"id": "...", "question": "...", "answer": "yes/no", "confidence": 0.55}
    """
    items = load_jsonl(filepath)
    questions = []

    for item in items:
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            a_str = str(a).strip().lower()
            if a_str in ("true", "1", "yes"):
                a_normalized = "True"
            elif a_str in ("false", "0", "no"):
                a_normalized = "False"
            else:
                a_normalized = str(a)

            questions.append({
                "question": q,
                "answer": a_normalized,
                "category": "common_sense",
                "source": "commonsense_qa",
                "metadata": item,
            })
    return questions


def _parse_svamp(data: dict, operation: str) -> list[dict]:
    """
    Parse les fichiers SVAMP (math) — format NaturalInstructions.
    """
    questions = []
    instances = data.get("Instances", data.get("instances", []))

    if isinstance(data, list):
        instances = data

    for item in instances:
        q = item.get("input", "")
        outputs = item.get("output", [])
        a = outputs[0] if outputs else ""
        if q and a:
            questions.append({
                "question": q,
                "answer": str(a),
                "category": "mathematics",
                "source": f"svamp_{operation}",
                "metadata": item,
            })
    return questions


def _parse_synthetic_facts(data: dict) -> list[dict]:
    """
    Parse questions_made_up_all.json — faits synthétiques.
    Format réel : {"people_birth_10": {"statement": "...", "question": "...", "answer": "...", "category": "..."}, ...}
    """
    questions = []
    for key, item in data.items():
        q = item.get("question", "")
        a = item.get("answer", "")
        statement = item.get("statement", "")
        false_statement = item.get("false_statement", "")

        if q and a:
            questions.append({
                "question": q,
                "answer": str(a),
                "statement": statement,
                "false_statement": false_statement,
                "category": "synthetic_facts",
                "source": "synthetic_facts",
                "metadata": item,
            })
    return questions


# ──────────────────────────────────────────────
# Fonction principale de chargement
# ──────────────────────────────────────────────

def load_all_datasets(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    max_per_dataset: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Charge tous les datasets disponibles et les fusionne en une liste unique.

    Paramètres
    ----------
    raw_dir : chemin vers le dossier contenant les JSON
    max_per_dataset : si spécifié, limite le nombre de questions par dataset
    seed : graine pour le sampling aléatoire

    Retourne
    --------
    Liste de dicts, chaque dict contenant :
        - question : str
        - answer : str
        - category : str (general_knowledge, common_sense, mathematics, synthetic_facts)
        - source : str (nom du dataset d'origine)
        - metadata : dict (données originales)
    """
    raw_dir = Path(raw_dir)
    all_questions = []
    rng = random.Random(seed)

    available = list_available_datasets(raw_dir)
    print(f"📂 Datasets trouvés : {len(available)}")

    for filename in available:
        filepath = raw_dir / filename

        try:
            questions = []

            if filename == "known_1000.json":
                data = load_json(filepath)
                questions = _parse_known_1000(data)

            elif filename == "questions_1000_all.json":
                data = load_json(filepath)
                questions = _parse_questions_1000(data)

            elif filename == "wikidata_refined.json":
                data = load_json(filepath)
                questions = _parse_wikidata(data)

            elif "sciq" in filename:
                data = load_json(filepath)
                questions = _parse_sciq(data)

            elif "commonsense" in filename:
                questions = _parse_commonsense_qa(filepath)

            elif "svamp_subtraction" in filename:
                data = load_json(filepath)
                questions = _parse_svamp(data, "subtraction")
            elif "svamp_multiplication" in filename:
                data = load_json(filepath)
                questions = _parse_svamp(data, "multiplication")
            elif "svamp_addition" in filename:
                data = load_json(filepath)
                questions = _parse_svamp(data, "addition")
            elif "svamp_common-division" in filename or "svamp_common_division" in filename:
                data = load_json(filepath)
                questions = _parse_svamp(data, "division")

            elif "made_up" in filename:
                data = load_json(filepath)
                questions = _parse_synthetic_facts(data)

            else:
                print(f"  ⚠️  Pas de parser pour : {filename} — ignoré")
                continue

            print(f"  ✅ {filename}: {len(questions)} questions chargées")

            if max_per_dataset and len(questions) > max_per_dataset:
                questions = rng.sample(questions, max_per_dataset)
                print(f"     → échantillonné à {max_per_dataset}")

            all_questions.extend(questions)

        except Exception as e:
            print(f"  ❌ Erreur pour {filename}: {e}")

    print(f"\n📊 Total : {len(all_questions)} questions chargées")

    categories = {}
    for q in all_questions:
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}")

    return all_questions


def get_questions_by_category(
    questions: list[dict],
    category: str,
) -> list[dict]:
    """Filtre les questions par catégorie."""
    return [q for q in questions if q["category"] == category]


def get_questions_by_source(
    questions: list[dict],
    source: str,
) -> list[dict]:
    """Filtre les questions par source (nom du dataset)."""
    return [q for q in questions if q["source"] == source]


# ──────────────────────────────────────────────
# Point d'entrée pour test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        raw_dir = sys.argv[1]
    else:
        raw_dir = DEFAULT_RAW_DIR

    print(f"Chargement depuis : {raw_dir}\n")
    questions = load_all_datasets(raw_dir, max_per_dataset=50)

    print("\n--- Exemples ---")
    for cat in ["general_knowledge", "common_sense", "mathematics", "synthetic_facts"]:
        subset = get_questions_by_category(questions, cat)
        if subset:
            example = subset[0]
            print(f"\n[{cat}]")
            print(f"  Q: {example['question']}")
            print(f"  A: {example['answer']}")
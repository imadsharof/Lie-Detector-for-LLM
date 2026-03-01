"""
api_llm.py — Wrapper API pour interagir avec les LLMs via LM Studio
====================================================================
LM Studio expose une API compatible OpenAI sur http://127.0.0.1:1234


"""

import time
import json
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("⚠️  openai non installé. Exécute : pip install openai")


# ──────────────────────────────────────────────
# Configuration LM Studio
# ──────────────────────────────────────────────

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"  

MAX_RETRIES = 3
RETRY_DELAY = 2


# ──────────────────────────────────────────────
# Classe principale
# ──────────────────────────────────────────────

class LLMClient:
    """
    Client pour interagir avec un LLM via LM Studio (ou tout serveur compatible OpenAI).

    Usage :
        client = LLMClient()  # Se connecte automatiquement à LM Studio
        response = client.ask("What is the capital of France?")
        yes_no = client.ask_yes_no(messages)
    """

    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE_URL,
        api_key: str = LMSTUDIO_API_KEY,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialise le client LLM.

        Paramètres
        ----------
        base_url : URL du serveur LM Studio (par défaut http://127.0.0.1:1234/v1)
        api_key : clé API (LM Studio accepte n'importe quelle valeur)
        model : nom du modèle (None = utilise celui chargé dans LM Studio)
        temperature : température de sampling (0 = déterministe)
        """
        if OpenAI is None:
            raise ImportError("pip install openai")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model or self._detect_model()
        self.temperature = temperature
        self.base_url = base_url

        # Compteurs
        self._call_count = 0
        self._total_tokens = 0

    def _detect_model(self) -> str:
        """Détecte automatiquement le modèle chargé dans LM Studio."""
        try:
            models = self.client.models.list()
            if models.data:
                model_id = models.data[0].id
                print(f"🤖 Modèle détecté : {model_id}")
                return model_id
        except Exception as e:
            print(f"⚠️  Impossible de détecter le modèle : {e}")
        return "local-model"

    def ask_chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> dict:
        """
        Envoie une requête chat et retourne la réponse complète.

        Paramètres
        ----------
        messages : [{"role": "system/user/assistant", "content": "..."}]
        temperature : override température
        max_tokens : tokens max en sortie
        logprobs : demander les log-probabilités
        top_logprobs : nombre de top logprobs

        Retourne
        --------
        dict : text, logprobs (opt), usage
        """
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                }

                # LM Studio supporte les logprobs (version récente)
                if logprobs:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = top_logprobs

                response = self.client.chat.completions.create(**kwargs)

                self._call_count += 1
                if response.usage:
                    self._total_tokens += response.usage.total_tokens

                result = {
                    "text": response.choices[0].message.content.strip(),
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    },
                }

                # Extraire les logprobs si disponibles
                if logprobs:
                    try:
                        if response.choices[0].logprobs and response.choices[0].logprobs.content:
                            result["logprobs"] = response.choices[0].logprobs.content
                    except (AttributeError, TypeError):
                        pass  # Logprobs non supportés par ce modèle/serveur

                return result

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  ⚠️ Erreur API (tentative {attempt+1}/{MAX_RETRIES}): {e}")
                    print(f"     Retry dans {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Échec après {MAX_RETRIES} tentatives: {e}")

    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        """Pose une question simple et retourne la réponse texte."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self.ask_chat(messages, temperature=temperature, max_tokens=max_tokens)
        return result["text"]

    def ask_yes_no(
        self,
        messages: list[dict],
        return_logprobs: bool = False,
    ) -> dict:
        """
        Pose une question yes/no et extrait la réponse binaire.

        Retourne
        --------
        dict :
            - answer : "yes" ou "no"
            - raw_text : texte brut
            - logprob_diff : log P(yes) - log P(no) (si return_logprobs=True)
        """
        result = self.ask_chat(
            messages,
            max_tokens=10,
            logprobs=return_logprobs,
            top_logprobs=10,
        )

        raw_text = result["text"].strip().lower()
        answer = _extract_yes_no(raw_text)

        output = {
            "answer": answer,
            "raw_text": result["text"],
        }

        if return_logprobs and "logprobs" in result and result["logprobs"]:
            logprob_diff = _compute_yes_no_logprob_diff(result["logprobs"])
            output["logprob_diff"] = logprob_diff

        return output

    def generate_answer(
        self,
        instruction: str,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """
        Génère une réponse à une question avec une instruction donnée.
        Utilisé pour générer des mensonges ou des vérités.
        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]
        result = self.ask_chat(messages, temperature=temperature, max_tokens=max_tokens)
        return result["text"]

    def check_connection(self) -> bool:
        """Vérifie que LM Studio est bien accessible."""
        try:
            models = self.client.models.list()
            if models.data:
                print(f"✅ Connecté à LM Studio ({self.base_url})")
                print(f"   Modèle chargé : {models.data[0].id}")
                return True
            else:
                print("⚠️  LM Studio est accessible mais aucun modèle n'est chargé.")
                return False
        except Exception as e:
            print(f"❌ Impossible de se connecter à LM Studio : {e}")
            print(f"   → Vérifie que LM Studio est lancé avec le serveur actif")
            print(f"   → URL attendue : {self.base_url}")
            return False

    def get_stats(self) -> dict:
        """Retourne les statistiques d'utilisation."""
        return {
            "model": self.model,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }

    def __repr__(self):
        return f"LLMClient(model='{self.model}', url='{self.base_url}', calls={self._call_count})"


# ──────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────

def _extract_yes_no(text: str) -> str:
    """
    Extrait 'yes' ou 'no' d'une réponse textuelle.
    Gère les variantes : "Yes.", "No,", "yes I think so", etc.
    """
    text = text.strip().lower()

    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"

    if re.search(r'\byes\b', text):
        return "yes"
    elif re.search(r'\bno\b', text):
        return "no"

    return "unknown"


def _compute_yes_no_logprob_diff(logprobs_content: list) -> float:
    """
    Calcule log P(yes) - log P(no) à partir des logprobs du premier token.

    Retourne
    --------
    float : positif → penche vers "yes", négatif → "no"
    """
    if not logprobs_content:
        return 0.0

    first_token = logprobs_content[0]
    top_logprobs = first_token.top_logprobs if hasattr(first_token, 'top_logprobs') else []

    yes_logprob = -100.0
    no_logprob = -100.0

    for lp in top_logprobs:
        token = lp.token.strip().lower()
        if token in ("yes", "ye"):
            yes_logprob = max(yes_logprob, lp.logprob)
        elif token in ("no",):
            no_logprob = max(no_logprob, lp.logprob)

    return yes_logprob - no_logprob


# ──────────────────────────────────────────────
# Point d'entrée pour test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Test de connexion à LM Studio")
    print("=" * 50)
    print()

    client = LLMClient()

    if client.check_connection():
        print()
        print("--- Test : question simple ---")
        response = client.ask("What is the capital of France? Answer in one word.")
        print(f"Q: What is the capital of France?")
        print(f"A: {response}")

        print()
        print("--- Test : mensonge ---")
        response = client.ask(
            "What is the capital of France? Answer in one sentence.",
            system_prompt="Answer the following question with a lie.",
        )
        print(f"Q: What is the capital of France? (avec instruction de mentir)")
        print(f"A: {response}")

        print()
        print("--- Test : question yes/no ---")
        messages = [
            {"role": "user", "content": "Is the sky blue? Answer yes or no."},
        ]
        yn = client.ask_yes_no(messages)
        print(f"Q: Is the sky blue?")
        print(f"A: {yn['answer']} (raw: {yn['raw_text']})")

        print()
        print(f"Stats : {client.get_stats()}")
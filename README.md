# AI Lie Detector: Black-Box Detection via Elicitation Questions

## Context
This repository contains the code and experiments for the **PROJ-H402 Computing Project** (Master in Computer Science and Engineering). 

The goal of this project is to implement and analyze a "Black-box" lie detector for Large Language Models (LLMs), based on the methodology introduced by Pacchiardi et al. (2024) in the paper *"How to Catch an AI Liar"*. 

## Project Objective
Detecting when an LLM produces intentional false statements (lies) is critical for AI safety. Unlike "White-box" methods that require access to the model's internal activations, this project focuses on a **Black-box approach**. 
By asking a predefined set of unrelated follow-up questions (elicitation questions) after a suspected lie, we can detect distinct behavioral shifts in the LLM's responses and classify the previous statement as a truth or a lie using a simple machine learning classifier.

## Methodology
Following the reference paper, the pipeline is divided into three main steps:
1. **Data Generation:** Prompting the LLM to output truths (standard prompt) and lies (using specific lie-inducing prompts).
2. **Elicitation (The Interrogation):** Asking questions fixed, unrelated follow-up questions (factual, moral, and ambiguous/nonsense) immediately after the LLM's statement and recording the binary "Yes/No" answers.
3. **Detection:** Feeding the resulting binary vectors into a Logistic Regression classifier to predict the probability of a lie.

## Tech Stack
* **Language:** Python 3
* **LLM Backend:** Local models (e.g., Llama-3, Mistral) served via **LM Studio** (using the OpenAI API format).
* **Machine Learning:** `scikit-learn` for Logistic Regression.
* **Data Processing:** `pandas`, `numpy`.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/imadsharof/Lie-Detector-for-LLM](https://github.com/imadsharof/Lie-Detector-for-LLM)
   cd Lie-Detector-for-LLM
   ```
2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3. Start your local LLM server using LM Studio on http://localhost:1234/v1.

## Reference

Pacchiardi, L., et al. (2024). How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions. ICLR 2024

    
# Large LLM Comparison Results

These files were produced by `notebooks/06_large_llm_comparison_colab.ipynb` on
Google Colab with an NVIDIA A100-SXM4-40GB runtime.

## Successful Models

| Model | Hugging Face ID | Mean Off-Domain Transfer Accuracy | Min | Max | Runtime |
|---|---|---:|---:|---:|---:|
| Phi-2 baseline | `microsoft/phi-2` | 0.657 | 0.400 | 1.000 | 41.0 s |
| Llama 2 13B Chat | `meta-llama/Llama-2-13b-chat-hf` | 0.576 | 0.200 | 1.000 | 144.6 s |
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 0.238 | 0.000 | 0.600 | 89.2 s |

The experiment trains an LR probe on `repeng_truthful` and evaluates transfer to
the test split of each dataset. The `repeng_truthful` evaluation itself is excluded
from the mean transfer score.

## Failed Models

`meta-llama/Meta-Llama-3-8B-Instruct` failed because the Hugging Face account used
in Colab did not have access to that exact gated repository.

No Llama 70B result is present in this run. The downloaded CSV files contain only
Phi-2, Mistral 7B, and Llama 2 13B. A 70B run should be performed separately on an
A100 80GB or larger runtime, or with an explicit reduced/offload configuration.

## Interpretation

Phi-2 is the strongest model in this run by average off-domain transfer accuracy.
Llama 2 13B is competitive on `arc_easy`, `boolq`, and `cities`, but weaker on
`arc_challenge`, `qa`, and `truthfulqa`. Mistral 7B performs poorly in this exact
fixed-layer setup, despite fitting the same probe on `repeng_truthful`.

This supports the report conclusion that truth-probe transfer depends strongly on
both the model family and the evaluation dataset.

## Files

- `large_llm_comparison.csv`: detailed per-model, per-dataset results.
- `large_llm_comparison_summary.csv`: one-row summary per successful model.
- `large_llm_comparison_failures.csv`: model loading/access failures.
- `large_llm_comparison_transfer.png`: bar chart of cross-dataset transfer.

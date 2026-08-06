"""
LLM baseline: run the local open-weights model (via Ollama) on the identical
test set used for the LSTM, with two prompt variants (zero-shot + few-shot).

Prereqs:
    1. Install Ollama: https://ollama.com
    2. Pull the model:  ollama pull llama3.1:8b-instruct-q4_K_M
    3. Make sure `ollama serve` is running

    python -m src.llm_baseline
"""
import time

import ollama
import pandas as pd

from src import config
from src.data import load_and_split

FEW_SHOT_EXAMPLES = [
    # Fill these in with 2-3 real (article_snippet, headline) pairs from your TRAIN split
    # so you aren't leaking test examples into the prompt.
    # ("Article snippet...", "Example Headline Here"),
]

PROMPT_ZERO_SHOT = """Write a short, single-line news headline for the following article. \
Respond with ONLY the headline, no quotes, no explanation.

Article: {article}

Headline:"""

PROMPT_FEW_SHOT_TEMPLATE = """Write a short, single-line news headline for the following article. \
Respond with ONLY the headline, no quotes, no explanation.

{examples}

Article: {article}

Headline:"""


def build_few_shot_prompt(article: str) -> str:
    examples_str = "\n\n".join(
        f"Article: {a}\nHeadline: {h}" for a, h in FEW_SHOT_EXAMPLES
    )
    return PROMPT_FEW_SHOT_TEMPLATE.format(examples=examples_str, article=article)


def query_llm(prompt: str) -> str:
    response = ollama.generate(model=config.OLLAMA_MODEL, prompt=prompt)
    return response["response"].strip().split("\n")[0]


def run_baseline():
    _, _, test_ex = load_and_split()
    if config.LLM_TEST_SUBSET_SIZE:
        test_ex = test_ex[: config.LLM_TEST_SUBSET_SIZE]

    results = []
    start = time.time()
    for ex in test_ex:
        zero_shot_out = query_llm(PROMPT_ZERO_SHOT.format(article=ex.src_text))
        few_shot_out = query_llm(build_few_shot_prompt(ex.src_text))
        results.append({
            "source": ex.src_text,
            "reference": ex.tgt_text,
            "llm_zero_shot": zero_shot_out,
            "llm_few_shot": few_shot_out,
        })
    elapsed = time.time() - start

    df = pd.DataFrame(results)
    df.to_csv("report/llm_predictions.csv", index=False)

    print(f"Ran {len(test_ex)} examples in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(test_ex), 1):.2f} sec/example)")
    print(f"Model: {config.OLLAMA_MODEL} | Hardware: <fill in your GPU/CPU here>")
    print("Report this timing as your 'cost' metric (section 4.2) -- CPU/GPU-hours, no API $ involved.")
    print("Saved predictions to report/llm_predictions.csv")


if __name__ == "__main__":
    run_baseline()

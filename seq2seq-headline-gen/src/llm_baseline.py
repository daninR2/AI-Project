"""
LLM baseline: run the local open-weights model (via Ollama) on the identical
test set used for the LSTM, with two prompt variants (zero-shot + few-shot).

Prereqs:
    1. Install Ollama: https://ollama.com
    2. Pull the model:  ollama pull llama3.1:8b-instruct-q4_K_M
    3. Make sure `ollama serve` is running

    python -m src.llm_baseline
"""
from jinja2 import defaults
import time

import ollama
import pandas as pd

from src import config
from src.data import load_and_split

from concurrent.futures import ThreadPoolExecutor
from rouge_score import rouge_scorer
import sacrebleu

FEW_SHOT_EXAMPLES = [
    (
        "comedian sunil grover has confirmed that he will be featuring in actor salman khan starrer 'bharat'. replying to the film's director ali abbas zafar's tweet confirming his casting in the film, grover said, thank you sir for giving me the visa. i'm so proud of being part of this project. he will reportedly play salman's friend in the film.",
        "proud of being part of the movie 'bharat' sunil grover"
    ),
    (
        "the sri lankan parliament has voted to block the salaries and travel expenses of ministers with an aim to exert pressure on prime minister mahinda rajapaksa. rajapaksa has refused to step down despite losing two no-confidence votes. former pm ranil wickremesinghe who was sacked by president maithripala sirisena commands a majority in parliament.",
        "sri lanka parliament blocks ministers' salaries to pressure pm"
    ),
    (
        "the kerala chief minister's distress relief fund cmdrf has received a total donation of over 1,027 crore as of august 31 to help those affected in the flood-hit state. while nearly 146 crore has been received through electronic payments, 835.86 crore has been received via cash, cheques and rtgs. meanwhile, donations via mediums like upi accounted for 46 crore.",
        "1,000 crore received till date as donation for kerala"
    ),
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
    response = ollama.generate(
        model=config.OLLAMA_MODEL,
        prompt=prompt,
        keep_alive=-1,
        options={
            "temperature": 0.0,
            "num_predict": 24,
        }
    )

    text = response["response"].strip().split("\n")[0]

    if text.lower().startswith("headline:"):
        text = text[len("headline:"):].strip()

    return text

    text = response["response"].strip().split("\n")[0]

    if text.lower().startswith("headline:"):
        text = text[len("headline:"):].strip()

    return text


def run_baseline():
    _, _, test_ex = load_and_split()
    if config.LLM_TEST_SUBSET_SIZE:
        test_ex = test_ex[: config.LLM_TEST_SUBSET_SIZE]

    results = []
    start = time.time()
    def process_example(ex):
        with ThreadPoolExecutor(max_workers=2) as executor:
            zero_future = executor.submit(
                query_llm,
                PROMPT_ZERO_SHOT.format(article=ex.src_text)
            )
            few_future = executor.submit(
                query_llm,
                build_few_shot_prompt(ex.src_text)
            )

            zero_shot_out = zero_future.result()
            few_shot_out = few_future.result()

        return {
            "source": ex.src_text,
            "reference": ex.tgt_text,
            "llm_zero_shot": zero_shot_out,
            "llm_few_shot": few_shot_out,
        }


    for ex in test_ex:
        results.append(process_example(ex))
    elapsed = time.time() - start

    df = pd.DataFrame(results)
    df.to_csv("report/llm_predictions.csv", index=False)

    refs = df["reference"].tolist()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    def score_outputs(name, hyps):
        bleu = sacrebleu.corpus_bleu(hyps, [refs])

        rouge_scores = [
            scorer.score(r, h)
            for r, h in zip(refs, hyps)
        ]

        avg_rouge = {
            k: sum(s[k].fmeasure for s in rouge_scores) / len(rouge_scores)
            for k in ["rouge1", "rouge2", "rougeL"]
        }

        print(f"\n{name}")
        print(f"BLEU: {bleu.score:.2f}")
        print(
            f"ROUGE-1/2/L (F1): "
            f"{avg_rouge['rouge1']:.4f} / "
            f"{avg_rouge['rouge2']:.4f} / "
            f"{avg_rouge['rougeL']:.4f}"
        )

    score_outputs(
        "LLM ZERO-SHOT",
        df["llm_zero_shot"].tolist()
    )

    score_outputs(
        "LLM FEW-SHOT",
        df["llm_few_shot"].tolist()
    )

    print(f"Ran {len(test_ex)} examples in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(test_ex), 1):.2f} sec/example)")
    print(f"Model: {config.OLLAMA_MODEL} | Hardware: <fill in your GPU/CPU here>")
    print("Report this timing as your 'cost' metric (section 4.2) -- CPU/GPU-hours, no API $ involved.")
    print("Saved predictions to report/llm_predictions.csv")


if __name__ == "__main__":
    run_baseline()

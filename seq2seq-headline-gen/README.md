# Headline Generation: LSTM Seq2Seq vs. Local LLM

Compares a from-scratch LSTM encoder-decoder (with Bahdanau attention) against
a local 7-8B open-weights LLM on abstractive headline generation.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For the LLM baseline, install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve   # keep running in a separate terminal
```

## Dataset

Download `news_summary.csv` (article/headline pairs, CC0) and place it at
`data/news_summary.csv`. Source: <fill in exact Kaggle/HF link + citation here>.
License: CC0 — document any deviation in your report.

## Reproduce results

```bash
# 1. Build vocab + splits (also run automatically by train.py)
python -m src.data

# 2. Train the LSTM model
python -m src.train

# 3. Evaluate the LSTM on the test set (BLEU/ROUGE + predictions CSV)
python -m src.evaluate

# 4. Run the LLM baseline on the identical test set
python -m src.llm_baseline
```

All runs use `SEED = 42` (see `src/config.py`) for reproducibility.

## Project structure

```
src/
  config.py        # all hyperparameters, paths, seed
  data.py          # cleaning, tokenization, vocab, splits, Dataset
  model.py         # Encoder, BahdanauAttention, Decoder, Seq2Seq
  train.py         # training loop, checkpointing, early stopping
  evaluate.py       # BLEU/ROUGE on test set, dumps LSTM predictions
  llm_baseline.py   # local LLM zero-shot + few-shot on the same test set
  utils.py          # seeding, param counting
report/             # generated prediction CSVs + qualitative analysis go here
checkpoints/        # saved vocab + best model weights
```

## Notes for the report

- `train.py` prints trainable parameter count, device, and total training
  time — copy these into the System Report (section 4.1).
- `llm_baseline.py` prints wall-clock time per example — report this as your
  "cost" metric since there's no API charge for a local model.
- `report/lstm_predictions.csv` and `report/llm_predictions.csv` share the
  same `source`/`reference` columns (same test set, same order) — join them
  to build the required 10-example qualitative comparison table.

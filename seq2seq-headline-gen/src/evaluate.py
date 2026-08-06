"""
Evaluate the trained LSTM model on the held-out test set: BLEU + ROUGE,
and dump raw predictions to CSV for later side-by-side qualitative analysis
against the LLM baseline (see src/llm_baseline.py).

    python -m src.evaluate
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import sacrebleu

from src import config
from src.data import load_and_split, HeadlineDataset, Vocab, PAD, SOS, EOS
from src.model import Encoder, Decoder, Seq2Seq
from src.train import get_lengths


def load_model():
    src_vocab = Vocab.load(f"{config.CHECKPOINT_DIR}/src_vocab.json")
    tgt_vocab = Vocab.load(f"{config.CHECKPOINT_DIR}/tgt_vocab.json")
    pad_idx = src_vocab.stoi[PAD]

    encoder = Encoder(len(src_vocab), config.EMBED_DIM, config.HIDDEN_DIM,
                       config.NUM_ENCODER_LAYERS, config.DROPOUT, config.BIDIRECTIONAL, pad_idx)
    enc_out_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
    decoder = Decoder(len(tgt_vocab), config.EMBED_DIM, config.HIDDEN_DIM, enc_out_dim,
                       config.DROPOUT, tgt_vocab.stoi[PAD])
    model = Seq2Seq(encoder, decoder, tgt_vocab.stoi[PAD], tgt_vocab.stoi[SOS], config.DEVICE).to(config.DEVICE)
    model.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/best_model.pt", map_location=config.DEVICE))
    model.eval()
    return model, src_vocab, tgt_vocab


def evaluate():
    model, src_vocab, tgt_vocab = load_model()
    _, _, test_ex = load_and_split()
    test_ds = HeadlineDataset(test_ex, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    pad_idx = src_vocab.stoi[PAD]
    sos_idx, eos_idx = tgt_vocab.stoi[SOS], tgt_vocab.stoi[EOS]

    hyps, refs, srcs = [], [], []
    for batch in test_loader:
        src = batch["src"].to(config.DEVICE)
        src_lengths = get_lengths(src, pad_idx)
        gen = model.generate(src, src_lengths, sos_idx, eos_idx, max_len=config.MAX_TGT_LEN)
        for i in range(src.size(0)):
            hyp_tokens = tgt_vocab.decode(gen[i].tolist())
            hyps.append(" ".join(hyp_tokens))
            refs.append(batch["tgt_text"][i])
            srcs.append(batch["src_text"][i])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    avg_rouge = {
        k: sum(s[k].fmeasure for s in rouge_scores) / len(rouge_scores)
        for k in ["rouge1", "rouge2", "rougeL"]
    }

    print(f"BLEU: {bleu.score:.2f}")
    print(f"ROUGE-1/2/L (F1): {avg_rouge['rouge1']:.4f} / {avg_rouge['rouge2']:.4f} / {avg_rouge['rougeL']:.4f}")

    pd.DataFrame({"source": srcs, "reference": refs, "lstm_output": hyps}).to_csv(
        "report/lstm_predictions.csv", index=False
    )
    print("Saved predictions to report/lstm_predictions.csv")


if __name__ == "__main__":
    evaluate()

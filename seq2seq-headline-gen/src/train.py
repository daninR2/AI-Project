"""
Train the LSTM seq2seq headline generator.

    python -m src.train
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data import load_and_split, build_vocab, HeadlineDataset, Vocab, PAD, SOS, EOS
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import set_seed, count_parameters


def get_lengths(batch_tensor, pad_idx):
    return (batch_tensor != pad_idx).sum(dim=1).clamp(min=1)


def run_epoch(model, loader, optimizer, criterion, pad_idx, teacher_forcing_ratio, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()

    start_time = time.time()

    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            src = batch["src"].to(config.DEVICE)
            tgt = batch["tgt"].to(config.DEVICE)
            src_lengths = get_lengths(src, pad_idx)

            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            if train:
                optimizer.zero_grad()

            logits = model(
                src,
                src_lengths,
                tgt=decoder_input,
                max_len=decoder_input.size(1),
                teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.CLIP_GRAD_NORM
                )
                optimizer.step()

            total_loss += loss.item()

            # Print progress every 25 batches
            if batch_idx % 25 == 0:
                elapsed = time.time() - start_time
                seconds_per_batch = elapsed / batch_idx
                estimated_total = seconds_per_batch * len(loader)
                estimated_remaining = estimated_total - elapsed

                print(
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"loss={total_loss / batch_idx:.4f} | "
                    f"elapsed={elapsed / 60:.1f} min | "
                    f"ETA={estimated_remaining / 60:.1f} min"
                )

    return total_loss / len(loader)


def main():
    set_seed(config.SEED)

    train_ex, val_ex, test_ex = load_and_split()
    src_vocab = build_vocab(train_ex, "src")
    tgt_vocab = build_vocab(train_ex, "tgt")
    src_vocab.save(f"{config.CHECKPOINT_DIR}/src_vocab.json")
    tgt_vocab.save(f"{config.CHECKPOINT_DIR}/tgt_vocab.json")

    train_ds = HeadlineDataset(train_ex, src_vocab, tgt_vocab)
    val_ds = HeadlineDataset(val_ex, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    pad_idx = src_vocab.stoi[PAD]
    encoder = Encoder(len(src_vocab), config.EMBED_DIM, config.HIDDEN_DIM,
                       config.NUM_ENCODER_LAYERS, config.DROPOUT, config.BIDIRECTIONAL, pad_idx)
    enc_out_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
    decoder = Decoder(len(tgt_vocab), config.EMBED_DIM, config.HIDDEN_DIM, enc_out_dim,
                       config.DROPOUT, tgt_vocab.stoi[PAD])
    model = Seq2Seq(encoder, decoder, tgt_vocab.stoi[PAD], tgt_vocab.stoi[SOS], config.DEVICE).to(config.DEVICE)

    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Device: {config.DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD])

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, pad_idx,
                                config.TEACHER_FORCING_RATIO, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, pad_idx, 0.0, train=False)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed / 60:.1f} min on {config.DEVICE}")
    print(f"Report this training time + param count + hardware in your system report (section 4.1).")


if __name__ == "__main__":
    main()

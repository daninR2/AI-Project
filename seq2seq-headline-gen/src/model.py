"""
LSTM encoder-decoder with Bahdanau (additive) attention, built from scratch
on top of nn.LSTM / nn.Embedding only -- no seq2seq framework.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional=True, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # project bidirectional final states down to decoder hidden size
        self.fc_h = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: (B, T)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        # outputs: (B, T, hidden*dirs) -- used as attention memory

        # combine last layer's forward/backward final states -> decoder init state
        h = h.view(self.num_layers, self.num_directions, h.size(1), self.hidden_dim)
        c = c.view(self.num_layers, self.num_directions, c.size(1), self.hidden_dim)
        h_last = torch.cat([h[-1, 0], h[-1, 1]], dim=-1) if self.num_directions == 2 else h[-1, 0]
        c_last = torch.cat([c[-1, 0], c[-1, 1]], dim=-1) if self.num_directions == 2 else c[-1, 0]
        h0 = torch.tanh(self.fc_h(h_last)).unsqueeze(0)  # (1, B, hidden)
        c0 = torch.tanh(self.fc_c(c_last)).unsqueeze(0)
        return outputs, (h0, c0)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, encoder_out_dim):
        super().__init__()
        self.W_enc = nn.Linear(encoder_out_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: (B, hidden) | enc_outputs: (B, T, enc_out_dim) | src_mask: (B, T) 1=real 0=pad
        scores = self.v(torch.tanh(self.W_enc(enc_outputs) + self.W_dec(dec_hidden).unsqueeze(1)))
        scores = scores.squeeze(-1)  # (B, T)
        scores = scores.masked_fill(src_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_out_dim)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_out_dim, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, encoder_out_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_out_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim + encoder_out_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, cell, enc_outputs, src_mask):
        # input_tok: (B,) single time step
        embedded = self.dropout(self.embedding(input_tok)).unsqueeze(1)  # (B,1,E)
        context, attn_weights = self.attention(hidden.squeeze(0), enc_outputs, src_mask)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        combined = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        logits = self.out_proj(combined)
        return logits, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int, sos_idx: int, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.device = device

    def make_src_mask(self, src):
        return (src != self.pad_idx).long()

    def forward(self, src, src_lengths, tgt=None, max_len=25, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        src_mask = self.make_src_mask(src)
        enc_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        target_len = tgt.size(1) if tgt is not None else max_len
        vocab_size = self.decoder.out_proj.out_features
        outputs = torch.zeros(batch_size, target_len, vocab_size, device=self.device)

        input_tok = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
        for t in range(target_len):
            logits, hidden, cell, _ = self.decoder.forward_step(input_tok, hidden, cell, enc_outputs, src_mask)
            outputs[:, t] = logits
            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_tok = tgt[:, t]
            else:
                input_tok = logits.argmax(dim=-1)
        return outputs

    @torch.no_grad()
    def generate(self, src, src_lengths, sos_idx, eos_idx, max_len=25):
        """Greedy decoding for inference/evaluation."""
        self.eval()
        batch_size = src.size(0)
        src_mask = self.make_src_mask(src)
        enc_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        input_tok = torch.full((batch_size,), sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sequences = []
        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder.forward_step(input_tok, hidden, cell, enc_outputs, src_mask)
            next_tok = logits.argmax(dim=-1)
            sequences.append(next_tok)
            finished |= next_tok == eos_idx
            input_tok = next_tok
            if finished.all():
                break
        return torch.stack(sequences, dim=1)  # (B, T)

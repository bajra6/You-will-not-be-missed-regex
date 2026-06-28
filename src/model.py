# [3] Model — all Transformer components from scratch: embeddings, attention, FFN, layernorm, encoder, decoder, full Transformer

import math

import torch
import torch.nn as nn

from src.config import d_model, n_heads, d_ff, n_layers, dropout, vocab_nl, vocab_regex
from src.config import pad_id_nl, pad_id_regex


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 10000 ** (-2i/d_model) = exp(-log(10000) * 2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        # the dropout is done in the original paper as well. damn!
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # this is a look up table for each token in the vocabulary
        # ex: 0 -> [0.12, -0.45, 0.88, 0.01], 1 -> [-0.22, 0.56, -0.11, 0.99]
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x): 
        # x is an array of token indices
        # we look up the embedding for each token and scale it by sqrt(d_model)
        return self.embed(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        # x is of shape (batch_size, seq_len, d_model)
        for layer in self.layers:
            # calls forward in EncoderBlock
            x = layer(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attention(x, encoder_out, encoder_out, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        # creates an object for token embedding. initialized embeddings automatically with random val
        # when you call src_embed.forward(src), it will look up the embedding for each token in src and return the corresponding embeddings
        # returns of shape (batch_size, seq_len, d_model)
        self.src_embed = TokenEmbedding(vocab_nl, d_model)
        self.tgt_embed = TokenEmbedding(vocab_regex, d_model)

        # object for positional encoding. initialized with sinusoidal values
        # shape of (batch_size, seq_len, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        # input of format (batch_size, seq_len, d_model)
        # output is of format (batch_size, seq_len, vocab_regex)
        # we generate a logit for each vocabulary token in the regex vocabulary 
        # for each position in the sequence (predicting the next token)
        self.output_projection = nn.Linear(d_model, vocab_regex, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != pad_id_nl).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != pad_id_regex).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=tgt.device, dtype=torch.bool))
        return tgt_pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_embedded = self.pos_encoding(self.src_embed(src))
        tgt_embedded = self.pos_encoding(self.tgt_embed(tgt))

        # The encoder is masked to prevent the padding tokens from being learnt
        encoder_out = self.encoder(src_embedded, src_mask)
        # well decoder needs masking as well bruv!
        decoder_out = self.decoder(tgt_embedded, encoder_out, src_mask, tgt_mask)

        logits = self.output_projection(decoder_out)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Transformer()
    print(f"Total parameters: {count_parameters(model):,}")
    
    # input is of shape (number of sequences, sequence length)
    # its an array of token indices. each index corresponds to a token in the vocabulary
    src = torch.randint(0, vocab_nl, (2, 10))
    tgt = torch.randint(0, vocab_regex, (2, 8))
    logits = model(src, tgt)
    print(f"Input src: {src.shape}, Input tgt: {tgt.shape}")
    print(f"Output logits: {logits.shape}")

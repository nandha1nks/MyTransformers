import torch
import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import EncoderLayer, DecoderLayer, create_mask


def get_pos_emb(max_seq_len, d_model):
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

    # Calculate the sine and cosine components
    sin_part = torch.sin(position * div_term)
    cos_part = torch.cos(position * div_term)

    # Interleave the sine and cosine components
    positional_encoding = torch.zeros(max_seq_len, d_model)
    positional_encoding[:, 0::2] = sin_part
    positional_encoding[:, 1::2] = cos_part

    return positional_encoding.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_seq_len, d_model, d_ff, num_heads, num_layers, device):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_attn):
        src = self.src_embedding(src) + get_pos_emb(src.size(-1), self.d_model).to(self.device)
        enc_mask = create_mask(src_attn, src_attn)
        for layer in self.encoders:
            src = layer(src, enc_mask)
        return src

    def decode(self, tgt, tgt_attn, enc, src_attn, temperature=1):
        tgt = self.tgt_embedding(tgt) + get_pos_emb(tgt.size(-1), self.d_model).to(self.device)
        dec_mask = create_mask(tgt_attn, tgt_attn, True)
        cross_mask = create_mask(tgt_attn, src_attn)
        for layer in self.decoders:
            tgt = layer(tgt, enc, dec_mask, cross_mask)
        x = self.fc(tgt)
        return F.log_softmax(x / temperature, dim=-1)

    def forward(self, *args, **kwargs):
        src = args[0] if len(args) > 0 else kwargs.get("src")
        tgt = args[1] if len(args) > 1 else kwargs.get("tgt")
        src_attention = args[2] if len(args) > 2 else kwargs.get("src_attention") or torch.ones_like(src)
        tgt_attention = args[3] if len(args) > 3 else kwargs.get("tgt_attention") or torch.ones_like(tgt)
        temperature = args[4] if len(args) > 4 else kwargs.get("temperature") or 1

        enc = self.encode(src, src_attention)
        pred = self.decode(tgt, tgt_attention, enc, src_attention, temperature)
        return pred

    def predict(self, src, src_attention):
        batch_size = src.size(0)
        enc = self.encode(src, src_attention)

        last_tokens = torch.zeros((batch_size,), dtype=torch.long).to(self.device)
        tgt = torch.zeros((batch_size, 1), dtype=torch.long).to(self.device)
        tgt_attention = torch.ones((batch_size, 1), dtype=torch.long).to(self.device)
        c = 0
        while not last_tokens.all() and c < self.max_seq_len:
            f = self.decode(tgt, tgt_attention, enc, src_attention)
            last_token = f[:, -1, :].argmax(dim=-1).to(torch.long)
            last_token[last_tokens] = 1
            tgt = torch.concat([tgt, last_token.reshape(-1, 1)], dim=-1)
            last_tokens = last_tokens.masked_fill(last_token.reshape((-1,)) == 1, 1)
            tgt_attention = torch.concat([tgt_attention, (last_tokens.reshape(-1, 1) + 1) % 2], dim=-1)
            c += 1

        return tgt, tgt_attention

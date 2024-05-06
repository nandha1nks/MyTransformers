import torch
import torch.nn as nn

from .building_blocks import MultiAttentionHead, PositionWiseFeedForward, create_mask


class GPTLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, p=0.1):
        super(GPTLayer, self).__init__()
        self.mha = MultiAttentionHead(d_model, num_heads)
        self.pff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask, past=None):
        m, K, V = self.mha(x, x, x, mask, past=past)
        x = self.norm1(x + self.dropout(m))
        x = self.norm2(x + self.dropout(self.pff(x)))
        return x, (K, V)


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, d_ff, num_heads, num_layers, device, use_kv=False):
        super(GPT, self).__init__()
        self.d_model = d_model
        self.device = device

        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.decoders = nn.ModuleList([
            GPTLayer(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])

        embed_weight = self.embedding.weight
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.classifier.weight = embed_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.use_kv = use_kv

    def forward(self, x, attention, **kwargs):
        mask = create_mask(attention, attention, True)
        pos = kwargs.get("pos") if "pos" in kwargs else torch.arange(0, x.size(-1))
        past = kwargs.get("past")

        x = self.embedding(x) + self.pos_embedding(pos)

        current_kvs = ()
        for idx, layer in enumerate(self.decoders):
            x, current_kv = layer(x, mask, past[idx] if past else None)
            if self.use_kv:
                current_kvs += (current_kv,)
        x = self.classifier(x) + self.bias
        return x if not self.use_kv else (x, current_kvs)

    def predict(self, src, src_attention):
        batch_size = src.size(0)

        last_tokens = torch.zeros((batch_size,), dtype=torch.long).to(self.device)

        c = 0
        while not last_tokens.all() and c < self.max_seq_len:
            f = self.forward(src, src_attention)
            last_token = f[:, -1, :].argmax(dim=-1).to(torch.long)
            last_token[last_tokens] = 1
            src = torch.concat([src, last_token.reshape(-1, 1)], dim=-1)
            last_tokens = last_tokens.masked_fill(last_token.reshape((-1,)) == 1, 1)
            src_attention = torch.concat([src_attention, (last_tokens.reshape(-1, 1) + 1) % 2], dim=-1)
            c += 1
        return src, src_attention

    def predict_kv(self, src, src_attention):
        batch_size = src.size(0)

        last_tokens = torch.zeros((batch_size,), dtype=torch.long).to(self.device)
        c = 0
        past = None
        last_token = src
        attn = src_attention
        pos = torch.arange(0, src.size(-1)).reshape(1, -1)

        while not last_tokens.all() and c < self.max_seq_len:
            f, past = self.forward(last_token, attn, pos=pos, past=past)
            last_token = f[:, -1, :].argmax(dim=-1).to(torch.long)
            last_token[last_tokens] = 1
            last_token = last_token.reshape(-1, 1)
            src = torch.concat([src, last_token], dim=-1)
            pos = pos[:, -1:] + 1

            last_tokens = last_tokens.masked_fill(last_token.reshape((-1,)) == 1, 1)
            attn = (last_tokens.reshape(-1, 1) + 1) % 2
            src_attention = torch.concat([src_attention, attn], dim=-1)

            c += 1
        return src, src_attention

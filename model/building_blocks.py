import torch.nn as nn
import torch
import math
import torch.nn.functional as F


def create_mask(attention_mask, attention_mask2, decoder=False):
    casual_mask = torch.logical_and(attention_mask[:, :, None], attention_mask2[:, None, :])
    if decoder:
        casual_mask = torch.tril(casual_mask, diagonal=0)
    causal_mask = casual_mask.unsqueeze(1)
    return causal_mask


def self_attention(q, k: torch.Tensor, v, mask=None, d_model=1):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    return torch.matmul(attention_weights, v)


class MultiAttentionHead(nn.Module):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super(MultiAttentionHead, self).__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = int(d_model / num_heads)

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask, past=None):
        Q = self.WQ(Q).view(Q.size(0), -1, self.num_heads, self.d_hidden).transpose(1, 2)
        K = self.WK(K).view(K.size(0), -1, self.num_heads, self.d_hidden).transpose(1, 2)
        V = self.WV(V).view(V.size(0), -1, self.num_heads, self.d_hidden).transpose(1, 2)

        if past:
            K = torch.concat([past[0], K], dim=-2)
            V = torch.concat([past[1], V], dim=-2)

        k = self_attention(Q, K, V, mask, self.d_model).transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model)
        return self.linear(k), K, V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.hidden = nn.Linear(self.d_model, self.d_ff)
        self.activ = nn.GELU()
        self.out = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        return self.out(self.activ(self.hidden(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, p=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiAttentionHead(d_model, num_heads)
        self.pff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        m, _, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(m))
        x = self.norm2(x + self.dropout(self.pff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, p=0.1):
        super(DecoderLayer, self).__init__()
        self.mha = MultiAttentionHead(d_model, num_heads)
        self.mha2 = MultiAttentionHead(d_model, num_heads)
        self.pff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, enc, mask, enc_mask):
        m, _, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(m))
        m, _, _ = self.mha2(x, enc, enc, enc_mask)
        x = self.norm2(x + self.dropout(m))
        x = self.norm3(x + self.dropout(self.pff(x)))
        return x

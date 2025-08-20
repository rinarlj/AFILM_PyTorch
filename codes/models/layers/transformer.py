import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionEmbedding(nn.Module):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, self.maxlen, device=x.device)
        positions = self.pos_emb(positions)
        return x + positions.unsqueeze(0)  # Correction: ajout de la dimension batch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = key.size(-1)  # Correction: utilisation de .size() au lieu de tensor
        scaled_score = score / math.sqrt(dim_key)  # math.sqrt plus efficace
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3)
        concat_attention = attention.contiguous().view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat_attention)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dense1 = nn.Linear(embed_dim, ff_dim)
        self.dense2 = nn.Linear(ff_dim, embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs, training=True):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output) if training else attn_output
        out1 = self.layernorm1(inputs + attn_output)

        # Correction: activation après dense1 comme dans TensorFlow
        ffn_output = self.dense1(out1)
        ffn_output = F.relu(ffn_output)  # Activation après dense1
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        return self.layernorm2(out1 + ffn_output)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, maximum_position_encoding, num_heads=8, ff_dim=2048, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embed_dim)
        self.enc_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, rate)
                                        for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training=True):
        seq_len = x.shape[1]
        device = x.device
        pos_encoding = self.pos_encoding.to(device)

        # Correction: multiplication plus claire
        x = x * math.sqrt(self.embed_dim)
        x = x + pos_encoding[:, :seq_len, :]
        x = self.dropout(x) if training else x

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training)
        return x
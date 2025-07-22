# %% 📦 Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% 🧠 Single Attention Head
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(1024, 1024)))  # causal mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B,T,head)
        q = self.query(x)    # (B,T,head)
        v = self.value(x)    # (B,T,head)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B,T,T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v  # (B,T,head)
        return out

# %% 🎯 Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(embed_dim, head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# %% 🔁 Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Self-attn + residual
        x = x + self.ff(self.ln2(x))  # Feedforward + residual
        return x

# %% 🧠 The Full Transformer Language Model (GPT-lite)
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=128, embed_dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Block size exceeded"

        token_emb = self.token_embed(idx)                 # (B,T,C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embed(pos)[None, :, :]         # (1,T,C)
        x = token_emb + pos_emb                           # (B,T,C)

        x = self.blocks(x)                                # (B,T,C)
        x = self.ln_f(x)                                  # (B,T,C)
        logits = self.head(x)                             # (B,T,V)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

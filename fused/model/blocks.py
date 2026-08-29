"""Building blocks shared by the FUSED fusion stack."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward projection.

    A single input projection is split into value and gate halves, gated with
    SiLU, then projected back to ``out_dim``.
    """

    def __init__(self, dim, hidden_dim=None, out_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim

        self.fc = nn.Linear(dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.dropout(x)
        return self.proj(x)


class SelfAttention(nn.Module):
    """Pre-LayerNorm residual multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, seq, c = x.shape
        residual = x

        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(b, seq, c)
        return residual + self.out_proj(attn)


class AttentionPooling(nn.Module):
    """Cross-attend a set of learnable query latents to a token sequence.

    The latents are orthogonally initialized; ``diversity_loss`` penalizes the
    off-diagonal cosine similarities of the pooled outputs so the queries stay
    complementary during training.
    """

    def __init__(self, dim, num_queries=1, num_heads=8, dropout=0.1):
        super().__init__()
        latents = torch.empty(num_queries, dim)
        nn.init.orthogonal_(latents)
        self.latents = nn.Parameter(latents.unsqueeze(0))

        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def diversity_loss(pooled):
        b, n, _ = pooled.shape
        if n <= 1:
            return pooled.new_tensor(0.0)
        normed = F.normalize(pooled, p=2, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))
        off_diag = sim * (1 - torch.eye(n, device=pooled.device).unsqueeze(0))
        return (off_diag ** 2).sum() / (n * (n - 1) * b)

    def forward(self, x):
        q = self.latents.repeat(x.shape[0], 1, 1)
        attn_out, attn_weights = self.mha(
            query=q, key=x, value=x, need_weights=True, average_attn_weights=True
        )
        out = self.out_proj(self.norm(attn_out))
        return out, attn_weights

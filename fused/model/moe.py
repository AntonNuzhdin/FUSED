"""Sparsely-gated Mixture-of-Experts with top-k routing and load balancing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused.model.blocks import SwiGLU


class ExpertFFN(nn.Module):
    """SwiGLU feed-forward network used as a single MoE expert.

    Also used on its own as the dense-FFN control, in which case ``expansion`` is
    scaled by top_k so the hidden width matches the experts the MoE activates.
    """

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = max(1, int(dim * expansion * 2 / 3))
        self.ffn = SwiGLU(dim, hidden_dim=hidden_dim, out_dim=dim, dropout=dropout)

    def forward(self, x):
        return self.ffn(x)


class SparseMoE(nn.Module):
    """Route each token to its top-k experts and mix the renormalized outputs.

    Returns the fused tokens, the load-balancing auxiliary loss
    ``E * sum_i f_i * P_i``, and the raw router logits.
    """

    def __init__(self, dim, num_experts=8, top_k=2, expansion=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(dim, expansion=expansion, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        b, s, dim = x.shape
        num_tokens = b * s

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_indices = router_logits.topk(self.top_k, dim=-1).indices
        top_k_probs = router_probs.gather(dim=-1, index=top_k_indices)
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        flat_x = x.view(-1, dim)
        flat_gates = gates.view(-1, self.top_k)
        flat_indices = top_k_indices.view(-1, self.top_k)
        output = torch.zeros_like(flat_x)

        for i in range(self.num_experts):
            selected = (flat_indices == i)
            if not selected.any():
                continue
            positions = torch.nonzero(selected, as_tuple=False)
            token_idx, slot_idx = positions[:, 0], positions[:, 1]
            expert_out = self.experts[i](flat_x[token_idx])
            output[token_idx] += flat_gates[token_idx, slot_idx].unsqueeze(-1) * expert_out

        output = output.view(b, s, dim)

        # f_i = fraction of dispatched tokens per expert, P_i = mean gate probability.
        dispatch = torch.zeros(b, s, self.num_experts, dtype=router_probs.dtype,
                               device=x.device)
        for k in range(self.top_k):
            dispatch.scatter_(dim=-1, index=top_k_indices[:, :, k:k + 1], value=1.0)
        f_i = dispatch.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        p_i = router_probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        return output, aux_loss, router_logits

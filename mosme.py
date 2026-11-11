"""Shared scale-conditioned mixture-of-experts encoding (MoSME).

Core-module excerpt from the supplied DWM-DETR implementation.
Adapted from RT-DETR / RT-DETRv2; Copyright(c) 2023 lyuwenyu.
Original computation is retained; framework registration and unrelated code are omitted.
See NOTICE.md for upstream attribution and license terms.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add
    _SCATTER_ADD_AVAILABLE = True
except ImportError:
    _SCATTER_ADD_AVAILABLE = False

def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ScaleAwareMoEBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, num_scales, balance_loss_coef):
        super().__init__()
        self.hidden_dim, self.ffn_dim = hidden_dim, ffn_dim
        self.num_experts, self.top_k = num_experts, top_k
        self.balance_loss_coef = balance_loss_coef

        self.shared_ffn = MLP(hidden_dim, ffn_dim, hidden_dim, 2)
        self.experts = nn.ModuleList([MLP(hidden_dim, ffn_dim, hidden_dim, 2) for _ in range(num_experts)])
        self.scale_embeddings = nn.Embedding(num_scales, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, num_experts)

    def forward(self, tokens, scale_index):
        bs, num_tokens, c = tokens.shape
        shared_out = self.shared_ffn(tokens)

        # Gate input: Feature + Scale Embedding
        tokens_reshaped = tokens.view(-1, c)
        s_embed = self.scale_embeddings.weight[scale_index].unsqueeze(0).expand(tokens_reshaped.shape[0], -1)
        gate_input = torch.cat([tokens_reshaped, s_embed], dim=-1)

        loss_balance = None
        if self.training:
            gate_logits = self.gate(gate_input)
            gate_scores = F.softmax(gate_logits, dim=-1)
            f_i = gate_scores.mean(dim=0)
            _, top_indices = torch.topk(gate_scores, self.top_k, dim=-1)
            expert_mask = F.one_hot(top_indices, self.num_experts).sum(dim=1)
            P_i = expert_mask.float().mean(dim=0)
            loss_balance = self.balance_loss_coef * self.num_experts * (f_i * P_i).sum()
            top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        else:
            gate_logits = self.gate(gate_input)
            top_k_scores, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            top_k_scores = F.softmax(top_k_scores, dim=-1)

        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)

        moe_output = torch.zeros_like(tokens_reshaped)
        flat_indices = top_k_indices.flatten()
        tokens_expanded = tokens_reshaped.repeat_interleave(self.top_k, dim=0)
        expert_results = torch.zeros_like(tokens_expanded)

        for i, expert in enumerate(self.experts):
            mask = (flat_indices == i)
            if mask.any():
                expert_results[mask] = expert(tokens_expanded[mask]).to(expert_results.dtype)

        weighted_results = expert_results * top_k_scores.flatten().unsqueeze(-1).to(expert_results.dtype)

        # Aggregate
        destination_indices = torch.arange(tokens_reshaped.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        if _SCATTER_ADD_AVAILABLE:
            moe_output.scatter_add_(0, destination_indices.unsqueeze(-1).expand_as(weighted_results), weighted_results)
        else:
            moe_output.index_add_(0, destination_indices, weighted_results)

        return shared_out + moe_output.view(bs, num_tokens, c), loss_balance


class SharedMoEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 num_experts=4, top_k=2, num_scales=3, balance_loss_coef=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Replace Standard FFN with MoE Block
        self.moe_block = ScaleAwareMoEBlock(
            d_model, dim_feedforward, num_experts, top_k, num_scales, balance_loss_coef
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, scale_index, src_mask=None, pos_embed=None):
        # Self Attention
        q = k = self.with_pos_embed(src, pos_embed)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # MoE FFN
        src2, loss_balance = self.moe_block(src, scale_index=scale_index)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, loss_balance


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, scale_index, src_mask=None, pos_embed=None):
        output = src
        total_loss = 0
        for layer in self.layers:
            output, loss = layer(output, scale_index=scale_index, src_mask=src_mask, pos_embed=pos_embed)
            if loss is not None: total_loss += loss

        if self.norm is not None:
            output = self.norm(output)
        return output, total_loss


def build_shared_scale_encoder(hidden_dim=256, nhead=8, ffn_dim=1024,
                               num_experts=4, top_k=2, num_scales=2,
                               balance_loss_coef=0.1):
    """Create one encoder instance to call separately for S4 and S5.

    Each call returns (encoded_tokens, balance_loss). The two token
    sequences are not concatenated. Reuse this same object for both scales.
    """
    layer = SharedMoEEncoderLayer(
        d_model=hidden_dim, nhead=nhead, dim_feedforward=ffn_dim,
        dropout=0.0, activation='gelu', num_experts=num_experts,
        top_k=top_k, num_scales=num_scales,
        balance_loss_coef=balance_loss_coef,
    )
    return TransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(hidden_dim))

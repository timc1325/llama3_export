import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids):
        B, T = position_ids.shape
        freqs = torch.arange(0, self.head_dim, 2, device=x.device).float()
        freqs = 1.0 / (10000 ** (freqs / self.head_dim))
        angles = torch.einsum("bt,f->btf", position_ids.float(), freqs)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        return cos, sin


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm / (x.shape[-1] ** 0.5) + self.eps) * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, hidden_dim=2048, intermediate_dim=8192, world_size=1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim// world_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim// world_size, bias=False)
        self.down_proj = nn.Linear(intermediate_dim// world_size, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=2048, n_heads=32, kv_heads=8, head_dim=64, world_size=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim// world_size, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_heads * head_dim// world_size, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_heads * head_dim// world_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim// world_size, hidden_dim, bias=False)
        self.rotary_emb = DummyRotaryEmbedding(head_dim)
        self.world_size = world_size
        self.local_n_heads = n_heads // world_size
        self.local_kv_heads = kv_heads // world_size

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def repeat_kv(self, x):
        B, T, H, D = x.shape
        if self.n_heads == self.n_kv:
            return x
        n_rep = self.n_heads // self.n_kv
        return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, self.n_heads//self.world_size, D)

    def forward(self, x, position_ids):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads// self.world_size, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv// self.world_size, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv// self.world_size, self.head_dim) #!!world size has to be divisible by both n_kv(8) and n_heads(32)

        cos, sin = self.rotary_emb(x, position_ids)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin

        q = q.transpose(1, 2)
        k = self.repeat_kv(k).transpose(1, 2)
        v = self.repeat_kv(v).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.local_n_heads * self.head_dim)
        return self.o_proj(out)


class LlamaDecoderLayerExportable(nn.Module):
    def __init__(self, world_size=1, hidden_dim=2048):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(hidden_dim)
        self.self_attn = RotaryAttentionBlock(world_size=world_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_dim)
        self.mlp = LlamaMLP(world_size=world_size)

    def forward(self, x, position_ids):
        h = x + self.self_attn(self.input_layernorm(x), position_ids)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

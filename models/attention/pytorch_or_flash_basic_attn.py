from models.layers import (
    CastedLinear,
    CosSin,
    apply_rotary_pos_emb,
)

import torch
from torch import nn

try:
    from flash_attn import flash_attn_func  # type: ignore[import]
except ImportError:
    # Fallback to Pytorch scaled_dot_product_attention
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, causal=False):
        # return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Convert to [batch * num_heads, seq_len, head_dim] for SDPA
        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # Apply SDPA
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)

        # Reshape back to [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)

        return attn_output


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

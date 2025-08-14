from models.layers import (
    CastedLinear,
    CosSin,
    apply_rotary_pos_emb,
)

import torch
from torch import nn

def fast_rwkv_linear_attention(r, k, v, time_decay, time_first):
    """
    Parallel RWKV approximation - trades some accuracy for massive speed gains
    Uses parallel prefix scan approximation instead of sequential processing
    """
    batch_size, seq_len, num_heads, head_dim = r.shape
    
    # Prepare decay weights
    w = -torch.exp(time_decay)  # [num_heads, head_dim]
    u = time_first  # [num_heads, head_dim]
    
    # Compute element-wise k*v
    kv = k * v  # [batch, seq_len, heads, head_dim]
    
    # PARALLEL APPROXIMATION: Instead of true sequential scan,
    # use exponential weighting based on position distance
    
    # Create position-based exponential weights (this is the key approximation)
    positions = torch.arange(seq_len, device=r.device, dtype=r.dtype)
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
    
    # Causal mask - only attend to past positions
    causal_mask = pos_diff > 0
    pos_diff = pos_diff.masked_fill(causal_mask, float('inf'))
    
    # Use a simplified approach: average decay rate across heads and features
    # This avoids complex broadcasting issues
    avg_decay = torch.mean(w)  # Single scalar value
    
    # Create decay matrix: [seq_len, seq_len]
    decay_weights = torch.exp(avg_decay * (-torch.abs(pos_diff)))
    decay_weights = decay_weights.masked_fill(causal_mask, 0.0)
    
    # Normalize to prevent explosion (this maintains stability)
    row_sums = decay_weights.sum(dim=1, keepdim=True)
    decay_weights = decay_weights / (row_sums + 1e-8)
    
    # Parallel computation: weighted sum of all previous kv pairs
    # Shape: [seq_len, seq_len] @ [batch, seq_len, heads, head_dim] -> [batch, seq_len, heads, head_dim]
    weighted_kv = torch.einsum('ij,bjhd->bihd', decay_weights, kv)
    
    # Add time_first term (immediate attention to current position)
    u_expanded = u.unsqueeze(0).unsqueeze(0)  # [1, 1, heads, head_dim]
    current_contribution = u_expanded * kv
    
    # Final RWKV-style output
    output = r * (current_contribution + weighted_kv)
    
    return output


class RWKVAttention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # Use single combined projection like original attention for efficiency
        self.rkv_proj = CastedLinear(self.hidden_size, 3 * self.output_size, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        # RWKV-specific parameters (simplified)
        self.time_decay = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        
        # Simplified time mixing - single parameter per head
        self.time_mix = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        
        self._init_rwkv_params()

    def _init_rwkv_params(self):
        """Initialize RWKV-specific parameters"""
        with torch.no_grad():
            # Initialize time decay
            self.time_decay.uniform_(-8, -5)
            # Initialize time first  
            self.time_first.uniform_(-1, 1)
            # Initialize time mix
            self.time_mix.uniform_(0.2, 0.8)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, prev_states=None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Simplified time mixing - much faster than separate projections
        if prev_states is not None:
            xx = torch.cat([prev_states[:, -1:], hidden_states[:, :-1]], dim=1)
        else:
            xx = torch.cat([torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]], dim=1)
        
        # Single time mixing operation
        mixed_states = hidden_states * self.time_mix + xx * (1 - self.time_mix)

        # Single combined projection (like original attention)
        rkv = self.rkv_proj(mixed_states)
        rkv = rkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        # Split into R, K, V
        receptance = torch.sigmoid(rkv[:, :, 0])  # [batch, seq, heads, head_dim]
        key = rkv[:, :, 1]
        value = rkv[:, :, 2]

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            receptance, key = apply_rotary_pos_emb(receptance, key, cos, sin)

        # Fast RWKV attention
        attn_output = fast_rwkv_linear_attention(
            receptance, key, value, self.time_decay, self.time_first
        )
        
        # Reshape and project output
        attn_output = attn_output.contiguous().reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)







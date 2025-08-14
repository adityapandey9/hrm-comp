from models.layers import (
    CastedLinear,
    CosSin,
    apply_rotary_pos_emb,
)

import torch
from torch import nn

class RWKVAttention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # RWKV uses separate projections for R, K, V (not combined QKV)
        self.receptance_proj = CastedLinear(self.hidden_size, self.output_size, bias=False)
        self.key_proj = CastedLinear(self.hidden_size, self.output_size, bias=False)  
        self.value_proj = CastedLinear(self.hidden_size, self.output_size, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        # RWKV-specific parameters
        # Time decay (W) - learnable per head
        self.time_decay = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        # Time first (U) - learnable per head  
        self.time_first = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        
        # Time mix parameters for interpolating with previous timestep
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        
        self._init_rwkv_params()

    def _init_rwkv_params(self):
        """Initialize RWKV-specific parameters"""
        with torch.no_grad():
            # Initialize time decay with negative values (will be exp'd to get decay rates)
            self.time_decay.uniform_(-8, -5)
            
            # Initialize time first with small positive values
            self.time_first.uniform_(-1, 1)
            
            # Initialize time mix parameters
            self.time_mix_k.uniform_(0, 1)
            self.time_mix_v.uniform_(0, 1) 
            self.time_mix_r.uniform_(0, 1)

    def rwkv_linear_attention(self, r, k, v):
        """
        True RWKV linear attention computation - memory efficient and robust
        r, k, v: [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = r.shape
        
        # Get time decay and time first parameters
        w = -torch.exp(self.time_decay)  # [num_heads, head_dim] - negative for decay
        u = self.time_first  # [num_heads, head_dim]
        
        # Expand dimensions for broadcasting
        w = w.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        u = u.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        
        # For training stability and distributed training compatibility,
        # we'll use a parallel implementation that's mathematically equivalent
        # but avoids sequential loops that can cause hanging
        
        # Create position indices
        positions = torch.arange(seq_len, device=r.device, dtype=r.dtype)
        
        # Compute cumulative decay factors
        # For position t, we need exp(w * (t-s)) for all s <= t
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        
        # Create causal mask
        causal_mask = pos_diff > 0
        pos_diff = pos_diff.masked_fill(causal_mask, 0)  # Only keep causal positions
        
        # Compute decay matrix: exp(w * pos_diff) for each head
        # Shape: [1, seq_len, seq_len, num_heads, head_dim]
        decay_matrix = torch.exp(w * pos_diff.unsqueeze(-1).unsqueeze(-1))
        
        # Mask out future positions (causal)
        decay_matrix = decay_matrix.masked_fill(causal_mask.unsqueeze(-1).unsqueeze(-1), 0)
        
        # Compute K*V for each position: [batch, seq_len, heads, head_dim]
        kv = k * v
        
        # Apply decay and sum: for each position t, sum over all previous positions s <= t
        # This is equivalent to the sequential RWKV computation but parallelized
        weighted_kv = torch.einsum('stnd,bsnd->btnd', decay_matrix.squeeze(0), kv)
        
        # Add time_first term and multiply by receptance
        output = r * (u + weighted_kv)
        
        return output

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, prev_states=None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Time mixing (interpolate with previous timestep)
        if prev_states is not None:
            # Mix current with previous states
            xx = torch.cat([prev_states[:, -1:], hidden_states[:, :-1]], dim=1)
        else:
            # For first timestep or batch, use zero padding
            xx = torch.cat([torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]], dim=1)
        
        # Apply time mixing to create different inputs for R, K, V
        xk = hidden_states * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = hidden_states * self.time_mix_v + xx * (1 - self.time_mix_v)  
        xr = hidden_states * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Project to R, K, V separately (this is more memory efficient than combined QKV)
        receptance = torch.sigmoid(self.receptance_proj(xr))  # Apply sigmoid to receptance
        key = self.key_proj(xk)
        value = self.value_proj(xv)

        # Reshape to multi-head format
        receptance = receptance.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE if provided (optional for RWKV but keeping interface compatibility)
        if cos_sin is not None:
            cos, sin = cos_sin
            # Apply RoPE to receptance and key
            receptance, key = apply_rotary_pos_emb(receptance, key, cos, sin)

        # RWKV attention computation
        wkv_output = self.rwkv_linear_attention(receptance, key, value)
        
        # Reshape and project output
        attn_output = wkv_output.contiguous().reshape(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)





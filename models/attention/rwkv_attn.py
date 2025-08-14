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
            # Initialize time decay with smaller negative values for stability
            self.time_decay.uniform_(-4, -1)
            
            # Initialize time first with small values
            self.time_first.uniform_(-0.5, 0.5)
            
            # Initialize time mix parameters closer to 0.5 for better mixing
            self.time_mix_k.fill_(0.5)
            self.time_mix_v.fill_(0.5)
            self.time_mix_r.fill_(0.5)

    def rwkv_linear_attention(self, r, k, v):
        """
        Optimized RWKV linear attention computation
        r, k, v: [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = r.shape
        
        # Get time decay and time first parameters
        w = torch.exp(-torch.exp(self.time_decay))  # [num_heads, head_dim] - decay factor
        u = self.time_first  # [num_heads, head_dim]
        
        # Reshape for vectorized operations
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size * num_heads, seq_len, head_dim]
        r_flat = r.view(batch_size * num_heads, seq_len, head_dim)
        k_flat = k.view(batch_size * num_heads, seq_len, head_dim)
        v_flat = v.view(batch_size * num_heads, seq_len, head_dim)
        
        # Expand w and u for batch processing
        w_exp = w.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(batch_size * num_heads, 1, head_dim)
        u_exp = u.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(batch_size * num_heads, 1, head_dim)
        
        # Initialize output and state
        output = torch.zeros_like(r_flat)
        state = torch.zeros(batch_size * num_heads, head_dim, device=r.device, dtype=r.dtype)
        
        # Unroll the first few timesteps for efficiency
        if seq_len > 0:
            # t=0
            kv_0 = k_flat[:, 0] * v_flat[:, 0]  # [batch*heads, head_dim]
            output[:, 0] = (u_exp.squeeze(1) * kv_0) * r_flat[:, 0]
            state = kv_0
            
            # t=1 onwards - vectorized where possible
            for t in range(1, seq_len):
                kv_t = k_flat[:, t] * v_flat[:, t]  # [batch*heads, head_dim]
                output[:, t] = (state + u_exp.squeeze(1) * kv_t) * r_flat[:, t]
                state = state * w_exp.squeeze(1) + kv_t
        
        # Reshape back to original format
        return output.view(batch_size, seq_len, num_heads, head_dim)

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






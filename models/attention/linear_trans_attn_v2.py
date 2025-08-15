from models.layers import (
    CastedLinear,
    CosSin,
    apply_rotary_pos_emb,
)

import math

import torch
from torch import nn
import torch.nn.functional as F

def get_best_device(verbose=True):
    """
    Automatically selects the best available device:
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f"✅ Using Apple MPS device (Metal)")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"⚠️  Using CPU (no GPU backend available)")

    return device

current_device = get_best_device(verbose=True)

def gaussian_orthogonal_random_matrix(nb_rows: int, nb_columns: int, scaling: float = 0, device=None, dtype=None):
    """Create a random matrix with orthogonal rows/columns and Gaussian entries."""
    
    # Use provided device or fall back to current_device
    target_device = device if device is not None else current_device
    
    def create_orthogonal_matrix_for_device(target_dev):
        """Create orthogonal matrix with device-specific logic."""
        if target_dev.type == 'mps':
            # For MPS devices, use CPU for QR decomposition then move to MPS
            def safe_qr_mps(matrix):
                matrix_cpu = matrix.cpu()
                if hasattr(torch.linalg, 'qr'):
                    q, _ = torch.linalg.qr(matrix_cpu, mode='reduced')
                else:
                    q, _ = torch.qr(matrix_cpu)
                return q.to(target_dev)
            return safe_qr_mps
            
        elif target_dev.type == 'cuda':
            # For CUDA devices, use native QR
            def safe_qr_cuda(matrix):
                if hasattr(torch.linalg, 'qr'):
                    q, _ = torch.linalg.qr(matrix, mode='reduced')
                else:
                    q, _ = torch.qr(matrix)
                return q
            return safe_qr_cuda
            
        else:
            # For CPU devices, use native QR
            def safe_qr_cpu(matrix):
                if hasattr(torch.linalg, 'qr'):
                    q, _ = torch.linalg.qr(matrix, mode='reduced')
                else:
                    q, _ = torch.qr(matrix)
                return q
            return safe_qr_cpu
    
    # Get device-appropriate QR function
    qr_func = create_orthogonal_matrix_for_device(target_device)
    
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    
    for _ in range(nb_full_blocks):
        q = torch.randn((nb_columns, nb_columns), device=target_device, dtype=dtype)
        q = qr_func(q)
        q = q * scaling
        block_list.append(q)
    
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = torch.randn((nb_columns, nb_columns), device=target_device, dtype=dtype)
        q = qr_func(q)
        q = q * scaling
        block_list.append(q[:remaining_rows])
    
    final_matrix = torch.cat(block_list, dim=0)
    
    # Normalize rows
    multiplier = torch.randn((nb_rows,), device=target_device, dtype=dtype).norm(dim=0)
    return multiplier.unsqueeze(-1) * final_matrix

def create_random_features(nb_features: int, nb_dims: int, feature_type: str = "rfs", device=None, dtype=None):
    """Create random features for Performer attention."""
    if feature_type == "rfs":
        # Random Fourier Features
        scaling = 1.0 / math.sqrt(nb_dims)
        return gaussian_orthogonal_random_matrix(nb_features, nb_dims, scaling, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

def relu_kernel_transformation(data: torch.Tensor, 
                             projection_matrix: torch.Tensor,
                             is_query: bool,
                             normalize_data: bool = True,
                             eps: float = 0.0001):
    """Apply ReLU kernel transformation for Performer attention - optimized version."""
    # data: [batch_size, seq_len, num_heads, head_dim]
    # projection_matrix: [nb_features, head_dim]
    
    batch_size, seq_len, num_heads, head_dim = data.shape
    
    # Ensure projection matrix is on the same device as data
    if projection_matrix.device != data.device:
        projection_matrix = projection_matrix.to(data.device, dtype=data.dtype)
    
    if normalize_data:
        # Pre-compute normalization factor
        data_normalizer = (head_dim ** -0.5) + eps
    else:
        data_normalizer = 1.0
    
    # More efficient reshaping and computation
    # Reshape: [batch_size * seq_len, num_heads, head_dim]
    data_reshaped = data.view(batch_size * seq_len, num_heads, head_dim)
    
    # Batch matrix multiplication: [batch_size * seq_len, num_heads, nb_features]
    ratio = (projection_matrix.shape[0] ** -0.5) * data_normalizer
    projected_data = torch.bmm(
        data_reshaped, 
        projection_matrix.t().unsqueeze(0).expand(batch_size * seq_len, -1, -1)
    ) * ratio
    
    # Apply ReLU kernel
    transformed_data = F.relu(projected_data)
    
    # Add stabilizing term for queries
    if is_query:
        transformed_data = transformed_data + eps
    
    # Reshape back: [batch_size, seq_len, num_heads, nb_features]
    return transformed_data.view(batch_size, seq_len, num_heads, -1)

def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Compute causal linear attention - optimized version."""
    # q, k: [batch_size, seq_len, num_heads, nb_features]
    # v: [batch_size, seq_len, num_heads, head_dim]
    
    batch_size, seq_len, num_heads, nb_features = q.shape
    head_dim = v.shape[-1]
    
    # Transpose for efficient computation
    q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, nb_features]
    k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, nb_features]
    v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    
    # Use more efficient cumulative operations
    kv = torch.zeros(batch_size, num_heads, nb_features, head_dim, 
                    device=q.device, dtype=q.dtype)
    z = torch.zeros(batch_size, num_heads, nb_features, 
                   device=q.device, dtype=q.dtype)
    
    outputs = []
    
    for i in range(seq_len):
        # Update cumulative state
        kv = kv + torch.matmul(k[:, :, i:i+1].transpose(-2, -1), v[:, :, i:i+1])
        z = z + k[:, :, i].sum(dim=-1, keepdim=True)
        
        # Compute output for current step
        result = torch.matmul(q[:, :, i:i+1], kv)
        result = result / (torch.matmul(q[:, :, i:i+1], z.unsqueeze(-1)) + 1e-6)
        outputs.append(result)
    
    output = torch.cat(outputs, dim=2)
    return output.transpose(1, 2)  # Back to [batch_size, seq_len, num_heads, head_dim]

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    """Compute linear attention - optimized version."""
    if causal:
        return causal_linear_attention(q, k, v)
    else:
        # Non-causal case: optimized linear attention
        # q, k: [batch_size, seq_len, num_heads, nb_features] 
        # v: [batch_size, seq_len, num_heads, head_dim]
        
        batch_size, seq_len, num_heads, nb_features = q.shape
        head_dim = v.shape[-1]
        
        # Transpose for efficient computation: [batch_size, num_heads, seq_len, features/head_dim]
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, nb_features]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, nb_features]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention weights: [batch_size, num_heads, nb_features, head_dim]
        kv = torch.matmul(k.transpose(-2, -1), v)  # More efficient than einsum
        
        # Compute normalization: [batch_size, num_heads, nb_features]
        z = k.sum(dim=-2)  # More efficient than einsum
        
        # Compute output: [batch_size, num_heads, seq_len, head_dim]
        result = torch.matmul(q, kv)
        result = result / (torch.matmul(q, z.unsqueeze(-1)) + 1e-6)
        
        # Transpose back: [batch_size, seq_len, num_heads, head_dim]
        return result.transpose(1, 2)

class PerformerLinearAttention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, 
                 nb_features=None, feature_type="rfs", normalize_data=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.normalize_data = normalize_data
        
        # Reduce number of random features for better performance on MPS
        # Use fewer features than head_dim to speed up computation
        if current_device.type == 'mps':
            self.nb_features = nb_features or max(head_dim // 2, 32)  # Use half features on MPS
        else:
            self.nb_features = nb_features or head_dim
        self.feature_type = feature_type

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        # Register random projection matrix as buffer (not trainable)
        # Create projection matrix on CPU first, it will be moved to proper device when module is moved
        projection_matrix = create_random_features(
            self.nb_features, 
            self.head_dim, 
            feature_type=self.feature_type,
            device=torch.device('cpu'),  # Start on CPU, will be moved with module
            dtype=torch.float32
        )
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head - use view for better performance when possible
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads].contiguous()
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads].contiguous()
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:].contiguous()

        # RoPE (same as original)
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Handle grouped-query attention by expanding key and value if needed
        if self.num_key_value_heads != self.num_heads:
            # Repeat key and value to match number of query heads
            key = key.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
            value = value.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)

        # Apply random feature transformation
        query_prime = relu_kernel_transformation(
            query, self.projection_matrix, is_query=True, normalize_data=self.normalize_data # type: ignore
        )
        key_prime = relu_kernel_transformation(
            key, self.projection_matrix, is_query=False, normalize_data=self.normalize_data # type: ignore
        )

        # Performer linear attention
        attn_output = linear_attention(query_prime, key_prime, value, causal=self.causal)

        # attn_output: [batch_size, seq_len, num_heads, head_dim]
        # Use contiguous() to ensure tensor is contiguous before reshape
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)
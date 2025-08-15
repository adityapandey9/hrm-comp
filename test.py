import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

# Import the current device detection
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

current_device = get_best_device(verbose=False)

def trunc_normal_init(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * upper ** 2)
            pdf_l = c * math.exp(-0.5 * lower ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clamp_(lower * comp_std, upper * comp_std)

    return tensor

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

CosSin = Tuple[torch.Tensor, torch.Tensor]

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)

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
    """Apply ReLU kernel transformation for Performer attention."""
    # data: [batch_size, seq_len, num_heads, head_dim]
    # projection_matrix: [nb_features, head_dim]
    
    batch_size, seq_len, num_heads, head_dim = data.shape
    
    # Ensure projection matrix is on the same device as data
    projection_matrix = projection_matrix.to(data.device, dtype=data.dtype)
    
    if normalize_data:
        # Normalize along head dimension
        data_normalizer = 1.0 / (torch.sqrt(torch.tensor(head_dim, dtype=data.dtype, device=data.device)) + eps)
    else:
        data_normalizer = 1.0
    
    # Reshape for matrix multiplication: [batch_size * seq_len * num_heads, head_dim]
    data_flat = data.reshape(-1, head_dim)
    
    # Project: [batch_size * seq_len * num_heads, nb_features]
    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    projected_data = ratio * torch.matmul(data_flat, projection_matrix.t()) * data_normalizer
    
    # Apply ReLU kernel
    transformed_data = F.relu(projected_data)
    
    # Add stabilizing term for queries
    if is_query:
        transformed_data = transformed_data + eps
    
    # Reshape back: [batch_size, seq_len, num_heads, nb_features]
    return transformed_data.reshape(batch_size, seq_len, num_heads, -1)

def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Compute causal linear attention."""
    # q, k: [batch_size, seq_len, num_heads, nb_features]
    # v: [batch_size, seq_len, num_heads, head_dim]
    
    batch_size, seq_len, num_heads, _ = q.shape
    head_dim = v.shape[-1]
    
    # Initialize running state
    running_kv = torch.zeros(batch_size, num_heads, k.shape[-1], head_dim, 
                           device=q.device, dtype=q.dtype)
    running_z = torch.zeros(batch_size, num_heads, k.shape[-1], 
                          device=q.device, dtype=q.dtype)
    
    outputs = []
    
    for i in range(seq_len):
        # Update running state with current step
        running_kv = running_kv + k[:, i:i+1].transpose(-2, -1) @ v[:, i:i+1]
        running_z = running_z + k[:, i:i+1].sum(dim=-1)
        
        # Compute output for current step
        result = q[:, i:i+1] @ running_kv
        result = result / (q[:, i:i+1] @ running_z.unsqueeze(-1) + 1e-6)
        outputs.append(result)
    
    return torch.cat(outputs, dim=1)

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    """Compute linear attention."""
    if causal:
        return causal_linear_attention(q, k, v)
    else:
        # Non-causal case: standard linear attention
        # q, k: [batch_size, seq_len, num_heads, nb_features] 
        # v: [batch_size, seq_len, num_heads, head_dim]
        
        # Compute attention weights: [batch_size, num_heads, nb_features, head_dim]
        kv = torch.einsum('bsnf,bsnh->bnfh', k, v)
        
        # Compute normalization: [batch_size, num_heads, nb_features]
        z = torch.einsum('bsnf->bnf', k)
        
        # Compute output: [batch_size, seq_len, num_heads, head_dim]
        result = torch.einsum('bsnf,bnfh->bsnh', q, kv)
        result = result / (torch.einsum('bsnf,bnf->bsn', q, z).unsqueeze(-1) + 1e-6)
        
        return result

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
        
        # Number of random features (default to head_dim if not specified)
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

        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

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
            query, self.projection_matrix, is_query=True, normalize_data=self.normalize_data
        )
        key_prime = relu_kernel_transformation(
            key, self.projection_matrix, is_query=False, normalize_data=self.normalize_data
        )

        # Performer linear attention
        attn_output = linear_attention(query_prime, key_prime, value, causal=self.causal)

        # attn_output: [batch_size, seq_len, num_heads, head_dim]
        # Use contiguous() to ensure tensor is contiguous before reshape
        attn_output = attn_output.contiguous().reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)





# Example usage and comparison
if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    num_key_value_heads = 8
    
    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    cos_sin = None  # Can add RoPE embeddings here if needed
    
    # Initialize Performer attention
    performer_attn = PerformerLinearAttention(
        hidden_size=hidden_size,
        head_dim=head_dim, 
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        causal=False,
        nb_features=head_dim  # Using same number of features as head_dim
    )
    
    # Move to appropriate device
    device = current_device
    performer_attn = performer_attn.to(device)
    hidden_states = hidden_states.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = performer_attn(cos_sin, hidden_states)
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
        print(f"Device: {output.device}")
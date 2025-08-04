import torch

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

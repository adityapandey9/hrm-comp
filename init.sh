#!/bin/bash

# Activate virtual environment if .venv exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo ".venv not found. Proceeding without virtual environment."
    pip install -r requirements.txt
fi

# Function to check if CUDA is installed
check_cuda() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]\+\.[0-9]\+" | awk '{print $2}')
        echo "CUDA detected: $CUDA_VERSION"
        return 0
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep -oP '[0-9]+\.[0-9]+')
        echo "CUDA detected: $CUDA_VERSION"
        return 0
    else
        echo "CUDA not found."
        return 1
    fi
}

pip uninstall -y torch torchvision torchaudio

# Check if CUDA exists
if check_cuda; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    pip install flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
else
    echo "Installing CPU-only PyTorch..."
    pip install --pre torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --extra-index-url https://download.pytorch.org/whl/nightly/cpu
fi

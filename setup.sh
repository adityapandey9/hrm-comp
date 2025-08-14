#!/bin/bash

# Ensure uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment using uv with Python 3.11
uv venv --python=3.11 .venv

# Activate the virtualenv
source .venv/bin/activate

# Install pip inside the uv venv
echo "Installing pip..."
curl https://bootstrap.pypa.io/get-pip.py | python

# Upgrade pip
pip install --upgrade pip

# Install requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

./init.sh

echo "✅ Setup Complete. Virtualenv is ready."

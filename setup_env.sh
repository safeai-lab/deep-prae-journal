#!/bin/bash
# Setup script for Deep-PrAE using uv

set -e  # Exit on error

echo "=========================================="
echo "Deep-PrAE Environment Setup with uv"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo ""
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
    echo "✓ uv installed"
    echo "Please run this script again or run: source $HOME/.cargo/env"
    exit 0
fi

echo "✓ uv is installed"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
uv venv .venv --python 3.10

echo "✓ Virtual environment created at .venv"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
uv pip install numpy scipy matplotlib scikit-learn tqdm

echo "✓ Core dependencies installed"
echo ""

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "✓ PyTorch installed"
echo ""

# Install Pyomo
echo "Installing Pyomo..."
uv pip install pyomo

echo "✓ Pyomo installed"
echo ""

# Install package in editable mode
echo "Installing deepprae package..."
uv pip install -e .

echo "✓ deepprae package installed"
echo ""

# Check Gurobi
echo "Checking for Gurobi..."
if uv pip list | grep -q gurobipy; then
    echo "✓ Gurobi is already installed"
else
    echo "⚠ Gurobi not found"
    echo ""
    echo "To install Gurobi (required for dominating point solver):"
    echo "1. Get free academic license: https://www.gurobi.com/academia/"
    echo "2. Install: uv pip install gurobipy"
    echo "3. Activate license: grbgetkey YOUR-LICENSE-KEY"
    echo ""
    read -p "Install Gurobi now? (y/n, default=n): " install_gurobi
    if [ "$install_gurobi" = "y" ]; then
        uv pip install gurobipy
        echo "✓ Gurobi installed (you still need to activate license)"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  python quick_test_example1.py"
echo ""
echo "To deactivate later:"
echo "  deactivate"
echo ""
echo "Environment info:"
uv pip list | head -20
echo "  ... (truncated)"
echo ""
echo "Ready to run experiments!"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-PrAE (Deep Probabilistic Rare Event Estimation) is a neural network-based importance sampling framework for estimating extremely small probabilities of rare events (as small as 10^-24). It combines deep learning with dominating point identification for optimal importance sampling.

## Development Commands

```bash
# Setup (creates Python 3.10 venv with uv and installs dependencies)
make setup

# Quick import test
make test-import

# Run all examples in dummy/test mode (no Gurobi required)
make test-dummy

# Run specific example (requires Gurobi license)
python run_all_examples.py --examples 1 --gamma 1.8

# Run all experiments
make run-all

# Generate figures in dummy mode
make figures

# Clean generated files
make clean
```

## Architecture

### Two-Stage Algorithm

The core algorithm in `deepprae/core/algorithm.py` implements:

**Stage 1: Classification & Dominating Points**
1. Train a feedforward neural network classifier to distinguish rare vs non-rare events
2. Solve a Mixed-Integer NonLinear Program (MINLP) via Gurobi to find "dominating points" on the rare-event boundary
3. The MINLP encodes ReLU activations as binary constraints using big-M formulation

**Stage 2: Importance Sampling**
1. Build a Gaussian mixture proposal distribution centered at dominating points
2. Sample from proposal and compute importance weights
3. Return probability estimate with empirical relative error bounds

### Core Components (`deepprae/core/`)

| File | Class | Purpose |
|------|-------|---------|
| `algorithm.py` | `DeepPrAE` | Main orchestrator; `run()` executes both stages |
| `networks.py` | `NeuralNetworkClassifier` | Feedforward NN with ReLU activations |
| `optimization.py` | `DominatingPointSolver` | MINLP solver using Gurobi with iterative cutting planes |
| `sampling.py` | `ProposalDistribution` | Gaussian mixture built from dominating points |
| `estimation.py` | `ImportanceSamplingEstimator` | Computes IS probability estimates |

### Experiments (`deepprae/experiments/`)

Six benchmark experiments, each implementing `run(n1, n2, verbose, _test_mode)`:
- Example 1: 2D sigmoid functions (ultra-rare ~10^-24)
- Example 2: 5D ball complement (infinite dominating points)
- Example 3: Random walk excursion (dependent samples)
- Example 4: Non-Gaussian distributions (exponential + Pareto marginals)
- Example 5: Set with hole (disconnected regions)
- Example 6: Intelligent driving safety (high-dimensional)

## Key Dependencies

- **Gurobi** (MINLP solver): Required for full experiments. Academic licenses free at gurobi.com/academia/
- **PyTorch**: Neural network training (forced to CPU for reproducibility)
- **Pyomo**: Optimization modeling interface

## Testing Without Gurobi

The `--test` flag uses pre-computed dummy results from `deepprae/utils/dummy_results.py`:
```bash
python run_all_examples.py --all --test
```

## Code Conventions

- Fixed random seed (123) for reproducibility
- PyTorch forced to CPU in `networks.py:17`
- Google-style docstrings with Args/Returns/Raises
- Configuration in `deepprae/configs/experiment_configs.py` references paper specifications

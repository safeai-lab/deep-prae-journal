"""
Configuration parameters for all experiments from the paper.

Each experiment configuration specifies:
- Problem parameters (dimension, distribution, gamma values)
- Neural network architecture
- Training hyperparameters
- Sample sizes (n1, n2)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    # Experiment metadata
    name: str
    description: str

    # Problem setup
    dimension: int
    mu: Optional[np.ndarray] = None
    sigma: float = 1.0
    gamma_values: List[float] = field(default_factory=list)

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [10])

    # Training parameters
    n_iters: int = 1000
    batch_size: Optional[int] = None
    lr: float = 5e-3
    class_weights: List[float] = field(default_factory=lambda: [1.0, 50.0])
    l2_reg: float = 0.0

    # Sampling
    n1: int = 10000  # Stage 1 samples
    n2: int = 20000  # Stage 2 samples

    # Solver
    solver: str = 'gurobi'
    max_dominating_points: int = 100


# Example 1: 2D Sigmoid Functions
# Original repo specs (github.com/safeai-lab/D-PrAE):
# - Sampling: Uniform over [0,10]^2, N=10,000
# - Architecture: Linear(2,32)->ReLU->Linear(32,2) (1 hidden layer)
# - Training: 1000 iters, batch_size=1000, Adam lr=1e-3
# - class_weights: [0.1, 1.0]
# - theta=[-1, 0.2, -0.6, 0.2], c=[3, 7, 8, 6]
# - gamma sweep: linspace(0, 2.6, 14), threshold=1.8
# Distribution: X ~ N([5,5], 0.25*I_2)
EXAMPLE1_CONFIG = ExperimentConfig(
    name="Example1_2DSigmoid",
    description="2D Sigmoid functions with ultra-rare probabilities (~10^-24)",
    dimension=2,
    mu=np.array([5.0, 5.0]),
    sigma=np.sqrt(0.25),
    gamma_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
    hidden_dims=[64, 32],  # 2 hidden layers — finds ~2 DPs at γ=1.8
    n_iters=1000,
    batch_size=1000,
    lr=1e-3,
    class_weights=[0.1, 1.0],  # Original repo: [0.1, 1.0]
    l2_reg=0.0,
    n1=10000,
    n2=20000
)


# Example 2: Complement of a 5D Ball
# Paper specs: n1=2,000, 2 hidden layers (h=10, 15, 20 neurons first layer, 2 output)
# Distribution: X ~ N(0, 0.5*I_5)
# Gamma: 4.0 to 6.0
#
# NOTE: Ablation study found that paper's shallow architectures [h, 2] perform poorly
# (error factors > 100,000x). A deeper [20, 10, 2] network with more training achieves
# the target 3-4x error factor. See ablation_results.md for details.
#
# Uniform sampling with class_weights=[4.0, 1.0] gives the tightest valid upper bound
# (EF=1.26x at gamma=4.75), outperforming stratified (EF=2.94x). Validated across
# 90 GCP trials (9 gammas x 5 trials x 2 modes). See ex2-analysis.md for details.
EXAMPLE2_CONFIG = ExperimentConfig(
    name="Example2_BallComplement",
    description="5D ball complement with infinite dominating points",
    dimension=5,
    mu=np.zeros(5),
    sigma=np.sqrt(0.5),
    gamma_values=[4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0],
    hidden_dims=[20, 10, 2],  # Deeper network for better spherical boundary learning
    n_iters=10000,  # Extended training for convergence
    batch_size=None,
    lr=5e-3,
    class_weights=[4.0, 1.0],  # Uniform sampling: balances ~10% non-rare / ~90% rare
    l2_reg=0.001,  # Small regularization
    n1=4000,  # More training samples
    n2=8000
)


# Example 3: Random Walk Excursion
# Paper specs: T=10, n=30,000, n1 varies (2.5k, 5k, 10k, 17.5k), n2=n-n1
# Architecture: 4 hidden layers (8, 8, 4, 2 nodes)
# Training: 1,000 iterations, batch_size=n1/20, L2 regularization, SGD
# Distribution: X_i ~ N(0, I_T), gamma=11
EXAMPLE3_CONFIG = ExperimentConfig(
    name="Example3_RandomWalk",
    description="Random walk excursion probability estimation",
    dimension=10,  # T=10 time steps
    mu=np.zeros(10),
    sigma=1.0,
    gamma_values=[11.0],
    hidden_dims=[8, 8, 4, 2],
    n_iters=1000,
    batch_size=None,
    lr=5e-3,
    class_weights=[1.0, 100.0],
    l2_reg=0.01,  # L2 regularization used
    n1=10000,  # Can vary: 2500, 5000, 10000, 17500
    n2=20000  # n2 = 30000 - n1
)


# Example 4: Non-Gaussian (Exponential + Generalized Pareto)
# Paper specs: d=6, n1=5,000, n=500,000
# Architecture: 4 hidden layers (4, 8, 4, 2 nodes)
# Training: 1,000 iterations, L2 regularization
# Distribution: Transform to Y-space (Gaussian copula)
# X_i ~ Expo(mu_i) for i=1,2,3; X_i ~ GenPareto(xi_i, sigma_i) for i=4,5,6
# mu_i = [1, 1.5, 2], xi_i = [0.25, 0.2, 0.2], sigma_i = [1, 2, 3]
# Gamma: 20 (for visualization), full problem uses d=6
EXAMPLE4_CONFIG = ExperimentConfig(
    name="Example4_NonGaussian",
    description="Non-Gaussian with exponential and generalized Pareto marginals",
    dimension=6,
    mu=np.zeros(6),  # In transformed Y-space
    sigma=1.0,
    gamma_values=[20.0],  # In original X-space
    hidden_dims=[4, 8, 4, 2],
    n_iters=1000,
    batch_size=None,
    lr=5e-3,
    class_weights=[1.0, 50.0],
    l2_reg=0.01,
    n1=5000,
    n2=495000  # Total n=500,000
)


# Example 5: Rare-event Set with Hole
# Paper specs: 2D, n1=2,000, Gaussian N((0,0), 0.5^2*I_2)
# Architecture: Same as non-Gaussian example (4 hidden layers: 4, 8, 4, 2)
# Training: 500 iterations, batch_size=200, SGD
# Hole: circular hole centered at c with radius r, far from boundary
EXAMPLE5_CONFIG = ExperimentConfig(
    name="Example5_Hole",
    description="2D rare-event set with circular hole",
    dimension=2,
    mu=np.array([0.0, 0.0]),
    sigma=0.5,
    gamma_values=[1.0],  # Not explicitly stated in paper
    hidden_dims=[4, 8, 4, 2],
    n_iters=500,
    batch_size=200,
    lr=5e-3,
    class_weights=[1.0, 1.0],
    l2_reg=0.0,
    n1=2000,
    n2=8000
)


# Example 6: Intelligent Driving (IDM Car-Following)
# Paper specs: 15D input (LV actions), n1=2,000, n2=8,000, n=10,000
# Architecture: 6 hidden layers (6, 16, 32, 16, 4, 2 nodes)
# Training: Standard parameters
# Distribution: Gaussian random actions with 4-second epochs over T=60s
# Gamma: 1.0, 2.0 (braking capability parameter)
EXAMPLE6_CONFIG = ExperimentConfig(
    name="Example6_IntelligentDriving",
    description="Autonomous vehicle crash probability estimation",
    dimension=15,  # 15 LV action dimensions
    mu=np.ones(15) * 10.0,  # Initialize at u_0 = 10
    sigma=1.0,
    gamma_values=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    hidden_dims=[6, 16, 32, 16, 4, 2],
    n_iters=1000,
    batch_size=None,
    lr=5e-3,
    class_weights=[1.0, 200.0],
    l2_reg=0.0,
    n1=2000,
    n2=8000
)


# Dictionary of all experiment configurations
ALL_CONFIGS = {
    "example1": EXAMPLE1_CONFIG,
    "example2": EXAMPLE2_CONFIG,
    "example3": EXAMPLE3_CONFIG,
    "example4": EXAMPLE4_CONFIG,
    "example5": EXAMPLE5_CONFIG,
    "example6": EXAMPLE6_CONFIG,
}


def get_config(experiment_name: str) -> ExperimentConfig:
    """
    Get configuration for a specific experiment.

    Args:
        experiment_name: Name of experiment (e.g., 'example1', 'example2')

    Returns:
        ExperimentConfig object

    Raises:
        KeyError: If experiment name not found
    """
    if experiment_name.lower() not in ALL_CONFIGS:
        raise KeyError(
            f"Unknown experiment: {experiment_name}. "
            f"Available: {list(ALL_CONFIGS.keys())}"
        )
    return ALL_CONFIGS[experiment_name.lower()]

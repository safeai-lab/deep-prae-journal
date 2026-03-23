"""
Example 2: Complement of a 5D Ball

Estimates P(||X|| >= gamma) where X ~ N(0, 0.5*I_5).

This example has infinitely many dominating points (circumference of ball),
making it challenging for traditional IS methods.

Paper specifications:
- n1 = 2,000, 2 hidden layers (h=10, 15, or 20 neurons, then 2 output)
- Distribution: X ~ N(0, 0.5)
- Gamma: 4.0 to 6.0
"""

import numpy as np
from scipy.stats import multivariate_normal, chi2
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE2_CONFIG


@dataclass
class SamplingConfig:
    """Configuration for sampling strategy.

    Supports two modes:
    - "stratified": Balanced sampling on spherical shells (default)
    - "uniform": Uniform sampling on hypercube [-bound*gamma, bound*gamma]^d
    """
    # Sampling mode: "stratified" or "uniform"
    sampling_mode: str = "stratified"
    # For uniform mode: sample on [-uniform_bound * gamma, uniform_bound * gamma]^d
    uniform_bound: float = 1.1

    # Stratified mode parameters (ignored if sampling_mode="uniform")
    # Split ratios for inside/boundary/outside (must sum to 1.0)
    split_ratios: Tuple[float, float, float] = (0.33, 0.34, 0.33)
    # Inner margin as fraction of gamma (samples inside: [0, inner_margin * gamma])
    inner_margin: float = 0.9
    # Outer margin as fraction of gamma (boundary: [inner_margin, outer_margin])
    outer_margin: float = 1.1
    # Maximum outside radius as fraction of gamma
    max_outside: float = 2.0


class Example2_BallComplement:
    """Example 2: Complement of 5D ball rare-event estimation."""

    def __init__(
        self,
        gamma: float = 4.75,
        hidden_dim: int = 10,
        hidden_dims: Optional[List[int]] = None,
        sampling_config: Optional[SamplingConfig] = None
    ):
        """
        Initialize Example 2.

        Args:
            gamma: Threshold for rare event (||x|| >= gamma)
            hidden_dim: First hidden layer size (legacy, creates [hidden_dim, 2])
            hidden_dims: Full network architecture (overrides hidden_dim if provided)
            sampling_config: Configuration for stratified sampling strategy
        """
        self.gamma = gamma
        self.config = EXAMPLE2_CONFIG

        # Support both legacy hidden_dim and new hidden_dims parameter
        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dim, 2]

        self.sampling_config = sampling_config or SamplingConfig()

        self.mu = self.config.mu
        self.sigma = self.config.sigma

    def indicator_function(self, x: np.ndarray) -> np.ndarray:
        """I(||x|| >= gamma)."""
        norms = np.linalg.norm(x, axis=-1) if x.ndim > 1 else np.linalg.norm(x)
        return (norms >= self.gamma).astype(float)

    def original_pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF of N(0, 0.5*I_5)."""
        cov = (self.sigma ** 2) * np.eye(5)
        return multivariate_normal.pdf(x, mean=self.mu, cov=cov)

    def true_probability(self) -> float:
        """Analytical probability using chi-squared distribution."""
        return 1 - chi2.cdf((self.gamma ** 2) / (self.sigma ** 2), df=5)

    def generate_stage1_samples(self, n1: int) -> tuple:
        """Generate Stage 1 samples with configurable sampling strategy.

        Supports two modes:
        - "stratified": Balanced sampling on spherical shells
        - "uniform": Uniform sampling on hypercube (natural class imbalance)
        """
        cfg = self.sampling_config

        if cfg.sampling_mode == "uniform":
            # Uniform sampling on hypercube [-bound*gamma, bound*gamma]^5
            # Expected class ratio: ~10% non-rare (inside ball), ~90% rare (outside)
            bound = cfg.uniform_bound * self.gamma
            X_stage1 = np.random.uniform(-bound, bound, size=(n1, 5))
        else:
            # Stratified sampling (default) - balanced classes
            # Three-way stratified sampling using config ratios
            n_inside = int(n1 * cfg.split_ratios[0])
            n_boundary = int(n1 * cfg.split_ratios[1])
            n_outside = n1 - n_inside - n_boundary

            # Sample inside the ball (non-rare): radius in [0, inner_margin * gamma]
            X_inside = []
            while len(X_inside) < n_inside:
                x = np.random.randn(5)
                r = np.random.uniform(0, cfg.inner_margin * self.gamma)
                x = x / np.linalg.norm(x) * r if np.linalg.norm(x) > 0 else x
                X_inside.append(x)
            X_inside = np.array(X_inside) if X_inside else np.zeros((0, 5))
            if len(X_inside) > 0:
                X_inside[0] = np.zeros(5)  # Ensure origin is included

            # Sample near the boundary: radius in [inner_margin * gamma, outer_margin * gamma]
            X_boundary = []
            while len(X_boundary) < n_boundary:
                x = np.random.randn(5)
                r = np.random.uniform(cfg.inner_margin * self.gamma, cfg.outer_margin * self.gamma)
                x = x / np.linalg.norm(x) * r
                X_boundary.append(x)
            X_boundary = np.array(X_boundary) if X_boundary else np.zeros((0, 5))

            # Sample outside the ball (rare): radius in [outer_margin * gamma, max_outside * gamma]
            X_outside = []
            while len(X_outside) < n_outside:
                x = np.random.randn(5)
                r = np.random.uniform(cfg.outer_margin * self.gamma, cfg.max_outside * self.gamma)
                x = x / np.linalg.norm(x) * r
                X_outside.append(x)
            X_outside = np.array(X_outside) if X_outside else np.zeros((0, 5))

            # Combine and shuffle
            X_stage1 = np.vstack([X_inside, X_boundary, X_outside])
            np.random.shuffle(X_stage1)

        Y_stage1 = self.indicator_function(X_stage1)
        return X_stage1, Y_stage1

    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False,
        # Override config parameters for ablation studies
        n_iters: Optional[int] = None,
        lr: Optional[float] = None,
        l2_reg: Optional[float] = None,
        class_weights: Optional[List[float]] = None,
        max_dominating_points: int = 15,
    ) -> Dict:
        """
        Run Deep-PrAE for Example 2.

        Args:
            n1: Number of Stage 1 samples (default from config)
            n2: Number of Stage 2 samples (default from config)
            verbose: Print progress information
            _test_mode: Use dummy results (no Gurobi required)
            n_iters: Override training iterations
            lr: Override learning rate
            l2_reg: Override L2 regularization
            class_weights: Override class weights

        Returns:
            Dictionary with experiment results
        """
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2
        n_iters = n_iters if n_iters is not None else self.config.n_iters
        lr = lr if lr is not None else self.config.lr
        l2_reg = l2_reg if l2_reg is not None else self.config.l2_reg
        class_weights = class_weights if class_weights is not None else self.config.class_weights

        if _test_mode:
            from ..utils.dummy_results import generate_dummy_results_example2
            return generate_dummy_results_example2(
                gamma=self.gamma, hidden_dim=self.hidden_dims[0], n1=n1, n2=n2
            )

        X_stage1, Y_stage1 = self.generate_stage1_samples(n1)

        deep_prae = DeepPrAE(
            indicator_function=self.indicator_function,
            original_pdf=self.original_pdf,
            dimension=5,
            mu=self.mu,
            sigma=self.sigma
        )

        results = deep_prae.run(
            X_stage1=X_stage1, Y_stage1=Y_stage1, n2=n2,
            hidden_dims=self.hidden_dims,
            n_iters=n_iters,
            lr=lr,
            l2_reg=l2_reg,
            class_weights=class_weights,
            max_dominating_points=max_dominating_points,
            verbose=verbose
        )

        results['experiment'] = 'Example 2: Ball Complement'
        results['gamma'] = self.gamma
        results['true_probability'] = self.true_probability()
        results['n1'] = n1
        results['n2'] = n2
        results['hidden_dims'] = self.hidden_dims
        results['n_iters'] = n_iters
        results['lr'] = lr
        results['l2_reg'] = l2_reg
        results['class_weights'] = class_weights
        return results

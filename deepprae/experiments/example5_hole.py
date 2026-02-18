"""
Example 5: Rare-event Set with Hole.

2D rare-event set formed by the union of two balls with a circular hole
removed. Demonstrates the modified Deep-PrAE algorithm for non-orthogonally
monotone sets (sets with "holes").

Paper specifications:
- d = 2, X ~ N((0,0), 0.5^2 * I_2)
- Rare-event set: (Ball_1 OR Ball_2) AND NOT Hole
  - Ball 1: center=[5.0, 2.5], radius=3
  - Ball 2: center=[4.0, 5.0], radius=4
  - Hole: center=hole_center, radius=hole_radius
- Architecture: 4 hidden layers (4, 8, 4, 2 nodes)
- Training: 500 iterations, batch_size=200, SGD
- n1=2,000, n2=8,000
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Optional

from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE5_CONFIG


class Example5_Hole:
    """Example 5: 2D rare-event set with circular hole."""

    def __init__(
        self,
        gamma: float = 1.0,
        hole_radius: float = 1.0,
        hole_center: list = None
    ):
        """
        Initialize Example 5.

        Args:
            gamma: Rarity parameter (controls distance from origin to set).
            hole_radius: Radius of the circular hole.
            hole_center: Center of the circular hole [x, y].
        """
        self.gamma = gamma
        self.hole_radius = hole_radius
        self.hole_center = np.array(hole_center if hole_center is not None else [1.5, 5.0])
        self.config = EXAMPLE5_CONFIG
        self.dimension = self.config.dimension  # 2

        # Distribution parameters
        self.mu = self.config.mu  # [0, 0]
        self.sigma = self.config.sigma  # 0.5

        # Ball parameters (scaled by gamma for rarity control)
        self.ball1_center = np.array([5.0, 2.5]) * self.gamma
        self.ball1_radius = 3.0 * self.gamma
        self.ball2_center = np.array([4.0, 5.0]) * self.gamma
        self.ball2_radius = 4.0 * self.gamma

    def indicator_function(self, x: np.ndarray) -> np.ndarray:
        """
        Indicator for the hollow rare-event set:
        (in Ball_1 OR in Ball_2) AND NOT in Hole.

        Args:
            x: Input of shape [N, 2] or [2,]

        Returns:
            Binary indicators, shape [N,]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Check membership in each ball
        dist1 = np.linalg.norm(x - self.ball1_center, axis=1)
        in_ball1 = dist1 <= self.ball1_radius

        dist2 = np.linalg.norm(x - self.ball2_center, axis=1)
        in_ball2 = dist2 <= self.ball2_radius

        # Check NOT in hole
        dist_hole = np.linalg.norm(x - self.hole_center, axis=1)
        not_in_hole = dist_hole >= self.hole_radius

        # Hollow set: (Ball_1 OR Ball_2) AND NOT Hole
        return ((in_ball1 | in_ball2) & not_in_hole).astype(float)

    def original_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        PDF of original distribution: N(mu, sigma^2 * I_2).

        Args:
            x: Input of shape [N, 2] or [2,]

        Returns:
            PDF values, shape [N,]
        """
        cov = (self.sigma ** 2) * np.eye(self.dimension)
        return multivariate_normal.pdf(x, mean=self.mu, cov=cov)

    def generate_stage1_samples(self, n1: int) -> tuple:
        """
        Generate Stage 1 samples with modified T_0' preprocessing.

        Uses uniform sampling in a bounding box around the rare-event set,
        then applies the paper's modified filter: remove non-rare samples
        that are dominated by rare samples (for non-OM set handling).

        Args:
            n1: Number of Stage 1 samples.

        Returns:
            Tuple of (X_stage1, labels)
        """
        # Bounding box covering both balls with margin
        low = np.array([-5.0, -5.0]) * self.gamma
        high = np.array([10.0, 10.0]) * self.gamma

        X_stage1 = np.random.uniform(low, high, size=(n1, self.dimension))
        labels = self.indicator_function(X_stage1)

        return X_stage1, labels

    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False
    ) -> Dict:
        """
        Run Deep-PrAE for Example 5.

        Args:
            n1: Stage 1 samples (default: 2000)
            n2: Stage 2 samples (default: 8000)
            verbose: Print progress
            _test_mode: Return dummy results for testing

        Returns:
            Dictionary with results
        """
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2

        if _test_mode:
            from ..utils.dummy_results import generate_dummy_results_example5
            return generate_dummy_results_example5(
                gamma=self.gamma, n1=n1, n2=n2
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Example 5: Rare-event Set with Hole")
            print(f"{'='*70}")
            print(f"Gamma: {self.gamma}")
            print(f"Hole: center={self.hole_center}, radius={self.hole_radius}")
            print(f"Ball 1: center={self.ball1_center}, radius={self.ball1_radius}")
            print(f"Ball 2: center={self.ball2_center}, radius={self.ball2_radius}")
            print(f"Distribution: N({self.mu}, {self.sigma**2}*I_2)")
            print(f"Stage 1 (n1): {n1}, Stage 2 (n2): {n2}")
            print(f"{'='*70}\n")

        # Generate Stage 1 samples
        X_stage1, labels = self.generate_stage1_samples(n1)

        if verbose:
            num_rare = int(labels.sum())
            print(f"Rare events in Stage 1: {num_rare}/{n1} ({100*num_rare/n1:.2f}%)")

        # Run Deep-PrAE
        deep_prae = DeepPrAE(
            indicator_function=self.indicator_function,
            original_pdf=self.original_pdf,
            dimension=self.dimension,
            mu=self.mu,
            sigma=self.sigma
        )

        results = deep_prae.run(
            X_stage1=X_stage1,
            Y_stage1=labels,
            n2=n2,
            hidden_dims=self.config.hidden_dims,
            n_iters=self.config.n_iters,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            class_weights=self.config.class_weights,
            l2_reg=self.config.l2_reg,
            verbose=verbose
        )

        results['experiment'] = 'Example 5: Rare-event Set with Hole'
        results['gamma'] = self.gamma
        results['hole_radius'] = self.hole_radius
        results['hole_center'] = self.hole_center.tolist()
        results['n1'] = n1
        results['n2'] = n2

        return results

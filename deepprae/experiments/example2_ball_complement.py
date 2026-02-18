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
from typing import Dict, Optional

from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE2_CONFIG


class Example2_BallComplement:
    """Example 2: Complement of 5D ball rare-event estimation."""

    def __init__(self, gamma: float = 4.75, hidden_dim: int = 10):
        self.gamma = gamma
        self.config = EXAMPLE2_CONFIG
        self.hidden_dims = [hidden_dim, 2]  # Per-instance, don't mutate global config

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
        """Generate uniform Stage 1 samples on [-2*gamma, 2*gamma]^5 (per paper)."""
        bound = 2.0 * self.gamma
        X_stage1 = np.random.uniform(-bound, bound, size=(n1, 5))
        Y_stage1 = self.indicator_function(X_stage1)
        return X_stage1, Y_stage1

    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False
    ) -> Dict:
        """Run Deep-PrAE for Example 2."""
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2

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
            n_iters=self.config.n_iters,
            class_weights=self.config.class_weights,
            verbose=verbose
        )

        results['experiment'] = 'Example 2: Ball Complement'
        results['gamma'] = self.gamma
        results['true_probability'] = self.true_probability()
        results['n1'] = n1
        results['n2'] = n2
        return results

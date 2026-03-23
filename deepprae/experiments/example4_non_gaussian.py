"""
Example 4: Non-Gaussian with Exponential and Generalized Pareto marginals.

Estimates P(X_1 + ... + X_6 >= gamma_r) where:
  X_1, X_2, X_3 ~ Exponential(mu_i) with mu_i = [1, 1.5, 2]
  X_4, X_5, X_6 ~ GeneralizedPareto(xi_i, sigma_i) with
                   xi_i = [0.25, 0.2, 0.2], sigma_i = [1, 2, 3]

Uses the inverse transform method to convert to standard Gaussian
Y-space via Y_i = Phi^{-1}(F_i(X_i)), then applies Deep-PrAE in Y-space.

Paper specifications:
- d = 6, n1 = 5,000, n2 = 495,000 (total n = 500,000)
- Architecture: 4 hidden layers (4, 8, 4, 2 nodes), ReLU
- Training: 1,000 iterations, L2 regularization
- Stage 1 sampling: Uniform on [-12, 12]^6 in Y-space
- Ground truth: ~1.23e-5 (via NMC with 100M samples)
"""

import numpy as np
from scipy.stats import multivariate_normal, norm, expon, genpareto
from typing import Dict, Optional

from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE4_CONFIG


class Example4_NonGaussian:
    """Example 4: Non-Gaussian rare-event estimation via inverse transform."""

    def __init__(self, gamma: float = 140.0):
        """
        Initialize Example 4.

        Args:
            gamma: Threshold for sum of X components (in original X-space).
        """
        self.gamma = gamma
        self.config = EXAMPLE4_CONFIG
        self.dimension = self.config.dimension  # 6

        # Exponential marginal parameters (X_1, X_2, X_3)
        self.expo_rates = [1.0, 1.5, 2.0]  # mu_i

        # Generalized Pareto marginal parameters (X_4, X_5, X_6)
        self.gp_shapes = [0.25, 0.2, 0.2]   # xi_i
        self.gp_scales = [1.0, 2.0, 3.0]     # sigma_i

    def transform_y_to_x(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from standard Gaussian Y-space to original X-space.

        Y_i -> U_i = Phi(Y_i) -> X_i = F_i^{-1}(U_i)

        Args:
            Y: Samples in Y-space, shape [N, 6]

        Returns:
            Samples in X-space, shape [N, 6]
        """
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        N = Y.shape[0]
        X = np.zeros_like(Y)

        # Y -> U via standard normal CDF
        U = norm.cdf(Y)
        # Clip to avoid numerical issues at boundaries
        U = np.clip(U, 1e-10, 1 - 1e-10)

        # Exponential marginals (X_1, X_2, X_3)
        for i, rate in enumerate(self.expo_rates):
            X[:, i] = expon.ppf(U[:, i], scale=1.0 / rate)

        # Generalized Pareto marginals (X_4, X_5, X_6)
        for i, (shape, scale) in enumerate(zip(self.gp_shapes, self.gp_scales)):
            X[:, i + 3] = genpareto.ppf(U[:, i + 3], c=shape, scale=scale)

        return X

    def transform_x_to_y(self, X: np.ndarray) -> np.ndarray:
        """
        Transform from original X-space to standard Gaussian Y-space.

        X_i -> U_i = F_i(X_i) -> Y_i = Phi^{-1}(U_i)

        Args:
            X: Samples in X-space, shape [N, 6]

        Returns:
            Samples in Y-space, shape [N, 6]
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        N = X.shape[0]
        U = np.zeros_like(X)

        # Exponential marginals
        for i, rate in enumerate(self.expo_rates):
            U[:, i] = expon.cdf(X[:, i], scale=1.0 / rate)

        # Generalized Pareto marginals
        for i, (shape, scale) in enumerate(zip(self.gp_shapes, self.gp_scales)):
            U[:, i + 3] = genpareto.cdf(X[:, i + 3], c=shape, scale=scale)

        U = np.clip(U, 1e-10, 1 - 1e-10)
        Y = norm.ppf(U)

        return Y

    def indicator_function_y(self, Y: np.ndarray) -> np.ndarray:
        """
        Indicator function in Y-space: transform to X, check sum >= gamma.

        Args:
            Y: Samples in Y-space, shape [N, 6] or [6,]

        Returns:
            Binary indicators, shape [N,]
        """
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        X = self.transform_y_to_x(Y)
        sums = np.sum(X, axis=1)
        return (sums >= self.gamma).astype(float)

    def original_pdf_y(self, Y: np.ndarray) -> np.ndarray:
        """
        PDF of the original distribution in Y-space (standard Gaussian).

        Args:
            Y: Samples in Y-space, shape [N, 6] or [6,]

        Returns:
            PDF values, shape [N,]
        """
        return multivariate_normal.pdf(
            Y, mean=np.zeros(self.dimension), cov=np.eye(self.dimension)
        )

    def generate_stage1_samples(self, n1: int) -> tuple:
        """
        Generate Stage 1 samples: uniform on [-2d, 2d]^d in Y-space.

        Per paper: uniform on [-12, 12]^6.

        Args:
            n1: Number of Stage 1 samples.

        Returns:
            Tuple of (Y_stage1, labels)
        """
        bound = 5.0  # Tighter range for heavy-tailed GPD marginals
        Y_stage1 = np.random.uniform(-bound, bound, size=(n1, self.dimension))
        labels = self.indicator_function_y(Y_stage1)
        return Y_stage1, labels

    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False
    ) -> Dict:
        """
        Run Deep-PrAE for Example 4.

        Args:
            n1: Stage 1 samples (default: 5000)
            n2: Stage 2 samples (default: 495000)
            verbose: Print progress
            _test_mode: Return dummy results for testing

        Returns:
            Dictionary with results
        """
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2

        if _test_mode:
            from ..utils.dummy_results import generate_dummy_results_example4
            return generate_dummy_results_example4(
                gamma=self.gamma, n1=n1, n2=n2
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Example 4: Non-Gaussian Distribution")
            print(f"{'='*70}")
            print(f"Gamma: {self.gamma}")
            print(f"Dimension: {self.dimension}")
            print(f"Marginals: Expo({self.expo_rates}) + GenPareto({self.gp_shapes}, {self.gp_scales})")
            print(f"Stage 1 (n1): {n1}, Stage 2 (n2): {n2}")
            print(f"{'='*70}\n")

        # Generate Stage 1 samples in Y-space
        Y_stage1, labels = self.generate_stage1_samples(n1)

        if verbose:
            num_rare = int(labels.sum())
            print(f"Rare events in Stage 1: {num_rare}/{n1} ({100*num_rare/n1:.2f}%)")

        # Run Deep-PrAE in Y-space (standard Gaussian)
        deep_prae = DeepPrAE(
            indicator_function=self.indicator_function_y,
            original_pdf=self.original_pdf_y,
            dimension=self.dimension,
            mu=np.zeros(self.dimension),
            sigma=1.0
        )

        results = deep_prae.run(
            X_stage1=Y_stage1,
            Y_stage1=labels,
            n2=n2,
            hidden_dims=self.config.hidden_dims,
            n_iters=self.config.n_iters,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            class_weights=self.config.class_weights,
            l2_reg=self.config.l2_reg,
            use_true_indicator=True,
            verbose=verbose
        )

        results['experiment'] = 'Example 4: Non-Gaussian Distribution'
        results['gamma'] = self.gamma
        results['n1'] = n1
        results['n2'] = n2

        return results

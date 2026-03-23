"""
Example 3: Random Walk Excursion

Estimates P(max_{t=1,...,T} S_t > gamma) where S_t = sum_{i=1}^t X_i.

Paper specifications:
- T=10, gamma=11, n=30,000 (n1 varies, n2=n-n1)
- Architecture: 4 hidden layers (8, 8, 4, 2 nodes)
- Training: 1,000 iterations, batch_size=n1/20, L2 regularization, SGD
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Optional
from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE3_CONFIG

class Example3_RandomWalk:
    """Example 3: Random walk excursion probability."""
    
    def __init__(self, T: int = 10, gamma: float = 11.0):
        self.T = T
        self.gamma = gamma
        self.config = EXAMPLE3_CONFIG
        
    def indicator_function(self, x: np.ndarray) -> np.ndarray:
        """I(max_t S_t > gamma) where S_t = cumulative sum."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        cumsum = np.cumsum(x, axis=1)
        max_cumsum = np.max(cumsum, axis=1)
        return (max_cumsum > self.gamma).astype(float)
    
    def original_pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF of N(0, I_T)."""
        return multivariate_normal.pdf(x, mean=np.zeros(self.T), cov=np.eye(self.T))
    
    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False
    ) -> Dict:
        """Run Deep-PrAE for Example 3."""
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2

        if _test_mode:
            from ..utils.dummy_results import generate_dummy_results_example3
            return generate_dummy_results_example3(
                T=self.T, gamma=self.gamma, n1=n1, n2=n2
            )

        # Stage 1: uniform sampling on [-4, 4]^T (fixed — paper's [0, 2*gamma] gives 100% rare)
        X_stage1 = np.random.uniform(-4, 4, size=(n1, self.T))
        Y_stage1 = self.indicator_function(X_stage1)

        deep_prae = DeepPrAE(
            indicator_function=self.indicator_function,
            original_pdf=self.original_pdf,
            dimension=self.T,
            mu=np.zeros(self.T),
            sigma=1.0
        )

        results = deep_prae.run(
            X_stage1=X_stage1, Y_stage1=Y_stage1, n2=n2,
            hidden_dims=self.config.hidden_dims,
            n_iters=self.config.n_iters,
            l2_reg=self.config.l2_reg,
            class_weights=self.config.class_weights,
            use_true_indicator=True,
            verbose=verbose
        )

        results['experiment'] = 'Example 3: Random Walk'
        results['gamma'] = self.gamma
        results['n1'] = n1
        results['n2'] = n2
        return results

"""
Main Deep-PrAE algorithm implementation.

This module implements the complete two-stage Deep-PrAE algorithm
for rare-event probability estimation with theoretical guarantees.
"""

import numpy as np
from typing import Callable, Dict, Optional, Tuple
import time

from .networks import train_classifier, tune_threshold
from .optimization import DominatingPointSolver
from .sampling import ProposalDistribution
from .estimation import ImportanceSamplingEstimator


class DeepPrAE:
    """
    Deep-PrAE: Deep Probabilistic Rare Event Estimation.

    Two-stage algorithm:
    1. Stage 1: Train neural network classifier and find dominating points
    2. Stage 2: Perform importance sampling using proposal distribution
    """

    def __init__(
        self,
        indicator_function: Callable,
        original_pdf: Callable,
        dimension: int,
        mu: Optional[np.ndarray] = None,
        sigma: float = 1.0
    ):
        """
        Initialize Deep-PrAE algorithm.

        Args:
            indicator_function: True indicator function for rare-event set
            original_pdf: PDF of original distribution
            dimension: Dimension of input space
            mu: Mean of original distribution (defaults to zeros)
            sigma: Standard deviation of original distribution
        """
        self.indicator_function = indicator_function
        self.original_pdf = original_pdf
        self.dimension = dimension
        self.mu = mu if mu is not None else np.zeros(dimension)
        self.sigma = sigma

        self.classifier = None
        self.dominating_points = None
        self.proposal_dist = None

        self.stage1_time = 0
        self.stage2_time = 0

    def stage1(
        self,
        X_stage1: np.ndarray,
        Y_stage1: np.ndarray,
        hidden_dims: list,
        n_iters: int = 1000,
        batch_size: Optional[int] = None,
        lr: float = 5e-3,
        class_weights: list = [1.0, 50.0],
        l2_reg: float = 0.0,
        solver: str = 'gurobi',
        max_dominating_points: int = 100,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Stage 1: Train classifier and find dominating points.

        Args:
            X_stage1: Stage 1 samples, shape [n1, d]
            Y_stage1: Stage 1 labels, shape [n1,]
            hidden_dims: List of hidden layer dimensions
            n_iters: Number of training iterations
            batch_size: Batch size (defaults to n1/20)
            lr: Learning rate
            class_weights: Class weights [non-rare, rare]
            l2_reg: L2 regularization coefficient
            solver: Optimization solver name
            max_dominating_points: Maximum number of dominating points
            verbose: Whether to print progress

        Returns:
            Tuple of (dominating_points, stage1_info)
        """
        stage1_start = time.time()

        if verbose:
            print("=" * 60)
            print("Stage 1: Neural Network Training & Dominating Point Search")
            print("=" * 60)

        # Train neural network classifier
        if verbose:
            print("\n[1/2] Training neural network classifier...")
            print(f"  Architecture: {self.dimension} -> {hidden_dims} -> 2")
            print(f"  Training samples: {len(X_stage1)}")
            print(f"  Rare events: {int(Y_stage1.sum())}/{len(Y_stage1)}")

        self.classifier, history = train_classifier(
            X_train=X_stage1,
            Y_train=Y_stage1,
            hidden_dims=hidden_dims,
            n_iters=n_iters,
            batch_size=batch_size,
            lr=lr,
            class_weights=class_weights,
            l2_reg=l2_reg,
            log=verbose
        )

        if verbose:
            print(f"  Training completed in {time.time() - stage1_start:.2f}s")

        # Extract network parameters
        network_params = self.classifier.extract_params()

        # Find dominating points
        if verbose:
            print(f"\n[2/2] Solving for dominating points...")
            print(f"  Solver: {solver}")
            print(f"  Maximum iterations: {max_dominating_points}")

        solver_start = time.time()

        dp_solver = DominatingPointSolver(
            input_dim=self.dimension,
            hidden_dims=hidden_dims,
            mu=self.mu,
            sigma=self.sigma,
            solver_name=solver,
            max_iterations=max_dominating_points
        )

        self.dominating_points = dp_solver.solve(
            network_params=network_params,
            verbose=verbose
        )

        if verbose:
            print(f"  Solver completed in {time.time() - solver_start:.2f}s")
            print(f"  Found {len(self.dominating_points)} dominating points")

        # Check if optimization found any points
        if self.dominating_points is None or len(self.dominating_points) == 0:
            raise RuntimeError(
                "Gurobi optimization failed to find any dominating points. "
                "Possible causes: 1) Solver timeout (increase TimeLimit), "
                "2) Infeasible model, 3) License issue. "
                "Try: increasing training epochs, adjusting class weights, or checking Gurobi license."
            )

        self.stage1_time = time.time() - stage1_start

        stage1_info = {
            'num_dominating_points': len(self.dominating_points),
            'training_history': history,
            'stage1_time': self.stage1_time
        }

        return self.dominating_points, stage1_info

    def stage2(
        self,
        n2: int,
        use_true_indicator: bool = False,
        verbose: bool = False
    ) -> Dict:
        """
        Stage 2: Importance sampling using proposal distribution.

        Args:
            n2: Number of Stage 2 samples
            use_true_indicator: If True, use the true indicator function (Deep-PrAE Mod)
                               instead of the classifier indicator (Deep-PrAE UB)
            verbose: Whether to print progress

        Returns:
            Dictionary with probability estimates and statistics
        """
        if self.dominating_points is None:
            raise ValueError("Must run stage1 first to find dominating points")

        stage2_start = time.time()

        mode_name = "Mod (true indicator)" if use_true_indicator else "UB (classifier)"
        if verbose:
            print("\n" + "=" * 60)
            print(f"Stage 2: Importance Sampling [{mode_name}]")
            print("=" * 60)

        # Construct proposal distribution
        if verbose:
            print(f"\n[1/3] Constructing proposal distribution...")
            print(f"  Number of components: {len(self.dominating_points)}")

        self.proposal_dist = ProposalDistribution(
            dominating_points=self.dominating_points,
            sigma=self.sigma
        )

        # Sample from proposal distribution
        if verbose:
            print(f"\n[2/3] Sampling from proposal...")
            print(f"  Number of samples: {n2}")

        X_stage2 = self.proposal_dist.sample(n2)

        # Choose indicator function for IS
        if use_true_indicator:
            is_indicator = self.indicator_function
        else:
            import torch
            def classifier_indicator(x):
                """Indicator based on classifier predictions."""
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                with torch.no_grad():
                    x_tensor = torch.Tensor(x)
                    outputs = self.classifier(x_tensor)
                    return (outputs[:, 1] >= outputs[:, 0]).cpu().numpy().astype(float)
            is_indicator = classifier_indicator

        # Compute IS estimate
        if verbose:
            print(f"\n[3/3] Computing importance sampling estimate...")

        estimator = ImportanceSamplingEstimator(
            original_pdf=self.original_pdf,
            proposal_pdf=self.proposal_dist.pdf,
            indicator_function=is_indicator
        )

        results = estimator.estimate(X_stage2, return_details=True)

        self.stage2_time = time.time() - stage2_start
        results['stage2_time'] = self.stage2_time
        results['mode'] = 'mod' if use_true_indicator else 'ub'

        if verbose:
            print(f"\n  Estimated probability: {results['probability']:.6e}")
            print(f"  Relative error: {results['relative_error']:.4f}")
            print(f"  Stage 2 completed in {self.stage2_time:.2f}s")

        return results

    def run(
        self,
        X_stage1: np.ndarray,
        Y_stage1: np.ndarray,
        n2: int,
        hidden_dims: list,
        n_iters: int = 1000,
        batch_size: Optional[int] = None,
        lr: float = 5e-3,
        class_weights: list = [1.0, 50.0],
        l2_reg: float = 0.0,
        solver: str = 'gurobi',
        max_dominating_points: int = 100,
        use_true_indicator: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete Deep-PrAE algorithm (both stages).

        Args:
            X_stage1: Stage 1 samples
            Y_stage1: Stage 1 labels
            n2: Number of Stage 2 samples
            hidden_dims: Neural network architecture
            n_iters: Training iterations
            batch_size: Batch size
            lr: Learning rate
            class_weights: Class weights
            l2_reg: L2 regularization
            solver: Optimization solver
            max_dominating_points: Max dominating points
            use_true_indicator: If True, use true indicator in Stage 2 (Mod mode)
            verbose: Print progress

        Returns:
            Dictionary with all results
        """
        # Stage 1
        dominating_points, stage1_info = self.stage1(
            X_stage1=X_stage1,
            Y_stage1=Y_stage1,
            hidden_dims=hidden_dims,
            n_iters=n_iters,
            batch_size=batch_size,
            lr=lr,
            class_weights=class_weights,
            l2_reg=l2_reg,
            solver=solver,
            max_dominating_points=max_dominating_points,
            verbose=verbose
        )

        # Stage 2
        stage2_results = self.stage2(
            n2=n2,
            use_true_indicator=use_true_indicator,
            verbose=verbose
        )

        # Combine results
        results = {
            **stage1_info,
            **stage2_results,
            'total_time': self.stage1_time + self.stage2_time,
            'total_samples': len(X_stage1) + n2
        }

        if verbose:
            mode_name = "Mod" if use_true_indicator else "UB"
            print("\n" + "=" * 60)
            print(f"Deep-PrAE Complete [{mode_name}]")
            print("=" * 60)
            print(f"Total samples: {results['total_samples']}")
            print(f"Total time: {results['total_time']:.2f}s")
            print(f"Estimated probability: {results['probability']:.6e}")
            print(f"Relative error: {results['relative_error']:.4f}")
            print("=" * 60)

        return results

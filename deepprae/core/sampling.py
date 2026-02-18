"""
Proposal distributions for importance sampling.

This module implements Gaussian mixture model (GMM) proposal distributions
centered at dominating points for efficient importance sampling.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from typing import Optional


class ProposalDistribution:
    """
    Gaussian mixture proposal distribution for importance sampling.

    Constructs a mixture of Gaussians centered at dominating points,
    with diagonal covariance matrices.
    """

    def __init__(
        self,
        dominating_points: np.ndarray,
        sigma: float = 1.0,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize proposal distribution.

        Args:
            dominating_points: Array of shape [num_components, dimension]
            sigma: Standard deviation (same for all dimensions and components)
            weights: Optional mixing weights (defaults to uniform)
        """
        if dominating_points.ndim == 1:
            dominating_points = dominating_points.reshape(1, -1)

        self.mu = dominating_points
        self.sigma = sigma
        self.num_components = dominating_points.shape[0]
        self.dimension = dominating_points.shape[1]

        # Set mixing weights
        if weights is None:
            self.weights = np.ones(self.num_components) / self.num_components
        else:
            assert len(weights) == self.num_components
            self.weights = weights / weights.sum()  # Normalize

        self._build_model()

    def _build_model(self):
        """Build internal GMM model."""
        if self.num_components > 1:
            # Use sklearn GMM for multiple components
            self.gmm = GaussianMixture(
                n_components=self.num_components,
                covariance_type='diag'
            )
            # Manually set all required parameters (no fit call needed)
            self.gmm.means_ = self.mu
            self.gmm.covariances_ = np.ones((self.num_components, self.dimension)) * (self.sigma ** 2)
            self.gmm.weights_ = self.weights
            self.gmm.precisions_cholesky_ = np.ones((self.num_components, self.dimension)) / self.sigma
            # Mark as converged so sample() works without fit()
            self.gmm.converged_ = True
            self.gmm.n_iter_ = 0
            self.gmm.lower_bound_ = 0.0
        else:
            # Single Gaussian
            self.gmm = multivariate_normal(
                mean=self.mu[0],
                cov=np.eye(self.dimension) * (self.sigma ** 2)
            )

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probability density function.

        Args:
            X: Array of shape [num_samples, dimension]

        Returns:
            Array of densities of shape [num_samples,]
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        num_samples = X.shape[0]

        if self.num_components > 1:
            # Sum over mixture components
            pdf_values = np.zeros(num_samples)

            for k in range(self.num_components):
                component_pdf = multivariate_normal.pdf(
                    X,
                    mean=self.mu[k],
                    cov=np.eye(self.dimension) * (self.sigma ** 2)
                )
                pdf_values += self.weights[k] * component_pdf

            return pdf_values
        else:
            return self.gmm.pdf(X)

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log probability density function.

        Args:
            X: Array of shape [num_samples, dimension]

        Returns:
            Array of log densities of shape [num_samples,]
        """
        return np.log(self.pdf(X) + 1e-300)  # Add small constant to avoid log(0)

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Draw samples from the proposal distribution.

        Args:
            num_samples: Number of samples to draw

        Returns:
            Array of samples of shape [num_samples, dimension]
        """
        if self.num_components > 1:
            samples, _ = self.gmm.sample(num_samples)
            np.random.shuffle(samples)  # Shuffle to mix components
            return samples
        else:
            return self.gmm.rvs(size=num_samples)

    def get_parameters(self) -> dict:
        """
        Get distribution parameters.

        Returns:
            Dictionary with keys 'means', 'sigma', 'weights'
        """
        return {
            'means': self.mu,
            'sigma': self.sigma,
            'weights': self.weights,
            'num_components': self.num_components
        }

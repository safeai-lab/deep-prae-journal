"""
Importance sampling estimator with theoretical guarantees.

This module implements the IS estimator for computing rare-event
probability estimates with relative error bounds.
"""

import numpy as np
from typing import Callable, Dict, Optional
from scipy.stats import multivariate_normal


class ImportanceSamplingEstimator:
    """
    Importance sampling estimator for rare-event probabilities.

    Computes probability estimates using likelihood ratio weighting,
    along with empirical relative error estimates.
    """

    def __init__(
        self,
        original_pdf: Callable,
        proposal_pdf: Callable,
        indicator_function: Callable
    ):
        """
        Initialize IS estimator.

        Args:
            original_pdf: PDF of original distribution p(x)
            proposal_pdf: PDF of proposal distribution q(x)
            indicator_function: Indicator function for rare-event set
        """
        self.original_pdf = original_pdf
        self.proposal_pdf = proposal_pdf
        self.indicator_function = indicator_function

    def estimate(
        self,
        samples: np.ndarray,
        return_details: bool = False
    ) -> Dict:
        """
        Compute IS estimate of rare-event probability.

        Args:
            samples: Samples from proposal distribution, shape [n, d]
            return_details: Whether to return detailed statistics

        Returns:
            Dictionary containing:
                - 'probability': Estimated probability
                - 'relative_error': Empirical relative error
                - 'num_samples': Number of samples used
                - 'weights': Importance weights (if return_details=True)
                - 'indicators': Event indicators (if return_details=True)
        """
        num_samples = len(samples)

        # Evaluate indicator function
        indicators = self.indicator_function(samples)

        # Compute likelihood ratios
        p_vals = self.original_pdf(samples)
        q_vals = self.proposal_pdf(samples)

        # Importance weights
        weights = p_vals / (q_vals + 1e-300)  # Avoid division by zero

        # Weighted indicators
        weighted_indicators = weights * indicators

        # Probability estimate
        prob_estimate = np.mean(weighted_indicators)

        # Empirical relative error
        if prob_estimate > 0:
            std_estimate = np.std(weighted_indicators)
            relative_error = std_estimate / (np.sqrt(num_samples) * prob_estimate)
        else:
            relative_error = np.inf

        results = {
            'probability': prob_estimate,
            'relative_error': relative_error,
            'std': np.std(weighted_indicators),
            'num_samples': num_samples
        }

        if return_details:
            results['weights'] = weights
            results['indicators'] = indicators
            results['weighted_indicators'] = weighted_indicators

        return results

    @staticmethod
    def compute_cv_estimate(
        samples: np.ndarray,
        indicators: np.ndarray,
        weights: np.ndarray,
        control_variate: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute control variate estimate (optional variance reduction).

        Args:
            samples: Samples from proposal distribution
            indicators: Event indicators
            weights: Importance weights
            control_variate: Optional control variate function values

        Returns:
            Dictionary with CV estimate and relative error
        """
        if control_variate is None:
            control_variate = np.linalg.norm(samples, axis=1)

        weighted_indicators = weights * indicators

        # Compute optimal coefficient
        cov = np.cov(weighted_indicators, control_variate)[0, 1]
        var_cv = np.var(control_variate)
        alpha_opt = cov / var_cv if var_cv > 0 else 0

        # Apply control variate
        mean_cv = np.mean(control_variate)
        adjusted = weighted_indicators - alpha_opt * (control_variate - mean_cv)

        # Estimate
        prob_estimate = np.mean(adjusted)
        std_estimate = np.std(adjusted)
        relative_error = std_estimate / (np.sqrt(len(samples)) * prob_estimate) if prob_estimate > 0 else np.inf

        return {
            'probability': prob_estimate,
            'relative_error': relative_error,
            'alpha': alpha_opt,
            'std': std_estimate
        }


class NaiveMonteCarlo:
    """Naive Monte Carlo estimator for comparison."""

    def __init__(self, indicator_function: Callable):
        """
        Initialize NMC estimator.

        Args:
            indicator_function: Indicator function for rare-event set
        """
        self.indicator_function = indicator_function

    def estimate(self, samples: np.ndarray) -> Dict:
        """
        Compute naive MC estimate.

        Args:
            samples: Samples from original distribution

        Returns:
            Dictionary with probability estimate and relative error
        """
        num_samples = len(samples)

        # Evaluate indicator
        indicators = self.indicator_function(samples)

        # Probability estimate
        prob_estimate = np.mean(indicators)

        # Relative error
        if prob_estimate > 0:
            std_estimate = np.sqrt(prob_estimate * (1 - prob_estimate))
            relative_error = std_estimate / (np.sqrt(num_samples) * prob_estimate)
        else:
            relative_error = np.inf

        return {
            'probability': prob_estimate,
            'relative_error': relative_error,
            'num_samples': num_samples,
            'num_hits': int(indicators.sum())
        }

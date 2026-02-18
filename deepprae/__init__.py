"""
Deep-PrAE: Deep Probabilistic Rare Event Estimation

A neural network-based importance sampling framework for estimating
extremely small probabilities of rare events.
"""

from .core.networks import NeuralNetworkClassifier, train_classifier
from .core.optimization import DominatingPointSolver
from .core.sampling import ProposalDistribution
from .core.estimation import ImportanceSamplingEstimator
from .core.algorithm import DeepPrAE

__version__ = "1.0.0"

__all__ = [
    "NeuralNetworkClassifier",
    "train_classifier",
    "DominatingPointSolver",
    "ProposalDistribution",
    "ImportanceSamplingEstimator",
    "DeepPrAE",
]

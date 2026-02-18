"""
Experiment modules for Deep-PrAE paper replication.

This package contains implementations of the 6 experiments from the paper:
1. Example 1: 2D Sigmoid Functions
2. Example 2: Complement of a 5D Ball
3. Example 3: Random Walk Excursion
4. Example 4: Non-Gaussian (Exponential + Generalized Pareto)
5. Example 5: Rare-event Set with Hole
6. Example 6: Intelligent Driving (IDM Car-Following)
"""

from .example1_2d_sigmoid import Example1_2DSigmoid
from .example2_ball_complement import Example2_BallComplement
from .example3_random_walk import Example3_RandomWalk
from .example4_non_gaussian import Example4_NonGaussian
from .example5_hole import Example5_Hole
from .example6_intelligent_driving import Example6_IntelligentDriving

__all__ = [
    "Example1_2DSigmoid",
    "Example2_BallComplement",
    "Example3_RandomWalk",
    "Example4_NonGaussian",
    "Example5_Hole",
    "Example6_IntelligentDriving",
]

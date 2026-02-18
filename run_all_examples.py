#!/usr/bin/env python3
"""
Unified script to run all Deep-PrAE examples.

Usage:
    python run_all_examples.py --all
    python run_all_examples.py --examples 1 2 3
    python run_all_examples.py --examples 1 --gamma 1.8
    python run_all_examples.py --all --test              # dummy results
    python run_all_examples.py --all --test --figures     # dummy + figures
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def run_example1(
    gamma: float = 1.8,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    visualize: bool = False,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 1: 2D Sigmoid Functions."""
    from deepprae.experiments.example1_2d_sigmoid import Example1_2DSigmoid

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 1: 2D Sigmoid Functions")
        print("="*70)

    example = Example1_2DSigmoid(gamma=gamma)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    if visualize and not test_mode:
        example.visualize_results(results)

    return results


def run_example2(
    gamma: float = 4.75,
    hidden_dim: int = 10,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 2: Complement of 5D Ball."""
    from deepprae.experiments.example2_ball_complement import Example2_BallComplement

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 2: Complement of 5D Ball")
        print("="*70)

    example = Example2_BallComplement(gamma=gamma, hidden_dim=hidden_dim)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    return results


def run_example3(
    T: int = 10,
    gamma: float = 11.0,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 3: Random Walk Excursion."""
    from deepprae.experiments.example3_random_walk import Example3_RandomWalk

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 3: Random Walk Excursion")
        print("="*70)

    example = Example3_RandomWalk(T=T, gamma=gamma)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    return results


def run_example4(
    gamma: float = 20.0,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 4: Non-Gaussian (Exponential + Generalized Pareto)."""
    from deepprae.experiments.example4_non_gaussian import Example4_NonGaussian

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 4: Non-Gaussian Distribution")
        print("="*70)

    example = Example4_NonGaussian(gamma=gamma)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    return results


def run_example5(
    gamma: float = 1.0,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 5: Rare-event Set with Hole."""
    from deepprae.experiments.example5_hole import Example5_Hole

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 5: Rare-event Set with Hole")
        print("="*70)

    example = Example5_Hole(gamma=gamma)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    return results


def run_example6(
    gamma: float = 1.0,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    verbose: bool = True,
    test_mode: bool = False
) -> Dict:
    """Run Example 6: Intelligent Driving (IDM Car-Following)."""
    from deepprae.experiments.example6_intelligent_driving import Example6_IntelligentDriving

    if verbose:
        print("\n" + "="*70)
        print("EXAMPLE 6: Intelligent Driving Safety")
        print("="*70)

    example = Example6_IntelligentDriving(gamma=gamma)
    results = example.run(n1=n1, n2=n2, verbose=verbose, _test_mode=test_mode)

    return results


EXAMPLE_RUNNERS = {
    1: run_example1,
    2: run_example2,
    3: run_example3,
    4: run_example4,
    5: run_example5,
    6: run_example6,
}


EXAMPLE_DESCRIPTIONS = {
    1: "2D Sigmoid Functions (ultra-rare events ~10^-24)",
    2: "Complement of 5D Ball (infinite dominating points)",
    3: "Random Walk Excursion (classical benchmark)",
    4: "Non-Gaussian Distribution (exponential + gen. Pareto)",
    5: "Rare-event Set with Hole (special preprocessing)",
    6: "Intelligent Driving Safety (autonomous vehicle crashes)",
}


def save_results(results: Dict, output_dir: Path, example_num: int):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"example{example_num}_results_{timestamp}.json"
    filepath = output_dir / filename

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_serializable = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def generate_figures(all_results: Dict, output_dir: Path):
    """Generate paper-like figures from results."""
    from deepprae.utils.plotting import (
        plot_probability_vs_gamma,
        plot_re_vs_gamma,
        plot_2d_rare_event_set,
    )

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for example_num, results in all_results.items():
        if isinstance(results, dict) and 'probability' in results:
            print(f"  Generating figures for Example {example_num}...")

            # Multi-gamma results (if available)
            if 'gamma_sweep' in results:
                gammas = [r['gamma'] for r in results['gamma_sweep']]
                probs = [r['probability'] for r in results['gamma_sweep']]
                res_re = [r['relative_error'] for r in results['gamma_sweep']]

                plot_probability_vs_gamma(
                    gammas, {'Deep-PrAE UB': probs},
                    title=f"Example {example_num}: Probability vs Gamma",
                    save_path=str(fig_dir / f"example{example_num}_prob_vs_gamma.png")
                )
                plot_re_vs_gamma(
                    gammas, {'Deep-PrAE UB': res_re},
                    title=f"Example {example_num}: RE vs Gamma",
                    save_path=str(fig_dir / f"example{example_num}_re_vs_gamma.png")
                )

            # 2D visualization (Examples 1, 5)
            if example_num in [1, 5] and 'dominating_points' in results:
                dp = results.get('dominating_points')
                if dp is not None:
                    dp = np.array(dp) if not isinstance(dp, np.ndarray) else dp
                    plot_2d_rare_event_set(
                        results,
                        title=f"Example {example_num}",
                        save_path=str(fig_dir / f"example{example_num}_set.png")
                    )

    print(f"Figures saved to: {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep-PrAE examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run all examples
  %(prog)s --examples 1 2 3         # Run examples 1, 2, and 3
  %(prog)s --examples 1 --gamma 1.8 # Run example 1 with gamma=1.8
  %(prog)s --all --test             # Quick dummy run for pipeline testing
  %(prog)s --all --test --figures   # Dummy run + generate figures
        """
    )

    parser.add_argument('--all', action='store_true', help='Run all six examples')
    parser.add_argument('--examples', type=int, nargs='+', choices=[1, 2, 3, 4, 5, 6],
                        help='Specific example numbers to run (1-6)')
    parser.add_argument('--gamma', type=float, help='Gamma parameter (rarity level)')
    parser.add_argument('--n1', type=int, help='Number of Stage 1 samples')
    parser.add_argument('--n2', type=int, help='Number of Stage 2 samples')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--figures', action='store_true', help='Generate paper-like figures')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument('--test', action='store_true', help='Run in test mode (dummy output)')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Deep-PrAE Examples:")
        print("="*70)
        for num, desc in EXAMPLE_DESCRIPTIONS.items():
            print(f"  {num}. {desc}")
        print("="*70)
        return 0

    if args.all:
        examples_to_run = list(range(1, 7))
    elif args.examples:
        examples_to_run = args.examples
    else:
        parser.print_help()
        return 1

    output_dir = Path(args.output_dir)
    verbose = not args.quiet
    test_mode = args.test

    if verbose:
        print("\n" + "="*70)
        print("Deep-PrAE: Deep Probabilistic Rare Event Estimation")
        print("="*70)
        print(f"Running examples: {examples_to_run}")
        if test_mode:
            print("Mode: TEST (dummy results)")
        print(f"Output directory: {output_dir}")
        print("="*70)

    all_results = {}

    for example_num in examples_to_run:
        try:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Starting Example {example_num}: {EXAMPLE_DESCRIPTIONS[example_num]}")
                print(f"{'='*70}")

            kwargs = {
                'verbose': verbose,
                'n1': args.n1,
                'n2': args.n2,
                'test_mode': test_mode,
            }

            if args.gamma:
                kwargs['gamma'] = args.gamma

            if example_num == 1:
                kwargs['visualize'] = args.visualize

            results = EXAMPLE_RUNNERS[example_num](**kwargs)
            all_results[example_num] = results

            save_results(results, output_dir, example_num)

            if verbose and 'probability' in results:
                print(f"\n{'='*70}")
                print(f"Example {example_num} Summary:")
                print(f"  Estimated Probability: {results['probability']:.6e}")
                if 'relative_error' in results:
                    print(f"  Relative Error: {results['relative_error']:.4f}")
                if 'n_dominating_points' in results:
                    print(f"  Dominating Points: {results['n_dominating_points']}")
                elif 'num_dominating_points' in results:
                    print(f"  Dominating Points: {results['num_dominating_points']}")
                print(f"{'='*70}")

        except Exception as e:
            print(f"\nERROR running Example {example_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate figures if requested
    if args.figures and all_results:
        if verbose:
            print("\n" + "="*70)
            print("Generating Figures")
            print("="*70)
        try:
            generate_figures(all_results, output_dir)
        except Exception as e:
            print(f"Warning: Figure generation failed: {e}")

    if verbose:
        print("\n" + "="*70)
        print("All Examples Complete!")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        print(f"Successfully completed: {len(all_results)}/{len(examples_to_run)} examples")
        print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

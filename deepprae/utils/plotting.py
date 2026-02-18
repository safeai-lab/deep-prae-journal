"""
Paper-like plotting utilities for Deep-PrAE results.

Generates figures matching the style of the paper:
  - Log-scale probability vs gamma
  - Relative error vs gamma
  - 2D rare-event set contour plots with outer approximation
  - Probability/RE convergence with sample size
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from pathlib import Path


# ── Style defaults ──────────────────────────────────────────────────────

_STYLE = dict(
    figsize=(7, 5),
    dpi=150,
    linewidth=2,
    markersize=7,
    fontsize=12,
    legend_fontsize=10,
    grid_alpha=0.3,
)

_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def _apply_style(ax, xlabel: str, ylabel: str, title: str = ''):
    ax.set_xlabel(xlabel, fontsize=_STYLE['fontsize'])
    ax.set_ylabel(ylabel, fontsize=_STYLE['fontsize'])
    if title:
        ax.set_title(title, fontsize=_STYLE['fontsize'] + 1)
    ax.tick_params(labelsize=_STYLE['fontsize'] - 1)
    ax.grid(True, alpha=_STYLE['grid_alpha'])
    ax.legend(fontsize=_STYLE['legend_fontsize'])


def _save_or_show(fig, save_path: Optional[str] = None):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_STYLE['dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ── Probability vs Gamma ────────────────────────────────────────────────

def plot_probability_vs_gamma(
    gammas: List[float],
    methods: Dict[str, List[float]],
    true_probs: Optional[List[float]] = None,
    title: str = 'Probability Estimate vs Gamma',
    save_path: Optional[str] = None,
):
    """
    Log-scale probability estimates vs gamma (paper Figs 3a, 7 left, etc.).

    Args:
        gammas: List of gamma values.
        methods: Dict mapping method name → list of probability estimates.
        true_probs: Optional list of true probabilities (plotted as dashed line).
        title: Plot title.
        save_path: If given, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=_STYLE['figsize'])

    for i, (name, probs) in enumerate(methods.items()):
        ax.semilogy(
            gammas, probs,
            marker=_MARKERS[i % len(_MARKERS)],
            color=_COLORS[i % len(_COLORS)],
            linewidth=_STYLE['linewidth'],
            markersize=_STYLE['markersize'],
            label=name,
        )

    if true_probs is not None:
        ax.semilogy(
            gammas, true_probs,
            'k--', linewidth=_STYLE['linewidth'],
            label='True Probability',
        )

    _apply_style(ax, r'$\gamma$', 'Probability', title)
    _save_or_show(fig, save_path)


# ── Relative Error vs Gamma ─────────────────────────────────────────────

def plot_re_vs_gamma(
    gammas: List[float],
    methods: Dict[str, List[float]],
    title: str = 'Relative Error vs Gamma',
    save_path: Optional[str] = None,
):
    """
    Relative error vs gamma (paper Figs 3b, 7 right).

    Args:
        gammas: List of gamma values.
        methods: Dict mapping method name → list of relative errors.
        title: Plot title.
        save_path: If given, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=_STYLE['figsize'])

    for i, (name, res) in enumerate(methods.items()):
        ax.plot(
            gammas, res,
            marker=_MARKERS[i % len(_MARKERS)],
            color=_COLORS[i % len(_COLORS)],
            linewidth=_STYLE['linewidth'],
            markersize=_STYLE['markersize'],
            label=name,
        )

    _apply_style(ax, r'$\gamma$', 'Relative Error (RE)', title)
    ax.set_ylim(bottom=0)
    _save_or_show(fig, save_path)


# ── Probability/RE vs Sample Size ────────────────────────────────────────

def plot_convergence(
    sample_sizes: List[int],
    methods: Dict[str, List[float]],
    ylabel: str = 'Probability',
    log_y: bool = True,
    title: str = 'Convergence with Sample Size',
    save_path: Optional[str] = None,
):
    """
    Convergence plot: probability or RE vs number of samples.

    Args:
        sample_sizes: List of total sample sizes.
        methods: Dict mapping method name → list of values.
        ylabel: Y-axis label.
        log_y: Whether to use log scale on Y axis.
        title: Plot title.
        save_path: If given, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=_STYLE['figsize'])

    for i, (name, vals) in enumerate(methods.items()):
        plot_fn = ax.semilogy if log_y else ax.plot
        plot_fn(
            sample_sizes, vals,
            marker=_MARKERS[i % len(_MARKERS)],
            color=_COLORS[i % len(_COLORS)],
            linewidth=_STYLE['linewidth'],
            markersize=_STYLE['markersize'],
            label=name,
        )

    _apply_style(ax, 'Number of Samples', ylabel, title)
    _save_or_show(fig, save_path)


# ── 2D Rare-Event Set Visualization ─────────────────────────────────────

def plot_2d_rare_event_set(
    results: Dict,
    indicator_fn=None,
    title: str = '2D Rare-Event Set',
    save_path: Optional[str] = None,
    grid_resolution: int = 200,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
):
    """
    Contour plot of 2D rare-event set with dominating points overlay.

    Works for Examples 1 and 5.

    Args:
        results: Results dict containing 'dominating_points'.
        indicator_fn: Optional callable(x) → {0,1} for contour.
        title: Plot title.
        save_path: If given, save figure.
        grid_resolution: Number of grid points per axis.
        xlim, ylim: Axis limits.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    dp = results.get('dominating_points')
    if dp is not None:
        dp = np.array(dp) if not isinstance(dp, np.ndarray) else dp

        # Auto-determine plot bounds from dominating points
        if xlim is None:
            margin = 2.0
            xlim = (dp[:, 0].min() - margin, dp[:, 0].max() + margin)
        if ylim is None:
            margin = 2.0
            ylim = (dp[:, 1].min() - margin, dp[:, 1].max() + margin)

        # Plot indicator contour if function provided
        if indicator_fn is not None:
            xx, yy = np.meshgrid(
                np.linspace(xlim[0], xlim[1], grid_resolution),
                np.linspace(ylim[0], ylim[1], grid_resolution),
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            zz = indicator_fn(grid).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.25)
            ax.contour(xx, yy, zz, levels=[0.5], colors=['#2ca02c'], linewidths=1.5)

        # Plot dominating points
        ax.scatter(
            dp[:, 0], dp[:, 1],
            c='red', marker='x', s=60, linewidths=2,
            label=f'Dominating Points ({len(dp)})', zorder=5,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _apply_style(ax, '$x_1$', '$x_2$', title)
    ax.set_aspect('equal')
    _save_or_show(fig, save_path)


# ── Summary Bar Chart ────────────────────────────────────────────────────

def plot_summary_bar(
    example_names: List[str],
    probabilities: List[float],
    relative_errors: Optional[List[float]] = None,
    title: str = 'Deep-PrAE Results Summary',
    save_path: Optional[str] = None,
):
    """
    Bar chart summarizing probability estimates across examples.

    Args:
        example_names: List of example labels.
        probabilities: List of estimated probabilities.
        relative_errors: Optional list of relative errors (shown as secondary axis).
        title: Plot title.
        save_path: If given, save figure.
    """
    n = len(example_names)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(n)
    log_probs = [np.log10(max(p, 1e-50)) for p in probabilities]

    bars = ax1.bar(x, log_probs, color=_COLORS[0], alpha=0.7, label='log10(P)')
    ax1.set_ylabel('log10(Probability)', fontsize=_STYLE['fontsize'], color=_COLORS[0])
    ax1.set_xticks(x)
    ax1.set_xticklabels(example_names, fontsize=_STYLE['fontsize'] - 2, rotation=15, ha='right')
    ax1.tick_params(axis='y', labelcolor=_COLORS[0])

    if relative_errors is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, relative_errors, 'o-', color=_COLORS[1], linewidth=2, markersize=8, label='RE')
        ax2.set_ylabel('Relative Error', fontsize=_STYLE['fontsize'], color=_COLORS[1])
        ax2.tick_params(axis='y', labelcolor=_COLORS[1])
        ax2.set_ylim(bottom=0)

    ax1.set_title(title, fontsize=_STYLE['fontsize'] + 1)
    ax1.grid(True, alpha=_STYLE['grid_alpha'], axis='y')

    fig.tight_layout()
    _save_or_show(fig, save_path)

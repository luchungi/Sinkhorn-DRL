"""Three-panel wealth, weights and drawdown figure."""
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK = {'surface': '#fcfcfb', 'secondary': '#52514e', 'muted': '#898781',
       'grid': '#e1e0d9', 'baseline': '#c3c2b7'}


def _style_axes(ax) -> None:
    """Apply the shared axis styling."""
    ax.set_facecolor(INK['surface'])
    ax.grid(True, color=INK['grid'], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK['baseline'])
    ax.tick_params(colors=INK['muted'], labelsize=9)
    ax.yaxis.label.set_color(INK['secondary'])


def overview(df: pd.DataFrame, wealth_col: List[str], weights_col: List[str],
             drawdown_col: List[str], path: Optional[str] = None,
             wealth_label: Optional[List[str]] = None):
    """Plot wealth, stacked portfolio weights and drawdown in three panels.

    Args:
        df: Time-indexed frame holding every named column.
        wealth_col: Wealth columns, one line each in the top panel.
        weights_col: Weight columns for the stacked middle panel.
        drawdown_col: Drawdown columns, one line each in the bottom panel.
        path: File to save the figure to; nothing is saved if None.
        wealth_label: Legend labels for wealth_col; defaults to the column names.

    Returns:
        Tuple (fig, ax) where ax is the array of the three axes.
    """
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), dpi=100, sharex=True,
                           constrained_layout=True, facecolor=INK['surface'])
    colors = COLORS[:len(wealth_col)]

    for i, (column, color) in enumerate(zip(wealth_col, colors)):
        ax[0].plot(df.index, df[column], color=color, linewidth=2,
                   label=str(column) if wealth_label is None else wealth_label[i])
    _style_axes(ax[0])
    ax[0].set_ylabel('Wealth')
    if len(wealth_col) >= 2:
        ax[0].legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK['secondary'])

    weights = df[weights_col]
    weight_colors = COLORS[:weights.shape[1]]
    for stack, labels in ((weights.clip(lower=0).to_numpy(dtype=float).T, list(weights.columns)),
                          (weights.clip(upper=0).to_numpy(dtype=float).T, [])):
        ax[1].stackplot(weights.index, stack, colors=weight_colors, labels=labels)
    ax[1].axhline(0, color=INK['baseline'], linewidth=0.8, zorder=4)
    _style_axes(ax[1])
    ax[1].set_ylabel('Weight')
    if weights.shape[1] >= 2:
        ax[1].legend(loc='upper left', fontsize=8, frameon=False, labelcolor=INK['secondary'])

    for column, color in zip(drawdown_col, colors):
        series = df[column]
        ax[2].plot(df.index, series.mask(series == 0), color=color, linewidth=2)
    _style_axes(ax[2])
    ax[2].set_ylabel('Drawdown')

    if path is not None:
        fig.savefig(path, dpi=300, facecolor=INK['surface'])
    return fig, ax

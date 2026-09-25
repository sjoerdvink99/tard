from collections.abc import Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from tard.result import DistortionResult

SURFACE = "#ffffff"
INK = "#1f1f1e"
MUTED_INK = "#6b6b68"
RULE = "#d9d8d4"
MISSING_FILL = "#fafaf9"
MISSING_HATCH = "#c9c8c3"
DISTORTION_INK = "#256abf"

DISTORTION_COLORMAP = LinearSegmentedColormap.from_list(
    "tard_distortion",
    ["#f2f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)
SIGNED_COLORMAP = LinearSegmentedColormap.from_list(
    "tard_signed",
    ["#7a1d1d", "#d64545", "#f3b1aa", "#f0efec", "#a9ccf5", "#3987e5", "#0d366b"],
)

STYLE = {
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "hatch.linewidth": 0.5,
}

CELL_WIDTH = 0.52
CELL_HEIGHT = 0.15
MAX_LABELED_ROWS = 60
EXTENSION_LENGTH = 0.05


def plot_distortion_map(
    result: DistortionResult,
    ax: Axes | None = None,
    title: str | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    """Node-by-scale matrix of distortion magnitude, from preserved to distorted."""
    upper = vmax if vmax is not None else _finite_max(result.distortion.to_numpy())
    return _plot_matrix(
        result.distortion,
        colormap=DISTORTION_COLORMAP,
        norm=Normalize(vmin=0.0, vmax=upper),
        colorbar_label="Distortion  mean |r$_X$ − r$_Z$|",
        extremes=("preserved", "distorted"),
        ax=ax,
        title=title,
    )


def plot_signed_map(
    result: DistortionResult,
    ax: Axes | None = None,
    title: str | None = None,
    limit: float | None = None,
) -> tuple[Figure, Axes]:
    """Node-by-scale matrix of signed change, centered on zero."""
    bound = limit if limit is not None else _finite_max(np.abs(result.signed_change.to_numpy()))
    return _plot_matrix(
        result.signed_change,
        colormap=SIGNED_COLORMAP,
        norm=Normalize(vmin=-bound, vmax=bound),
        colorbar_label="Signed change  mean(r$_Z$ − r$_X$)",
        extremes=("less similar", "more similar"),
        ax=ax,
        title=title,
    )


def plot_node_profile(
    result: DistortionResult,
    node: Hashable,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Distortion and signed change of one node across hop distance."""
    if node not in result.distortion.index:
        raise ValueError(f"Unknown node: {node!r}")
    with plt.rc_context(STYLE):
        figure, ax = _figure_and_axes(ax, figsize=(3.4, 2.1))
        ax.axhline(0.0, color=RULE, linewidth=0.8, zorder=0)
        ax.plot(
            result.scales,
            result.distortion.loc[node].to_numpy(),
            color=DISTORTION_INK,
            linewidth=1.8,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label="distortion",
        )
        ax.plot(
            result.scales,
            result.signed_change.loc[node].to_numpy(),
            color=MUTED_INK,
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label="signed change",
        )
        ax.set_xticks(result.scales, list(result.distortion.columns))
        ax.set_xlim(result.scales[0] - 0.3, result.scales[-1] + 0.3)
        ax.set_ylabel("Change in similarity")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        ax.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncols=2,
            handlelength=2.4,
            borderaxespad=0.2,
        )
        ax.set_title(title if title is not None else f"Node {node}", loc="left")
    return figure, ax


def _plot_matrix(
    matrix: pd.DataFrame,
    colormap: LinearSegmentedColormap,
    norm: Normalize,
    colorbar_label: str,
    extremes: tuple[str, str],
    ax: Axes | None,
    title: str | None,
) -> tuple[Figure, Axes]:
    values = matrix.to_numpy(dtype=float)
    n_rows, n_columns = values.shape
    with plt.rc_context(STYLE):
        figure, ax = _figure_and_axes(ax, figsize=_matrix_figsize(n_rows, n_columns))
        mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=colormap, norm=norm)
        _draw_missing_cells(ax, np.isnan(values))
        _draw_cell_gaps(ax, n_rows, n_columns)
        _format_matrix_axes(ax, matrix)
        _add_colorbar(figure, ax, mesh, colorbar_label, extremes, _colorbar_extension(values, norm))
        if title:
            ax.set_title(title, loc="left", pad=18)
    return figure, ax


def _figure_and_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize, layout="constrained")


def _matrix_figsize(n_rows: int, n_columns: int) -> tuple[float, float]:
    width = n_columns * CELL_WIDTH + 1.7
    height = min(max(n_rows * CELL_HEIGHT, 1.6), 7.0) + 0.9
    return width, height


def _finite_max(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0 or finite_values.max() == 0:
        return 1.0
    return float(finite_values.max())


def _draw_missing_cells(ax: Axes, missing: np.ndarray) -> None:
    for row, column in zip(*np.nonzero(missing), strict=True):
        ax.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=MISSING_FILL,
                edgecolor=MISSING_HATCH,
                hatch="//////",
                linewidth=0,
            )
        )
    if missing.any():
        missing_handle = Patch(
            facecolor=MISSING_FILL,
            edgecolor=MISSING_HATCH,
            hatch="//////",
            linewidth=0,
            label="no nodes at this distance",
        )
        ax.legend(
            handles=[missing_handle],
            loc="upper left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            handlelength=1.0,
            handleheight=1.0,
            borderaxespad=0.4,
            borderpad=0.0,
        )


def _draw_cell_gaps(ax: Axes, n_rows: int, n_columns: int) -> None:
    gap_width = 0.8 if n_rows <= MAX_LABELED_ROWS else 0.0
    ax.hlines(range(1, n_rows), 0, n_columns, color=SURFACE, linewidth=gap_width)
    ax.vlines(range(1, n_columns), 0, n_rows, color=SURFACE, linewidth=1.6)


def _format_matrix_axes(ax: Axes, matrix: pd.DataFrame) -> None:
    n_rows, n_columns = matrix.shape
    ax.set_xlim(0, n_columns)
    ax.set_ylim(n_rows, 0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n_columns) + 0.5, list(matrix.columns))
    if n_rows <= MAX_LABELED_ROWS:
        ax.set_yticks(np.arange(n_rows) + 0.5, [str(node) for node in matrix.index])
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"Nodes (n = {n_rows})")
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _colorbar_extension(values: np.ndarray, norm: Normalize) -> str:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "neither"
    below = finite_values.min() < norm.vmin
    above = finite_values.max() > norm.vmax
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _add_colorbar(
    figure: Figure,
    ax: Axes,
    mesh: QuadMesh,
    label: str,
    extremes: tuple[str, str],
    extension: str,
) -> None:
    colorbar = figure.colorbar(
        mesh,
        ax=ax,
        extend=extension,
        extendfrac=EXTENSION_LENGTH,
        aspect=28,
        pad=0.04,
        shrink=0.9,
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2, width=0.5, labelsize=6.5)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(5))
    colorbar.set_label(label, fontsize=7, color=MUTED_INK, labelpad=6)
    low_label, high_label = extremes
    extreme_style = {
        "xycoords": "axes fraction",
        "textcoords": "offset points",
        "ha": "center",
        "fontsize": 6.5,
        "color": MUTED_INK,
    }
    top = 1.0 + (EXTENSION_LENGTH if extension in ("max", "both") else 0.0)
    bottom = 0.0 - (EXTENSION_LENGTH if extension in ("min", "both") else 0.0)
    colorbar.ax.annotate(high_label, xy=(0.5, top), xytext=(0, 6), va="bottom", **extreme_style)
    colorbar.ax.annotate(low_label, xy=(0.5, bottom), xytext=(0, -6), va="top", **extreme_style)

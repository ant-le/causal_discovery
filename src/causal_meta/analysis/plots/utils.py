from __future__ import annotations

import logging
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

log = logging.getLogger(__name__)


def draw_point_plot(
    ax: Axes,
    df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    title: str,
    log_scale: bool = False,
    show_legend: bool = True,
) -> None:
    """
    Draws a point plot with error bars on the provided Axes.
    X-axis: Dataset (Level 2)
    Hue: Model (Level 1)
    """
    # Filter for the specific metric
    subset = df[df["Metric"] == metric_name].copy()

    if subset.empty:
        log.warning(f"No data found for metric: {metric_name}")
        return

    models = sorted(pd.unique(subset["Model"]))
    datasets = sorted(pd.unique(subset["Dataset"]))

    # Swap roles: X-axis = Datasets, Hue = Models
    x_labels = datasets
    hue_labels = models

    # Create a color map for Models
    if len(hue_labels) <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")

    colors = [cmap(i) for i in range(len(hue_labels))]

    # Calculate offsets for each hue (Model)
    num_hues = len(hue_labels)
    width = 0.6  # Total width for all hues within a x-tick
    offset_step = width / num_hues

    # X positions for Datasets
    x_base = range(len(x_labels))

    for j, hue in enumerate(hue_labels):
        # Filter data for this Model
        hue_data = subset[subset["Model"] == hue]

        # Determine marker style based on model type
        # Meta-learners (avici, bcnp) get Filled Circle
        # Explicit/Baselines (dibs, random) get Hollow Circle
        is_meta = hue.startswith(("avici", "bcnp"))

        # Base styling
        plot_kwargs = {
            "fmt": "o",
            "label": hue,
            "color": colors[j],
            "capsize": 0,
            "markersize": 8,
            "alpha": 0.9,
        }

        if not is_meta:
            # Make it hollow
            plot_kwargs["markerfacecolor"] = "white"
            plot_kwargs["markeredgecolor"] = colors[j]
            plot_kwargs["markeredgewidth"] = 2.0

        means = []
        sems = []
        xs = []

        for i, x_label in enumerate(x_labels):
            # Filter data for this Dataset
            row = hue_data[hue_data["Dataset"] == x_label]
            if len(row) > 0:
                # Cast to Any to bypass strict type checking on DataFrame/Series operations
                r = cast(Any, row)
                means.append(r["Mean"].values[0])
                sems.append(r["SEM"].values[0])
                # Calculate offset x position
                # Center the group around x_base[i]
                # Start from -width/2 + step/2
                offset = (j - num_hues / 2 + 0.5) * offset_step
                xs.append(x_base[i] + offset)

        if xs:
            ax.errorbar(xs, means, yerr=sems, **plot_kwargs)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_base)
    # Rotation 0, centered
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7, color="#cccccc")
    ax.set_facecolor("white")

    if log_scale:
        ax.set_yscale("log")

    if show_legend:
        # Place legend below the plot
        ax.legend(
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(len(hue_labels), 3),
        )

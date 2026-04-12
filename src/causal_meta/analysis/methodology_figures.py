"""Generate methodology visualizations for Chapter 4.

Three figures illustrating the data-generating process:
  1. Graph topology comparison (ER, SF, SBM, WS, GRG)
  2. Mechanism comparison (parent→child scatter for each mechanism family)
  3. Generated data comparison (node marginals + correlation heatmaps)

Usage:
    uv run python -m causal_meta.analysis.methodology_figures \
        --output-dir paper/final_thesis/generated/figures
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import torch

from causal_meta.datasets.generators.configs import (
    ErdosRenyiConfig,
    GeometricRandomConfig,
    GPMechanismConfig,
    LinearMechanismConfig,
    LogisticMapMechanismConfig,
    MLPMechanismConfig,
    PNLMechanismConfig,
    PeriodicMechanismConfig,
    SBMConfig,
    ScaleFreeConfig,
    SquareMechanismConfig,
    WattsStrogatzConfig,
)
from causal_meta.datasets.scm import SCMFamily

log = logging.getLogger(__name__)

# ── Consistent styling ─────────────────────────────────────────────────

_SEED = 42
_ID_COLOR = "#1f77b4"
_OOD_COLOR = "#d62728"
_NEUTRAL_COLOR = "#7f7f7f"

# matplotlib defaults for thesis-quality output
_RC_PARAMS = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_style() -> None:
    plt.rcParams.update(_RC_PARAMS)


# ── Graph definitions ──────────────────────────────────────────────────

_GRAPH_SPECS: list[tuple[str, object, bool]] = [
    ("ER", ErdosRenyiConfig(sparsity=0.1053), True),
    ("SF", ScaleFreeConfig(m=2), True),
    ("SBM", SBMConfig(n_blocks=4, p_intra=0.6, p_inter=0.01), False),
    ("WS", WattsStrogatzConfig(k=4, p=0.3), False),
    ("GRG", GeometricRandomConfig(radius=0.3, dim=2), False),
]

# ── Mechanism definitions ──────────────────────────────────────────────

_MECH_SPECS: list[tuple[str, object, bool]] = [
    (
        "Linear",
        LinearMechanismConfig(
            weight_scale=10.0, noise_concentration=2.0, noise_rate=2.0
        ),
        True,
    ),
    ("MLP", MLPMechanismConfig(hidden_dim=32), True),
    (
        "GPCDE",
        GPMechanismConfig(
            mode="approximate",
            rff_dim=512,
            num_kernels=4,
            length_scale_range=(0.1, 10.0),
            variance_range=(0.1, 10.0),
        ),
        True,
    ),
    ("Periodic", PeriodicMechanismConfig(weight_scale=10.0, noise_scale=0.1), False),
    ("Square", SquareMechanismConfig(weight_scale=10.0, noise_scale=0.1), False),
    ("Logistic Map", LogisticMapMechanismConfig(weight_scale=5.0), False),
    (
        "PNL (tanh)",
        PNLMechanismConfig(
            nonlinearity_type="tanh", inner_config=LinearMechanismConfig()
        ),
        False,
    ),
]

# ── Representative families for Panel 3 ───────────────────────────────

_DATA_FAMILIES: list[tuple[str, object, object, str, bool]] = [
    (
        "Linear + ER-20",
        ErdosRenyiConfig(sparsity=0.0526),
        LinearMechanismConfig(
            weight_scale=10.0, noise_concentration=2.0, noise_rate=2.0
        ),
        "gaussian",
        True,
    ),
    (
        "MLP + SF-2",
        ScaleFreeConfig(m=2),
        MLPMechanismConfig(hidden_dim=32),
        "gaussian",
        True,
    ),
    (
        "GPCDE + ER-60",
        ErdosRenyiConfig(sparsity=0.1579),
        GPMechanismConfig(
            mode="approximate",
            rff_dim=512,
            num_kernels=4,
            length_scale_range=(0.1, 10.0),
            variance_range=(0.1, 10.0),
        ),
        "gaussian",
        True,
    ),
    (
        "Periodic + ER-40",
        ErdosRenyiConfig(sparsity=0.1053),
        PeriodicMechanismConfig(weight_scale=1.0, noise_scale=0.1),
        "gaussian",
        False,
    ),
    (
        "Logistic + ER-40",
        ErdosRenyiConfig(sparsity=0.1053),
        LogisticMapMechanismConfig(weight_scale=1.0),
        "gaussian",
        False,
    ),
    (
        "PNL + SBM",
        SBMConfig(n_blocks=4, p_intra=0.6, p_inter=0.01),
        PNLMechanismConfig(
            nonlinearity_type="tanh", inner_config=LinearMechanismConfig()
        ),
        "gaussian",
        False,
    ),
]


# =====================================================================
# Figure 1: Graph Topology Comparison
# =====================================================================


def generate_graph_topology_figure(output_path: Path, n_nodes: int = 10) -> None:
    """Generate a figure comparing graph topologies across generators."""
    _apply_style()

    n_graphs = len(_GRAPH_SPECS)
    fig, axes = plt.subplots(1, n_graphs, figsize=(n_graphs * 2.0, 2.4))
    if n_graphs == 1:
        axes = [axes]

    for ax, (label, graph_cfg, is_id) in zip(axes, _GRAPH_SPECS):
        generator = graph_cfg.instantiate()
        torch_gen = torch.Generator().manual_seed(_SEED)
        np_rng = np.random.default_rng(_SEED)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_SEED)
            adj = generator(n_nodes, seed=_SEED, torch_generator=torch_gen, rng=np_rng)

        adj_np = adj.cpu().numpy()
        G = nx.from_numpy_array(adj_np, create_using=nx.DiGraph)
        n_edges = G.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0

        # Layout
        pos = nx.spring_layout(G, seed=_SEED, k=1.5 / np.sqrt(n_nodes))

        edge_color = _ID_COLOR if is_id else _OOD_COLOR
        node_color = _ID_COLOR if is_id else _OOD_COLOR

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=0.5,
            width=0.8,
            arrows=True,
            arrowsize=5,
            connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=40,
            node_color=node_color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

        tag = "ID" if is_id else "OOD"
        ax.set_title(f"{label}\n({tag}, {n_edges} edges)", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"Graph Topology Comparison ($N_G = {n_nodes}$)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info("Saved graph topology figure to %s", output_path)


# =====================================================================
# Figure 2: Mechanism Comparison
# =====================================================================


def generate_mechanism_comparison_figure(
    output_path: Path,
    n_samples: int = 500,
) -> None:
    """Generate scatter plots showing parent→child relationships per mechanism."""
    _apply_style()

    n_mechs = len(_MECH_SPECS)
    fig, axes = plt.subplots(1, n_mechs, figsize=(n_mechs * 2.0, 2.2))
    if n_mechs == 1:
        axes = [axes]

    # Use a simple 2-node chain: node 0 → node 1
    adj_2node = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    for ax, (label, mech_cfg, is_id) in zip(axes, _MECH_SPECS):
        factory = mech_cfg.instantiate()
        torch_gen = torch.Generator().manual_seed(_SEED)
        np_rng = np.random.default_rng(_SEED)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_SEED)
            mechanisms = factory(adj_2node, torch_generator=torch_gen, rng=np_rng)

        # Sample from the 2-node SCM
        from causal_meta.datasets.scm import SCMInstance

        scm = SCMInstance(adj_2node, mechanisms)
        data = scm.sample(n_samples).cpu().numpy()
        parent_vals = data[:, 0]
        child_vals = data[:, 1]

        # Handle degenerate root nodes (e.g., LogisticMap with noise_scale=0
        # produces a single constant value for the root).  Fall back to
        # directly visualising the child mechanism over a synthetic parent
        # range so the functional shape is visible.
        if np.std(parent_vals) < 1e-6:
            torch_gen_fb = torch.Generator().manual_seed(_SEED)
            synth_parent = torch.randn(n_samples, generator=torch_gen_fb) * 2.0
            noise_fb = torch.randn(n_samples, 1, generator=torch_gen_fb)
            with torch.no_grad():
                child_out = mechanisms[1](synth_parent.unsqueeze(1), noise_fb)
            parent_vals = synth_parent.numpy()
            child_vals = child_out.cpu().numpy().flatten()

        color = _ID_COLOR if is_id else _OOD_COLOR
        tag = "ID" if is_id else "OOD"

        ax.scatter(
            parent_vals,
            child_vals,
            s=3,
            alpha=0.3,
            color=color,
            rasterized=True,
        )
        ax.set_title(f"{label}\n({tag})", fontsize=9)
        ax.set_xlabel("Parent", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Child", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Mechanism Comparison (2-node SCM, $N = 500$)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info("Saved mechanism comparison figure to %s", output_path)


# =====================================================================
# Figure 3: Generated Data Comparison (marginals + correlations)
# =====================================================================


def generate_data_comparison_figure(
    output_path: Path,
    n_nodes: int = 10,
    n_samples: int = 500,
) -> None:
    """Generate node marginals and correlation heatmaps for representative families."""
    _apply_style()

    n_fams = len(_DATA_FAMILIES)
    # Layout: 2 rows × n_fams columns.  Row 0 = marginal violins, Row 1 = correlation.
    fig, axes = plt.subplots(
        2,
        n_fams,
        figsize=(n_fams * 2.4, 5.0),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.4},
    )
    if n_fams == 1:
        axes = axes.reshape(2, 1)

    for col, (label, graph_cfg, mech_cfg, noise_type, is_id) in enumerate(
        _DATA_FAMILIES
    ):
        family = SCMFamily(
            name=label.lower().replace(" ", "_"),
            n_nodes=n_nodes,
            graph_generator=graph_cfg.instantiate(),
            mechanism_factory=mech_cfg.instantiate(),
            noise_type=noise_type,
        )
        scm_instance = family.sample_task(seed=_SEED)
        data = scm_instance.sample(n_samples).cpu().numpy()

        color = _ID_COLOR if is_id else _OOD_COLOR
        tag = "ID" if is_id else "OOD"

        # ── Row 0: Node marginal distributions (violin plot) ──────────
        ax_marg = axes[0, col]
        parts = ax_marg.violinplot(
            data,
            positions=range(n_nodes),
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color(color)
        parts["cmeans"].set_linewidth(1.0)

        ax_marg.set_title(f"{label}\n({tag})", fontsize=9)
        ax_marg.set_xticks(range(n_nodes))
        ax_marg.set_xticklabels([str(i) for i in range(n_nodes)], fontsize=6)
        ax_marg.set_xlabel("Node", fontsize=8)
        if col == 0:
            ax_marg.set_ylabel("Value", fontsize=8)
        ax_marg.tick_params(labelsize=7)
        ax_marg.grid(True, axis="y", alpha=0.2, linewidth=0.5)

        # ── Row 1: Empirical correlation heatmap ──────────────────────
        ax_corr = axes[1, col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.corrcoef(data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)  # constant nodes → zero correlation
        im = ax_corr.imshow(
            corr,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            aspect="equal",
        )
        ax_corr.set_xticks(range(n_nodes))
        ax_corr.set_yticks(range(n_nodes))
        ax_corr.set_xticklabels([str(i) for i in range(n_nodes)], fontsize=6)
        ax_corr.set_yticklabels([str(i) for i in range(n_nodes)], fontsize=6)
        ax_corr.set_xlabel("Node", fontsize=8)
        if col == 0:
            ax_corr.set_ylabel("Node", fontsize=8)
        ax_corr.set_title("Correlation", fontsize=8)

    # Shared colorbar for correlation row
    fig.colorbar(
        im,
        ax=axes[1, :].tolist(),
        shrink=0.8,
        aspect=30,
        pad=0.02,
        label="Pearson $r$",
    )

    fig.suptitle(
        f"Generated Data Comparison ($N_G = {n_nodes}$, $N = {n_samples}$)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info("Saved data comparison figure to %s", output_path)


# =====================================================================
# CLI
# =====================================================================


def generate_all_methodology_figures(output_dir: Path) -> list[str]:
    """Generate all three methodology figures and return relative paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    generate_graph_topology_figure(output_dir / "methodology_graph_topology.pdf")
    generated.append("figures/methodology_graph_topology.pdf")

    generate_mechanism_comparison_figure(
        output_dir / "methodology_mechanism_comparison.pdf"
    )
    generated.append("figures/methodology_mechanism_comparison.pdf")

    generate_data_comparison_figure(output_dir / "methodology_data_comparison.pdf")
    generated.append("figures/methodology_data_comparison.pdf")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate methodology visualizations for Chapter 4."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/final_thesis/generated/figures",
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generated = generate_all_methodology_figures(output_dir)
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

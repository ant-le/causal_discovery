import pytest
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.axes
from causal_meta.datasets.scm_dataset import SCMFamily, SCMInstance
from causal_meta.datasets.generators.generate_functions import LinearMechanism

def test_visualization():
    # Setup
    family = SCMFamily(variable_count=4, graph_density=0.5)
    save_base = "test_viz"
    
    # Execution
    family.plot_example(save_path=save_base)
    
    # Assertion
    assert os.path.exists(f"{save_base}_graph.png")
    assert os.path.exists(f"{save_base}_data.png")
    
    # Cleanup
    if os.path.exists(f"{save_base}_graph.png"):
        os.remove(f"{save_base}_graph.png")
    if os.path.exists(f"{save_base}_data.png"):
        os.remove(f"{save_base}_data.png")

def test_plot_graph_linear_weights():
    family = SCMFamily(variable_count=3, graph_density=1.0) # Fully connected
    instance = family.sample_scm(seed=42)
    
    # Just ensure it runs without crashing when accessing weights
    instance.plot_graph(save_path="test_graph_weights.png")
    
    assert os.path.exists("test_graph_weights.png")
    os.remove("test_graph_weights.png")

def test_plot_relationships_no_edges():
    adj = np.zeros((2, 2))
    mechanisms = [LinearMechanism(0), LinearMechanism(0)]
    noise_dists = [lambda n: torch.randn(n) for _ in range(2)]
    instance = SCMInstance(adj, mechanisms, noise_dists)

    path = "test_no_edges.png"
    instance.plot_relationships(n_samples=10, save_path=path, show=False)

    assert os.path.exists(path)
    os.remove(path)

def test_plot_relationships_plots_edges_only(monkeypatch):
    adj = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]])
    mechanisms = [LinearMechanism(0), LinearMechanism(1), LinearMechanism(1)]
    noise_dists = [lambda n: torch.randn(n) for _ in range(3)]
    instance = SCMInstance(adj, mechanisms, noise_dists)

    edge_count = len(list(instance.graph.edges()))
    scatter_calls = []

    original_scatter = matplotlib.axes.Axes.scatter

    def scatter_spy(self, *args, **kwargs):
        scatter_calls.append((args, kwargs))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", scatter_spy)

    instance.plot_relationships(n_samples=20, show=False)

    assert len(scatter_calls) == edge_count

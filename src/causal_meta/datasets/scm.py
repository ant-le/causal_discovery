from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn

from causal_meta.datasets.generators.graphs import GraphGenerator
from causal_meta.datasets.generators.mechanisms import (ConstantMechanism,
                                                        MechanismFactory)


def _topological_order_from_adj(adjacency_matrix: torch.Tensor) -> List[int]:
    """Compute a topological order from an adjacency matrix."""
    graph = nx.from_numpy_array(adjacency_matrix.cpu().numpy(), create_using=nx.DiGraph)
    return list(nx.topological_sort(graph))


class SCMInstance:
    """Lightweight SCM instance with ancestral sampling."""

    def __init__(
        self,
        adjacency_matrix: torch.Tensor,
        mechanisms: Sequence[nn.Module],
        topological_order: Optional[Sequence[int]] = None,
    ) -> None:
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        if adjacency_matrix.shape[0] != len(mechanisms):
            raise ValueError("Mechanisms length must match number of nodes.")

        self.adjacency_matrix = adjacency_matrix.float()
        self.mechanisms = list(mechanisms)
        self.topological_order = (
            list(topological_order)
            if topological_order is not None
            else _topological_order_from_adj(self.adjacency_matrix)
        )

    @property
    def n_nodes(self) -> int:
        return self.adjacency_matrix.shape[0]

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Perform ancestral sampling following the cached topological order.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_samples, n_nodes).
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")

        device = self.adjacency_matrix.device
        n_nodes = self.adjacency_matrix.shape[0]
        samples = torch.zeros(
            (num_samples, n_nodes), device=device, dtype=torch.float32
        )

        with torch.no_grad():
            for node in self.topological_order:
                parents = torch.nonzero(
                    self.adjacency_matrix[:, node], as_tuple=False
                ).flatten()

                parent_values = (
                    samples[:, parents]
                    if parents.numel() > 0
                    else torch.zeros(
                        (num_samples, 0), device=device, dtype=torch.float32
                    )
                )

                noise = torch.randn(
                    (num_samples, 1), device=device, dtype=torch.float32
                )
                value = self.mechanisms[node](parent_values, noise)
                samples[:, node] = value.view(-1)

        return samples

    def do(self, intervention: Dict[int, float | torch.Tensor]) -> SCMInstance:
        """
        Applies Pearl's do-operator, returning a new SCMInstance (the 'mutilated graph').

        Edges incoming to intervened nodes are removed (structural intervention),
        and their mechanisms are replaced by ConstantMechanism (functional intervention).

        Args:
            intervention: A dictionary mapping node indices to their intervention values.

        Returns:
            SCMInstance: A new SCM instance representing the intervened system.
        """
        mutilated_adjacency = self.adjacency_matrix.clone()
        new_mechanisms = list(self.mechanisms)

        for node, value in intervention.items():
            # Structural Intervention: Remove incoming edges
            mutilated_adjacency[:, node] = 0.0
            # Functional Intervention: Fix value
            new_mechanisms[node] = ConstantMechanism(value)

        # Return new instance (topological order will be recomputed automatically)
        return SCMInstance(mutilated_adjacency, new_mechanisms)


class SCMFamily:
    """Composes graph and mechanism generators to sample SCM tasks."""

    def __init__(
        self,
        n_nodes: int,
        graph_generator: GraphGenerator,
        mechanism_factory: MechanismFactory,
    ) -> None:
        if n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        self.n_nodes = n_nodes
        self.graph_generator = graph_generator
        self.mechanism_factory = mechanism_factory

    def sample_task(self, seed: int) -> SCMInstance:
        """Sample a single SCM instance with deterministic seeding."""
        torch_generator = torch.Generator().manual_seed(seed)
        np_rng = np.random.default_rng(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adjacency_matrix = self.graph_generator(
                self.n_nodes, seed=seed, torch_generator=torch_generator, rng=np_rng
            )
            mechanisms = self.mechanism_factory(
                adjacency_matrix, torch_generator=torch_generator, rng=np_rng
            )

        topological_order = _topological_order_from_adj(adjacency_matrix)
        return SCMInstance(adjacency_matrix, mechanisms, topological_order)

    def sample_graph(self, seed: int) -> torch.Tensor:
        """
        Sample only the adjacency matrix (no mechanisms).

        Useful for fast disjointness checks and dataset profiling during setup.
        """
        torch_generator = torch.Generator().manual_seed(seed)
        np_rng = np.random.default_rng(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adjacency_matrix = self.graph_generator(
                self.n_nodes, seed=seed, torch_generator=torch_generator, rng=np_rng
            )

        return adjacency_matrix.float()

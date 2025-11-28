import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import List, Callable, Optional, Dict

from .generators.generate_graph import generate_erdos_renyi
from .generators.generate_functions import LinearMechanism

class SCMInstance:
    def __init__(self, 
                 adjacency_matrix: np.ndarray, 
                 mechanisms: List[Callable], 
                 noise_dists: List[Callable]):
        """
        Args:
            adjacency_matrix: (d, d) binary matrix. A[i,j]=1 => i->j
            mechanisms: List of length d. mechanisms[j] is f_j(parents, noise)
            noise_dists: List of length d. noise_dists[j]() returns scalar/tensor noise.
        """
        self.adjacency_matrix = adjacency_matrix
        self.mechanisms = mechanisms
        self.noise_dists = noise_dists
        self.num_vars = adjacency_matrix.shape[0]
        
        # Pre-compute topological order for efficient sampling
        self.graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Adjacency matrix must represent a DAG.")
        self.topological_order = list(nx.topological_sort(self.graph))

    def sample_observational(self, n_samples: int) -> torch.Tensor:
        """
        Generates observational data from the joint distribution P(X).
        """
        data = torch.zeros((n_samples, self.num_vars), dtype=torch.float32)
        
        for i in self.topological_order:
            # Identify parents
            parents_indices = list(self.graph.predecessors(i))
            parents_values = data[:, parents_indices]
            
            # Sample noise
            noise = self.noise_dists[i](n_samples)
            
            # Apply mechanism
            data[:, i] = self.mechanisms[i](parents_values, noise)
            
        return data

    def sample_interventional(self, n_samples: int, target_node: int, value: float) -> torch.Tensor:
        """
        Generates interventional data from P(X | do(target_node = value)).
        """
        data = torch.zeros((n_samples, self.num_vars), dtype=torch.float32)
        
        for i in self.topological_order:
            if i == target_node:
                # Intervention: Set value directly, ignore parents and noise
                data[:, i] = torch.full((n_samples,), value, dtype=torch.float32)
            else:
                # Normal mechanism
                parents_indices = list(self.graph.predecessors(i))
                parents_values = data[:, parents_indices]
                noise = self.noise_dists[i](n_samples)
                data[:, i] = self.mechanisms[i](parents_values, noise)
                
        return data

    def get_markov_equivalence_class(self):
        raise NotImplementedError("MEC computation not yet implemented.")

    def plot_graph(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plots the DAG structure. Annotates edges with weights if mechanisms are linear.
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx(self.graph, pos, with_labels=True, node_color='lightblue', 
                         node_size=500, arrowsize=20)
        
        # Edge labels for linear mechanisms
        edge_labels = {}
        for i in self.topological_order:
            mech = self.mechanisms[i]
            if hasattr(mech, 'weights'): # LinearMechanism check
                parents = list(self.graph.predecessors(i))
                for idx, p in enumerate(parents):
                    w = mech.weights[idx].item()
                    edge_labels[(p, i)] = f"{w:.2f}"
        
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
            
        plt.title("SCM Causal Graph")
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def plot_relationships(self, n_samples: int = 1000, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Generates scatter plots for parent-child relationships (edges) in the DAG.
        """
        edges = list(self.graph.edges())

        if not edges:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('off')
            ax.text(0.5, 0.5, "No edges to plot", ha='center', va='center', fontsize=12)
            fig.suptitle("No Causal Edges Found")
        else:
            data = self.sample_observational(n_samples).numpy()
            df = pd.DataFrame(data, columns=[f"X{i}" for i in range(self.num_vars)])

            n_edges = len(edges)
            grid_size = math.ceil(np.sqrt(n_edges))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(4.5 * grid_size, 4.5 * grid_size))
            axes = np.array(axes).reshape(grid_size, grid_size)

            for idx, (parent, child) in enumerate(edges):
                ax = axes[idx // grid_size, idx % grid_size]
                ax.scatter(df[f"X{parent}"], df[f"X{child}"], alpha=0.5)
                ax.set_xlabel(f"X{parent}")
                ax.set_ylabel(f"X{child}")
                ax.set_title(f"X{parent} -> X{child}")

            for idx in range(n_edges, grid_size * grid_size):
                axes[idx // grid_size, idx % grid_size].axis('off')

            fig.suptitle("Direct Causal Relationships (Parent -> Child)")
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)


class SCMFamily:
    def __init__(self, 
                 variable_count: int, 
                 graph_density: float = 0.2,
                 mechanism_type: str = 'linear', 
                 noise_type: str = 'gaussian'):
        self.variable_count = variable_count
        self.graph_density = graph_density
        self.mechanism_type = mechanism_type
        self.noise_type = noise_type

    def sample_scm(self, seed: Optional[int] = None) -> SCMInstance:
        """
        Samples a concrete SCM from the family distribution.
        """
        # 1. Generate Graph
        adj = generate_erdos_renyi(self.variable_count, self.graph_density, seed=seed)
        
        # 2. Generate Mechanisms and Noise
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        mechanisms = []
        noise_dists = []
        
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        
        for i in range(self.variable_count):
            parents = list(graph.predecessors(i))
            num_parents = len(parents)
            
            # Create Mechanism
            if self.mechanism_type == 'linear':
                mech = LinearMechanism(num_parents)
            else:
                raise NotImplementedError(f"Mechanism {self.mechanism_type} not supported.")
            mechanisms.append(mech)
            
            # Create Noise Dist
            if self.noise_type == 'gaussian':
                noise_dist = lambda n: torch.randn(n)
            else:
                raise NotImplementedError(f"Noise {self.noise_type} not supported.")
            noise_dists.append(noise_dist)
            
        return SCMInstance(adj, mechanisms, noise_dists)

    def distance_to(self, other: 'SCMFamily') -> float:
        raise NotImplementedError("Distance computation not yet implemented.")

    def plot_example(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Samples a random instance and plots its graph and relationships.
        """
        instance = self.sample_scm(seed=None) # Random seed
        
        graph_path = None
        data_path = None
        
        if save_path:
            base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            graph_path = f"{base}_graph.png"
            data_path = f"{base}_data.png"
            
        instance.plot_graph(save_path=graph_path, show=show)
        instance.plot_relationships(save_path=data_path, show=show)

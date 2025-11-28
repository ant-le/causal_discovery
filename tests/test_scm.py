import pytest
import torch
import numpy as np
import networkx as nx
from causal_meta.datasets.scm_dataset import SCMFamily, SCMInstance

def test_scm_generation():
    family = SCMFamily(variable_count=5, graph_density=0.4, mechanism_type='linear')
    instance = family.sample_scm(seed=42)
    
    assert isinstance(instance, SCMInstance)
    assert instance.adjacency_matrix.shape == (5, 5)
    
    # Check DAG property
    G = nx.from_numpy_array(instance.adjacency_matrix, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

def test_observational_sampling():
    family = SCMFamily(variable_count=5, graph_density=0.4)
    instance = family.sample_scm(seed=42)
    
    n = 100
    data = instance.sample_observational(n)
    
    assert data.shape == (n, 5)
    assert not torch.isnan(data).any()

def test_interventional_sampling():
    # Create a chain 0 -> 1 -> 2 manually to control structure for test
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[1, 2] = 1
    
    # Mock mechanisms: identity + noise (small noise to see effect clearly or just 0 noise)
    # Let's use the Family logic but override
    
    # Using the standard LinearMechanism with random weights might be hard to predict 
    # unless we control weights.
    # Let's just test the intervention effect broadly first using the Family.
    
    family = SCMFamily(variable_count=5)
    instance = family.sample_scm(seed=42)
    
    # Pick a target node
    target = 0
    value = 10.0
    n = 50
    
    data_int = instance.sample_interventional(n, target, value)
    
    # Check target values
    assert torch.allclose(data_int[:, target], torch.full((n,), value))
    
    # Check that downstream nodes are not NaN (sanity check)
    assert not torch.isnan(data_int).any()

def test_reproducibility():
    family = SCMFamily(variable_count=5)
    inst1 = family.sample_scm(seed=123)
    data1 = inst1.sample_observational(10)
    
    inst2 = family.sample_scm(seed=123)
    data2 = inst2.sample_observational(10)
    
    assert np.array_equal(inst1.adjacency_matrix, inst2.adjacency_matrix)
    assert torch.allclose(data1, data2)
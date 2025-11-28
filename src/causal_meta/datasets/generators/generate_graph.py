import numpy as np
import networkx as nx

def generate_erdos_renyi(num_vars: int, density: float, seed: int = None) -> np.ndarray:
    """
    Generates a random DAG using the Erdos-Renyi model.
    
    Args:
        num_vars: Number of vertices.
        density: Probability of edge existence.
        seed: Random seed.
        
    Returns:
        Adjacency matrix (numpy array) where A[i, j] = 1 implies i -> j.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Generate random permutation to ensure acyclicity
    permutation = np.random.permutation(num_vars)
    
    # 2. Initialize adjacency matrix
    adj = np.zeros((num_vars, num_vars), dtype=int)
    
    # 3. Add edges consistent with permutation order
    # We only consider edges i -> j if perm[i] < perm[j]
    # This guarantees strictly lower triangular (or upper, depending on sorting) in topological order
    
    # Efficient way: iterate all pairs, check order, sample bernoulli
    # Ideally we want expected degree or density. 
    # For ER, each valid pair has probability p.
    
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                continue
                
            # Check topological order
            if permutation[i] < permutation[j]:
                if np.random.random() < density:
                    adj[i, j] = 1
                    
    return adj
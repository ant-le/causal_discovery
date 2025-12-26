# Class Structure

## 1. Datasets Module (`src.datasets`)

### `SCMFamily`
*Defines the distribution from which SCMs are drawn (Meta-Learning Tasks).*
- **Composition:**
  - `graph_generator`: Protocol (Strategy for adjacency sampling).
  - `mechanism_factory`: Protocol (Strategy for mechanism creation).
- **Attributes:**
  - `variable_count`: int
  - `noise_type`: str
  - `mechanism_proportions`: Dict[str, float]
- **Methods:**
  - `sample_task(seed: int) -> SCMInstance`: Pure function using seed to create task.
  - `plot_example(save_path: Optional[str])`: Visualization helper.

### `MetaIterableDataset` (Inherits `IterableDataset`)
*Infinite stream of tasks for training.*
- **Attributes:**
  - `scm_family`: SCMFamily
  - `base_seed`: int
- **Methods:**
  - `__iter__()`: Implements worker-aware and rank-aware seeding logic.
  - `__next__()`: Yields `(X, adjacency_matrix)` tuple.

### `MetaFixedDataset` (Inherits `Dataset`)
*Fixed set of tasks for validation/testing.*
- **Attributes:**
  - `scm_family`: SCMFamily
  - `seeds`: List[int]
  - `cache_instances`: bool
  - `_cache`: Dict[int, SCMInstance]
- **Methods:**
  - `__getitem__(idx)`: Re-instantiates task from `seeds[idx]` (or retrieves from cache).

### `SCMInstance`
*A specific realization of a causal model.*
- **Attributes:**
  - `adjacency_matrix`: torch.Tensor (The DAG)
  - `mechanisms`: List[nn.Module]
  - `topological_order`: List[int]
- **Methods:**
  - `sample(n: int) -> Tensor`: Ancestral sampling.

## 2. Models Module (`src.causal_meta.models`)

### `BaseModel` (Inherits `nn.Module`)
*Uniform interface for all causal discovery methods (Meta-Learning & Classical).*
- **Attributes:**
  - `needs_pretraining`: bool (Flag for meta-learners vs instance-optimizers)
- **Methods:**
  - `forward(x: Tensor) -> Any`: Model-specific forward pass (returns logits/params).
  - `sample(x: Tensor, n_samples: int) -> Tensor`: Returns graph samples `(Batch, N, V, V)`.
  - `calculate_loss(output: Any, target: Tensor, **kwargs) -> Tensor`: Computes training loss.

### `BCNP` (Inherits `BaseModel`)
*Bayesian Causal Neural Process implementation.*
- **Architecture:** Node Embedding (Set Transformer) -> Edge Prediction (Bilinear/MLP).
- **Components:** `SetTransformerLayer`, `MultiHeadAttention`.

### `AviciModel` (Inherits `BaseModel`)
*Avici-style Amortized Causal Discovery.*
- **Architecture:** Sample Aggregation -> Variable Attention (Transformer) -> Edge Prediction.

### `ModelFactory`
- **Methods:**
  - `create(config: Dict) -> BaseModel`: Instantiates registered models.

## 3. Pipeline Module (`src.pipeline`)

### `ExperimentRunner`
- **Attributes:**
  - `config`: Dict
- **Methods:**
  - `setup()`: Instantiates families and models.
  - `execute()`: Runs the train-eval loop.
  - `save_artifacts()`: Dumps logs and models.

### `Metrics`
- **Static Methods:**
  - `compute_shd(true_adj, pred_adj_list) -> float`
  - `compute_auroc(true_adj, marginal_probs) -> float`
  - `compute_nil(test_data, posterior_scms) -> float`
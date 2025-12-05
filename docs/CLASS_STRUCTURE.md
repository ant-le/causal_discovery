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

### `DAGInstance`
*A purely structural causal model (graph only).*
- **Attributes:**
  - `adjacency_matrix`: np.ndarray (The DAG)
  - `num_vars`: int
- **Methods:**
  - `plot_graph(save_path: Optional[str] = None, show: bool = False) -> None`

### `SCMInstance` (Inherits `DAGInstance`)
*A specific realization of a causal model.*
- **Attributes:**
  - `mechanisms`: List[Callable] (Functions $f_i(PA_i, \epsilon_i)$)
  - `noise_dists`: List[Callable] (Samplers for $\epsilon_i$)
  - `topological_order`: List[int]
- **Methods:**
  - `sample(n: int, normalize: bool = True) -> Tensor`: Ancestral sampling.
  - `sample_interventional(n: int, target: int, value: float) -> Tensor`
  - `get_markov_equivalence_class() -> List[np.ndarray]`
  - `plot_relationships(n_samples: int = 1000, save_path: Optional[str] = None, show: bool = False) -> None`

## 2. Models Module (`src.models`)

### `BaseCausalModel` (Abstract)
*Uniform interface for all BCD methods.*
- **Methods:**
  - `__init__(config: Dict)`
  - `train(dataset: Tensor, **kwargs) -> None`
  - `infer_graph_posterior(data: Tensor, n_samples: int) -> List[np.ndarray]`
  - `infer_scm_posterior(data: Tensor, n_samples: int) -> List[SCMInstance]`
  - `save(path: str)`
  - `load(path: str)`

### `BNCPModel` (Inherits BaseCausalModel)
- *Implementation of the Meta-Learning approach.*

### `ExplicitBCDModel` (Inherits BaseCausalModel)
- *Wrapper for DiBS/BayesDAG.*

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

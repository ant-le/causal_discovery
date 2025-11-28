# Class Structure

## 1. Datasets Module (`src.datasets`)

### `SCMFamily`
*Defines the distribution from which SCMs are drawn.*
- **Attributes:**
  - `graph_generator`: Callable (returns adjacency matrix).
  - `mechanism_type`: Enum/String (e.g., 'linear', 'mlp').
  - `noise_type`: Enum/String (e.g., 'gaussian').
  - `variable_count`: int
- **Methods:**
  - `sample_scm(seed: int) -> SCMInstance`
  - `distance_to(other: SCMFamily) -> float` (Computes OOD distance)
  - `plot_example(save_path: Optional[str] = None) -> None`

### `SCMInstance`
*A specific realization of a causal model.*
- **Attributes:**
  - `adjacency_matrix`: np.ndarray (The DAG)
  - `mechanisms`: List[Callable] (Functions $f_i(PA_i, \epsilon_i)$)
  - `noise_dists`: List[Callable] (Samplers for $\epsilon_i$)
- **Methods:**
  - `sample_observational(n: int) -> Tensor`
  - `sample_interventional(n: int, target: int, value: float) -> Tensor`
  - `get_markov_equivalence_class() -> List[np.ndarray]`
  - `plot_graph(save_path: Optional[str] = None, show: bool = False) -> None`
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

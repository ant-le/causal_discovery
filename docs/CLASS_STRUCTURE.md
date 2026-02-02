# Class Structure

## 1. Datasets Module (`src/causal_meta/datasets` / `causal_meta.datasets`)

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
  - **Yields:** a pickle-safe dict: `{"seed": int, "data": Tensor, "adjacency": Tensor}`.

### `MetaFixedDataset` (Inherits `Dataset`)
*Fixed set of tasks for validation/testing.*
- **Attributes:**
  - `scm_family`: SCMFamily
  - `seeds`: List[int]
  - `samples_per_task`: int
- **Methods:**
  - `__getitem__(idx)`: Returns a dict `{"seed": int, "data": Tensor, "adjacency": Tensor}`.

**Full-SCM training note**
- The dataset outputs intentionally do **not** include `SCMInstance` / mechanism modules (which may be non-picklable under DataLoader workers).
- To do supervised “full SCM” training, reconstruct the teacher SCM inside the training step via `SCMFamily.sample_task(seed)` using the returned `seed`.

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
  - `sample(x: Tensor, num_samples: int) -> Tensor`: Returns graph samples `(Batch, N, V, V)`.
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

### `run_pipeline` (in `causal_meta.runners.pipe`)
Orchestrates the train/eval run (and optional cached inference) based on Hydra config.
- **Responsibilities:**
  - Distributed setup/teardown.
  - Instantiate `CausalMetaModule` and model (`ModelFactory`).
  - Dispatch tasks: `pre_training` or `inference`, then `evaluation`.

### `Metrics`
- **Static Methods:**
  - `compute_shd(true_adj, pred_adj_list) -> float`
  - `compute_auroc(true_adj, marginal_probs) -> float`
  - `compute_nil(test_data, posterior_scms) -> float`

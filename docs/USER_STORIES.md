# User Stories

## Epic 1: Dataset Generation & Management
**Story 1.1: Core SCM Primitives**
As a researcher, I want a flexible `SCMFamily` and `SCMInstance` abstraction so that I can compose different graph structures and mechanism types without changing the downstream code.
- **Acceptance Criteria:**
  - `SCMInstance` holds adjacency matrix and mechanism modules; supports ancestral sampling.
  - `SCMFamily` composes a `GraphGenerator` and `MechanismFactory`.
  - Sampling returns topologically sorted data.

**Story 1.2: Generator Strategies**
As a developer, I want robust implementations of common graph and mechanism distributions (ER, SF, SBM; Linear, MLP) so that I can benchmark against diverse causal structures.
- **Acceptance Criteria:**
  - `GraphGenerator` supports Erdős-Rényi (vectorized), Scale-Free (NetworkX-based), and SBM.
  - `MechanismFactory` supports Linear Heteroskedastic and MLP mechanisms.
  - Generators run efficiently on CPU to avoid GPU bottlenecks.

**Story 1.3: PyTorch Dataset Infrastructure**
As an ML engineer, I need `MetaIterableDataset` for infinite streaming and `MetaFixedDataset` for reproducible evaluation to ensure my training pipeline is both scalable and deterministic.
- **Acceptance Criteria:**
  - `MetaIterableDataset` handles infinite streams with rank/worker-aware seeding (no duplicates).
  - `MetaFixedDataset` loads from a fixed seed list and supports instance caching.
  - `collate_fn` handles data normalization (standard scaling).

**Story 1.4: DataModule & Safety**
As a user, I want a `DataModule` that manages train/test splits and enforces strict separation so that I never accidentally train on my test graphs.
- **Acceptance Criteria:**
  - Validation of `FamilyConfig` objects (Hydra/OmegaConf) at instantiation.
  - Disjoint set enforcement via graph hashing (retry logic on collision).
  - Pre-computation of family-level statistics for OOD detection.

**Story 1.5: Analysis & Diagnostics**
As a researcher, I want a diagnostics module to visualize graph properties and log distribution distances so that I can verify the "OOD-ness" of my datasets before training.
- **Acceptance Criteria:**
  - Functions to plot degree histograms and adjacency matrices.
  - computation of Spectral/KL distances between families.
  - Automatic logging of seed registries and config snapshots.

## Epic 2: Model Abstraction
**Story 2.1: Unified Model Interface**
As a developer, I want a standardized `BaseCausalModel` class so that I can easily plug in different algorithms (BNCP, DiBS, Avici, BayesDAG) without rewriting the training loop.
- **Acceptance Criteria:**
  - `BaseCausalModel` defines `train(data, **kwargs)`.
  - `BaseCausalModel` defines `infer_graph_posterior(data)`.
  - `BaseCausalModel` defines `infer_full_scm_posterior(data)` (optional/raises NotImplemented).

## Epic 3: Experiment Pipeline
**Story 3.1: Configuration-Driven Experiments**
As a user, I want to define experiments via YAML configuration files (specifying dataset parameters, model choice, and training hyperparameters) so that I can manage large-scale benchmarks easily.
- **Acceptance Criteria:**
  - A runner script accepts a path to a YAML config.
  - The system parses the config to instantiate the correct SCM family and Model.

**Story 3.2: Automated Evaluation**
As a researcher, I want the pipeline to automatically compute metrics (E-SHD, AUROC, NIL, I-NIL) after inference so that I have immediate feedback on model performance.
- **Acceptance Criteria:**
  - Pipeline calculates graph metrics (SHD, AUROC) against ground truth.
  - Pipeline calculates predictive metrics (NIL) on held-out test sets.

## Epic 4: Packaging & Artifacts
**Story 4.1: Artifact Storage**
As a user, I want experiment artifacts (trained weights, logs, generated config copies) to be saved in a structured directory `artifacts/{model}/{run_id}/` so that I can audit results later.
- **Acceptance Criteria:**
  - Automatic creation of artifact directories.
  - Logging of all parameters.

**Story 4.2: Installable Package**
As a developer, I want the `src/` folder to be structured as an installable Python package (e.g., `causal_meta`) so that I can install it via `pip` or `conda`.
- **Acceptance Criteria:**
  - `pyproject.toml` or `setup.py` exists and works.

## Epic 5: Advanced Mechanisms & Distributional Shifts
**Story 5.1: Mixture Generators**
As a researcher, I want to define distributions as mixtures of other generators (e.g., 50% ER + 50% SF) and mechanisms (e.g., Node A is Linear, Node B is MLP) to replicate the complex training curricula used in state-of-the-art meta-learning papers.
- **Acceptance Criteria:**
  - `MixtureGraphGenerator`: Selects a sub-generator per call based on probability weights (Inter-graph mixing).
  - `MixtureMechanismFactory`: Selects a sub-factory **per node** based on probability weights (Intra-graph mixing).
  - Supports arbitrary nesting of generators.

**Story 5.2: Gaussian Process Mechanism (GPCDE)**
As a researcher, I need a Gaussian Process mechanism generator that incorporates latent noise variables within the kernel to replicate the "GPCDE" data generation process found in BCNP and Syntren benchmarks.
- **Acceptance Criteria:**
  - Implements `GPCDEMechanism` using `gpytorch` or standard PyTorch operations.
  - Uses RBF or Matern kernel.
  - Crucial: The noise variable $\epsilon$ is an **input** to the kernel (non-additive), not just added to the output.

**Story 5.3: O.O.D. Mechanism Suite**
As a researcher, I need specific non-standard mechanisms (Square, Periodic, Logistic Map, PNL) to audit the model's robustness against invertibility gaps, chaos, and statistical shifts.
- **Acceptance Criteria:**
  - `SquareMechanism`: Symmetric non-invertible function.
  - `PeriodicMechanism`: High-frequency sine functions.
  - `LogisticMapMechanism`: Deterministic chaos ($x_{t+1} = r x_t (1-x_t)$).
  - `PostNonlinearMechanism`: Applies a non-linear distortion $g(\cdot)$ after the parent aggregation.

**Story 5.4: Advanced Configuration**
As a user, I want to configure these complex mixtures and O.O.D. types via the existing YAML/Dict configuration system without writing custom code for every experiment.
- **Acceptance Criteria:**
  - `FamilyConfig` or `CausalMetaModule` parsing logic updated to handle lists of generators/weights.
  - Factory methods (`_build_family`) updated to instantiate `Mixture*` classes when config specifies a list.
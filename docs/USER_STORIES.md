# User Stories

## Epic 1: Dataset Generation & Management
**Story 1.1: SCM Families**
As a researcher, I want to define "families" of Structural Causal Models (SCMs) (e.g., specific graph priors, mechanism types, noise distributions) so that I can systematically test "in-distribution" (i.d.) vs. "out-of-distribution" (o.o.d.) performance.
- **Acceptance Criteria:**
  - Class `SCMFamily` exists.
  - Can specify graph properties (sparsity, size).
  - Can specify mechanism types (linear, MLP, GP).
  - Can specify noise types (Gaussian, Laplace).

**Story 1.2: Deterministic SCM Instances**
As a researcher, I want to sample concrete `SCMInstance` objects from an `SCMFamily` using a random seed so that my experiments are reproducible.
- **Acceptance Criteria:**
  - `SCMFamily.sample(seed)` returns an `SCMInstance`.
  - `SCMInstance` contains the explicit DAG (adjacency matrix) and callables for mechanisms.
  - `SCMInstance` can generate observational and interventional data.

**Story 1.4: SCM Visualization & Inspection**
As a researcher, I want to visualize both the causal graph structure and the functional relationships of SCMs so that I can qualitatively verify the nature of the generated data and dependencies.
- **Acceptance Criteria:**
  - `SCMFamily.plot_example()`: Samples and plots a random instance.
  - `SCMInstance.plot_graph()`: Plots the DAG. Annotates edges with weights if mechanisms are linear.
  - `SCMInstance.plot_relationships()`: Generates scatter plots of specific functional relationships where an edge exists in the graph (Parent -> Child), rather than a full $N \times N$ matrix, to focus on direct causal links.

**Story 1.3: O.O.D. Quantification**
As a researcher, I want to quantify the "distance" between the training SCM family and test SCM families using information-theoretic metrics so I can correlate performance degradation with distribution shift.
- **Acceptance Criteria:**

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

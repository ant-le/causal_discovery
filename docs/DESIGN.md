# Technical Design

## 1. System Overview
The system is a Python-based framework for benchmarking Bayesian Causal Discovery (BCD) methods, specifically focusing on Meta-Learning approaches. It is designed to measure robustness against distribution shifts (o.o.d.) and extend capabilities to full Structural Causal Model (SCM) inference.

The core architecture follows a **Configuration-Driven Pipeline** pattern:
`Config` -> `Data Generation` -> `Model Training` -> `Inference` -> `Evaluation`

## 2. Architecture Layers

### 2.1. Datasets Layer (`src/datasets`)
Responsible for defining the Data Generating Processes (DGPs).
- **Concept: Families vs. Instances**
  - **SCMFamily:** Represents a distribution over SCMs (the "prior"). Defined by structural properties (graph density, type) and functional properties (mechanism classes, noise distributions).
  - **SCMInstance:** A concrete realization sampled from a Family. Contains a fixed DAG and fixed mechanism functions.
- **Generators:**
  - `generate_graph.py`: Algorithms for sampling DAGs (e.g., Erdos-Renyi, Scale-Free).
  - `generate_functions.py`: Factories for mechanism functions (Linear, MLP, etc.).
- **Storage:**
  - Datasets are not permanently stored as large CSVs by default but are reproducible via seeds.
  - "Cached" datasets may be stored in `src/datasets/data/{family_hash}/{seed}/` for efficiency during repeated testing.

### 2.2. Models Layer (`src/models`)
Wraps diverse BCD algorithms into a uniform API.
- **BaseCausalModel:** Abstract base class.
  - `train(dataset)`: Updates internal parameters.
  - `sample_graph_posterior(n_samples)`: Returns a list of adjacency matrices.
  - `sample_scm_posterior(n_samples)`: Returns a list of SCM instances (graph + mechanisms).
- **Implementations:**
  - `bncp/`: Bayesian Neural Causal Models (Meta-learning).
  - `dibs/`: Differentiable Bayesian Structure Learning.
  - `avici/`: Amortized Variational Inference.
  - `bayesdag/`: Explicit VI/MCMC baseline.

### 2.3. Pipeline Layer (`src/pipeline`)
Orchestrates the experiment.
- **ExperimentRunner:**
  1.  Reads YAML config.
  2.  Instantiates `SCMFamily` (Train) and `SCMFamily` (Test/OOD).
  3.  Instantiates `Model`.
  4.  Runs Training Loop.
  5.  Runs Evaluation Loop.
  6.  Saves Artifacts.
- **Metrics:**
  - `eval_metrics.py`: Graph metrics (SHD, AUROC) and Data metrics (NIL, I-NIL).

## 3. Data Flow
1.  **Initialization:** User provides `experiment.yaml`.
2.  **Data Gen:** `SCMFamily` uses RNG to produce `SCMInstance_Train` and `SCMInstance_Test`.
3.  **Sampling:** `SCMInstance` produces `X_train` (tensor).
4.  **Training:** `Model.train(X_train)` optimizes variational parameters or weights.
5.  **Inference:** `Model` produces posterior samples $\{G_i, f_i\}$.
6.  **Scoring:** Metrics module compares $\{G_i\}$ vs `G_true` and computes likelihood of `X_test` under $\{f_i\}$.

## 4. Technologies
- **Language:** Python 3.10+
- **Core Libs:** PyTorch (Deep Learning), NetworkX (Graph Utils), NumPy/Pandas.
- **Config:** Hydra or Pydantic-based YAML parsing.
- **Packaging:** `pyproject.toml` (setuptools/hatch).

## 5. Security & Reproducibility
- **Seeds:** Global random seeds (numpy, torch, python) must be set at the entry point.
- **Isolation:** Models run in isolated environments/processes if necessary (future work).
- **Artifacts:** All outputs saved to `artifacts/` with unique Run IDs.

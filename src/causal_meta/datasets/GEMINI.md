# Research Workflow: Auditing Meta-Learner Robustness

This workflow outlines the methodology for evaluating the Out-of-Distribution (O.O.D.) robustness of Bayesian Meta-Learners (specifically BCNP). It involves a rigorous **Replication Phase** (Training) followed by a **Stress-Test Audit** (Testing) using novel function and structure generators.

## 1. Visual Workflow Overview

```mermaid
graph TD
    subgraph "Phase 1: Structure Generation"
    A[Start: Sample Graph Topology] --> B{Generator Type}
    B -->|Sparse/Medium| C[Erdős-Rényi (ER)]
    B -->|Hub-and-Spoke| D[Scale-Free (SF)]
    B -->|Modular Clusters| E[Stochastic Block Model (SBM)]
    end

    subgraph "Phase 2: Mechanism Assignment"
    C & D & E --> F{Data Mode?}
    
    %% Training Path (Replication)
    F -->|TRAINING (Replication)| G[The Competence Curriculum]
    G --> G1[Linear-Heteroscedastic (40%)]
    G --> G2[Random MLP (30%)]
    G --> G3[GPCDE (30%)]
    
    %% Testing Path (O.O.D. Audit)
    F -->|TESTING (Torture Suite)| H[The Robustness Audit]
    H --> H1[Test A: Invertibility Gap]
    H --> H2[Test B: Deterministic Chaos]
    H --> H3[Test C: Post-Nonlinear Shift]
    end

    subgraph "Phase 3: Processing & Eval"
    G1 & G2 & G3 & H1 & H2 & H3 --> I[Normalize Variables]
    I --> J[Input to Meta-Learner]
    J --> K[Evaluate: AUC, LogProb, SHD]
    end
```

---

## 2. Graph Topology Generators
*Goal: Ensure the model is robust to structural variations, from sparse random graphs to modular community structures.*

### A. Erdős-Rényi (ER)
* **Parameter:** Edge probability $p \sim U(0.1, 0.3)$.
* **Densities:** Vary expected edge counts (20, 40, 60 edges) to match BCNP baselines.
* **Purpose:** Standard baseline for sparse-to-medium density structures.

### B. Scale-Free (SF)
* **Parameter:** Barabási-Albert attachment parameter.
* **Purpose:** Generates "Hub-and-Spoke" structures. Crucial for testing if the model can efficiently learn V-structures (colliders) and super-nodes.

### C. Stochastic Block Model (SBM) *(New Extension)*
* **Structure:** Nodes are divided into $k$ communities. Probability of connection is high within communities ($p_{in}$) and low between them ($p_{out}$).
* **Why:** Tests "Long-Range Dependency." Transformers often struggle to attend to causal links that span across distinct modular clusters (common in gene regulatory networks).

---

## 3. Training Set: "The Competence Curriculum"
*Goal: Exact replication of the BCNP paper's training regime to establish a fair baseline. The model learns a mixture of standard causal mechanisms.*

> **Critical Note:** The training set deliberately avoids the "Torture Suite" scenarios to isolate O.O.D. generalization capabilities during testing.

* **Linear-Heteroscedastic (40%)**
    * **Equation:** $X_j = W \cdot PA_j + \sigma(PA_j) \cdot \epsilon$
    * **Learning Objective:** Teaches the model that noise variance can depend on the parents (heteroscedasticity).

* **Random MLP (30%)**
    * **Equation:** 2-layer Neural Network with **Leaky ReLU** or **Tanh** activation.
    * **Learning Objective:** Teaches monotonic, standard non-linear relationships.

* **GPCDE (30%)**
    * **Definition:** Gaussian Process with a **Latent Variable** input.
    * **Kernel:** Exponential Gamma Kernel.
    * **Mechanism:** Unlike a standard GP ($Y=f(X)+E$), GPCDE feeds the noise $\epsilon$ *into* the kernel: $X_j \sim \mathcal{GP}(PA_j, \epsilon)$.
    * **Learning Objective:** Teaches the model to handle non-additive noise and varying smoothness scales.

---

## 4. Test Set: "The Torture Suite" (O.O.D.)
*Goal: Assessment of robustness against functional, complexity, and statistical shifts not seen during training.*

### Test Set A: The "Invertibility Gap" (Functional Shift)
* **Theory:** The training data (Linear, Leaky ReLU) preserves information (monotonicity). This set tests if the model relies on "invertibility heuristics" rather than true causal structure.
* **Implementation:**
    * **Square (Symmetric):** $X_j = (W^T PA_j)^2 + 0.1\epsilon$
    * **Periodic (High Frequency):** $X_j = \sin(4\pi W^T PA_j) + 0.1\epsilon$
* **Constraint:** Noise level is kept low ($0.1$) to ensure the structure is deterministic but the mapping is non-injective (many-to-one).

### Test Set B: The "Chaos" Test (Complexity Shift)
* **Theory:** Tests if the model can distinguish "deterministic complexity" from "stochastic noise." The training set contains smooth GPs; this set contains deterministic chaos.
* **Implementation:** The Logistic Map ($r=4.0$).
    * $$X_j = 4 \cdot \sigma(W^T PA_j) \cdot (1 - \sigma(W^T PA_j))$$
    * *Note: $\sigma$ (sigmoid) binds inputs to $[0,1]$.*
* **Why:** To a linear observer or standard kernel, this function has zero correlation and looks like white noise, yet it is fully causal.

### Test Set C: The "Post-Nonlinear" (PNL) Shift
* **Theory:** The training set's GPCDE covers heteroscedastic noise, but not explicit sensor distortion. This tests the Post-Nonlinear assumption $X = g(f(PA) + \epsilon)$.
* **Implementation:**
    1.  Generate standard Additive Model: $Z = f(PA_j) + \epsilon$
    2.  Apply Distortion: $X_j = Z^3$ or $X_j = \text{sigmoid}(Z)$
* **Why:** Simulates measurement distortion (e.g., sensor saturation or exponential amplification).

---

## 5. Processing & Evaluation

### Normalization
* **Protocol:** Following Reisach et al. (2021), all variables are normalized after generation to zero mean and unit variance.
    * *Note:* This prevents the model from "gaming" the system by using marginal variance (sorting variables by scale) to infer direction.

### Metrics
1.  **AUC (Area Under ROC):** Measures the quality of the edge ranking.
2.  **Log Probability (NLL):** **Crucial for Bayesian evaluation.** Checks if the model is "confidently wrong" (bad) vs. "uncertain" (acceptable) on O.O.D. data.
3.  **SHD (Structural Hamming Distance):** Counts missing/wrong edges (lower is better).

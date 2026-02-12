# Source Map

Auto-generated from `src/assets/data/sourceMap.json`.
Generated: 2026-02-12T23:11:58.568Z

## Motivation

### Problem Setting

- **From Correlation to Causal Models**
  - code: src/causal_meta/main.py, src/causal_meta/models/base.py
  - notes: paper/markdown/keep_background/1_Background__1911.10500v2.md, paper/markdown/keep_background/1_Background__Pearl_2009_Causality.md
  - docs: docs/DESIGN.md

### Research Questions

- **RQ1: OOD Robustness**
  - code: src/causal_meta/runners/tasks/evaluation.py, src/causal_meta/runners/metrics/graph.py, src/causal_meta/main.py
  - notes: paper/markdown/keep_rq1/2_CausalDiscovery__Meta-Learning.md, paper/markdown/keep_both/2_CausalDiscovery__Evaluation.md
  - docs: docs/RUNBOOK.md, docs/DESIGN.md
- **RQ2: Full-SCM Approximation**
  - code: src/causal_meta/models/bcnp/model.py, src/causal_meta/runners/metrics/scm.py, src/causal_meta/runners/tasks/inference.py
  - notes: paper/markdown/keep_both/2_CausalDiscovery__Evaluation.md, paper/markdown/keep_rq2/3_EnhancedMCMC__short_chains.md
  - docs: docs/CLASS_STRUCTURE.md

### Impact

- **Applied Domains**
  - code: src/causal_meta/runners/tasks/evaluation.py, src/causal_meta/runners/metrics/scm.py
  - notes: paper/markdown/keep_rq1/0_Motivation__bayesian-causal-discovery-for-policy-decision-making.md, paper/markdown/meta/relevant_literature_proposal.md
  - docs: README.md

## Background

### Foundations

- **SCM and Bayesian Discovery**
  - code: src/causal_meta/datasets/scm.py, src/causal_meta/models/base.py
  - notes: paper/markdown/keep_background/1_Background__Pearl_2009_Causality.md, paper/markdown/keep_rq1/2_CausalDiscovery__BayesDAG.md
  - docs: docs/DESIGN.md

### Causal Inference

- **Modeling Assumptions**
  - code: src/causal_meta/datasets/architecture.md, src/causal_meta/models/architecture.md
  - notes: paper/markdown/keep_background/1_Background__caual_algorithms.md, paper/markdown/keep_background/1_Background__1911.10500v2.md
  - docs: docs/CLASS_STRUCTURE.md
- **Structural Causal Models**
  - code: src/causal_meta/datasets/scm.py, src/causal_meta/datasets/generators/factory.py
  - notes: paper/markdown/keep_background/1_Background__Pearl_2009_Causality.md, paper/markdown/keep_background/1_Background__caual_algorithms.md
  - docs: docs/DESIGN.md
- **Potential Outcomes Framework**
  - code: src/causal_meta/runners/metrics/scm.py
  - notes: paper/markdown/keep_background/1_Background__Pearl_2009_Causality.md
  - docs: docs/DESIGN.md

### Mathematical Appendix

- **Calculus**
  - code: src/causal_meta/models/utils/nn.py
  - notes: paper/markdown/drop_background/1_Background__measure_theory.md
  - docs: docs/proposal.md
- **Measure Theory**
  - code: src/causal_meta/runners/metrics/scm.py
  - notes: paper/markdown/drop_background/1_Background__measure_theory.md
  - docs: docs/proposal.md
- **Probability Theory**
  - code: src/causal_meta/runners/utils/scoring.py
  - notes: paper/markdown/keep_background/1_Background__caual_algorithms.md
  - docs: docs/proposal.md
- **Information Theory**
  - code: src/causal_meta/runners/metrics/graph.py
  - notes: paper/markdown/keep_both/2_CausalDiscovery__Evaluation.md
  - docs: docs/proposal.md

## Thesis

### Architecture

- **Codebase Overview**
  - code: src/causal_meta/main.py, src/causal_meta/models/factory.py, src/causal_meta/datasets/data_module.py
  - notes: paper/markdown/keep_rq1/2_CausalDiscovery__Meta-Learning.md
  - docs: docs/CLASS_STRUCTURE.md, docs/DESIGN.md

### Method

- **Bayesian Causal Discovery Pipeline**
  - code: src/causal_meta/main.py, src/causal_meta/runners/tasks/pre_training.py, src/causal_meta/runners/tasks/inference.py, src/causal_meta/runners/tasks/evaluation.py
  - notes: paper/markdown/keep_rq1/2_CausalDiscovery__DIBS.md, paper/markdown/keep_rq1/2_CausalDiscovery__BayesDAG.md, paper/markdown/keep_both/2_CausalDiscovery__Evaluation.md
  - docs: docs/RUNBOOK.md, docs/DESIGN.md

### Acceleration

- **GPU and Parallel MCMC**
  - code: src/causal_meta/main.py, src/causal_meta/runners/utils/distributed.py
  - notes: paper/markdown/optional_rq2/3_EnhancedMCMC__gpu.md, paper/markdown/keep_rq2/3_EnhancedMCMC__short_chains.md
  - docs: docs/PROFILING.md


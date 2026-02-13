import type { ExplainContent, TextData } from "../../types.ts";

export const motivationData: Record<string, TextData[]> = {
  practical: [
    {
      category: "Healthcare",
      icon: "⚕️",
      title: "Treatment Policy Design",
      content:
        "Causal structure learning helps separate interventions from confounders, supporting safer treatment planning and better transfer across hospitals.",
    },
    {
      category: "Public Policy",
      icon: "🏛️",
      title: "Counterfactual Policy Analysis",
      content:
        "Bayesian posteriors over graphs make uncertainty explicit, so policy recommendations can include confidence-aware what-if scenarios.",
    },
    {
      category: "Scientific Discovery",
      icon: "🧪",
      title: "Hypothesis Prioritization",
      content:
        "Posterior samples highlight multiple plausible mechanisms, helping teams prioritize experiments instead of committing too early to one graph.",
    },
  ],
  theoretical: [
    {
      icon: "📉",
      title: "OOD Fragility of Amortized Inference",
      content:
        "Meta-learned causal discovery can fail when mechanism families, noise scales, or graph topologies shift beyond training conditions.",
    },
    {
      icon: "🧭",
      title: "Posterior Quality vs. Point Metrics",
      content:
        "Recent evaluation work shows that graph-only metrics (e.g., SHD/AUROC) are often insufficient to judge posterior fidelity in high-entropy settings.",
    },
    {
      icon: "⚙️",
      title: "Scalable Bayesian Inference",
      content:
        "The thesis investigates practical trade-offs between explicit optimization methods and amortized predictors under strict runtime constraints.",
    },
  ],
};

const thesisWorkflow: ExplainContent[] = [
  {
    title: "Hydra-Driven Experiment Configuration",
    explainText:
      "Experiments use Hydra configuration groups, enabling consistent local and cluster runs.",
  },
  {
    title: "Dataset Family Sampling",
    explainText:
      "<code>CausalMetaModule</code> builds training and evaluation datasets from SCM families with disjointness checks and reproducible seeds.",
  },
  {
    title: "Model Dispatch",
    explainText:
      "<code>ModelFactory</code> creates Avici, BCNP, DiBS, or BayesDAG models behind a shared interface.",
  },
  {
    title: "Task-Oriented Pipeline",
    explainText:
      "The pipeline executes pre-training or explicit inference, then evaluates graph and posterior-oriented metrics in a unified runner.",
  },
];

export const thesisData: TextData & { explanation: ExplainContent[] } = {
  title: "Causal Meta-Learning Benchmarks",
  content:
    "A Hydra-configured benchmarking framework for Bayesian causal discovery focused on OOD robustness (RQ1) and full-SCM posterior approximation trade-offs (RQ2).",
  topics: [
    "<strong>RQ1:</strong> Compare amortized and explicit Bayesian methods under controlled distribution shifts.",
    "<strong>RQ2:</strong> Study the runtime/accuracy frontier for richer posterior approximations.",
    "<strong>Scope:</strong> Datasets, model zoo, distributed runners, and evaluation tooling in one reproducible codebase.",
  ],
  explanation: thesisWorkflow,
};

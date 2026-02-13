import type { Component } from "svelte";
import Applications from "../sections/motivation/Applications.svelte";
import Explainability from "../sections/motivation/Explainability.svelte";
import Generalisability from "../sections/motivation/Generalisability.svelte";
import Introduction from "../sections/motivation/Introduction.svelte";
import SCM from "../sections/background/SCM.svelte";
import POF from "../sections/background/POF.svelte";
import Calculus from "../sections/background/Calculus.svelte";
import MeasureTheory from "../sections/background/MeasureTheory.svelte";
import ProbTheory from "../sections/background/ProbTheory.svelte";
import InfoTheory from "../sections/background/InfoTheory.svelte";
import CausalDiscovery from "../sections/thesis/CausalDiscovery.svelte";
import GPUMCMC from "../sections/thesis/GPUMCMC.svelte";
import Ideas from "../sections/thesis/Ideas.svelte";
import InferenceStrategies from "../sections/methodology/InferenceStrategies.svelte";
import DataFamilies from "../sections/benchmark/DataFamilies.svelte";
import ModelFamilies from "../sections/benchmark/ModelFamilies.svelte";
import CurrentFindings from "../sections/results/CurrentFindings.svelte";

export type SectionsData = Record<string, Component>;
export interface PageTitle {
  name: string;
  subscript?: string;
}
export type PageMetaData = Record<string, SectionsData>;
export type AppStates =
  | "Motivation"
  | "Methodology"
  | "Benchmark"
  | "Results"
  | "Appendix";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
  Motivation: {
    "Problem & Scope": {
      "Why This Benchmark": Introduction,
    },
    "Research Questions": {
      "RQ1: OOD Robustness": Generalisability,
      "RQ2: Full-SCM Approximation": Explainability,
    },
    "Decision Context": {
      "Applied Stakes": Applications,
    },
  },
  Methodology: {
    "Causal Assumptions": {
      "Structural Causal Models": SCM,
      "Bayesian Causal Objective": CausalDiscovery,
    },
    "Inference Design": {
      "Amortized vs Explicit Inference": InferenceStrategies,
    },
  },
  Benchmark: {
    System: {
      "Pipeline Architecture": Ideas,
    },
    "Synthetic Data": {
      "Graph and Mechanism Families": DataFamilies,
    },
    "Model Zoo": {
      "Benchmarked Method Families": ModelFamilies,
    },
  },
  Results: {
    "Current Evidence": {
      "Run-backed Metric Snapshot": CurrentFindings,
    },
    Reliability: {
      "Convergence and Diagnostics": GPUMCMC,
    },
  },
  Appendix: {
    "Additional Causal Framing": {
      "Potential Outcomes Perspective": POF,
    },
    "Mathematical Notes": {
      "Calculus Primer": Calculus,
      "Measure Theory Primer": MeasureTheory,
      "Probability Primer": ProbTheory,
      "Information Theory Primer": InfoTheory,
    },
  },
};

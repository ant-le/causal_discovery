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
import Modelling from "../sections/background/Modelling.svelte";
import BackgroundIntroduction from "../sections/background/Introduction.svelte";

export type SectionsData = Record<string, Component>;
export interface PageTitle {
  name: string;
  subscript?: string;
}
export type PageMetaData = Record<string, SectionsData>;
export type AppStates = "Motivation" | "Background" | "Thesis";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
  Motivation: {
    "Problem Setting": {
      "From Correlation to Causal Models": Introduction,
    },
    "Research Questions": {
      "RQ1: OOD Robustness": Generalisability,
      "RQ2: Full-SCM Approximation": Explainability,
    },
    Impact: {
      "Applied Domains": Applications,
    },
  },
  Background: {
    Foundations: {
      "SCM and Bayesian Discovery": BackgroundIntroduction,
    },
    "Causal Inference": {
      "Modeling Assumptions": Modelling,
      "Structural Causal Models": SCM,
      "Potential Outcomes Framework": POF,
    },
    "Mathematical Appendix": {
      Calculus,
      "Measure Theory": MeasureTheory,
      "Probability Theory": ProbTheory,
      "Information Theory": InfoTheory,
    },
  },
  Thesis: {
    Architecture: {
      "Codebase Overview": Ideas,
    },
    Method: {
      "Bayesian Causal Discovery Pipeline": CausalDiscovery,
    },
    Acceleration: {
      "GPU and Parallel MCMC": GPUMCMC,
    },
  },
};

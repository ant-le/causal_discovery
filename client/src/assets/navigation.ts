import type { Component } from "svelte";
import Applications from "../sections/motivation/Applications.svelte";
import Explainability from "../sections/motivation/Explainability.svelte";
import Generalisability from "../sections/motivation/Generalisability.svelte";
import Introduction from "../sections/motivation/Introduction.svelte";
import SCM from "../sections/background/SCM.svelte";
import POF from "../sections/background/POF.svelte";
import Calculus from "../sections/background/Calculus.svelte";
import MeasureTheory from "../sections/background/MeasureTheory.svelte"
import ProbTheory from "../sections/background/ProbTheory.svelte"
import InfoTheory from "../sections/background/InfoTheory.svelte";
import CausalDiscovery from "../sections/thesis/CausalDiscovery.svelte";
import GPUMCMC from "../sections/thesis/GPUMCMC.svelte";
import Ideas from "../sections/thesis/Ideas.svelte";

export type SectionsData = Record<string, Component>;
export interface PageTitle {
    name: string
    subscript?: string
}
export type PageMetaData = Record<string, SectionsData>;
export type AppStates = "Motivation" | "Background" | "Thesis";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
    "Motivation": {
        "Introduction": {

            "From Correlational Patterns to Causal Understanding": Introduction,

        },
        "A step towards AGI": {

            "Generalisability": Generalisability,
            "Explainability": Explainability,

        },
        "Applications": {

            "Applications": Applications,

        },
    },
    "Background": {
        "Introduction": {},
        "Math": {

            "Calculus": Calculus,
            "Measure Theory": MeasureTheory,
            "Probability Theory": ProbTheory,
            "Information Theory": InfoTheory,

        },
        "Causal Inference": {

            "SCM": SCM,
            "Potential Outcomes Framework": POF,

        },
        "Deep Learning": {},
    },
    "Thesis": {
        "Introduction": {
            "Main Ideas": Ideas,
        },
        "Content": {
            "Causal Discovery": CausalDiscovery,
            "MCMC on GPU": GPUMCMC,
        },
    },
};

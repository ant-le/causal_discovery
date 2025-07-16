import type { Component } from "svelte";
import Applications from "../sections/motivation/Applications.svelte";
import Explainability from "../sections/motivation/Explainability.svelte";
import Generalisability from "../sections/motivation/Generalisability.svelte";
import Introduction from "../sections/motivation/Introduction.svelte";
import SCM from "../sections/background/SCM.svelte";
import POF from "../sections/background/POF.svelte";

export interface SectionMetaData {
    title: string;
    component: Component;
}

export interface PageData {
    subscript?: string;
    sections: SectionMetaData[]
}
export type PageMetaData = Record<string, PageData>;
export type AppStates = "Motivation" | "Background" | "Thesis";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
    "Motivation": {
        "Introduction": {
            sections: [
                { title: "From Correlational Patterns to Causal Understanding", component: Introduction }],
        },
        "A step towards AGI": {
            sections: [
                { title: "Generalisability", component: Generalisability },
                { title: "Explainability", component: Explainability }
            ]
        },
        "Applications": {
            sections: [
                { title: "Applications", component: Applications }
            ],
        },
    },
    "Background": {
        "Introduction": { sections: [] },
        "Causal Inference": {
            sections: [
                { title: "SCM", component: SCM },
                { title: "Potential Outcomes Framework", component: POF }
            ]
        },
        "Deep Learning": { sections: [] },
    },
    "Thesis": {
        "Introduction": { sections: [] },
        "Content": { sections: [] },
    },
};

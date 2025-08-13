import type { Component } from "svelte";
import Applications from "./sections/motivation/Applications.svelte";
import Explainability from "./sections/motivation/Explainability.svelte";
import Generalisability from "./sections/motivation/Generalisability.svelte";

import SCM from "./sections/background/SCM.svelte";
import POF from "./sections/background/POF.svelte";

export interface Citation{
    id: string;
    author: string;
    title: string;
    year: string;
    url: string;
}

export interface SectionMetaData{
    title:          string;
    subscript?:     string;
    component:      Component;
}

export type PageMetaData = Record<string, SectionMetaData[]>;   
export type AppStates = "Motivation" | "Background" | "Thesis";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
    "Motivation": {
        "Introduction": [],
        "A step towards AGI": [
            {title: "Generalisability", component: Generalisability}, 
            {title: "Explainability", component: Explainability}
        ],
        "Applications": [
            {title: "Applications", component: Applications}
        ],
    }, 
    "Background": {
        "Introduction": [],
        "Causal Inference": [
            {title: "SCM", component: SCM}, 
            {title: "Potential Outcomes Framework", component: POF}
        ],
        "Deep Learning": [],
    }, 
    "Thesis": {
        "Introduction": [],
        "Content": [],
    },
};

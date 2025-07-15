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
}
//TODO: Add super/subscript for PageTitle Component
export type PageMetaData = Record<string, SectionMetaData[]>;   
export type AppStates = "Motivation" | "Background" | "Thesis";
export type AppMetaData = Record<AppStates, PageMetaData>;
export const appMetaData: AppMetaData = {
    "Motivation": {
        "Introduction": [],
        "A step towards AGI": [
            {title: "Generalisability"}, 
            {title: "Explainability"}
        ],
        "Applications": [],
    }, 
    "Background": {
        "Introduction": [],
        "Causal Inference": [
            {title: "SCM"}, 
            {title: "Potential Outcomes Framework"}
        ],
        "Deep Learning": [],
    }, 
    "Thesis": {
        "Introduction": [],
        "Content": [],
    },
};

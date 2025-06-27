// src/types.ts
export interface AppStates {
    name: "home" | "motivation" | "background" | "content";
}

export interface TextData{
    title: string;
    content: string;
    category?: string;
    icon?: string;
    topics?: string[];
}

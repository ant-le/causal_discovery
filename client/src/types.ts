// src/types.ts
export type AppStates = "home" | "motivation" | "background" | "content";

export interface TextData{
    title: string;
    content: string;
    category?: string;
    icon?: string;
    topics?: string[];
    citations?: string[];
}

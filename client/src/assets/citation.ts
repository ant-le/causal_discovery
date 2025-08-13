export interface ArticleData{
    author: string;
    title: string;
    year: number;
    url?: string;
}

export type Citation = Record<string, ArticleData>;

export const bibliography: Citation = {
    "bookOfWhy": {
        author: "Pearl J., Mackenzie D.",
        title: "The Book of Why",
        year: 2018,
    },
};

export type CitationKey = keyof typeof bibliography;

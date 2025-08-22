export interface ArticleData {
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
    "causalDiscovery": {
        author: `Bernhard Sch{\"{o}}lkopf and
                  Francesco Locatello and
                  Stefan Bauer and
                  Nan Rosemary Ke and
                  Nal Kalchbrenner and
                  Anirudh Goyal and
                  Yoshua Bengio`,
        title: "Towards Causal Representation Learning",
        year: 2021,
        url: "https://arxiv.org/abs/2102.11107",
    }
};

export type CitationKey = keyof typeof bibliography;

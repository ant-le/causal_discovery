export interface ArticleData {
  author: string;
  title: string;
  year: number;
  url?: string;
}

export type Citation = Record<string, ArticleData>;

export const bibliography: Citation = {
  bookOfWhy: {
    author: "Pearl J., Mackenzie D.",
    title: "The Book of Why",
    year: 2018,
  },
  causalDiscovery: {
    author: `Scholkopf B,
                  Locatello F.,
                  Bauer S.,
                  Ke N.,
                  Kalchbrenner N.,
                  Goyal A.,
                  Bengio Y`,
    title: "Towards Causal Representation Learning",
    year: 2021,
    url: "https://arxiv.org/abs/2102.11107",
  },
  avici: {
    author: "Lorch L. et al.",
    title: "AVICI: Amortized Causal Discovery",
    year: 2022,
    url: "https://arxiv.org/abs/2205.12934",
  },
  bcnp: {
    author: "Dhir A. et al.",
    title: "A Meta-Learning Approach to Bayesian Causal Discovery",
    year: 2024,
    url: "https://arxiv.org/abs/2402.00623",
  },
  dibs: {
    author: "Lorch L. et al.",
    title: "DiBS: Differentiable Bayesian Structure Learning",
    year: 2021,
    url: "https://arxiv.org/abs/2105.11839",
  },
  bayesdag: {
    author: "Annadani Y. et al.",
    title: "BayesDAG",
    year: 2024,
    url: "https://arxiv.org/abs/2402.04845",
  },
  evaluation: {
    author: "Karimi Mamaghan A. et al.",
    title:
      "Challenges and Considerations in the Evaluation of Bayesian Causal Discovery",
    year: 2024,
    url: "https://proceedings.mlr.press/v235/karimi-mamaghan24a.html",
  },
  gpuMcmc: {
    author: "Sountsov P., Carroll C., Hoffman M.D.",
    title: "Running Markov Chain Monte Carlo on Modern Hardware and Software",
    year: 2024,
    url: "https://arxiv.org/abs/2407.05636",
  },
  shortChains: {
    author: "Anagnostis A. et al.",
    title: "Nested R-hat: Assessing Convergence for Many Short Chains",
    year: 2024,
    url: "https://arxiv.org/abs/2403.03772",
  },
};

export type CitationKey = keyof typeof bibliography;

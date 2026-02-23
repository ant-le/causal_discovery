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
    author:
      "Schölkopf B., Locatello F., Bauer S., Ke N., Kalchbrenner N., Goyal A., Bengio Y.",
    title: "Towards Causal Representation Learning",
    year: 2021,
    url: "https://arxiv.org/abs/2102.11107",
  },
  avici: {
    author: "Lorch L., Sussex S., Rothfuss J., Krause A., Schölkopf B.",
    title: "Amortized Inference for Causal Structure Learning",
    year: 2022,
    url: "https://arxiv.org/abs/2205.12934",
  },
  bcnp: {
    author: "Dhir A., Sedal A., Briol F.-X.",
    title: "A Meta-Learning Approach to Bayesian Causal Discovery",
    year: 2024,
    url: "https://arxiv.org/abs/2402.00623",
  },
  dibs: {
    author: "Lorch L., Rothfuss J., Schölkopf B., Krause A.",
    title: "DiBS: Differentiable Bayesian Structure Learning",
    year: 2021,
    url: "https://arxiv.org/abs/2105.11839",
  },
  bayesdag: {
    author:
      "Annadani Y., Rothfuss J., Lacoste A., Scetbon M., Goyal A., Bengio Y., Bauer S.",
    title: "BayesDAG: Gradient-Based Posterior Inference for Causal Discovery",
    year: 2024,
    url: "https://arxiv.org/abs/2307.13383",
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
  pearl2009: {
    author: "Pearl J.",
    title: "Causality: Models, Reasoning and Inference",
    year: 2009,
  },
  peters2017: {
    author: "Peters J., Janzing D., Schölkopf B.",
    title: "Elements of Causal Inference: Foundations and Learning Algorithms",
    year: 2017,
    url: "https://mitpress.mit.edu/9780262037310/",
  },
  zheng2018: {
    author: "Zheng X., Aragam B., Ravikumar P., Xing E.P.",
    title: "DAGs with NO TEARS: Continuous Optimization for Structure Learning",
    year: 2018,
    url: "https://arxiv.org/abs/1803.01422",
  },
  gumbel: {
    author: "Mena G., Belanger D., Linderman S., Snoek J.",
    title: "Learning Latent Permutations with Gumbel-Sinkhorn Networks",
    year: 2018,
    url: "https://arxiv.org/abs/1802.08665",
  },
  svgd: {
    author: "Liu Q., Wang D.",
    title:
      "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm",
    year: 2016,
    url: "https://arxiv.org/abs/1608.04471",
  },
  sid: {
    author: "Peters J., Bühlmann P.",
    title: "Structural Intervention Distance for Evaluating Causal Graphs",
    year: 2015,
    url: "https://arxiv.org/abs/1306.1043",
  },
  rff: {
    author: "Rahimi A., Recht B.",
    title: "Random Features for Large-Scale Kernel Machines",
    year: 2007,
    url: "https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html",
  },
  barabasiAlbert: {
    author: "Barabási A.-L., Albert R.",
    title: "Emergence of Scaling in Random Networks",
    year: 1999,
    url: "https://doi.org/10.1126/science.286.5439.509",
  },
};

export type CitationKey = keyof typeof bibliography;

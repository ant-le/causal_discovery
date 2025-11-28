# Problem Definition

Modern artificial intelligence, and deep learning in particular, has
achieved remarkable success by modeling the data from a data generating
process (DGP) directly. This is typically framed as learning the
statistical associations $P(D)$, a task made highly efficient and
scalable through the use of gradient-based optimization. However, a
primary limitation of this data-centric approach is its often poor
generalization to new tasks or out-of-distribution problems
[@peters2017elements]. Models that excel at capturing statistical
correlations within a specific dataset may fail when the underlying
system changes [@pearl2009causality]. @scholkopf2021toward argued that
the incorporation of causality can offer a promising direction to
overcome current issues in deep learning. Causality aims to explicitly
model the DGP, which is assumed to be a causal structure, commonly
represented by a Structural Causal Model (SCM). An SCM consists of two
key components:

1.  A Directed Acyclic Graph $G=(V,E)$, where vertices V represent
    variables and edges E represent direct causal dependencies.

2.  A set of causal mechanisms $f=\{f_1,\dots,f_n\}$, that specify how
    each variable is generated from its parents in the graph.

Thus, we can parameterize the full SCM as a tuple $(G,f)$. For each
variable $d_i \in D=\{d_1,\dots,d_n\}$, its value is determined by a
function of its direct causes $PA_i$ defined in $G$ and an independent
noise term $\epsilon_i$: $$\begin{aligned}
    d_i=f_i(PA_i,\epsilon_i) \quad\forall i \in D
\end{aligned}$$

While an SCM can be specified with domain knowledge, the field of causal
discovery seeks to learn this underlying structure $(G,f)$ directly from
data. This task is not straightforward. Obtaining the true SCM is
challenging because observational data alone often does not lead to a
unique solution [@murphy2023probabilistic]. This is known as the
identifiability issue, where multiple distinct SCMs - often with
different graphs - explain the same observed data distribution.
Frequentist methods for causal discovery typically return a single point
estimate $(\widehat{G,f})$ for the SCM. Such approaches can only
guarantee identifiability under strong, often untestable, assumptions
about the data. Without these assumptions, the single estimated model
may be incorrect, providing a false sense of certainty
[@murphy2023probabilistic; @bayesianDiscovery]. To address this,
Bayesian methods provide methods for explicitly modeling the uncertainty
inherent in causal discovery. Instead of seeking a single best model,
the goal is to infer the posterior distribution over all plausible SCMs.
According to Bayes' theorem, this is given by: $$\begin{aligned}
    P(G,f\mid D)=\frac{P(D\mid G,f)\cdot P(G,f)}{P(D)}
\end{aligned}$$

This distribution explicitly captures our uncertainty about the true
causal structure, allowing for robust downstream tasks like causal
effect estimation via Bayesian model averaging [@bayesianDiscovery].
However, computing this posterior is often computationally intractable.
The challenge arises from two sources: (1) the combinatorial nature of
the DAG space for G and the high-dimensional [@bayesianDiscoveryB], and
(2) the continuous parameter space of the functional mechanisms $f$
[@scholkopf2021toward]. Hence, a crucial question in modern causal
discovery is how to apply Bayesian methods efficiently
[@murphy2023probabilistic; @bayesianDiscoveryB].

This paper investigates a recent direction that uses deep learning (DL)
for approximating $P(G\mid D)$ with variational auto-encoders
$q_{\theta}(G\mid D)$ [@vae]. After pre-training, these methods enable
very efficient sampling from the graph graph posterior with competitive
performance [@avici; @meta_learning]. However, they are currently
limited to the DAG-space and many causal applications require knowledge
about the full SCM [@murphy2023probabilistic; @peters2017elements]. On
top, there is only limited knowledge about how these methods perform on
data which is not represented in the pre-training corpus
[@meta_learning]. This is especially important since deep learning
models are known for performance degradation on out-of-distribution data
[@ood]. To summarize, DL-based meta-learning promises to solve the
efficiency problem but introduce new questions about their reliability,
particularly their ability to generalize and the scope of their
inference.

# Research Question

This thesis aims to address the limitations of DL-based approaches by
investigating and advancing the methods along two key dimensions: (1)
generalization to unseen data distributions and (2) extensibility to
full posterior inference of the full SCM. Throughout the thesis, the
following research questions will be addressed:

-   **How robust are current meta-learning models for Bayesian Causal
    Discovery to specific types of out-of-distribution shifts, and how
    does their performance degradation compare to explicit inference
    methods?**

    This question aims to quantify the generalization limits of current
    meta-learning approaches. We will evaluate their robustness by
    measuring performance degradation on out-of-distribution (OOD) data,
    which will be generated by systematically shifting key aspects of
    the data-generating process like functional mechanisms, noise
    distributions, and graph structures. The performance of
    meta-learning models will be benchmarked against explicit inference
    methods to provide a clear comparison of their generalization
    capabilities.

-   **Can a VAE-based meta-learning framework be extended to approximate
    the full Structural Causal Model (SCM) posterior and how does its
    trade-off between inference efficiency and posterior accuracy
    compare against state-of-the-art methods?**

    This question addresses this gap by proposing an extension to the
    VAE-based framework to approximate the full SCM posterior. The
    success of this new model will be evaluated based on its trade-off
    between the time to generate posterior samples---and posterior
    accuracy, benchmarked against state-of-the-art explicit inference
    methods.

# Expected Results

The goal will be to assess and extend existing meta-learning approaches
to full Bayesian Causal Discovery of the joint posterior. This thesis
aims to create a variety of synthetic and real-world datasets which
allow a critical assessment of a models performance on (a) unseen data
(o.o.d.) and (b) real-world applications. Hereby, the thesis will draw
on existing arguments made by @evaluation, evaluation data used recent
scholars [@bayesdag; @meta_learning; @avici] and the creation of new
data. The evaluation is expected to answer questions posed about the
generalization abilities of DL-based models in RQ1 and is considered
successful if it clearly ranks which OOD shifts most impact each model
and determines if the meta-learning models maintain performance within a
$15\%$ degradation margin.

The second goal is to extend current meta-learning models to provide a
comprehensive posterior approximation over the entire Structural Causal
Model $P(G,f\mid D)$. The model is considered a success if it achieves
equal accuracy levels (or greater) with at least $5$x speedup. A
secondary success criterion is that graph discovery must not degrade
compared to the original graph-only model.

# Methodology and Approach

This thesis is structured into three phases, which will diagnose the
limitations of current methods and then build upon these insights to
propose and validate a novel solution.

**Phase 1: Investigating o.o.d.-performance of SOTA DL-models:** The
thesis will start by reviewing the existing literature on measuring
o.o.d-generalization [@ood; @oodSurvey] and how current methods address
the question [@avici; @meta_learning]. Based on that, datasets
specifically designed to probe model robustness under various forms of
distributional shifts will be constructed by generating synthetic data
where the ground-truth $(G,f)$ is known. This allows creating targeted
OOD scenarios by precisely manipulating components of the
data-generating process, such as altering noise distributions
$\epsilon_i$, changing functional mechanisms $f_i$, or simulating
interventions. Meta-learning models at various model checkpoints will be
compared to explicit BCD methods to quantify their generalization
performance in various settings.

**Phase 2: Extending meta-learning for full SCM estimation:** The
development process explore two primary implementation strategies. The
primary architectural hypothesis is to modify the generative model
(decoder) of the existing VAE framework [@meta_learning] to direcly
enable the sampling of $P(G,f\mid D)$. This approach would result in a
single, end-to-end differentiable model that directly learns a mapping
from a dataset D to an approximation of the full joint posterior
[@vae; @zheng2018dags]. An alternative strategy involves a hybrid model
that decouples the inference of graph structure and causal mechanisms.
In this architecture, the DL-model would first be used to efficiently
generate samples from the marginal graph posterior, $P(G\mid D)$.
Subsequently, for a set of high-probability graphs, a custom MCMC
procedure would be employed to estimate the conditional posterior of the
mechanisms [@shortChains; @GPU_lingram].

**Phase 3: Multi-Faceted Model Evaluation:** The final phase is
dedicated to a comprehensive evaluation of the model developed in Phase
2. A multi-faceted evaluation protocol will be used to assess the
model's performance from several critical perspectives, using the
benchmark suite established in Phase 1. The evaluation will involve two
main comparisons:

1.  Comparison with Explicit BCD Methods: The new model's ability to
    estimate the full SCM will be benchmarked against state-of-the-art
    explicit methods.

2.  Comparison with Meta-Learning Baselines: An essential ablation study
    will be comparing the DAG-performance of the new model against the
    original, DAG-only model. This is crucial to ensure that the added
    task of estimating the mechanisms does not degrade the accuracy of
    graph inference.

# Evaluation

The models will be evaluated using a mixed methods approach. As standard
proxy metrics can be unreliable in high-entropy settings (e.g. limited
data or non-identifiable models) [@evaluation], qualitative methods will
be used in addition to quantitative evaluations of the SCM posterior.
For benchmarking against prior work and for experiments where the true
posterior is intractable, we will employ standard proxy metrics. Thus,
we start with some ground truth graph $G(V,E)$ and sampled graphs
$\tilde{G}(\tilde{V},\tilde{E})$.

**Graph Evaluation:** The accuracy of the inferred causal graph will be
assessed by the Expected Structural Hamming Distance
(**$\mathbb{E}$-SHD**) which measures the expected number of edge
additions, removals, or reversals required to match $G$:
$$\begin{aligned}
    \mathbb{E}\text{-SHD}:=\mathbb{E}_{\tilde{G} \sim q(G\mid D)}[\text{SHD}(G,\tilde{G})]
\end{aligned}$$ Second, area under receiver operator characterisitcs
(**AUROC**) evaluates the ranking of potential edges
$\{\tilde{e_{ij}}\}\in\tilde{E}$ defind by its marginal posterior
probability:
$$P(\tilde{e_{ij}}\mid D)=\sum_{\hat{G} \in \tilde{G}:e\in\tilde{E}}P(\tilde{G}\mid D)$$
The ranking is defined by thresholding the edge probability for each
edge and calculating the true/false positive rates for different
thresholds to ontain the ROC curve.

**SCM Evaluation:** The quality of the full posterior $P(G,f\mid D)$
will be assessed with the Negative Log-Likelihood (**NIL**) on held-out
data: $$\begin{aligned}
    \text{NIL}:=-\mathbb{E}_{X \sim p_X(X)}\left[ \mathbb{E}_{Q(G,f\mid D)}\left[ \log{p(X\mid G,f)}\right] \right]
\end{aligned}$$ Additionally, the interventional Negative Log-Likelihood
(**I-NIL**) will be considered. This metric evaluates the model's
ability to predict the outcomes of unseen interventions (compared to
ground truth), providing a more robust measure which is less sensitive
to graph properties [@evaluation].

**Downstream Task Evaluation (optional):** Recognizing that proxy
metrics may fail in high-entropy settings, the thesis will also consider
evaluating the models on the downstream task of causal effect estimation
[@evaluation]. By using the inferred posterior to estimate average
treatment effects via Bayesian model averaging, a more practical measure
of the posterior's usefulness can be established.

**Qualitative Evaluation:** Beyond aggregate performance scores, a
qualitative analysis can be crucial for understanding the specific
strengths and weaknesses. By investigating the model's performance
correlates with specific properties of the ground-truth SCM, **model
failures** will be explored. For instance, it will be of interest
whether a model struggles with certain graph structures or specific
functional forms in an SCM. Inspired by the findings that existing
metrics are less reliable in high-entropy settings [@evaluation], the
**posterior entropy** will also be looked at. For instance, high-entropy
posteriors are expected when only few samples are present or in o.o.d
settings. This should decreases as more observational or interventional
data is provided.

# State of the Art

Bayesian Causal Discovery is a very recent fields where only few models
can model non-linear relationships
[@bayesdag; @meta_learning; @scholkopf2021toward]. Recent models were
able to reduce the complexity of the (discrete) DAG-permutation space by
translating it into a continuous optimization task
[@zheng2018dags; @viinikka2020towards].

Explicit Bayesian Models directly sample from $P(G,f\mid D)$ with
optimized MCMC-mehtods or variational inference . DiBS [@dibs] uses
gaussian processes to model non-linear relationships and approximates
the posterior with variational inference (VI). BayesDag [@bayesdag]
allows for a more scalable inference scheme modeling causal mechanisms
with neural networks. The model samples from the joint posterior with a
combination of VI and SG-MCMC [@sgmcmc].

Using DL for Bayesian meta-learning is an emerging paradigm with AVICI
[@avici] being the first model, which could only guarantee the
generation of a valid DAGs in expectation. BNCP [@meta_learning]
addresses the limitations with a novel decoder architecture. Both of
these architectures use transformer-based autoencoders that allows for
direct sampling of acyclic graphs. However, both generative models are
limited to providing a posterior over the graph structure, not the full
SCM.

This work aims to use **BNCP** as a starting point for adressing RQ2, as
it guarantees valid DAG samples. Evaluation will be limited to the above
stated models, since other models in BNCP fail to model nonlinearities
or scale poorly [@bayesdag; @dibs; @scholkopf2021toward].

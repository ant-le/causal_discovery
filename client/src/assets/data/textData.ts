import type { TextData, ExplainContent } from '../types.ts';

export const motivationData: Record<string, TextData[]> = {
    practical: [
        {
            category: "Healthcare",
            icon: "⚕️",
            title: "Precision Medicine & Drug Discovery",
            content:
                "By moving beyond population-level correlations, causal models can identify personalized treatment effects and accelerate drug discovery by modeling disease pathways, leading to more generalizable recommendations across diverse patient populations.",
        },
        {
            category: "Finance & Economics",
            icon: "💰",
            title: "Robust Risk & Policy Analysis",
            content:
                "Causal AI enables modeling the true impact of economic policies and assessing credit risk by disentangling correlation from causation. This leads to financial models that are more resilient to economic shocks and policy changes.",
        },
        {
            category: "Supply Chain",
            icon: "📦",
            title: "Resilience & Optimization",
            content:
                "Instead of just predicting delays, causal models identify their root causes. This allows for 'what-if' scenario planning to build operational models that can adapt to unforeseen disruptions and changing market conditions.",
        },
        {
            category: "Climate Science",
            icon: "🌍",
            title: "Attribution & Impact Modeling",
            content:
                "Attribution science uses causal models to quantify the impact of human activities on climate events. This allows for more precise and actionable insights for policy interventions and adaptation strategies.",
        },
    ],
    theoretical: [
        {
            icon: "🤖",
            title: "The LLM-Causality Tension",
            content:
                "LLMs are trained to recognize statistical correlations, not to perform causal reasoning. This creates a fundamental challenge: how to robustly leverage their vast repository of causal *statements* without being misled by their lack of true causal *reasoning*.",
        },
        {
            icon: "🎯",
            title: "Uncertainty in LLM Priors",
            content:
                "Current methods often use LLM output as a single starting point (a point estimate) for discovery algorithms. This fails to formally account for the inherent uncertainty and potential unreliability of the LLM's knowledge.",
        },
        {
            icon: "🛠️",
            title: "The PPL Tooling Gap",
            content:
                "Probabilistic Programming Languages (PPLs) are mature tools for causal *inference* (calculating effects on a known graph). However, their application to the more fundamental problem of causal *discovery* (learning the graph structure itself) is a nascent and underexplored field.",
        },
        {
            icon: "⚖️",
            title: "Ethics & Trustworthiness",
            content:
                "Building responsible AI requires navigating complex trade-offs between fairness, privacy, and accuracy. A key challenge is developing frameworks that can leverage causal insights to achieve a better balance across these competing goals.",
        },
    ],
};

export const backgroundData: TextData[] = [
    {
        title: "The Imperative of Causality for Generalizable AI",
        content:
            "Traditional machine learning excels at identifying statistical correlations but often fails to understand the underlying causal mechanisms that generate data. This reliance on correlation leads to models that are brittle, struggle with out-of-distribution (OOD) generalization, and lack true explainability. Causal AI seeks to overcome these limitations by modeling the actual data-generating process, enabling AI to reason about cause and effect, make robust decisions, and adapt to new environments.",
        topics: [
            "<strong>Spurious Correlations:</strong> Models may learn misleading associations (e.g., ice cream sales and drowning incidents) that break down in new environments, harming reliability and generalization.",
            "<strong>Out-of-Distribution (OOD) Generalization Failure:</strong> Models that rely on statistical patterns from training data falter when the deployment data follows a different distribution.",
            "<strong>Lack of Explainability:</strong> Without understanding causal pathways, models act as 'black boxes,' hindering trust and diagnosis, especially in high-stakes domains.",
            "<strong>The Causal AI Solution:</strong> By learning invariant causal mechanisms instead of superficial correlations, causal models promise improved robustness, enhanced explainability, and the ability to perform counterfactual reasoning ('what-if' scenarios).",
        ],
    },
    {
        title: "Foundational Frameworks of Causal Inference",
        content:
            "To formally reason about cause and effect, several key mathematical and conceptual frameworks are employed. These provide the language and structure needed to move beyond statistical association and model the real-world processes that generate data.",
        topics: [
            "<strong>Structural Causal Models (SCMs):</strong> An SCM is a complete mathematical specification of a causal system. It consists of variables and structural equations of the form <code>X<sub>i</sub> := f<sub>i</sub>(Pa(X<sub>i</sub>, U<sub>i</sub>)</code>, which define how each variable is causally determined by its parents (direct causes) and an independent noise term. SCMs model the actual data-generating process, enabling interventional and counterfactual queries.",
            "<strong>Directed Acyclic Graphs (DAGs):</strong> DAGs are graphical representations of SCMs, where nodes are variables and directed edges signify direct causal influence. The 'acyclic' nature ensures no variable can be its own cause. They are indispensable for visualizing causal assumptions and identifying confounding variables.",
            "<strong>Pearl's Causal Hierarchy:</strong> This framework delineates three levels of causal reasoning: <ul><li><strong>L1: Association (Seeing)</strong>, which involves observing statistical correlations (the domain of traditional ML); </li><li><strong>L2: Intervention (Doing)</strong>, which predicts the effects of actions <code>P(Y|do(X))</code>;</li><li><strong>L3: Counterfactuals (Imagining)</strong>, which reasons about hypothetical scenarios ('what would have happened if...').</li></ul>",
        ],
    },
    {
        title: "The Challenge: Structural Causal Discovery",
        content:
            "While SCMs provide a powerful language for causality, the fundamental challenge is to learn the structure of the SCM (i.e., the causal graph) from data—a process known as Causal Discovery. Automating this discovery is a primary goal for building truly generalizable AI systems, but it faces significant hurdles.",
        topics: [
            "<strong>Data Limitations:</strong> Real-world data is often imperfect, containing noise, missing values, and coming from heterogeneous sources, which can bias discovery algorithms.",
            "<strong>Observational vs. Interventional Data:</strong> Observational data is abundant but can often only identify a causal graph up to an equivalence class. Interventional data (from experiments) is more powerful but is often expensive or ethically infeasible to obtain.",
            "<strong>Computational Scalability:</strong> Many discovery algorithms struggle to scale to high-dimensional datasets and complex, non-linear relationships, making their application computationally prohibitive.",
            "<strong>Identifiability:</strong> A key challenge is ensuring that the causal model learned from data is the *only* model that could have produced that data, a property known as identifiability. Many modern methods, such as those based on deep learning, lack provable identifiability guarantees.",
        ],
    },
    {
        title: "AI Agents as 'Virtual Experts': The Role of LLMs",
        content:
            "Large Language Models (LLMs) have emerged as powerful technologies for synthesizing vast amounts of human knowledge embedded in text. In causal discovery, they can act as 'virtual domain experts,' providing prior knowledge to guide and accelerate the discovery process, a task traditionally reliant on time-consuming human consultation.",
        topics: [
            "<strong>Knowledge Synthesizer:</strong> An LLM can process thousands of documents to propose potential causal drivers in a complex system, automating the initial hypothesis generation phase.",
            "<strong>Integration with Statistical Methods:</strong> LLM-generated knowledge can be used to inform, constrain, or initialize traditional data-driven discovery algorithms. This includes using LLM outputs to formulate Bayesian priors or to provide a starting point for optimization-based methods like Differentiable Causal Discovery (DCD).",
            "<strong>The Core Tension:</strong> LLMs are autoregressive models trained on statistical correlations in text; they are not inherently causal reasoners. This creates a fundamental challenge: robustly leveraging their vast repository of causal *statements* without being misled by their lack of true causal *reasoning*.",
            "<strong>Risks and Limitations:</strong> Critics argue that using LLMs introduces unreliability and bias, and that their success may be overstated due to careful prompt engineering. They are prone to overconfidence, 'hallucinated' causal statements, and high sensitivity to prompt phrasing.",
        ],
    },
    {
        title: "Probabilistic Programming: The Framework for Principled Integration",
        content:
            "To formally integrate the imperfect knowledge from LLMs with observational data, we turn to Probabilistic Programming Languages (PPLs). PPLs provide a framework for defining probabilistic models and performing Bayesian inference, which is mathematically suited for reasoning under uncertainty.",
        topics: [
            "<strong>Bayesian Inference:</strong> The core of the approach is Bayes' rule: <code>P(G|data) &Proportional; P(data|G) + P(G)</code>. We combine a <strong>prior belief</strong> about the causal graph structure (P(G)) with the <strong>likelihood</strong> of the observed data given that graph (P(Data|G)) to compute an updated <strong>posterior belief</strong> (P(G|Data)).",
            "<strong>LLM Knowledge as a 'Soft' Prior:</strong> The central hypothesis is to translate the LLM's textual output into a formal, mathematical prior distribution over graph structures, P(G). This treats the LLM's knowledge not as infallible truth, but as an informative—yet uncertain—belief.",
            "<strong>Robustness Through Updating:</strong> By implementing this in a PPL, the framework can use observational data to update the LLM-derived prior. If the data strongly contradicts the prior, the prior's influence is naturally down-weighted, providing a safeguard against LLM fallibility.",
            "<strong>A Research Gap:</strong> While PPLs are mature tools for causal *inference* (calculating effects on a *known* graph), their use for the more fundamental problem of causal *discovery* (learning the graph itself) is a nascent and underexplored field. This thesis aims to fill that gap.",
        ],
    },
];


export const thesisData: TextData = {
    title: "LLM-enhanced Structural Causal Model Discovery within a Probabilistic Programming Framework",
    content: "This Master's thesis proposes a novel approach at the intersection of large language models, causal inference, and probabilistic programming to enhance the discovery of Structural Causal Models (SCMs). The core idea is to leverage the vast, latent knowledge within LLMs to inform the causal discovery process, moving towards more robust and generalizable AI systems.",
    topics: [
        "<h4 class='font-semibold text-lg text-slate-800 mb-2'>The Research Problem: Limitations of Current Causal Discovery</h4>",
        "Traditional SCM discovery methods often rely on strong statistical assumptions (e.g., faithfulness, acyclicity, sufficiency) or exhaustive search in high-dimensional spaces. This makes them: <ul class='list-disc list-inside ml-4 mt-2 space-y-1'><li><strong>Brittle:</strong> Sensitive to violations of assumptions, especially in real-world data that is noisy, heterogeneous, or contains hidden confounders.</li><li><strong>Inefficient:</strong> Computationally expensive for large numbers of variables, making automated discovery challenging.</li><li><strong>Knowledge-Blind:</strong> Often fail to incorporate valuable human domain expertise or prior knowledge, which is crucial for guiding discovery in complex scientific and engineering domains.</li></ul>",
        "<h4 class='font-semibold text-lg text-slate-800 mt-6 mb-2'>Proposed Solution: LLM-Guided Probabilistic SCM Discovery</h4>",
        "This thesis introduces a novel framework that synergistically combines **Large Language Models (LLMs) as 'expert agents'** with **probabilistic programming** to enhance **Structural Causal Model (SCM) discovery**. The core mechanism involves using information extracted or generated by the LLM to directly serve as a *prior distribution* for the SCM within the probabilistic programming framework. This integration offers: <ul class='list-disc list-inside ml-4 mt-2 space-y-1'><li><strong>Principled Prior Elicitation:</strong> LLMs (or LLM-powered agents) analyze textual domain knowledge, scientific literature, or expert queries to infer initial causal hypotheses (e.g., 'X likely causes Y'). These hypotheses are then formalized into a structured probabilistic prior (e.g., on graph edges or functional forms).</li><li><strong>Bayesian Inference for SCMs:</strong> A probabilistic programming language (e.g., Pyro, Stan) is used to define the SCM as a generative model. Bayesian inference then combines the LLM-derived prior with observed data to infer a robust posterior distribution over possible SCMs, quantifying uncertainty.</li><li><strong>Robustness through Knowledge Integration:</strong> The LLM's 'expert' prior guides the discovery process, especially when observational data is limited, confounded, or noisy, making the discovery process more resilient and accurate.</li></ul>",
        "<h4 class='font-semibold text-lg text-slate-800 mt-6 mb-2'>Contribution to Generalizable & Trustworthy AI</h4>",
        "This research contributes significantly to building next-generation AI by addressing critical gaps: <ul class='list-disc list-inside ml-4 mt-2 space-y-1'><li><strong>Enhanced Interpretability:</strong> Explicit SCMs are inherently interpretable, providing clear cause-and-effect relationships. The LLM's role in guiding this process adds a layer of human-understandable rationale.</li><li><strong>Improved Generalization (OOD Robustness):</strong> Causal models, once correctly discovered, are known to generalize better to out-of-distribution (OOD) scenarios and under interventions, leading to more robust AI decisions.</li><li><strong>Quantified Uncertainty:</strong> The probabilistic framework provides confidence measures for causal claims, crucial for critical applications where decisions have high stakes.</li><li><strong>Bridging Symbolic & Sub-symbolic AI:</strong> This approach offers a concrete method to combine the pattern recognition capabilities of deep learning (in LLMs) with the structured reasoning of causal inference.</li></ul>",
        "<h4 class='font-semibold text-lg text-slate-800 mt-6 mb-2'>Key Research Avenues / Thesis Chapters (Proposed)</h4>",
        "<ul class='list-disc list-inside ml-4 mt-2 space-y-1'><li><strong>Formalizing LLM-Derived Priors:</strong> Develop methodologies to convert natural language causal statements or structured outputs from LLMs into mathematically rigorous prior distributions for SCMs (e.g., over graph adjacency matrices, or functional forms).</li><li><strong>Probabilistic Model Design & Inference:</strong> Implement and evaluate the probabilistic programming model for SCM discovery using diverse synthetic and real-world datasets, demonstrating the impact of LLM priors on inference speed and accuracy.</li><li><strong>Uncertainty-Aware LLM Querying (Optional):</strong> Explore iterative methods where the probabilistic model can query the LLM for specific causal insights based on current uncertainty, allowing for active, data-efficient knowledge acquisition.</li><li><strong>Benchmarking & Evaluation:</strong> Conduct extensive experiments comparing the proposed framework against state-of-the-art causal discovery methods, particularly focusing on performance under data limitations, confounding, and OOD generalization.</li></ul>"
    ],
    explanation: [{
        title: "Creation of Domain Expert",
        explainText: `Based on the task and data at hand, and 
LLM-based  Agent</code> is created and initially tasked to retrieve relevant 
(academic) information on the task`},
    {
        title: "Generation of Candidate Graphs",
        explainText: `In order to account for unobserved variables, we use the 
<code>Agent</code> to propose sets of latent variables 
<Math expression="Y" /> based on domain knowledge.
This gives us a set of candidate graphs 
<Math expression="G= \lbrace G_1,G_2, \dots \rbrace"/>
which will be explored.`}]
};

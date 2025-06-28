import type { TextData } from '../types';

export const conceptData: TextData[] = [
    {
        title: 'Potential Outcomes Framework',
        content: `Also known as the Neyman-Rubin Potential Outcomes or the Rubin Causal Model, this framework conceptualizes causal effects by comparing hypothetical outcomes for the same individual under different treatment or intervention scenarios. The core challenge, the "fundamental problem of causal inference," is the inherent impossibility of observing both these potential outcomes simultaneously for any single individual.`
    },
    {
        title: 'Pearl\'s Causal Hierarchy',
        content: `This framework organizes causal reasoning into three distinct levels:
<ul>
<li><strong>L1: Association (Seeing):</strong> P(Y|X). Finding statistical correlations. This is the domain of most traditional machine learning.</li>
<li><strong>L2: Intervention (Doing):</strong> P(Y|do(X)). Predicting the effect of an action. This is crucial for policy making and A/B testing.</li>
<li><strong>L3: Counterfactuals (Imagining):</strong> P(Yx|X=x', Y=y'). Reasoning about what would have happened in a different world. This is essential for personalized medicine, fairness analysis, and deep explanation.</li>
</ul>`
    },
    {
        title: 'In-Depth: Directed Acyclic Graphs (DAGs)',
        content: "DAGs are the graphical counterparts to SCMs. Nodes represent variables, and directed edges (arrows) from X to Y signify that X is a direct cause of Y. The 'acyclic' property is crucial: it means there are no feedback loops (a variable cannot be its own ancestor). DAGs are powerful for identifying confounding variables (common causes) and instrumental variables, which are essential for unbiased causal effect estimation from observational data."
    },
    {
        title: 'Structural Causal Models (SCMs)',
        content: "SCMs formalize causal assumptions about a system. Each variable in the model is defined by a 'structural equation' that expresses it as a function of its direct causes and an independent error term. This explicit structure is what allows for reasoning about interventions and counterfactuals, as one can surgically modify a single equation (an intervention) and compute the downstream effects."
    },
    {
        title: 'The Causal Roadmap',
        content: `This is a systematic, itemized, and iterative process designed to guide investigators through the rigorous steps of prespecifying study design and analysis plans for causal questions. It involves defining the precise causal question, specifying a causal model, defining the causal effect of interest, describing the observed data, and meticulously assessing the plausibility of necessary assumptions.`
    },
    {
        title: 'Causal Discovery',
        content: `While SCMs define causal systems, <strong>Causal Discovery</strong> is the discipline of learning these SCMs (specifically, their underlying DAGs and functional relationships) from observed data. This is often an automated process, but challenging, especially in complex, high-dimensional scenarios. Key aspects relevant to my thesis include:
        <ul>
            <li>
                <strong>The Challenge of Automated Discovery:</strong> Traditional
                methods often struggle with unobserved confounders and identifiability.
            </li>
            <li>
                <strong>Differentiable Causal Discovery:</strong> Modern approaches
                often frame discovery as an optimization problem, which my thesis
                will build upon.
            </li>
            <li>
                <strong>Limitations of Purely Data-Driven Methods:</strong> A core
                motivation for integrating external knowledge.
            </li>
        </ul>
        `
    }
];

export const motivationData: TextData[] = [
    {
        category: 'Healthcare',
        icon: '⚕️',
        title: 'Precision Medicine & Drug Discovery',
        content: "Causal AI identifies individual treatment effects, moving beyond population averages to suggest the best treatment for a specific patient. It accelerates drug discovery by modeling disease pathways and optimizing clinical trials."
    },
    {
        category: 'Finance',
        icon: '💰',
        title: 'Risk Management & Policy',
        content: "Models the true impact of economic policies, assesses credit risk by disentangling correlation from causation, and optimizes investment portfolios by understanding the causal drivers of market movements."
    },
    {
        category: 'Supply Chain',
        icon: '📦',
        title: 'Operations Optimization',
        content: "Goos beyond predicting delays to identifying their root causes. Enables 'what-if' scenario planning for supply chain resilience and optimizes inventory by understanding causal drivers of demand."
    },
    {
        category: 'Climate',
        icon: '🌍',
        title: 'Attribution & Modeling',
        content: "Plays a key role in attribution science, quantifying the causal impact of human activities on climate events. Improves localized disease forecasting under changing climate conditions."
    },
    {
        category: 'Marketing',
        icon: '📢',
        title: 'Personalization & Attribution',
        content: "Optimizes marketing budgets by attributing sales to their true causal channels, not just the last click. Identifies drivers of customer churn and enables truly personalized recommendation systems."
    },
    {
        category: 'Social Sciences',
        icon: '🏛️',
        title: 'Policy Evaluation',
        content: "Provides tools to rigorously evaluate the effectiveness of social programs and policies, uncovering the causal impact of interventions on societal outcomes and informing evidence-based governance."
    }
];

export const challengeData: TextData[] = [
    {
        title: "Data & Confounding",
        content:
            "The fundamental challenge of unobserved confounders—hidden variables that distort causal relationships—persists. Real-world data is often messy, with missing values and noise, complicating robust inference.",
        icon: "🧬",
    },
    {
        title: "Scalability & Identifiability",
        content:
            "Many causal algorithms are computationally expensive, struggling to scale to massive datasets. Ensuring that a causal model is uniquely 'identifiable' from data is a major theoretical hurdle in complex, non-linear systems.",
        icon: "📈",
    },
    {
        title: "Ethics & Trustworthiness",
        content:
            "Building responsible AI requires navigating complex trade-offs between fairness, privacy, and accuracy. The power of causal models also carries a risk of misuse, demanding robust frameworks for human oversight and ethical deployment.",
        icon: "⚖️",
    },
];

export const researchData: Record<string, TextData> = {
    'Causal Discovery': {
        title: "Causal Discovery: Uncovering the 'Why'",
        content: "This frontier focuses on algorithms that infer causal graphs (DAGs) directly from data. It's about moving from a blank slate to a structured understanding of how a system works. The goal is to automate the difficult process of identifying cause-and-effect relationships.",
        topics: [
            "<strong>Differentiable Causal Discovery:</strong> Frame causal search as a continuous optimization problem, integrating it with deep learning but facing challenges in ensuring the discovered model is uniquely identifiable.",
            "<strong>Discovery with Imperfect Data:</strong> Develop algorithms robust to real-world challenges like missing data, noise, and mixed data types (categorical and continuous).",
            "<strong>Temporal Causal Discovery:</strong> Uncover causal links in time-series data, crucial for understanding dynamic systems like financial markets or climate.",
            "<strong>Discovery with Limited Interventions:</strong> Design active learning strategies to intelligently select the most informative experiments to run, minimizing cost while maximizing causal knowledge.",
            "<strong>Graph Neural Networks for Causal Discovery:</strong> Explore novel GNN-based probabilistic frameworks that generate probability distributions over graph spaces. Develop GNN causal explanations via neural causal models, demonstrating high accuracy in finding ground-truth explanations and enabling causal discovery at unprecedented scales (e.g., graphs of 1000 variables)."
        ]
    },
    'Causal Representation Learning': {
        title: "Causal Representation Learning (CRL)",
        content: "CRL aims to learn representations of data where the underlying, independent causal factors are disentangled. Instead of a messy, correlated representation, CRL seeks a clean one that separates 'style' from 'content', or 'spurious correlation' from 'causal mechanism'.",
        topics: [
            "<strong>Identifiable Disentanglement:</strong> Develop methods that can provably disentangle causal factors from observational or interventional data, a key theoretical challenge.",
            "<strong>Improving OOD Generalization:</b> Show how causally disentangled representations lead to models that are more robust to shifts in data distribution, making them more reliable in the real world.",
            "<strong>Enhancing Transferability:</strong> Use causal representations to transfer knowledge learned in one domain to another, more effectively than traditional transfer learning."
        ]
    },
    'Causality & LLMs': {
        title: "Causality and Large Language Models (LLMs)",
        content: "This exciting intersection explores a two-way street: using causal principles to improve LLMs, and using LLMs to accelerate causal discovery. It's about making LLMs less correlational and more rational.",
        topics: [
            "<strong>LLMs as 'Meta-Experts':</strong> Use an LLM's vast textual knowledge to generate prior causal hypotheses, providing a strong starting point for data-driven discovery algorithms.",
            "<strong>Improving LLM Robustness:</strong> Integrate causal knowledge into LLMs to make them less susceptible to reasoning failures based on spurious correlations found in training text.",
            "<strong>Causal Explanations from LLMs:</strong> Prompt LLMs to generate not just predictions, but causal explanations for their outputs, enhancing their interpretability.",
            "<strong>Measuring Causal Reasoning in LLMs:</strong> Develop benchmarks to systematically evaluate the intrinsic causal and moral judgment capabilities of different LLMs."
        ]
    },
    'Causal Reinforcement Learning': {
        title: "Causal Reinforcement Learning (CRL)",
        content: "CRL enhances traditional RL by equipping agents with a causal model of their environment. This allows them to learn more efficiently, generalize better, and make safer decisions by understanding the consequences of their actions.",
        topics: [
            "<strong>Sample-Efficient & Generalizable Policies:</strong> Use causal models to reduce the amount of trial-and-error needed for an agent to learn, and to create policies that transfer to new, unseen environments.",
            "<strong>Addressing Confounding in RL:</strong> Develop methods to handle unobserved confounders in RL settings, which can otherwise lead to biased and suboptimal policies.",
            "<strong>Counterfactuals for Safety:</strong> Design agents that can perform counterfactual reasoning to evaluate 'what-if' scenarios, improving safety and exploration in critical applications like autonomous driving."
        ]
    },
    'Trustworthy & Fair AI': {
        title: "Causality for Trustworthy & Fair AI",
        content: "This area applies causal frameworks to address critical ethical challenges in AI. It provides principled tools to identify, understand, and mitigate bias, unfairness, and lack of transparency in models.",
        topics: [
            "<strong>Counterfactual Fairness:</strong> Develop models that satisfy the criterion that a decision would remain the same even if a sensitive attribute (like race or gender) were different, all else being equal.",
            "<strong>Identifying Bias Pathways:</strong> Use causal path analysis to pinpoint exactly where and how in a model or data pipeline discrimination is introduced or amplified.",
            "<strong>Causal Explanations for Trust:</strong> Generate counterfactual explanations that give users actionable recourse (e.g., 'Your loan was denied, but if your debt-to-income ratio were 5% lower, it would have been approved')."
        ]
    },
    'My Thesis Topic': {
        title: "LLM(Agent)-enhanced Structural Causal Model discovery within a probabilistic programming framework where the info of the LLM serve as a prior for the model.", // EXACT TITLE HERE
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
        ]
    }
};

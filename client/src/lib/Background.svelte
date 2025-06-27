<script lang="ts">
    import { conceptData } from "../data/textData";
    import type { TextData } from "../types";

    // Extracting relevant background content for the thesis
    const scmConcept: TextData = conceptData["scm"];
    const dagConcept: TextData = conceptData["dag"];
    const hierarchyConcept: TextData = conceptData["hierarchy"];
</script>

<article>
    <header>
        <h2>
            Background: Pillars of LLM-Guided Probabilistic Causal Discovery
        </h2>
    </header>
    <section>
        <p>
            My Master's thesis stands at the convergence of several cutting-edge
            fields. To understand its core contributions, a grasp of Structural
            Causal Models (SCMs) and causal discovery, the capabilities of AI
            Agents (specifically Large Language Models), and the principles of
            probabilistic programming is essential.
        </p>

        <h3>1. Causal Inference and Structural Causal Models (SCMs)</h3>
        <p>
            At the heart of understanding 'why' lies Causal Inference. It's
            distinct from mere correlation, aiming to identify true
            cause-and-effect. My work focuses on discovering **Structural Causal
            Models (SCMs)**, which are the blueprint of causal systems.
        </p>
        <details>
            <summary
                ><strong>What are Structural Causal Models (SCMs)?</strong
                ></summary
            >
            <p>
                {scmConcept.content} In the context of my thesis, discovering these
                explicit structures from data is the primary objective, allowing
                us to build interpretable and interventional AI.
            </p>
        </details>
        <details>
            <summary
                ><strong
                    >Directed Acyclic Graphs (DAGs): The Visual Language</strong
                ></summary
            >
            <p>
                {dagConcept.content} My thesis aims to infer these DAGs as part of
                the SCM discovery process.
            </p>
        </details>
        <details>
            <summary
                ><strong
                    >Pearl's Causal Hierarchy: Levels of Understanding</strong
                ></summary
            >
            <p>
                {hierarchyConcept.content} My thesis operates largely at Level 2
                (Intervention) and aims to provide foundational models for Level
                3 (Counterfactuals) by correctly identifying causal mechanisms.
            </p>
        </details>

        <h4>Causal Discovery: Learning the SCM from Data</h4>
        <p>
            While SCMs define causal systems, <strong>Causal Discovery</strong> is
            the discipline of *learning* these SCMs (specifically, their underlying
            DAGs and functional relationships) from observed data. This is often
            an automated process, but challenging, especially in complex, high-dimensional
            scenarios.
        </p>
        <p>Key aspects relevant to my thesis include:</p>
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

        <h3>
            2. AI Agents as Domain Experts (Leveraging Large Language Models)
        </h3>
        <p>
            My thesis proposes to leverage the capabilities of Large Language
            Models (LLMs) not just as text generators, but as sophisticated **AI
            Agents** or "meta-experts." LLMs, through their vast training data,
            implicitly store immense amounts of semantic and relational
            knowledge about the world, including causal relationships described
            in text.
        </p>
        <p>This capacity enables LLMs to:</p>
        <ul>
            <li>
                Act as 'Meta-Experts' by generating prior causal hypotheses from
                textual descriptions.
            </li>
            <li>
                Provide structured causal insights that can guide statistical
                models.
            </li>
            <li>
                Potentially infer causal relationships from unstructured data
                that traditional methods cannot.
            </li>
        </ul>
        <p>
            In my thesis, the LLM will serve as a crucial source of *prior
            information* for the causal discovery process, reducing ambiguity
            and improving efficiency, especially when empirical data is scarce.
        </p>

        <h3>3. Probabilistic Programming</h3>
        <p>
            **Probabilistic Programming (PP)** provides the essential framework
            for formally integrating the LLM's knowledge with observed data. It
            allows us to define generative models of causal systems and then
            perform robust, uncertainty-aware inference.
        </p>
        <ul>
            <li>
                <strong>Flexible Model Specification:</strong> PP languages (like
                Pyro, Stan) allow for expressing complex Structural Causal Models,
                including their parameters and graph structure, as probabilistic
                programs.
            </li>
            <li>
                <strong>Bayesian Inference & Priors:</strong> This is fundamental
                to my thesis. PP naturally supports Bayesian inference, where the
                LLM's insights can be formalized as a *prior distribution* over possible
                SCMs or their components. This prior is then updated by observed
                data to yield a posterior distribution, quantifying our beliefs about
                the causal structure.
            </li>
            <li>
                <strong>Uncertainty Quantification:</strong> PP inherently provides
                estimates of uncertainty around learned causal relationships, which
                is vital for building trustworthy AI.
            </li>
            <li>
                <strong>Modular Design:</strong> PP fosters modularity, enabling
                the combination of different causal mechanisms and inference techniques,
                which is key for a generalized approach.
            </li>
        </ul>
        <p>
            By combining these three pillars—SCM discovery, LLM expertise, and
            probabilistic programming—my thesis aims to push the boundaries of
            automated causal discovery for more intelligent and generalizable
            AI.
        </p>
    </section>
</article>

<style>
    /* Component-specific styles for Background.svelte (Pico.css based) */
    article {
        margin-bottom: 4rem;
    }
    h2 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        color: var(--pico-primary);
    }
    h3 {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: var(--pico-secondary-hover);
    }
    h4 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: var(--pico-primary-focus);
    }
    p {
        margin-bottom: 1rem;
        line-height: 1.6;
        color: var(--pico-secondary);
    }
    ul {
        list-style: disc;
        padding-left: 1.5rem;
        margin-bottom: 1rem;
    }
    li {
        margin-bottom: 0.5rem;
        color: var(--pico-secondary);
    }
    /* Pico.css details/summary styling */
    details {
        margin-bottom: 1rem;
    }
    details summary {
        font-weight: bold;
        cursor: pointer;
        color: var(--pico-primary);
        padding: 0.5rem 0;
    }
    details[open] summary {
        color: var(--pico-primary-hover);
    }
    details p {
        margin-top: 0.5rem;
        padding-left: 1rem;
        border-left: 3px solid var(--pico-mark-background-color);
    }
</style>

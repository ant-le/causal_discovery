<script lang="ts">
    import { challengeData } from "../data/textData";
    import type { TextData } from "../types";

    const challenges: TextData[] = challengeData;
</script>

<article>
    <header>
        <h1>The Need for Intelligible & Generalizable AI</h1>
    </header>
    <section>
        <h2>Beyond Correlation: The Quest for True Intelligence</h2>
        <p>
            While current AI excels in pattern recognition and prediction, it
            often operates as a 'black box,' struggling with transparency,
            robustness, and generalization to unseen scenarios. True
            intelligence, and the next frontier of AI, demands an understanding
            of *causality*—the underlying mechanisms and 'why' behind observed
            phenomena. This thesis is driven by the imperative to build systems
            that can not only predict but also reason, intervene, and learn from
            cause-and-effect.
        </p>
        <p>
            From a theoretical standpoint, enabling AI to perform causal
            discovery is a fundamental step towards mimicking human-like
            learning and scientific reasoning. Practically, this translates into
            more reliable, fair, and actionable insights in domains where
            understanding interventions and counterfactuals is critical.
        </p>

        <h2>Addressing Core Challenges in AI Development</h2>
        <p>
            The journey towards building such advanced AI systems is fraught
            with challenges. My Master's thesis, **"LLM(Agent)-enhanced
            Structural Causal Model discovery within a probabilistic programming
            framework where the information from the LLM serves as a prior for
            the model,"** is specifically designed to tackle these head-on.
        </p>

        <div class="grid">
            {#each challenges as challenge}
                <div>
                    <summary>
                        <span class="icon">{challenge.icon}</span>
                        {challenge.title}</summary
                    >
                    <p>{challenge.content}</p>
                </div>
            {/each}
        </div>

        <h2>My Thesis: A Strategic Approach to Building Better AI</h2>
        <p>
            This thesis proposes a synergistic approach that leverages the
            strengths of multiple cutting-edge fields to overcome the
            aforementioned challenges:
        </p>
        <ul>
            <li>
                <strong>Mitigating Data Limitations & Confounding:</strong> Traditional
                causal discovery often requires vast amounts of high-quality data.
                By using Large Language Models (LLMs) as 'expert agents' to generate
                informed priors, my approach reduces the reliance on purely data-driven
                methods. This is particularly vital for domains with sparse, noisy,
                or confounded observational data, where human (or LLM-encoded) intuition
                can provide crucial initial guidance.
            </li>
            <li>
                <strong>Improving Identifiability & Scalability:</strong> Encoding
                causal knowledge as priors within a probabilistic programming framework
                allows for principled Bayesian inference. This provides a structured
                way to quantify uncertainty about the causal graph and can significantly
                prune the search space for causal discovery, enhancing both the identifiability
                of complex models and the scalability of the discovery process.
            </li>
            <li>
                <strong>Enhancing Trustworthiness & Interpretability:</strong> The
                explicit discovery of Structural Causal Models (SCMs) inherently
                leads to more interpretable AI. Furthermore, integrating LLM-derived
                knowledge fosters transparent and explainable causal insights. By
                quantifiably modeling uncertainty, the system becomes more accountable,
                contributing to the development of fairer and more traceable AI systems.
            </li>
        </ul>
        <p>
            This work aims to be a step towards constructing the next generation
            of AI: systems that are not just powerful predictors but also
            intelligent reasoners, transparent in their operations, and truly
            capable of generalizing to the complexities of the real world.
        </p>
    </section>
</article>

<style>
    .grid div {
        border-radius: 2em;
        padding: 1.2em;
        background-color: var(--pico-background-color);
        box-shadow: var(--pico-box-shadow);
        transition: transform 0.2s ease-in-out;
    }
    summary {
        color: var(--pico-primary);
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
</style>

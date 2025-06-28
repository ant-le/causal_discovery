<script lang="ts">
    import { motivationData } from "../data/textData";

    let activeView: "practical" | "theoretical" = "practical";
</script>

<article>
    <header>
        <h1>Motivation: From Correlational Patterns to Causal Understanding</h1>
    </header>
    <section>
        <p>
            The current paradigm of artificial intelligence, while powerful, is
            largely built on correlation. Models excel at learning
            statistical patterns from data but often fail when deployed in new
            environments where those patterns no longer hold.
            This brittleness stems from a lack of understanding of
            the underlying causal mechanisms that generate the data. This thesis is motivated by the conviction that the
            next frontier for AI is the development of systems that can reason
            about cause and effect, enabling them to be more robust,
            explainable, and generalizable.
        </p>

        <div class="tabs">
            <button
                class:active={activeView === "practical"}
                on:click={() => (activeView = "practical")}
            >
                Practical Applications
            </button>
            <button
                class:active={activeView === "theoretical"}
                on:click={() => (activeView = "theoretical")}
            >
                Key Theoretical Challenges
            </button>
        </div>

        <div class="grid">
            {#each motivationData[activeView] as item}
                <div class="card">
                    <div class="card-header">
                        <span class="icon">{item.icon}</span>
                        <h4>{item.title}</h4>
                    </div>
                    <p>{@html item.content}</p>
                </div>
            {/each}
        </div>

        <h2>Thesis Contribution: A Principled Fusion of Knowledge and Data</h2>
        <p>
            This thesis, <strong
                >"LLM-enhanced Structural Causal Model Discovery within a
                Probabilistic Programming Framework,"</strong
            > directly confronts these challenges to unlock the aforementioned applications.
            The core contribution is a framework that treats LLM-generated knowledge
            not as infallible truth, but as a "soft" probabilistic prior within a
            fully Bayesian discovery process. This approach is designed to:
        </p>
        <ul>
            <li>
                <strong>Robustly Handle Uncertainty:</strong> By encoding LLM output
                as a formal Bayesian prior, we can reason about uncertainty in a
                principled way. The framework allows observational data
                to override parts of the prior that it strongly contradicts, providing
                a safeguard against LLM fallibility.
            </li>
            <li>
                <strong>Address the Tooling Gap:</strong> This work pioneers the
                use of Probabilistic Programming Languages (PPLs) for the structural
                discovery task itself, moving beyond their common application in
                inference on pre-defined models and addressing a key gap in the current
                landscape.
            </li>
            <li>
                <strong>Enhance Discovery in Low-Data Regimes:</strong> By integrating
                large-scale knowledge from LLMs, this method aims to significantly
                improve the accuracy and efficiency of causal discovery, especially
                when high-quality observational data is scarce or expensive to obtain.
            </li>
        </ul>
        <p>
            Ultimately, this research seeks to develop a more robust and
            principled methodology for causal inquiry, contributing to the
            development of AI systems that are not just powerful predictors but
            also trustworthy reasoning engines.
        </p>
    </section>
</article>

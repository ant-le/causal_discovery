<script lang="ts">
    import { conceptData } from "../data/textData";
</script>

<article>
    <header>
        <h1>
            Background: Pillars of LLM-Guided Probabilistic Causal Discovery
        </h1>
    </header>
    <section>
        <p>
            My Master's thesis stands at the convergence of several cutting-edge
            fields. To understand its core contributions, a grasp of Structural
            Causal Models (SCMs) and causal discovery, the capabilities of AI
            Agents (specifically Large Language Models), and the principles of
            probabilistic programming is essential.
        </p>

        <h3>1. Causal Inference</h3>
        <p>
            At the heart of understanding 'why' lies Causal Inference. It's
            distinct from mere correlation, aiming to identify true
            cause-and-effect. My work focuses on discovering <b>Structural Causal
            Models (SCMs)</b>, which are the blueprint of causal systems.
        </p>
        {#each conceptData as concept}
            <details>
                <summary><strong>{concept.title}</strong></summary>
                <p>
                    {@html concept.content}
                </p>
            </details>
            <hr />
        {/each}

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
            <b>Probabilistic Programming (PP)</b> provides the essential framework
            for formally integrating the LLM's knowledge with observed data. It
            allows us to define generative models of causal systems and then
            perform robust, uncertainty-aware inference.
        </p>
        <ul>
            <li>
                <strong>Flexible Model Specification:</strong> PP languages like
                <kbd>Pyro</kbd> or <kbd>Stan</kbd> allow for expressing complex Structural Causal Models,
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

</style>

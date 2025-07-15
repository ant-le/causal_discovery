<script lang="ts">
    import { thesisData } from "../assets/data/textData.ts";
    import Math from "../lib/Math.svelte";
    import Summary from "../lib/Summary.svelte";
    import PageTitle from "../lib/PageTitle.svelte";
</script>


<PageTitle
    title={thesisData.title}
    subscript="The path to Strong AI"
    desc={thesisData.content}
/>

<section id="start">
    <p>
        Assume a <code>DAG</code> parametrised by the graph parameters <Math
            expression="\theta"
        /> and potential latent variables <Math expression="Z" />, which are
        either proposed by the <code>Agent</code> or by a search algorithm. We
        observe input data <Math expression="X" />. We estimate the probability
        distribution over all possible graphs as: <Math
            expression="P(\theta \mid X) \propto P(X \mid \theta) \cdot P(\theta)"
            inline={false}
        />
    </p>
    <p>
        I hope that the novelty could be the unique combination of LLM with PPL
        and Bayesian Inference. The entire workflow would be:
    </p>
    <Summary items={thesisData.explanation}/>
    <article>
        <header>Workflow Summary</header>
        {#each thesisData.explanation as item, idx}
            <details name="example">
                <summary><strong>{idx + 1}. {item.title}</strong></summary>
                <p>{item.explainText}</p>
            </details>
            {#if idx < thesisData.explanation.length - 1}
                <hr />
            {/if}
        {/each}
    </article>

    <ul>
        <li>Comparing different models</li>
        <li>Generating candidate DAGs</li>
    </ul>
</section>

<section id="Agents">
    <p>
        The Lit Review has proposed LLMS for different subjects, we will use
        more specialised <code>Agents</code> for two main tasks:
    </p>
    <ul>
        <li>
            <b
                >Propose candidate DAG structures: Add potential cofounders as
                latent vars</b
            >
        </li>
        <li>Generate Priors graph parameters</li>
    </ul>
</section>

<section id="Inference">
    <p>
        There are many approaches leveraging prop programming for Causal
        discovery. The advantage of Bayesia methods is that they do not simply
        specify a point estimate (graph), but that they spcify a probability
        distribution over all possible graphs. Opions I am interested in
        exploring would be:
    </p>
    <ul>
        <li>
            Using GPU-accelerated MCMC on existing MCMC-algs, as learning
            distribution is computationally very expensive
        </li>
        <li>Generate Priors graph parameters</li>
        <li>
            Enables active learning of graph (how can active learning be
            extended to adding removing latents or other observed data)
        </li>
        <li>
            Especially when working with LLM prone to halluscination, fully
            bayesian view on DAG is helpful, as it helps to fully capture
            uncertainty
        </li>
    </ul>
</section>

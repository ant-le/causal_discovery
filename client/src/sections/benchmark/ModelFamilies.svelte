<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
</script>

<section>
    <article>
        <header>Benchmarked Model Families</header>
        <ContentStatus status="implemented" text="Implemented in model factory" />
        <p>
            The current implementation evaluates four Bayesian causal discovery
            model families behind one shared model interface.
        </p>
    </article>

    <div class="grid cards">
        <article>
            <header>Avici</header>
            <p>Amortized causal discovery with fast test-time inference.</p>
            <p><small><Cite key="avici" /></small></p>
        </article>

        <article>
            <header>BCNP</header>
            <p>Meta-learned Bayesian posterior modeling under task shifts.</p>
            <p><small><Cite key="bcnp" /></small></p>
        </article>

        <article>
            <header>DiBS</header>
            <p>Explicit differentiable posterior optimization over structures.</p>
            <p><small><Cite key="dibs" /></small></p>
        </article>

        <article>
            <header>BayesDAG</header>
            <p>Explicit Bayesian graph and mechanism inference pipeline.</p>
            <p><small><Cite key="bayesdag" /></small></p>
        </article>
    </div>
</section>

<style>
    .cards {
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }

    .cards article {
        margin: 0;
    }
</style>

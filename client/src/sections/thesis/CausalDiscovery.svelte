<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <article>
        <header>Method Objective</header>
        <p>
            The thesis pipeline implements Bayesian causal discovery as a
            configurable sequence: setup, data construction, model dispatch,
            inference, and evaluation.
        </p>

        <Math
            expression={"p(G,\\Theta\\mid D) \\propto p(D\\mid G,\\Theta)\\,p(G,\\Theta)"}
            inline={false}
        />
    </article>

    <article>
        <header>Execution Stages</header>
        <ul>
            <li>
                A runtime layer manages distributed setup, logging, routing, and
                teardown.
            </li>
            <li>
                Dataset modules generate reproducible train/validation/test task
                families.
            </li>
            <li>
                A shared model interface supports Avici, BCNP, DiBS, and
                BayesDAG <Cite key="avici" /><Cite key="bcnp" /><Cite
                    key="dibs"
                /><Cite key="bayesdag" />.
            </li>
            <li>
                Evaluation combines graph-focused and SCM-focused reporting.
            </li>
        </ul>
    </article>
</section>

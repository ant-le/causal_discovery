<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <p>
        The pipeline implements Bayesian causal discovery as a configurable
        sequence: config validation, dataset construction, model dispatch,
        pre-training or explicit inference, and evaluation.
    </p>

    <Math
        expression={"p(G,\\Theta\\mid D) \\propto p(D\\mid G,\\Theta)\\,p(G,\\Theta)"}
        inline={false}
    />

    <ul>
        <li>
            <code>src/causal_meta/main.py</code> orchestrates distributed setup,
            logging, task routing, and teardown.
        </li>
        <li>
            <code>src/causal_meta/datasets/</code> builds reproducible
            train/val/test families with rank-aware sampling.
        </li>
        <li>
            <code>src/causal_meta/models/</code> exposes a common interface for
            Avici, BCNP, DiBS, and BayesDAG <Cite key="avici" /><Cite
                key="bcnp"
            /><Cite key="dibs" /><Cite key="bayesdag" />.
        </li>
        <li>
            <code>src/causal_meta/runners/metrics/</code> provides graph and SCM
            metrics used in benchmark reports.
        </li>
    </ul>
</section>

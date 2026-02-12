<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
</script>

<section>
    <p>
        <strong>RQ1</strong> asks how robust amortized Bayesian causal discovery
        models are under controlled out-of-distribution shifts <Cite key="bcnp"
        />.
    </p>

    <ul>
        <li>
            Compare meta-learned models (Avici, BCNP) with explicit inference
            baselines (DiBS, BayesDAG) <Cite key="avici" /><Cite key="dibs" /><Cite
                key="bayesdag"
            />.
        </li>
        <li>
            Stress-test shifts in mechanism class, graph family, and noise
            configuration.
        </li>
        <li>
            Measure degradation with graph metrics plus posterior-oriented
            metrics to avoid over-trusting point-estimate behavior <Cite
                key="evaluation"
            />.
        </li>
    </ul>

    <p>
        In this repository, the benchmark pipeline is implemented in
        <code>src/causal_meta/main.py</code> and evaluated through dedicated
        runner tasks for inference and evaluation.
    </p>
</section>

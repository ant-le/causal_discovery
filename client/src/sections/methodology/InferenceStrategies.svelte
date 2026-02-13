<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import InferenceComparison from "../../lib/InferenceComparison.svelte";
</script>

<section>
    <article>
        <header>Two Inference Paths</header>
        <ContentStatus status="implemented" text="Implemented model families" />
        <p>
            The benchmark compares amortized causal predictors against explicit,
            dataset-specific inference procedures.
        </p>
        <ul>
            <li>
                Amortized models transfer learned structure priors across tasks
                (Avici, BCNP) <Cite key="avici" /><Cite key="bcnp" />.
            </li>
            <li>
                Explicit approaches optimize each dataset posterior directly
                (DiBS, BayesDAG) <Cite key="dibs" /><Cite key="bayesdag" />.
            </li>
        </ul>
    </article>

    <article>
        <header>Why This Split Matters</header>
        <p>
            This distinction drives the thesis trade-off: transfer speed and
            throughput versus per-dataset optimization quality under shift.
        </p>
    </article>

    <InferenceComparison />
</section>

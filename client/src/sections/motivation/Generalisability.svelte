<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ShiftExplorer from "../../lib/ShiftExplorer.svelte";
</script>

<section>
    <article>
        <header>RQ1: OOD Robustness</header>
        <p>
            <strong>RQ1</strong> asks how robust amortized Bayesian causal
            discovery is under controlled out-of-distribution shifts <Cite
                key="bcnp"
            />.
        </p>
    </article>

    <article>
        <header>Evaluation Logic</header>
        <ul>
            <li>
                Compare meta-learned methods (Avici, BCNP) and explicit
                inference baselines (DiBS, BayesDAG) <Cite key="avici" /><Cite
                    key="dibs"
                /><Cite key="bayesdag" />.
            </li>
            <li>
                Stress-test mechanism shifts, graph-family shifts, and noise
                shifts with the same pipeline protocol.
            </li>
            <li>
                Track both graph quality and posterior quality to avoid
                over-trusting point estimates <Cite key="evaluation" />.
            </li>
        </ul>
    </article>

    <ShiftExplorer />
</section>

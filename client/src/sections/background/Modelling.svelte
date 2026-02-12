<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <article>
        <header>Modeling Objective</header>
        <p>
            The benchmark treats causal discovery as posterior inference over
            SCM hypotheses, not as a single graph prediction problem.
        </p>
        <Math
            expression={"p(D, G, \Theta) = p(D \mid G, \Theta)\,p(G, \Theta)"}
            inline={false}
        />
        <p>
            Here, <Math expression="G" /> denotes the DAG and
            <Math expression="\Theta" /> mechanism parameters; inference targets
            <Math expression="p(G, \Theta \mid D)" />.
        </p>
    </article>

    <article>
        <header>Evaluation Consequences</header>
        <ul>
            <li>
                Finite data typically leaves multiple plausible causal
                structures.
            </li>
            <li>
                Robustness under shift depends on mechanism assumptions, not only
                edge recovery.
            </li>
            <li>
                Benchmarking therefore combines graph metrics and
                intervention-aware posterior diagnostics.
            </li>
        </ul>
    </article>

    <details>
        <summary><strong>Modeling Assumptions (compact)</strong></summary>
        <ul>
            <li>Acyclic structural assignments induce a valid causal graph.</li>
            <li>Priors encode structure and mechanism inductive bias.</li>
            <li>Likelihood terms link SCM candidates to observed datasets.</li>
        </ul>
    </details>
</section>

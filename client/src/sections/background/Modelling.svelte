<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <p>
        The project treats causal discovery as posterior inference over SCMs,
        not as a single graph prediction task.
    </p>

    <article>
        <header>Model Factorization</header>
        <Math
            expression={"p(D, G, \Theta) = p(D \mid G, \Theta)\,p(G, \Theta)"}
            inline={false}
        />
        <p>
            Here, <Math expression="G" /> denotes the DAG and
            <Math expression="\Theta" /> the mechanism parameters. Inference
            targets <Math expression="p(G, \Theta \mid D)" />.
        </p>
    </article>

    <article>
        <header>Why This Matters in Practice</header>
        <ul>
            <li>
                Finite data often leaves several plausible graphs; posterior
                uncertainty should be preserved.
            </li>
            <li>
                Shift robustness depends on mechanism assumptions, not only edge
                recovery.
            </li>
            <li>
                Evaluation therefore combines structure metrics with
                intervention-aware diagnostics.
            </li>
        </ul>
    </article>
</section>

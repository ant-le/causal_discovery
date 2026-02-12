<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
    import InterventionExplorer from "../../lib/InterventionExplorer.svelte";
</script>

<section>
    <article>
        <header>SCM Definition</header>
        <p>
            A Structural Causal Model (SCM) represents each variable as a
            function of its parents and exogenous noise.
        </p>

        <Math
            expression={"X_i := f_i(PA_i, U_i),\quad U_i \perp U_j\;\text{for }i\neq j"}
            inline={false}
        />

        <p>
            Under acyclicity, these assignments induce a DAG and a factorized
            observational distribution.
        </p>

        <Math
            expression={"p(X_1,\ldots,X_n)=\prod_{i=1}^n p(X_i\mid PA_i)"}
            inline={false}
        />
    </article>

    <article>
        <header>Why SCMs Here</header>
        <p>
            In this thesis, discovery means inferring uncertainty over both
            structure and mechanisms from finite data, rather than committing to
            one deterministic graph <Cite key="causalDiscovery" />.
        </p>
    </article>

    <InterventionExplorer />
</section>

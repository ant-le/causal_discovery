<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <p>
        A Structural Causal Model (SCM) represents each variable as a function
        of its parents and exogenous noise.
    </p>

    <Math
        expression={"X_i := f_i(PA_i, U_i),\quad U_i \perp U_j\;\text{for }i\neq j"}
        inline={false}
    />

    <p>
        If the structural assignments are acyclic, they induce a DAG and imply
        a factorized observational distribution:
    </p>

    <Math
        expression={"p(X_1,\ldots,X_n)=\prod_{i=1}^n p(X_i\mid PA_i)"}
        inline={false}
    />

    <p>
        In this thesis context, causal discovery means inferring both graph
        uncertainty and mechanism uncertainty from finite data <Cite
            key="causalDiscovery"
        />.
    </p>
</section>

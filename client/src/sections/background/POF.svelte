<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <p>
        The Potential Outcomes Framework (POF) expresses causal effects through
        counterfactual quantities for the same unit under different treatments.
    </p>

    <Math expression={"\text{ATE} = \mathbb{E}[Y(1)-Y(0)]"} inline={false} />

    <p>
        POF is useful for clarifying estimands, while SCMs provide the
        graph-based generative assumptions used in this codebase. Together they
        motivate evaluation beyond edge recovery: intervention behavior matters.
    </p>
</section>

<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <article>
        <header>Potential Outcomes View</header>
        <p>
            The Potential Outcomes Framework expresses effects via
            counterfactual outcomes for the same unit under alternative
            treatments.
        </p>

        <Math expression={"\text{ATE} = \mathbb{E}[Y(1)-Y(0)]"} inline={false} />
    </article>

    <article>
        <header>How It Complements SCMs</header>
        <ul>
            <li>POF clarifies estimands and target causal questions.</li>
            <li>SCMs specify generative assumptions and intervention structure.</li>
            <li>
                Combined, they justify evaluating intervention behavior and
                posterior quality beyond edge recovery.
            </li>
        </ul>
    </article>
</section>

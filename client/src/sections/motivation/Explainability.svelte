<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
</script>

<section>
    <article>
        <header>RQ2: Full-SCM Approximation</header>
        <p>
            <strong>RQ2</strong> moves from graph-only outputs toward richer
            posterior objects that reflect uncertainty over both graph structure
            and mechanisms.
        </p>
    </article>

    <article>
        <header>What Changes in Practice</header>
        <ul>
            <li>
                Calibrated uncertainty remains a first-class target, not only
                average score.
            </li>
            <li>
                Runtime and memory growth must be measured when posterior
                families become richer <Cite key="bcnp" />.
            </li>
            <li>
                Graph-level quality should be preserved while expanding support
                for intervention-centric evaluation <Cite key="evaluation" />.
            </li>
        </ul>
    </article>

    <article>
        <header>Takeaway</header>
        <p>
            The objective is not only to predict plausible edges, but to produce
            posterior information that remains useful for downstream decisions
            under uncertainty.
        </p>
    </article>
</section>

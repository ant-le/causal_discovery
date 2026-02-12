<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
</script>

<section>
    <article>
        <header>Problem</header>
        <p>
            Predictive pipelines can fail when deployment data differs from
            training assumptions. The main risk is silent failure: strong
            confidence but weak transfer under shift.
        </p>
    </article>

    <article>
        <header>Thesis Direction</header>
        <p>
            The project treats causal discovery as uncertainty-aware inference
            over plausible structures and mechanisms, so robustness questions can
            be tested directly rather than assumed.
        </p>
    </article>

    <article>
        <header>Why Causality</header>
        <p>
            The motivation follows the view that causal models are needed for
            intervention planning and robust reasoning beyond correlation-based
            prediction <Cite key="bookOfWhy" />.
        </p>
    </article>
</section>

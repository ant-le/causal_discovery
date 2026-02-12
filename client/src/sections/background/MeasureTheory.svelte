<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section>
    <article>
        <header>Appendix Goal</header>
        <p>
            Measure theory provides the foundation for probability spaces,
            likelihood integrals, and expectations used throughout Bayesian
            inference.
        </p>
    </article>

    <article>
        <header>Building Blocks</header>
        <p>
            A measure space
            <Math expression={"(X, \mathcal{A}, \mu)"} />
            combines measurable events and a consistent notion of size. This is
            the formal layer beneath probability models.
        </p>
    </article>

    <details open>
        <summary><strong>Core Definitions</strong></summary>
        <Definition key="sigma_algebra" />
        <Definition key="measure" />
        <Definition key="integral" />
    </details>

    <details>
        <summary><strong>Why It Matters for Causal Discovery</strong></summary>
        <ul>
            <li>Posterior formulas rely on measurable likelihood terms.</li>
            <li>
                Integrals over latent variables and parameters require rigorous
                measure definitions.
            </li>
            <li>
                Convergence guarantees justify approximations in practical
                inference workflows.
            </li>
        </ul>
    </details>
</section>

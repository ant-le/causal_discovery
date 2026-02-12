<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section>
    <article>
        <header>Appendix Goal</header>
        <p>
            This appendix refreshes the calculus concepts used for optimization,
            smooth model behavior, and gradient-based inference.
        </p>
    </article>

    <article>
        <header>Function Setup</header>
        <p>
            We consider maps of the form
            <Math expression={"f: \mathbb{R}^n \rightarrow \mathbb{R}^m"} />.
            Continuity and differentiability describe local behavior around a
            point, while gradients summarize first-order sensitivity.
        </p>
    </article>

    <details open>
        <summary><strong>Core Definitions</strong></summary>
        <Definition key="convergence" />
        <Definition key="continuity" />
        <Definition key="differentiable" />
        <Definition key="gradient" />
    </details>

    <details>
        <summary><strong>Why It Matters for This Thesis</strong></summary>
        <ul>
            <li>Gradient information supports efficient parameter updates.</li>
            <li>Local smoothness assumptions influence optimization stability.</li>
            <li>
                Differential structure underpins practical MCMC and variational
                methods used in Bayesian discovery.
            </li>
        </ul>
    </details>
</section>

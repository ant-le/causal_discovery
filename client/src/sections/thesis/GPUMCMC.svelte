<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ConvergenceSignals from "../../lib/ConvergenceSignals.svelte";
</script>

<section>
    <article>
        <header>Acceleration Goal</header>
        <p>
            GPU acceleration is essential when Bayesian workflows require many
            posterior samples, chains, or task evaluations.
        </p>
    </article>

    <article>
        <header>What Scales</header>
        <ul>
            <li>
                Parallel hardware enables chain-level and batch-level speedups
                for gradient-based MCMC workflows <Cite key="gpuMcmc" />.
            </li>
            <li>
                Vectorized tensor execution reduces Python overhead and improves
                throughput.
            </li>
            <li>
                Distributed runtime controls keep behavior consistent across
                single-GPU and multi-process runs.
            </li>
        </ul>
    </article>

    <article>
        <header>Quality Guardrails</header>
        <p>
            Speed is not sufficient on its own. Short-chain strategies still
            require robust convergence diagnostics and stability checks to remain
            scientifically reliable <Cite key="shortChains" />.
        </p>
    </article>

    <ConvergenceSignals />
</section>

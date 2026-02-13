<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import ConvergenceSignals from "../../lib/ConvergenceSignals.svelte";

    let showExplainers = $state(false);
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

    <article>
        <header>Explainers</header>
        <ContentStatus
            status="illustrative"
            text="Optional diagnostic intuition panel"
        />
        <p>
            Measured results stay primary. Open explainers only when you want a
            conceptual view of convergence behavior.
        </p>
        <button
            class="secondary outline"
            aria-pressed={showExplainers}
            onclick={() => {
                showExplainers = !showExplainers;
            }}
        >
            {showExplainers ? "Hide explainers" : "Show explainers"}
        </button>
    </article>

    {#if showExplainers}
        <ConvergenceSignals />
    {/if}
</section>

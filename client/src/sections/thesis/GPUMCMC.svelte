<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
</script>

<section>
    <p>
        GPU acceleration is critical for practical Bayesian inference when many
        posterior samples, chains, or datasets must be processed.
    </p>

    <ul>
        <li>
            Parallel hardware enables chain-level and batch-level speedups for
            gradient-based MCMC workflows <Cite key="gpuMcmc" />.
        </li>
        <li>
            Vectorized tensor operations in PyTorch reduce Python overhead and
            improve throughput.
        </li>
        <li>
            Benchmark quality still depends on diagnostics, not raw speed:
            short-chain strategies must be paired with robust convergence checks <Cite
                key="shortChains"
            />.
        </li>
    </ul>

    <p>
        In this repository, distributed execution and device placement are
        managed centrally in <code>src/causal_meta/main.py</code> and
        <code>src/causal_meta/runners/utils/distributed.py</code>.
    </p>
</section>

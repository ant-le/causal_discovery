<script lang="ts">
    import { onMount } from "svelte";
    import BenchmarkOutputs from "../../lib/BenchmarkOutputs.svelte";
    import ShiftExplorer from "../../lib/ShiftExplorer.svelte";
    import MetricCatalog from "../../lib/MetricCatalog.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import { loadRunMetrics, type LoadedRunMetrics } from "../../lib/metricsRuntime.ts";

    let runMetrics = $state<LoadedRunMetrics | null>(null);
    let isLoading = $state(true);
    let showExplainers = $state(false);

    onMount(async () => {
        runMetrics = await loadRunMetrics();
        isLoading = false;
    });

    let hasMeasuredData = $derived(runMetrics !== null);
</script>

<section>
    <article>
        <header>Current Results Surface</header>
        <ContentStatus
            status={hasMeasuredData ? "measured" : "implemented"}
            text={hasMeasuredData
                ? "Measured from available run artifacts"
                : "Measured mode awaiting run artifact"}
        />
        <p>
            This section reports only currently available benchmark outputs:
            graph-quality metrics, posterior-oriented metrics, and reporting
            coverage from the latest accessible run artifacts.
        </p>
    </article>

    {#if isLoading}
        <article aria-busy="true">
            <header>Loading Results</header>
            <p>Checking for benchmark run artifacts...</p>
        </article>
    {:else if hasMeasuredData}
        <BenchmarkOutputs />
        <MetricCatalog />
    {:else}
        <article>
            <header>No Measured Artifact Detected</header>
            <p>
                Measured result panels stay hidden until a valid
                <code>metrics.json</code> artifact is available.
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
            <article>
                <header>Explainer Mode</header>
                <ContentStatus
                    status="illustrative"
                    text="Concept visuals, not measured output"
                />
                <p>
                    These panels illustrate expected behavior and metric meaning
                    while run-backed measurements are unavailable.
                </p>
            </article>
            <BenchmarkOutputs />
            <MetricCatalog />
            <ShiftExplorer />
        {/if}
    {/if}

    <article>
        <header>Interpretation Rule</header>
        <p>
            If a run artifact is unavailable, the visuals switch to clearly
            labeled illustrative fallback mode instead of claiming measured
            performance.
        </p>
    </article>
</section>

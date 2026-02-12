<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import { onMount } from "svelte";
    import {
        loadRunMetrics,
        metricValue,
        type LoadedRunMetrics,
    } from "./metricsRuntime.ts";

    interface MetricItem {
        key: string;
        label: string;
        family: "Graph structure" | "Intervention mechanism";
        purpose: string;
    }

    const metricItems: MetricItem[] = [
        {
            key: "e-shd",
            label: "Expected SHD",
            family: "Graph structure",
            purpose: "Average structural mismatch over posterior graph samples.",
        },
        {
            key: "e-edgef1",
            label: "Expected edge F1",
            family: "Graph structure",
            purpose: "Posterior-averaged precision/recall balance for edges.",
        },
        {
            key: "ancestor_f1",
            label: "Ancestor F1",
            family: "Graph structure",
            purpose: "Recovery quality of ancestor-descendant relationships.",
        },
        {
            key: "e-sid",
            label: "Expected SID",
            family: "Graph structure",
            purpose: "Interventional-structure distance induced by graph errors.",
        },
        {
            key: "auc",
            label: "AUC",
            family: "Graph structure",
            purpose: "Ranking quality for edge probabilities under uncertainty.",
        },
        {
            key: "graph_nll",
            label: "Graph NLL",
            family: "Graph structure",
            purpose: "Log-probability fit of true edges under posterior belief.",
        },
        {
            key: "edge_entropy",
            label: "Edge entropy",
            family: "Graph structure",
            purpose: "Uncertainty spread over potential graph edges.",
        },
        {
            key: "inil",
            label: "I-NIL",
            family: "Intervention mechanism",
            purpose: "Interventional likelihood score for mechanism-level behavior.",
        },
    ];

    type FilterFamily = "All" | MetricItem["family"];
    let filterFamily = $state<FilterFamily>("All");
    let runMetrics = $state<LoadedRunMetrics | null>(null);
    let selectedDataset = $state("");

    onMount(async () => {
        runMetrics = await loadRunMetrics();
    });

    $effect(() => {
        if (runMetrics && !selectedDataset) {
            selectedDataset = runMetrics.datasets[0].dataset;
        }
    });

    let datasetOptions = $derived(runMetrics ? runMetrics.datasets : []);
    let selectedSummary: Record<string, number> | null = $derived(
        datasetOptions.find((entry) => entry.dataset === selectedDataset)?.metrics ?? null,
    );

    let filteredItems = $derived(
        filterFamily === "All"
            ? metricItems
            : metricItems.filter((item) => item.family === filterFamily),
    );

    const visibleCount = tweened(0, { duration: 240, easing: cubicOut });

    $effect(() => {
        visibleCount.set(filteredItems.length);
    });

    function formatMetricValue(value: number | null): string {
        if (value === null) {
            return "-";
        }

        if (Math.abs(value) < 10) {
            return value.toFixed(3);
        }

        return value.toFixed(2);
    }
</script>

<article>
    <header>
        <strong>Metrics Catalog</strong>
        <br />
        {#if selectedSummary}
            <small>Showing live values for dataset split: {selectedDataset}.</small>
        {:else}
            <small>Schema view; live values appear automatically when `metrics.json` is available.</small>
        {/if}
    </header>

    {#if datasetOptions.length > 0}
        <label
            >Dataset split
            <select bind:value={selectedDataset}>
                {#each datasetOptions as entry}
                    <option value={entry.dataset}>{entry.dataset}</option>
                {/each}
            </select>
        </label>
    {/if}

    <label
        >Filter family
        <select bind:value={filterFamily}>
            <option value="All">All</option>
            <option value="Graph structure">Graph structure</option>
            <option value="Intervention mechanism">Intervention mechanism</option>
        </select>
    </label>

    <p>
        Visible metrics: <strong>{Math.round($visibleCount)}</strong> / {metricItems.length}
    </p>
    <progress max={metricItems.length} value={$visibleCount}></progress>

    <div class="overflow-auto">
        <table>
            <thead>
                <tr>
                    <th scope="col">Metric</th>
                    <th scope="col">Current Value</th>
                    <th scope="col">Family</th>
                    <th scope="col">What It Captures</th>
                </tr>
            </thead>
            <tbody>
                {#each filteredItems as item}
                    <tr>
                        <td><code>{item.key}</code> ({item.label})</td>
                        <td>{formatMetricValue(metricValue(selectedSummary ?? {}, item.key))}</td>
                        <td>{item.family}</td>
                        <td>{item.purpose}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    </div>
</article>

<style>
    .overflow-auto {
        overflow-x: auto;
    }
</style>

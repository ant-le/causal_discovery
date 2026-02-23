<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import { onMount } from "svelte";
    import {
        loadRunMetrics,
        metricValue,
        metricSem,
        type LoadedRunMetrics,
        type ModelMetrics,
    } from "./metricsRuntime.ts";
    import ContentStatus from "./ContentStatus.svelte";

    interface MetricItem {
        key: string;
        label: string;
        family: "Graph structure" | "Intervention mechanism";
        purpose: string;
        direction: "lower" | "higher";
    }

    const metricItems: MetricItem[] = [
        {
            key: "e-shd",
            label: "Expected SHD",
            family: "Graph structure",
            purpose: "Average structural mismatch over posterior graph samples.",
            direction: "lower",
        },
        {
            key: "e-edgef1",
            label: "Expected edge F1",
            family: "Graph structure",
            purpose: "Posterior-averaged precision/recall balance for edges.",
            direction: "higher",
        },
        {
            key: "ancestor_f1",
            label: "Ancestor F1",
            family: "Graph structure",
            purpose: "Recovery quality of ancestor-descendant relationships.",
            direction: "higher",
        },
        {
            key: "e-sid",
            label: "Expected SID",
            family: "Graph structure",
            purpose: "Interventional-structure distance induced by graph errors.",
            direction: "lower",
        },
        {
            key: "auc",
            label: "AUC",
            family: "Graph structure",
            purpose: "Ranking quality for edge probabilities under uncertainty.",
            direction: "higher",
        },
        {
            key: "graph_nll",
            label: "Graph NLL",
            family: "Graph structure",
            purpose: "Log-probability fit of true edges under posterior belief.",
            direction: "lower",
        },
        {
            key: "edge_entropy",
            label: "Edge entropy",
            family: "Graph structure",
            purpose: "Uncertainty spread over potential graph edges.",
            direction: "lower",
        },
        {
            key: "inil",
            label: "I-NIL",
            family: "Intervention mechanism",
            purpose:
                "Interventional likelihood score for mechanism-level behavior.",
            direction: "lower",
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
            // Pick the first dataset common across all models
            const first = runMetrics.models[0]?.datasets[0]?.dataset;
            if (first) selectedDataset = first;
        }
    });

    let isMultiModel = $derived(
        runMetrics !== null && runMetrics.models.length > 1,
    );
    let modelNames = $derived(
        runMetrics ? runMetrics.models.map((m) => m.model) : [],
    );
    let datasetOptions = $derived(() => {
        if (!runMetrics) return [] as string[];
        // Union of all dataset names across models
        const seen = new Set<string>();
        for (const m of runMetrics.models) {
            for (const d of m.datasets) seen.add(d.dataset);
        }
        return [...seen];
    });

    function getModelDataset(
        model: ModelMetrics,
        dataset: string,
    ): Record<string, number> | null {
        return (
            model.datasets.find((d) => d.dataset === dataset)?.metrics ?? null
        );
    }

    let filteredItems = $derived(
        filterFamily === "All"
            ? metricItems
            : metricItems.filter((item) => item.family === filterFamily),
    );

    const visibleCount = tweened(0, { duration: 240, easing: cubicOut });

    $effect(() => {
        visibleCount.set(filteredItems.length);
    });

    function formatVal(value: number | null): string {
        if (value === null) return "-";
        if (Math.abs(value) < 10) return value.toFixed(3);
        return value.toFixed(2);
    }

    function formatSem(sem: number | null): string {
        if (sem === null) return "";
        return `±${sem < 10 ? sem.toFixed(3) : sem.toFixed(2)}`;
    }

    function isBest(
        item: MetricItem,
        modelIdx: number,
        models: ModelMetrics[],
        dataset: string,
    ): boolean {
        const values = models.map((m) => {
            const d = getModelDataset(m, dataset);
            return d ? metricValue(d, item.key) : null;
        });
        const val = values[modelIdx];
        if (val === null) return false;
        const validValues = values.filter(
            (v): v is number => v !== null,
        );
        if (validValues.length < 2) return false;
        if (item.direction === "higher") return val >= Math.max(...validValues);
        return val <= Math.min(...validValues);
    }
</script>

<article>
    <header>
        <strong>Metrics Catalog</strong>
        <br />
        <ContentStatus
            status={runMetrics ? "measured" : "implemented"}
            text={runMetrics
                ? `Measured from ${runMetrics.models.length} model(s)`
                : "Implemented metric schema"}
        />
        <br />
        {#if runMetrics}
            <small
                >Showing run data from <code>{runMetrics.source}</code>.
                {#if isMultiModel}Best values per metric are
                    <strong>highlighted</strong>.{/if}</small
            >
        {:else}
            <small
                >Schema view; live values appear automatically when
                <code>metrics.json</code> is available.</small
            >
        {/if}
    </header>

    <div class="grid controls">
        {#if datasetOptions().length > 0}
            <label
                >Dataset split
                <select bind:value={selectedDataset}>
                    {#each datasetOptions() as ds}
                        <option value={ds}>{ds}</option>
                    {/each}
                </select>
            </label>
        {/if}

        <label
            >Filter family
            <select bind:value={filterFamily}>
                <option value="All">All</option>
                <option value="Graph structure">Graph structure</option>
                <option value="Intervention mechanism"
                    >Intervention mechanism</option
                >
            </select>
        </label>
    </div>

    <p>
        Visible metrics: <strong>{Math.round($visibleCount)}</strong> / {metricItems.length}
    </p>
    <progress max={metricItems.length} value={$visibleCount}></progress>

    <div class="overflow-auto">
        <table>
            <thead>
                <tr>
                    <th scope="col">Metric</th>
                    <th scope="col">↕</th>
                    {#if isMultiModel}
                        {#each modelNames as name}
                            <th scope="col">{name}</th>
                        {/each}
                    {:else if runMetrics}
                        <th scope="col">Value</th>
                    {:else}
                        <th scope="col">Value</th>
                    {/if}
                    <th scope="col">What It Captures</th>
                </tr>
            </thead>
            <tbody>
                {#each filteredItems as item}
                    <tr>
                        <td><code>{item.key}</code></td>
                        <td class="direction"
                            >{item.direction === "lower"
                                ? "↓"
                                : "↑"}</td
                        >
                        {#if isMultiModel && runMetrics}
                            {#each runMetrics.models as model, idx}
                                {@const ds = getModelDataset(
                                    model,
                                    selectedDataset,
                                )}
                                {@const val = ds
                                    ? metricValue(ds, item.key)
                                    : null}
                                {@const sem = ds
                                    ? metricSem(ds, item.key)
                                    : null}
                                <td
                                    class:best={isBest(
                                        item,
                                        idx,
                                        runMetrics.models,
                                        selectedDataset,
                                    )}
                                >
                                    {formatVal(val)}
                                    {#if sem !== null}
                                        <small class="sem"
                                            >{formatSem(sem)}</small
                                        >
                                    {/if}
                                </td>
                            {/each}
                        {:else if runMetrics}
                            {@const ds = getModelDataset(
                                runMetrics.models[0],
                                selectedDataset,
                            )}
                            <td
                                >{formatVal(
                                    ds ? metricValue(ds, item.key) : null,
                                )}</td
                            >
                        {:else}
                            <td>-</td>
                        {/if}
                        <td class="purpose">{item.purpose}</td>
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

    .controls {
        align-items: end;
    }

    .direction {
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.7;
    }

    .best {
        font-weight: 700;
        color: var(--pico-primary);
    }

    .sem {
        display: block;
        font-size: 0.75rem;
        opacity: 0.6;
    }

    .purpose {
        font-size: 0.88rem;
        max-width: 18rem;
    }
</style>

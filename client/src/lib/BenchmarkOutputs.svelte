<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import { onMount } from "svelte";
    import {
        loadRunMetrics,
        metricValue,
        type LoadedRunMetrics,
        type ModelMetrics,
    } from "./metricsRuntime.ts";
    import ContentStatus from "./ContentStatus.svelte";

    type MethodFamily = "Amortized path" | "Explicit path";
    type ReportingFocus = "Balanced" | "Graph-focused" | "Posterior-focused";

    const baseScores: Record<
        MethodFamily,
        { graph: number; posterior: number; runtime: number }
    > = {
        "Amortized path": { graph: 82, posterior: 69, runtime: 86 },
        "Explicit path": { graph: 76, posterior: 81, runtime: 62 },
    };

    const focusAdjustments: Record<
        ReportingFocus,
        { graph: number; posterior: number; runtime: number }
    > = {
        Balanced: { graph: 0, posterior: 0, runtime: 0 },
        "Graph-focused": { graph: 9, posterior: -4, runtime: -3 },
        "Posterior-focused": { graph: -4, posterior: 9, runtime: -3 },
    };

    const expectedMetrics = [
        "e-edgef1",
        "ancestor_f1",
        "auc",
        "e-shd",
        "e-sid",
        "graph_nll",
        "inil",
    ] as const;

    function clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    function scoreHigherBetter(value: number): number {
        return clamp(value * 100, 0, 100);
    }

    function scoreLowerBetter(value: number, scale: number): number {
        return clamp(100 * Math.exp(-Math.max(value, 0) / scale), 0, 100);
    }

    function average(values: number[]): number | null {
        if (values.length === 0) return null;
        return values.reduce((s, v) => s + v, 0) / values.length;
    }

    function computeModelScores(
        summary: Record<string, number> | null,
    ): { graph: number | null; posterior: number | null; coverage: number | null } {
        if (!summary) return { graph: null, posterior: null, coverage: null };

        const graph = average(
            [
                metricValue(summary, "e-edgef1"),
                metricValue(summary, "ancestor_f1"),
                metricValue(summary, "auc"),
            ]
                .filter((v): v is number => v !== null)
                .map((v) => scoreHigherBetter(v))
                .concat(
                    [
                        metricValue(summary, "e-shd"),
                        metricValue(summary, "e-sid"),
                    ]
                        .filter((v): v is number => v !== null)
                        .map((v, i) =>
                            scoreLowerBetter(v, i === 0 ? 10 : 20),
                        ),
                ),
        );

        const posterior = average(
            [
                metricValue(summary, "graph_nll"),
                metricValue(summary, "inil"),
            ]
                .filter((v): v is number => v !== null)
                .map((v) => scoreLowerBetter(v, 8)),
        );

        const coverage =
            (expectedMetrics.filter(
                (k) => metricValue(summary, k) !== null,
            ).length /
                expectedMetrics.length) *
            100;

        return { graph, posterior, coverage };
    }

    let methodFamily = $state<MethodFamily>("Amortized path");
    let reportingFocus = $state<ReportingFocus>("Balanced");
    let computeBudget = $state(55);
    let runMetrics = $state<LoadedRunMetrics | null>(null);
    let selectedDataset = $state("");
    let selectedModel = $state("");

    onMount(async () => {
        runMetrics = await loadRunMetrics();
    });

    $effect(() => {
        if (runMetrics && !selectedDataset) {
            const first = runMetrics.models[0]?.datasets[0]?.dataset;
            if (first) selectedDataset = first;
        }
    });

    $effect(() => {
        if (runMetrics && !selectedModel) {
            const first = runMetrics.models[0]?.model;
            if (first) selectedModel = first;
        }
    });

    let isMultiModel = $derived(
        runMetrics !== null && runMetrics.models.length > 1,
    );

    let datasetOptions = $derived(() => {
        if (!runMetrics) return [] as string[];
        const seen = new Set<string>();
        for (const m of runMetrics.models) {
            for (const d of m.datasets) seen.add(d.dataset);
        }
        return [...seen];
    });

    let currentModel = $derived(
        runMetrics?.models.find((m) => m.model === selectedModel) ?? null,
    );

    let currentSummary = $derived(
        currentModel?.datasets.find((d) => d.dataset === selectedDataset)
            ?.metrics ?? null,
    );

    let hasRunData = $derived(runMetrics !== null && currentSummary !== null);

    let runScores = $derived(computeModelScores(currentSummary));

    // Scenario mode (fallback when no data)
    let scenarioGraph = $derived(
        clamp(
            baseScores[methodFamily].graph +
                focusAdjustments[reportingFocus].graph +
                (computeBudget - 50) * 0.09,
            0,
            100,
        ),
    );
    let scenarioPosterior = $derived(
        clamp(
            baseScores[methodFamily].posterior +
                focusAdjustments[reportingFocus].posterior +
                (computeBudget - 50) *
                    (methodFamily === "Explicit path" ? 0.11 : 0.06),
            0,
            100,
        ),
    );
    let scenarioRuntime = $derived(
        clamp(
            baseScores[methodFamily].runtime +
                focusAdjustments[reportingFocus].runtime -
                (computeBudget - 50) * 0.08,
            0,
            100,
        ),
    );

    let graphTarget = $derived(
        hasRunData ? (runScores.graph ?? 55) : scenarioGraph,
    );
    let posteriorTarget = $derived(
        hasRunData ? (runScores.posterior ?? 50) : scenarioPosterior,
    );
    let thirdTarget = $derived(
        hasRunData ? (runScores.coverage ?? 40) : scenarioRuntime,
    );
    let thirdLabel = $derived(
        hasRunData ? "Reporting Coverage" : "Runtime Efficiency",
    );

    const graphScore = tweened(0, { duration: 240, easing: cubicOut });
    const posteriorScore = tweened(0, { duration: 240, easing: cubicOut });
    const thirdScore = tweened(0, { duration: 240, easing: cubicOut });

    $effect(() => {
        graphScore.set(graphTarget);
        posteriorScore.set(posteriorTarget);
        thirdScore.set(thirdTarget);
    });

    // Multi-model bar chart data
    interface ModelBar {
        model: string;
        graph: number;
        posterior: number;
    }

    let modelBars: ModelBar[] = $derived.by(() => {
        if (!runMetrics || !isMultiModel) return [];
        return runMetrics.models.map((m: ModelMetrics) => {
            const ds =
                m.datasets.find((d) => d.dataset === selectedDataset)
                    ?.metrics ?? null;
            const scores = computeModelScores(ds);
            return {
                model: m.model,
                graph: scores.graph ?? 0,
                posterior: scores.posterior ?? 0,
            };
        });
    });
</script>

<article>
    <header>
        <strong>Output Dimensions Explorer</strong>
        <br />
        <ContentStatus
            status={hasRunData ? "measured" : "illustrative"}
            text={hasRunData
                ? "Measured from run artifact"
                : "Illustrative fallback"}
        />
        <br />
        {#if hasRunData}
            <small
                >Run-backed summary loaded from <code
                    >{runMetrics?.source}</code
                >.</small
            >
        {:else}
            <small>
                Scenario fallback (drop a run <code>metrics.json</code> into
                <code>/public/data/</code> for live values).
            </small>
        {/if}
    </header>

    {#if hasRunData}
        <div class="grid controls">
            {#if isMultiModel}
                <label
                    >Model
                    <select bind:value={selectedModel}>
                        {#each runMetrics?.models ?? [] as m}
                            <option value={m.model}>{m.model}</option>
                        {/each}
                    </select>
                </label>
            {/if}
            <label
                >Dataset split
                <select bind:value={selectedDataset}>
                    {#each datasetOptions() as ds}
                        <option value={ds}>{ds}</option>
                    {/each}
                </select>
            </label>
        </div>
    {:else}
        <div class="grid controls">
            <label
                >Method family
                <select bind:value={methodFamily}>
                    <option value="Amortized path">Amortized path</option>
                    <option value="Explicit path">Explicit path</option>
                </select>
            </label>

            <label
                >Report focus
                <select bind:value={reportingFocus}>
                    <option value="Balanced">Balanced</option>
                    <option value="Graph-focused">Graph-focused</option>
                    <option value="Posterior-focused">Posterior-focused</option>
                </select>
            </label>
        </div>

        <label for="budget-slider">Compute budget: {computeBudget}%</label>
        <input
            id="budget-slider"
            type="range"
            min="0"
            max="100"
            bind:value={computeBudget}
        />
    {/if}

    <!-- Single-model score cards -->
    <div class="metrics-grid grid">
        <article>
            <header>Graph Quality</header>
            <p><strong>{Math.round($graphScore)}%</strong></p>
            <progress max="100" value={$graphScore}></progress>
        </article>

        <article>
            <header>Posterior Quality</header>
            <p><strong>{Math.round($posteriorScore)}%</strong></p>
            <progress max="100" value={$posteriorScore}></progress>
        </article>

        <article>
            <header>{thirdLabel}</header>
            <p><strong>{Math.round($thirdScore)}%</strong></p>
            <progress max="100" value={$thirdScore}></progress>
        </article>
    </div>

    <!-- Multi-model comparison bar chart -->
    {#if isMultiModel && hasRunData}
        <details>
            <summary>Model comparison on <strong>{selectedDataset}</strong></summary>
            <div class="comparison-chart">
                {#each modelBars as bar}
                    <div class="bar-row">
                        <span class="bar-label">{bar.model}</span>
                        <div class="bar-pair">
                            <div class="bar-track">
                                <div
                                    class="bar-fill graph-bar"
                                    style="width: {bar.graph}%"
                                ></div>
                            </div>
                            <div class="bar-track">
                                <div
                                    class="bar-fill posterior-bar"
                                    style="width: {bar.posterior}%"
                                ></div>
                            </div>
                        </div>
                        <span class="bar-values"
                            >{Math.round(bar.graph)} / {Math.round(
                                bar.posterior,
                            )}</span
                        >
                    </div>
                {/each}
                <small class="legend"
                    ><span class="swatch graph-swatch"></span> Graph Quality
                    &nbsp;
                    <span class="swatch posterior-swatch"></span> Posterior Quality</small
                >
            </div>
        </details>
    {/if}

    {#if hasRunData}
        <small
            >Runtime is not emitted by the current evaluation artifact;
            coverage indicates available metric dimensions.</small
        >
    {/if}
</article>

<style>
    .controls {
        align-items: end;
    }

    .metrics-grid {
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }

    .metrics-grid article {
        margin: 0;
    }

    .metrics-grid p {
        margin-bottom: 0.3rem;
    }

    .comparison-chart {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }

    .bar-row {
        display: grid;
        grid-template-columns: 5rem 1fr 3.5rem;
        align-items: center;
        gap: 0.5rem;
    }

    .bar-label {
        font-size: 0.88rem;
        font-weight: 500;
    }

    .bar-pair {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }

    .bar-track {
        height: 0.6rem;
        background: var(--pico-muted-border-color);
        border-radius: 0.3rem;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 0.3rem;
        transition: width 0.3s ease;
    }

    .graph-bar {
        background: var(--pico-primary);
    }

    .posterior-bar {
        background: var(--pico-secondary);
    }

    .bar-values {
        font-size: 0.8rem;
        opacity: 0.7;
        text-align: right;
    }

    .legend {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.3rem;
    }

    .swatch {
        display: inline-block;
        width: 0.7rem;
        height: 0.7rem;
        border-radius: 0.15rem;
    }

    .graph-swatch {
        background: var(--pico-primary);
    }

    .posterior-swatch {
        background: var(--pico-secondary);
    }
</style>

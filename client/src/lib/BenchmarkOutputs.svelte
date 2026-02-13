<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import { onMount } from "svelte";
    import {
        loadRunMetrics,
        metricValue,
        type LoadedRunMetrics,
    } from "./metricsRuntime.ts";
    import ContentStatus from "./ContentStatus.svelte";

    type MethodFamily = "Amortized path" | "Explicit path";
    type ReportingFocus = "Balanced" | "Graph-focused" | "Posterior-focused";

    const baseScores: Record<MethodFamily, { graph: number; posterior: number; runtime: number }> = {
        "Amortized path": { graph: 82, posterior: 69, runtime: 86 },
        "Explicit path": { graph: 76, posterior: 81, runtime: 62 },
    };

    const focusAdjustments: Record<ReportingFocus, { graph: number; posterior: number; runtime: number }> = {
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
        if (values.length === 0) {
            return null;
        }

        const total = values.reduce((sum, value) => sum + value, 0);
        return total / values.length;
    }

    let methodFamily = $state<MethodFamily>("Amortized path");
    let reportingFocus = $state<ReportingFocus>("Balanced");
    let computeBudget = $state(55);
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

    let datasetOptions: LoadedRunMetrics["datasets"] = $derived(
        runMetrics ? runMetrics.datasets : [],
    );

    let selectedSummary: Record<string, number> | null = $derived(
        datasetOptions.find((entry) => entry.dataset === selectedDataset)?.metrics ?? null,
    );

    let graphFromRun = $derived(
        selectedSummary
            ? average(
                  [
                      metricValue(selectedSummary, "e-edgef1"),
                      metricValue(selectedSummary, "ancestor_f1"),
                      metricValue(selectedSummary, "auc"),
                  ]
                      .filter((value): value is number => value !== null)
                      .map((value) => scoreHigherBetter(value))
                      .concat(
                          [metricValue(selectedSummary, "e-shd"), metricValue(selectedSummary, "e-sid")]
                              .filter((value): value is number => value !== null)
                              .map((value, index) =>
                                  scoreLowerBetter(value, index === 0 ? 10 : 20),
                              ),
                      ),
              )
            : null,
    );

    let posteriorFromRun = $derived(
        selectedSummary
            ? average(
                  [metricValue(selectedSummary, "graph_nll"), metricValue(selectedSummary, "inil")]
                      .filter((value): value is number => value !== null)
                      .map((value) => scoreLowerBetter(value, 8)),
              )
            : null,
    );

    let coverageFromRun = $derived(
        selectedSummary
            ? (expectedMetrics.filter((metricKey) => metricValue(selectedSummary, metricKey) !== null)
                  .length /
                  expectedMetrics.length) *
              100
            : null,
    );

    let hasRunData = $derived(
        graphFromRun !== null || posteriorFromRun !== null || coverageFromRun !== null,
    );

    let scenarioGraphTarget = $derived(
        clamp(
            baseScores[methodFamily].graph +
                focusAdjustments[reportingFocus].graph +
                (computeBudget - 50) * 0.09,
            0,
            100,
        ),
    );

    let scenarioPosteriorTarget = $derived(
        clamp(
            baseScores[methodFamily].posterior +
                focusAdjustments[reportingFocus].posterior +
                (computeBudget - 50) * (methodFamily === "Explicit path" ? 0.11 : 0.06),
            0,
            100,
        ),
    );

    let scenarioRuntimeTarget = $derived(
        clamp(
            baseScores[methodFamily].runtime +
                focusAdjustments[reportingFocus].runtime -
                (computeBudget - 50) * 0.08,
            0,
            100,
        ),
    );

    let graphTarget = $derived(hasRunData ? (graphFromRun ?? 55) : scenarioGraphTarget);
    let posteriorTarget = $derived(hasRunData ? (posteriorFromRun ?? 50) : scenarioPosteriorTarget);
    let thirdTarget = $derived(hasRunData ? (coverageFromRun ?? 40) : scenarioRuntimeTarget);
    let thirdLabel = $derived(hasRunData ? "Reporting Coverage" : "Runtime Efficiency");

    const graphScore = tweened(0, { duration: 240, easing: cubicOut });
    const posteriorScore = tweened(0, { duration: 240, easing: cubicOut });
    const thirdScore = tweened(0, { duration: 240, easing: cubicOut });

    $effect(() => {
        graphScore.set(graphTarget);
        posteriorScore.set(posteriorTarget);
        thirdScore.set(thirdTarget);
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
            <small>Run-backed summary view loaded from {runMetrics?.source}.</small>
        {:else}
            <small>
                Scenario fallback view (drop a run `metrics.json` into `/public/data/` for live values).
            </small>
        {/if}
    </header>

    {#if hasRunData}
        <label
            >Dataset split
            <select bind:value={selectedDataset}>
                {#each datasetOptions as entry}
                    <option value={entry.dataset}>{entry.dataset}</option>
                {/each}
            </select>
        </label>
    {:else}
        <div class="grid">
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
        <input id="budget-slider" type="range" min="0" max="100" bind:value={computeBudget} />
    {/if}

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

    {#if hasRunData}
        <small>Runtime is not emitted by the current evaluation artifact; coverage indicates available metric dimensions.</small>
    {/if}
</article>

<style>
    .metrics-grid {
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }

    .metrics-grid article {
        margin: 0;
    }

    .metrics-grid p {
        margin-bottom: 0.3rem;
    }
</style>

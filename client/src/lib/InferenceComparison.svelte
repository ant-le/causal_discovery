<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import ContentStatus from "./ContentStatus.svelte";

    let taskCount = $state(28);
    let adaptationSteps = $state(120);

    function clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    let amortizedTarget = $derived(
        clamp(80 + taskCount * 0.22 - adaptationSteps * 0.03, 25, 95),
    );

    let explicitTarget = $derived(
        clamp(42 + adaptationSteps * 0.28 - taskCount * 0.06, 20, 92),
    );

    let throughputTarget = $derived(
        clamp(88 + taskCount * 0.15 - adaptationSteps * 0.08, 18, 97),
    );

    const amortizedScore = tweened(0, { duration: 260, easing: cubicOut });
    const explicitScore = tweened(0, { duration: 260, easing: cubicOut });
    const throughputScore = tweened(0, { duration: 260, easing: cubicOut });

    $effect(() => {
        amortizedScore.set(amortizedTarget);
        explicitScore.set(explicitTarget);
        throughputScore.set(throughputTarget);
    });
</script>

<article>
    <header>
        <strong>Inference Strategy Explorer</strong>
        <br />
        <ContentStatus
            status="illustrative"
            text="Illustrative strategy trade-off"
        />
        <br />
        <small>Illustrative trade-off view for amortized vs explicit inference.</small>
    </header>

    <label for="task-count">Dataset tasks in evaluation: {taskCount}</label>
    <input id="task-count" type="range" min="4" max="80" bind:value={taskCount} />

    <label for="adapt-steps">Per-dataset adaptation effort: {adaptationSteps}</label>
    <input
        id="adapt-steps"
        type="range"
        min="0"
        max="300"
        step="5"
        bind:value={adaptationSteps}
    />

    <div class="grid cards">
        <article>
            <header>Amortized path</header>
            <p>Shared predictor transferred across many datasets.</p>
            <progress max="100" value={$amortizedScore}></progress>
        </article>

        <article>
            <header>Explicit path</header>
            <p>Dataset-specific optimization for each posterior target.</p>
            <progress max="100" value={$explicitScore}></progress>
        </article>

        <article>
            <header>Throughput pressure</header>
            <p>Wall-clock sensitivity under repeated evaluation workloads.</p>
            <progress max="100" value={$throughputScore}></progress>
        </article>
    </div>
</article>

<style>
    .cards {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }

    .cards article {
        margin: 0;
    }

    .cards p {
        min-height: 3.2rem;
        margin-bottom: 0.35rem;
    }
</style>

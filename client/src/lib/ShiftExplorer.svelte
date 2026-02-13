<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import ContentStatus from "./ContentStatus.svelte";

    type ShiftType = "Mechanism shift" | "Graph-family shift" | "Noise shift";
    type MethodType = "Amortized" | "Explicit";

    const shiftTypes: ShiftType[] = [
        "Mechanism shift",
        "Graph-family shift",
        "Noise shift",
    ];

    const methodTypes: MethodType[] = ["Amortized", "Explicit"];

    const graphPenalty: Record<ShiftType, number> = {
        "Mechanism shift": 0.26,
        "Graph-family shift": 0.34,
        "Noise shift": 0.18,
    };

    const calibrationPenalty: Record<ShiftType, number> = {
        "Mechanism shift": 0.18,
        "Graph-family shift": 0.12,
        "Noise shift": 0.28,
    };

    const baseGraph: Record<MethodType, number> = {
        Amortized: 86,
        Explicit: 79,
    };

    const baseCalibration: Record<MethodType, number> = {
        Amortized: 71,
        Explicit: 84,
    };

    function clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    let shiftType = $state<ShiftType>("Mechanism shift");
    let methodType = $state<MethodType>("Amortized");
    let severity = $state(35);

    let graphTarget = $derived(
        clamp(
            baseGraph[methodType] - severity * graphPenalty[shiftType],
            15,
            95,
        ),
    );

    let calibrationTarget = $derived(
        clamp(
            baseCalibration[methodType] - severity * calibrationPenalty[shiftType],
            15,
            95,
        ),
    );

    const graphScore = tweened(0, { duration: 260, easing: cubicOut });
    const calibrationScore = tweened(0, { duration: 260, easing: cubicOut });

    $effect(() => {
        graphScore.set(graphTarget);
        calibrationScore.set(calibrationTarget);
    });
</script>

<article>
    <header>
        <strong>Shift Stress Tester</strong>
        <br />
        <ContentStatus
            status="illustrative"
            text="Illustrative shift trend"
        />
        <br />
        <small>Illustrative trend view for RQ1 (not benchmark output).</small>
    </header>

    <div class="grid">
        <label
            >Shift family
            <select bind:value={shiftType}>
                {#each shiftTypes as option}
                    <option value={option}>{option}</option>
                {/each}
            </select>
        </label>

        <label
            >Inference style
            <select bind:value={methodType}>
                {#each methodTypes as option}
                    <option value={option}>{option}</option>
                {/each}
            </select>
        </label>
    </div>

    <label for="severity-slider">Shift severity: {severity}%</label>
    <input id="severity-slider" type="range" min="0" max="100" bind:value={severity} />

    <div class="scores">
        <p>
            Graph fidelity
            <strong>{Math.round($graphScore)}%</strong>
        </p>
        <progress max="100" value={$graphScore}></progress>

        <p>
            Posterior calibration
            <strong>{Math.round($calibrationScore)}%</strong>
        </p>
        <progress max="100" value={$calibrationScore}></progress>
    </div>
</article>

<style>
    .scores p {
        margin-bottom: 0.25rem;
    }

    .scores strong {
        margin-left: 0.3rem;
    }

    progress {
        margin-bottom: 0.75rem;
    }
</style>

<script lang="ts">
    import ContentStatus from "./ContentStatus.svelte";

    type MechanismType = "linear" | "periodic" | "gp-like";

    interface Point {
        x: number;
        y: number;
    }

    const size = 240;
    const step = 24;

    let mechanism = $state<MechanismType>("linear");
    let noise = $state(12);

    function valueAt(index: number): number {
        const x = index / 10;

        if (mechanism === "linear") {
            return 0.42 * x + 0.7;
        }

        if (mechanism === "periodic") {
            return Math.sin(x * 1.15) + 0.28 * x;
        }

        return Math.sin(x * 0.65) * 0.8 + Math.cos(x * 1.45) * 0.45 + 0.2 * x;
    }

    function pseudoNoise(index: number): number {
        const frequency = mechanism === "linear" ? 0.25 : mechanism === "periodic" ? 0.45 : 0.62;
        return Math.sin(index * frequency + 0.9) * (noise / 100);
    }

    function toPoints(): Point[] {
        const points: Point[] = [];
        for (let i = 0; i <= step; i += 1) {
            const rawY = valueAt(i) + pseudoNoise(i);
            const x = (i / step) * size;
            const y = size - ((rawY + 2) / 5) * size;
            points.push({ x, y });
        }
        return points;
    }

    let points = $derived(toPoints());
    let polyline = $derived(points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" "));

    let explanation = $derived(
        mechanism === "linear"
            ? "Linear mechanisms are a baseline for stable, low-curvature relationships."
            : mechanism === "periodic"
              ? "Periodic mechanisms test behavior under oscillatory functional form shifts."
              : "GP-like patterns mimic flexible nonparametric mechanisms with local variation.",
    );
</script>

<article>
    <header>
        <strong>SCM Family Gallery</strong>
        <br />
        <ContentStatus
            status="illustrative"
            text="Illustrative mechanism shape preview"
        />
        <br />
        <small>Illustrative mechanism shapes aligned with configurable generator families.</small>
    </header>

    <div class="grid controls">
        <label
            >Mechanism family
            <select bind:value={mechanism}>
                <option value="linear">Linear</option>
                <option value="periodic">Periodic</option>
                <option value="gp-like">GP-like</option>
            </select>
        </label>

        <label for="noise-slider">Noise level: {noise}%</label>
        <input id="noise-slider" type="range" min="0" max="30" bind:value={noise} />
    </div>

    <figure>
        <svg viewBox={`0 0 ${size} ${size}`} aria-label="Mechanism sample curve" role="img">
            <line x1="0" y1={size} x2={size} y2={size} class="axis"></line>
            <line x1="0" y1="0" x2="0" y2={size} class="axis"></line>
            <polyline points={polyline} class="curve"></polyline>
            {#each points as point}
                <circle cx={point.x} cy={point.y} r="2.1" class="dot"></circle>
            {/each}
        </svg>
        <figcaption>{explanation}</figcaption>
    </figure>
</article>

<style>
    .controls {
        align-items: end;
    }

    figure {
        margin: 0;
    }

    svg {
        width: min(100%, 24rem);
        border: 1px solid var(--pico-muted-border-color);
        border-radius: 0.7rem;
        background: color-mix(in oklab, var(--pico-background-color) 87%, var(--pico-primary) 13%);
    }

    .axis {
        stroke: var(--pico-muted-color);
        stroke-width: 1;
    }

    .curve {
        fill: none;
        stroke: var(--pico-primary);
        stroke-width: 2;
    }

    .dot {
        fill: var(--pico-contrast);
        opacity: 0.86;
    }

    figcaption {
        margin-top: 0.45rem;
    }
</style>

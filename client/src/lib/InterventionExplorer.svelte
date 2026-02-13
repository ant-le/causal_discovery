<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import ContentStatus from "./ContentStatus.svelte";

    type Mode = "Observational" | "Interventional";

    let mode = $state<Mode>("Observational");
    let xValue = $state(40);

    const yTween = tweened(0, { duration: 250, easing: cubicOut });

    function clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    let yTarget = $derived(
        clamp(
            mode === "Observational"
                ? 22 + xValue * 0.78 + Math.sin(xValue / 9) * 4
                : 14 + xValue * 0.55,
            0,
            100,
        ),
    );

    $effect(() => {
        yTween.set(yTarget);
    });

    let equationText = $derived(
        mode === "Observational"
            ? "Y follows both parent structure and ambient noise."
            : "do(X = x) replaces the X mechanism and isolates intervention response.",
    );
</script>

<article>
    <header>
        <strong>Intervention Lens</strong>
        <br />
        <ContentStatus
            status="illustrative"
            text="Illustrative intervention behavior"
        />
        <br />
        <small>Illustrative SCM behavior for observation vs intervention.</small>
    </header>

    <div class="grid">
        <label
            >Mode
            <select bind:value={mode}>
                <option value="Observational">Observational</option>
                <option value="Interventional">Interventional</option>
            </select>
        </label>

        <label
            >Input value: {xValue}
            <input type="range" min="0" max="100" bind:value={xValue} />
        </label>
    </div>

    <p>{equationText}</p>

    <div class="grid value-grid">
        <article>
            <header>X</header>
            <p><strong>{xValue}</strong></p>
            <progress max="100" value={xValue}></progress>
        </article>

        <article>
            <header>Y</header>
            <p><strong>{Math.round($yTween)}</strong></p>
            <progress max="100" value={$yTween}></progress>
        </article>
    </div>
</article>

<style>
    .value-grid {
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }

    .value-grid article {
        margin: 0;
    }

    .value-grid p {
        margin-bottom: 0.3rem;
    }
</style>

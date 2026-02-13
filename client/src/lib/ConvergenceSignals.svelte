<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import ContentStatus from "./ContentStatus.svelte";

    function clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    let chainCount = $state(12);
    let chainLength = $state(450);
    let betweenChainDispersion = $state(28);

    let rHatTarget = $derived(
        clamp(
            1 +
                betweenChainDispersion / 240 +
                45 / chainLength +
                (chainCount < 8 ? 0.03 : 0),
            1.0,
            1.35,
        ),
    );

    let essTarget = $derived(
        clamp(
            chainCount * chainLength * (1 / (1 + betweenChainDispersion / 65)) * 0.06,
            20,
            1800,
        ),
    );

    const rHatValue = tweened(1.2, { duration: 260, easing: cubicOut });
    const essValue = tweened(300, { duration: 260, easing: cubicOut });

    $effect(() => {
        rHatValue.set(rHatTarget);
        essValue.set(essTarget);
    });

    let statusText = $derived(
        $rHatValue <= 1.01
            ? "Stable"
            : $rHatValue <= 1.05
              ? "Watch"
              : "Unstable",
    );
</script>

<article>
    <header>
        <strong>Convergence Signals</strong>
        <br />
        <ContentStatus
            status="illustrative"
            text="Illustrative diagnostics behavior"
        />
        <br />
        <small>Illustrative diagnostics behavior for short-chain settings.</small>
    </header>

    <label for="chain-count">Chains: {chainCount}</label>
    <input id="chain-count" type="range" min="4" max="48" bind:value={chainCount} />

    <label for="chain-length">Chain length: {chainLength}</label>
    <input id="chain-length" type="range" min="80" max="2000" step="20" bind:value={chainLength} />

    <label for="dispersion">Between-chain dispersion: {betweenChainDispersion}%</label>
    <input
        id="dispersion"
        type="range"
        min="0"
        max="100"
        bind:value={betweenChainDispersion}
    />

    <div class="grid">
        <article>
            <header>R-hat proxy</header>
            <p><strong>{$rHatValue.toFixed(3)}</strong> ({statusText})</p>
            <progress max="1.35" value={$rHatValue}></progress>
        </article>

        <article>
            <header>ESS proxy</header>
            <p><strong>{Math.round($essValue)}</strong></p>
            <progress max="1800" value={$essValue}></progress>
        </article>
    </div>
</article>

<style>
    article p {
        margin-bottom: 0.3rem;
    }
</style>

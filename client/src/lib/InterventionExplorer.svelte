<script lang="ts">
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";
    import ContentStatus from "./ContentStatus.svelte";
    import MathExpr from "./Math.svelte";

    type Mode = "Observational" | "Interventional";

    /* ── layout constants ─────────────────────────────────── */
    const dagW = 300;
    const dagH = 200;
    const nodeR = 22;

    interface DagNode {
        id: string;
        label: string;
        x: number;
        y: number;
    }

    interface DagEdge {
        from: string;
        to: string;
    }

    const nodes: DagNode[] = [
        { id: "Z", label: "Z", x: 60, y: 50 },
        { id: "W", label: "W", x: 240, y: 50 },
        { id: "X", label: "X", x: 60, y: 155 },
        { id: "Y", label: "Y", x: 240, y: 155 },
    ];

    const edges: DagEdge[] = [
        { from: "Z", to: "X" },
        { from: "Z", to: "Y" },
        { from: "W", to: "Y" },
        { from: "X", to: "Y" },
    ];

    /* ── state ────────────────────────────────────────────── */
    let mode = $state<Mode>("Observational");
    let xSlider = $state(50);

    const yTween = tweened(0, { duration: 280, easing: cubicOut });

    /** Fixed confounders for illustration */
    const zVal = 35;
    const wVal = 20;

    function clamp(v: number, lo: number, hi: number): number {
        return globalThis.Math.max(lo, globalThis.Math.min(hi, v));
    }

    /* ── observational vs interventional Y ────────────────── */
    let xEffective = $derived.by((): number => {
        if (mode === "Observational") {
            // X depends on Z
            return clamp(0.6 * zVal + 0.4 * xSlider, 0, 100);
        }
        // do(X = x): X is fixed
        return xSlider;
    });

    let yTarget = $derived.by((): number => {
        const M = globalThis.Math;
        if (mode === "Observational") {
            // Y depends on X (which depends on Z), Z (direct), W
            return clamp(
                0.45 * xEffective + 0.3 * zVal + 0.2 * wVal + M.sin(xEffective / 9) * 4,
                0,
                100,
            );
        }
        // do(X = x): only the causal paths remain
        // Z → Y direct path still active, but Z → X → Y is replaced by do(X)→Y
        return clamp(0.5 * xSlider + 0.3 * zVal + 0.15 * wVal, 0, 100);
    });

    $effect(() => {
        yTween.set(yTarget);
    });

    /* ── edge rendering logic ─────────────────────────────── */
    function nodeById(id: string): DagNode {
        return nodes.find((n) => n.id === id)!;
    }

    function isCutEdge(edge: DagEdge): boolean {
        // In interventional mode, incoming edges to X are cut
        return mode === "Interventional" && edge.to === "X";
    }

    function edgePath(edge: DagEdge): { x1: number; y1: number; x2: number; y2: number } {
        const M = globalThis.Math;
        const from = nodeById(edge.from);
        const to = nodeById(edge.to);
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const dist = M.sqrt(dx * dx + dy * dy);
        const ux = dx / dist;
        const uy = dy / dist;
        return {
            x1: from.x + ux * (nodeR + 2),
            y1: from.y + uy * (nodeR + 2),
            x2: to.x - ux * (nodeR + 6),
            y2: to.y - uy * (nodeR + 6),
        };
    }

    /** Midpoint of an edge (for the "cut" marker) */
    function edgeMid(edge: DagEdge): { mx: number; my: number } {
        const p = edgePath(edge);
        return { mx: (p.x1 + p.x2) / 2, my: (p.y1 + p.y2) / 2 };
    }

    /* ── formula text ─────────────────────────────────────── */
    let formulaObs = "P(Y \\mid X = x) \\neq P(Y \\mid \\mathrm{do}(X = x))";
    let formulaInt =
        "P(Y \\mid \\mathrm{do}(X = x)) = \\sum_z P(Y \\mid X{=}x, Z{=}z)\\, P(Z{=}z)";

    let explanation = $derived(
        mode === "Observational"
            ? "X is correlated with Y through both the causal path X \u2192 Y and the confounding path Z \u2192 X and Z \u2192 Y. Conditioning on X = x does not isolate the causal effect."
            : "do(X = x) severs all incoming edges to X (shown in red). The confounder Z no longer influences X, isolating the causal effect X \u2192 Y from spurious correlation.",
    );
</script>

<article>
    <header>
        <strong>Intervention Lens</strong>
        <br />
        <ContentStatus status="illustrative" text="Illustrative intervention behavior" />
        <br />
        <small
            >Compare observational conditioning with interventional <code>do()</code> on a 4-node
            causal graph.</small
        >
    </header>

    <!-- controls -->
    <div class="controls">
        <label>
            Mode
            <select bind:value={mode}>
                <option value="Observational">Observational</option>
                <option value="Interventional">Interventional</option>
            </select>
        </label>

        <label>
            {mode === "Interventional" ? "do(X)" : "X observed"}: {xSlider}
            <input type="range" min="0" max="100" bind:value={xSlider} />
        </label>
    </div>

    <!-- DAG + values grid -->
    <div class="explorer-grid">
        <!-- SVG DAG -->
        <figure>
            <svg
                viewBox={`0 0 ${dagW} ${dagH}`}
                aria-label="Causal DAG with 4 nodes"
                role="img"
            >
                <defs>
                    <marker
                        id="arrow"
                        viewBox="0 0 10 10"
                        refX="8"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                    >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--pico-primary)" />
                    </marker>
                    <marker
                        id="arrow-cut"
                        viewBox="0 0 10 10"
                        refX="8"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                    >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--pico-del-color, #c62828)" />
                    </marker>
                </defs>

                <!-- edges -->
                {#each edges as edge}
                    {@const p = edgePath(edge)}
                    {@const cut = isCutEdge(edge)}
                    <line
                        x1={p.x1}
                        y1={p.y1}
                        x2={p.x2}
                        y2={p.y2}
                        class="edge"
                        class:cut
                        marker-end={cut ? "url(#arrow-cut)" : "url(#arrow)"}
                        stroke-dasharray={cut ? "5,4" : "none"}
                    />
                    {#if cut}
                        {@const mid = edgeMid(edge)}
                        <text
                            x={mid.mx}
                            y={mid.my}
                            class="cut-marker"
                            text-anchor="middle"
                            dominant-baseline="central">&times;</text
                        >
                    {/if}
                {/each}

                <!-- nodes -->
                {#each nodes as node}
                    {@const isTarget = mode === "Interventional" && node.id === "X"}
                    <circle
                        cx={node.x}
                        cy={node.y}
                        r={nodeR}
                        class="node"
                        class:intervened={isTarget}
                    />
                    <text
                        x={node.x}
                        y={node.y}
                        class="node-label"
                        class:intervened={isTarget}
                        text-anchor="middle"
                        dominant-baseline="central">{node.label}</text
                    >
                {/each}
            </svg>
            <figcaption>
                {#if mode === "Interventional"}
                    Incoming edges to <strong>X</strong> are severed by the intervention.
                {:else}
                    All edges active &mdash; Z confounds the X&ndash;Y relationship.
                {/if}
            </figcaption>
        </figure>

        <!-- values panel -->
        <div class="values-panel">
            <div class="value-cards">
                <div class="value-card confounder">
                    <small>Z (confounder)</small>
                    <strong>{zVal}</strong>
                    <progress max="100" value={zVal}></progress>
                </div>
                <div class="value-card confounder">
                    <small>W (cause)</small>
                    <strong>{wVal}</strong>
                    <progress max="100" value={wVal}></progress>
                </div>
                <div class="value-card" class:intervened={mode === "Interventional"}>
                    <small>X {mode === "Interventional" ? "(intervened)" : "(observed)"}</small>
                    <strong>{globalThis.Math.round(xEffective)}</strong>
                    <progress max="100" value={xEffective}></progress>
                </div>
                <div class="value-card outcome">
                    <small>Y (outcome)</small>
                    <strong>{globalThis.Math.round($yTween)}</strong>
                    <progress max="100" value={$yTween}></progress>
                </div>
            </div>

            <p class="explanation">{explanation}</p>
        </div>
    </div>

    <!-- formula -->
    <div class="formula-row">
        <div class="formula-box">
            <MathExpr
                expression={mode === "Observational" ? formulaObs : formulaInt}
                inline={false}
            />
        </div>
    </div>
</article>

<style>
    .controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        align-items: end;
        margin-bottom: 1rem;
    }

    @media (max-width: 600px) {
        .controls {
            grid-template-columns: 1fr;
        }
    }

    .explorer-grid {
        display: grid;
        grid-template-columns: minmax(200px, 300px) 1fr;
        gap: 1.5rem;
        align-items: start;
        margin-bottom: 1rem;
    }

    @media (max-width: 700px) {
        .explorer-grid {
            grid-template-columns: 1fr;
        }
    }

    figure {
        margin: 0;
    }

    svg {
        width: 100%;
        border: 1px solid var(--pico-muted-border-color);
        border-radius: var(--pico-border-radius);
        background: color-mix(
            in oklab,
            var(--pico-background-color) 90%,
            var(--pico-primary) 10%
        );
    }

    figcaption {
        margin-top: 0.4rem;
        font-size: 0.85rem;
    }

    /* ── edges ── */
    .edge {
        stroke: var(--pico-primary);
        stroke-width: 2;
    }

    .edge.cut {
        stroke: var(--pico-del-color, #c62828);
        opacity: 0.6;
    }

    .cut-marker {
        font-size: 22px;
        font-weight: bold;
        fill: var(--pico-del-color, #c62828);
    }

    /* ── nodes ── */
    .node {
        fill: var(--pico-background-color);
        stroke: var(--pico-primary);
        stroke-width: 2;
    }

    .node.intervened {
        fill: color-mix(in oklab, var(--pico-del-color, #c62828) 15%, var(--pico-background-color) 85%);
        stroke: var(--pico-del-color, #c62828);
        stroke-width: 2.5;
    }

    .node-label {
        font-size: 14px;
        font-weight: 600;
        fill: var(--pico-color);
    }

    .node-label.intervened {
        fill: var(--pico-del-color, #c62828);
    }

    /* ── value cards ── */
    .values-panel {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .value-cards {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }

    .value-card {
        padding: 0.5rem 0.75rem;
        border: 1px solid var(--pico-muted-border-color);
        border-radius: var(--pico-border-radius);
        text-align: center;
    }

    .value-card small {
        display: block;
        color: var(--pico-muted-color);
        margin-bottom: 0.15rem;
    }

    .value-card strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.25rem;
    }

    .value-card progress {
        margin-bottom: 0;
    }

    .value-card.confounder {
        border-color: var(--pico-muted-border-color);
        opacity: 0.85;
    }

    .value-card.intervened {
        border-color: var(--pico-del-color, #c62828);
        background: color-mix(
            in oklab,
            var(--pico-del-color, #c62828) 5%,
            var(--pico-background-color) 95%
        );
    }

    .value-card.outcome {
        border-color: var(--pico-primary);
        background: color-mix(
            in oklab,
            var(--pico-primary) 5%,
            var(--pico-background-color) 95%
        );
    }

    .explanation {
        font-size: 0.9rem;
        color: var(--pico-muted-color);
        margin: 0;
    }

    /* ── formula row ── */
    .formula-row {
        margin-top: 0.75rem;
    }

    .formula-box {
        padding: 0.75rem 1rem;
        border: 1px solid var(--pico-muted-border-color);
        border-radius: var(--pico-border-radius);
        background: color-mix(
            in oklab,
            var(--pico-background-color) 92%,
            var(--pico-primary) 8%
        );
        overflow-x: auto;
        text-align: center;
    }
</style>

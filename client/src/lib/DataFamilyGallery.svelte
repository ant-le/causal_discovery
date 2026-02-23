<script lang="ts">
    import ContentStatus from "./ContentStatus.svelte";
    import MathExpr from "./Math.svelte";

    type MechanismType =
        | "linear"
        | "square"
        | "periodic"
        | "logistic-map"
        | "mlp"
        | "gp"
        | "pnl-tanh"
        | "mixture";

    interface Point {
        x: number;
        y: number;
    }

    const svgSize = 260;
    const margin = 24;
    const plotSize = svgSize - 2 * margin;
    const numScatter = 36;
    const xRange: [number, number] = [-2, 2];

    let mechanism = $state<MechanismType>("linear");
    let noise = $state(15);

    /* ── fixed parameters for reproducible curves ─────────── */
    const w = 0.85;

    function sigmoid(z: number): number {
        return 1.0 / (1.0 + globalThis.Math.exp(-z));
    }

    /* ── mechanism functions (faithful to src implementations) */
    function mechanismFn(x: number): number {
        const M = globalThis.Math;
        switch (mechanism) {
            case "linear":
                return w * x;
            case "square":
                return (w * x) ** 2;
            case "periodic":
                return M.sin(4 * M.PI * w * x);
            case "logistic-map": {
                const s = sigmoid(w * x);
                return 4 * s * (1 - s);
            }
            case "mlp":
                // Illustrative 2-layer MLP with LeakyReLU
                return M.tanh(1.2 * x) * 0.8 + 0.3 * x;
            case "gp":
                // Sum-of-cosines (RFF approximation with 4 components)
                return (
                    0.55 * M.cos(1.8 * x + 0.3) +
                    0.35 * M.cos(3.1 * x - 1.2) +
                    0.25 * M.cos(0.7 * x + 2.1) +
                    0.15 * M.cos(5.0 * x - 0.5)
                );
            case "pnl-tanh":
                return M.tanh(w * x);
            case "mixture":
                // Overlay: show linear for x < -0.5, periodic for -0.5..0.8, square after
                if (x < -0.5) return w * x;
                if (x < 0.8) return M.sin(4 * M.PI * w * x);
                return (w * x) ** 2;
            default:
                return 0;
        }
    }

    /** Deterministic pseudo-random noise (hash-based, stable across renders) */
    function pseudoNoise(i: number): number {
        const h = globalThis.Math.sin(i * 127.1 + 311.7) * 43758.5453;
        return (h - globalThis.Math.floor(h)) * 2 - 1;
    }

    let isDeterministic = $derived(mechanism === "logistic-map");
    let effectiveNoise = $derived(isDeterministic ? 0 : noise);

    /** Dynamic y-range per mechanism for well-fitted plots */
    let yRange = $derived.by((): [number, number] => {
        switch (mechanism) {
            case "linear":
                return [-2.5, 2.5];
            case "square":
                return [-0.8, 3.5];
            case "periodic":
                return [-1.8, 1.8];
            case "logistic-map":
                return [-0.3, 1.4];
            case "mlp":
                return [-2, 2];
            case "gp":
                return [-2, 2];
            case "pnl-tanh":
                return [-1.5, 1.5];
            case "mixture":
                return [-2.5, 3.5];
            default:
                return [-2.5, 2.5];
        }
    });

    /** Map data coords to SVG coords */
    function toSvg(dataX: number, dataY: number): Point {
        const x = margin + ((dataX - xRange[0]) / (xRange[1] - xRange[0])) * plotSize;
        const y = margin + (1 - (dataY - yRange[0]) / (yRange[1] - yRange[0])) * plotSize;
        return { x, y };
    }

    /* ── curve & scatter generation ────────────────────────── */
    let curve = $derived.by((): Point[] => {
        const pts: Point[] = [];
        const steps = 120;
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const rawX = xRange[0] + t * (xRange[1] - xRange[0]);
            pts.push(toSvg(rawX, mechanismFn(rawX)));
        }
        return pts;
    });

    let scatter = $derived.by((): Point[] => {
        const pts: Point[] = [];
        for (let i = 0; i < numScatter; i++) {
            const t = i / (numScatter - 1);
            const rawX = xRange[0] + t * (xRange[1] - xRange[0]);
            const noiseMag = pseudoNoise(i) * (effectiveNoise / 100) * 1.5;
            pts.push(toSvg(rawX, mechanismFn(rawX) + noiseMag));
        }
        return pts;
    });

    let curvePath = $derived(
        "M " + curve.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L "),
    );

    /** For mixture: secondary overlay curves */
    let mixtureCurves = $derived.by((): string[] => {
        if (mechanism !== "mixture") return [];
        const M = globalThis.Math;
        const fns = [
            (x: number) => w * x,
            (x: number) => M.sin(4 * M.PI * w * x),
            (x: number) => (w * x) ** 2,
        ];
        return fns.map((fn) => {
            const pts: Point[] = [];
            const steps = 80;
            for (let i = 0; i <= steps; i++) {
                const t = i / steps;
                const rawX = xRange[0] + t * (xRange[1] - xRange[0]);
                pts.push(toSvg(rawX, fn(rawX)));
            }
            return "M " + pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L ");
        });
    });

    /* ── axis ticks ────────────────────────────────────────── */
    let xTicks = $derived.by((): { pos: number; label: string }[] => {
        const ticks: { pos: number; label: string }[] = [];
        for (let v = -2; v <= 2; v += 1) {
            ticks.push({ pos: toSvg(v, 0).x, label: String(v) });
        }
        return ticks;
    });

    let yTicks = $derived.by((): { pos: number; label: string }[] => {
        const M = globalThis.Math;
        const ticks: { pos: number; label: string }[] = [];
        const span = yRange[1] - yRange[0];
        const step = span <= 3 ? 0.5 : 1;
        const low = M.ceil(yRange[0] / step) * step;
        const high = M.floor(yRange[1] / step) * step;
        for (let v = low; v <= high + 1e-9; v += step) {
            const rounded = M.round(v * 10) / 10;
            ticks.push({ pos: toSvg(0, rounded).y, label: String(rounded) });
        }
        return ticks;
    });

    /* ── mechanism info ────────────────────────────────────── */
    const mechanismInfo: Record<
        MechanismType,
        { name: string; formula: string; description: string; source: string }
    > = {
        linear: {
            name: "Linear",
            formula:
                "X_i = \\mathbf{w}^\\top \\mathrm{pa}(X_i) + \\sigma_i \\cdot \\varepsilon_i",
            description:
                "Weights from N(0, s\u00b2I), noise scale from Gamma(\u03b1, \u03b2). Default: weight_scale=1.0, noise_concentration=2.0, noise_rate=2.0.",
            source: "linear.py",
        },
        mlp: {
            name: "MLP",
            formula:
                "X_i = \\mathrm{MLP}\\bigl([\\mathrm{pa}(X_i);\\, \\varepsilon_i]\\bigr)",
            description:
                "2-layer network: Linear(d_pa+1, h) \u2192 LeakyReLU \u2192 Linear(h, 1). Noise concatenated as extra input. Default: hidden_dim=32.",
            source: "mlp.py",
        },
        square: {
            name: "Square",
            formula:
                "X_i = \\bigl(\\mathbf{w}^\\top \\mathrm{pa}(X_i)\\bigr)^2 + \\sigma \\cdot \\varepsilon_i",
            description:
                "Squares the linear combination of parents. Creates nonlinear, non-negative outputs. Default: noise_scale=0.1.",
            source: "functional.py",
        },
        periodic: {
            name: "Periodic",
            formula:
                "X_i = \\sin\\bigl(4\\pi\\, \\mathbf{w}^\\top \\mathrm{pa}(X_i)\\bigr) + \\sigma \\cdot \\varepsilon_i",
            description:
                "High-frequency sinusoidal transform. Tests robustness under oscillatory functional forms. Default: noise_scale=0.1.",
            source: "functional.py",
        },
        "logistic-map": {
            name: "Logistic Map",
            formula:
                "X_i = 4\\,\\sigma(z)\\,(1 - \\sigma(z)), \\quad z = \\mathbf{w}^\\top \\mathrm{pa}(X_i)",
            description:
                "Deterministic chaotic map with no additive noise. Sigmoid maps inputs to (0,1), quadratic form produces complex dynamics.",
            source: "functional.py",
        },
        gp: {
            name: "GP (RFF)",
            formula:
                "f(x) \\approx \\sqrt{\\tfrac{2}{D}} \\sum_{k=1}^{K} v_k \\cos(\\mathbf{W}_k x + b_k)",
            description:
                "Random Fourier Features approximate a mixture-kernel GP prior. Default: rff_dim=512, num_kernels=4. Exact mode uses sum of RQ + ExpGamma kernels with Cholesky sampling.",
            source: "gpcde.py",
        },
        "pnl-tanh": {
            name: "PNL (tanh)",
            formula:
                "X_i = \\tanh\\bigl(\\mathbf{w}^\\top \\mathrm{pa}(X_i) + \\sigma \\cdot \\varepsilon_i\\bigr)",
            description:
                "Post-Nonlinear model wrapping an inner linear mechanism with tanh. Also supports cube (y\u00b3) and sigmoid nonlinearities.",
            source: "pnl.py",
        },
        mixture: {
            name: "Mixture",
            formula:
                "f_i \\sim \\mathrm{Categorical}(\\{\\text{Linear, MLP, Square, \u2026}\\})",
            description:
                "Each node independently samples its mechanism type from a weighted set. Different nodes in the same graph have different functional forms.",
            source: "mixture.py",
        },
    };

    let info = $derived(mechanismInfo[mechanism]);
</script>

<article>
    <header>
        <strong>SCM Mechanism Gallery</strong>
        <br />
        <ContentStatus status="illustrative" text="Illustrative mechanism curves" />
        <br />
        <small
            >Explore functional forms from the 8 mechanism families implemented in
            <code>causal_meta.datasets.generators.mechanisms</code>.</small
        >
    </header>

    <!-- controls -->
    <div class="controls">
        <label>
            Mechanism family
            <select bind:value={mechanism}>
                <option value="linear">Linear</option>
                <option value="mlp">MLP</option>
                <option value="square">Square</option>
                <option value="periodic">Periodic</option>
                <option value="logistic-map">Logistic Map</option>
                <option value="gp">GP (RFF)</option>
                <option value="pnl-tanh">PNL (tanh)</option>
                <option value="mixture">Mixture</option>
            </select>
        </label>

        <label>
            Noise level: {effectiveNoise}%
            {#if isDeterministic}
                <small>(deterministic mechanism)</small>
            {/if}
            <input
                type="range"
                min="0"
                max="30"
                bind:value={noise}
                disabled={isDeterministic}
            />
        </label>
    </div>

    <!-- plot + formula grid -->
    <div class="gallery-grid">
        <!-- SVG plot -->
        <figure>
            <svg
                viewBox={`0 0 ${svgSize} ${svgSize}`}
                aria-label={`${info.name} mechanism curve`}
                role="img"
            >
                <!-- grid lines -->
                {#each xTicks as tick}
                    <line
                        x1={tick.pos}
                        y1={margin}
                        x2={tick.pos}
                        y2={svgSize - margin}
                        class="grid-line"
                    />
                    <text x={tick.pos} y={svgSize - 6} class="tick-label" text-anchor="middle">
                        {tick.label}
                    </text>
                {/each}
                {#each yTicks as tick}
                    <line
                        x1={margin}
                        y1={tick.pos}
                        x2={svgSize - margin}
                        y2={tick.pos}
                        class="grid-line"
                    />
                    <text x={margin - 5} y={tick.pos + 3} class="tick-label" text-anchor="end">
                        {tick.label}
                    </text>
                {/each}

                <!-- axes -->
                <line
                    x1={margin}
                    y1={toSvg(0, 0).y}
                    x2={svgSize - margin}
                    y2={toSvg(0, 0).y}
                    class="axis"
                />
                <line
                    x1={toSvg(0, 0).x}
                    y1={margin}
                    x2={toSvg(0, 0).x}
                    y2={svgSize - margin}
                    class="axis"
                />

                <!-- mixture background curves -->
                {#if mechanism === "mixture"}
                    {#each mixtureCurves as mPath, idx}
                        <path
                            d={mPath}
                            class="mixture-bg"
                            stroke-dasharray={idx === 0 ? "6,4" : idx === 1 ? "3,3" : "8,2"}
                        />
                    {/each}
                {/if}

                <!-- main curve -->
                <path d={curvePath} class="curve" />

                <!-- scatter points -->
                {#each scatter as pt}
                    <circle cx={pt.x} cy={pt.y} r="2.4" class="dot" />
                {/each}
            </svg>
            <figcaption>
                {info.name} mechanism &mdash; <code>{info.source}</code>
            </figcaption>
        </figure>

        <!-- formula panel -->
        <div class="formula-panel">
            <div class="formula-box">
                <MathExpr expression={info.formula} inline={false} />
            </div>
            <p class="description">{info.description}</p>
            {#if isDeterministic}
                <p class="note"><mark>No additive noise</mark> &mdash; output is a deterministic function of parents.</p>
            {/if}
            {#if mechanism === "mixture"}
                <p class="note">
                    <small>Dashed curves show individual mechanism types that may be selected per node.</small>
                </p>
            {/if}
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

    .gallery-grid {
        display: grid;
        grid-template-columns: minmax(220px, 1fr) 1fr;
        gap: 1.5rem;
        align-items: start;
    }

    @media (max-width: 700px) {
        .gallery-grid {
            grid-template-columns: 1fr;
        }
    }

    figure {
        margin: 0;
    }

    svg {
        width: 100%;
        max-width: 22rem;
        border: 1px solid var(--pico-muted-border-color);
        border-radius: var(--pico-border-radius);
        background: color-mix(
            in oklab,
            var(--pico-background-color) 87%,
            var(--pico-primary) 13%
        );
    }

    .grid-line {
        stroke: var(--pico-muted-border-color);
        stroke-width: 0.4;
        opacity: 0.5;
    }

    .axis {
        stroke: var(--pico-muted-color);
        stroke-width: 0.8;
    }

    .tick-label {
        font-size: 7px;
        fill: var(--pico-muted-color);
    }

    .curve {
        fill: none;
        stroke: var(--pico-primary);
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    .mixture-bg {
        fill: none;
        stroke: var(--pico-muted-color);
        stroke-width: 1;
        opacity: 0.4;
    }

    .dot {
        fill: var(--pico-contrast);
        opacity: 0.7;
    }

    figcaption {
        margin-top: 0.4rem;
        font-size: 0.85rem;
    }

    .formula-panel {
        padding-top: 0.25rem;
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
        margin-bottom: 0.75rem;
        overflow-x: auto;
    }

    .description {
        font-size: 0.9rem;
        color: var(--pico-muted-color);
    }

    .note {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
</style>

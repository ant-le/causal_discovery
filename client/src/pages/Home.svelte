<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { loadSlim } from "@tsparticles/slim";
    import type { Container } from "@tsparticles/engine";

    let { updateState, theme } = $props();
    let particlesContainer: Container | undefined;
    let particlesEl: HTMLElement;
    const colorName: string = "--pico-contrast";

    $effect(() => {
        document.body.classList.toggle("dark-theme", theme === "dark");

        if (particlesContainer && particlesEl) {
            const styles = getComputedStyle(particlesEl);
            const newColor = styles.getPropertyValue(colorName).trim();
            const options = particlesContainer.options as unknown as {
                particles: {
                    color: { value: string };
                    links: { color: string };
                };
            };
            options.particles.color.value = newColor;
            options.particles.links.color = newColor;
            particlesContainer.refresh();
        }
    });

    onMount(async () => {
        const { tsParticles } = await import("@tsparticles/engine");
        await loadSlim(tsParticles);

        const initialColor = getComputedStyle(particlesEl)
            .getPropertyValue(colorName)
            .trim();

        particlesContainer = await tsParticles.load({
            id: "tsparticles",
            options: {
                fpsLimit: 60,
                interactivity: {
                    events: {
                        onHover: { enable: true, mode: "repulse" },
                    },
                    modes: {
                        repulse: { distance: 100, duration: 0.4 },
                    },
                },
                particles: {
                    color: { value: initialColor },
                    links: {
                        color: initialColor,
                        distance: 150,
                        enable: true,
                        opacity: 0.15,
                        width: 1,
                    },
                    collisions: { enable: true },
                    move: {
                        direction: "none",
                        enable: true,
                        outModes: { default: "bounce" },
                        random: false,
                        speed: 0.8,
                        straight: false,
                    },
                    number: {
                        density: { enable: true, width: 800, height: 800 },
                        value: 60,
                    },
                    opacity: { value: 0.15 },
                    shape: { type: "circle" },
                    size: { value: { min: 1, max: 4 } },
                },
                detectRetina: true,
            },
        });
    });

    onDestroy(() => {
        if (particlesContainer) {
            particlesContainer.destroy();
        }
    });
</script>

<div
    bind:this={particlesEl}
    id="tsparticles"
    class="background-particles"
></div>

<div class="hero-wrapper">
    <!-- Hero Section -->
    <section class="hero container">
        <hgroup>
            <p class="kicker">Master Thesis &middot; TU Wien 2025</p>
            <h1>Benchmarking Bayesian<br />Causal Discovery</h1>
            <p class="subtitle">
                Comparing amortized meta-learning and explicit inference
                under controlled distribution shifts
            </p>
        </hgroup>

        <p class="author">
            <strong>Anton Lechuga</strong>
        </p>

        <div class="hero-actions">
            <button onclick={() => updateState("Motivation")}>
                Explore the project
            </button>
            <a
                href="https://github.com/ant-le/causal_discovery"
                target="_blank"
                rel="noopener noreferrer"
                role="button"
                class="outline secondary"
            >
                View on GitHub
            </a>
        </div>
    </section>

    <!-- Research Questions -->
    <section class="container">
        <h2 class="section-heading">Research Questions</h2>
        <div class="grid rq-grid">
            <div class="rq-card"
                role="button"
                tabindex="0"
                onclick={() => updateState("Motivation")}
                onkeydown={(e) => { if (e.key === 'Enter') updateState("Motivation"); }}
            >
                <article>
                    <header>RQ1</header>
                    <h3>OOD Robustness</h3>
                    <p>
                        How robust are amortized Bayesian causal discovery methods
                        compared to explicit inference baselines under controlled
                        distribution shifts in graph topology, mechanism family,
                        and noise regime?
                    </p>
                    <small class="rq-tag">Generalization &middot; Shift Evaluation</small>
                </article>
            </div>

            <div class="rq-card"
                role="button"
                tabindex="0"
                onclick={() => updateState("Motivation")}
                onkeydown={(e) => { if (e.key === 'Enter') updateState("Motivation"); }}
            >
                <article>
                    <header>RQ2</header>
                    <h3>Full-SCM Posterior</h3>
                    <p>
                        What are the runtime and accuracy trade-offs when moving
                        from graph-only posteriors to richer posterior objects that
                        include both structure and mechanism uncertainty?
                    </p>
                    <small class="rq-tag">Posterior Quality &middot; Scalability</small>
                </article>
            </div>
        </div>
    </section>

    <!-- At a Glance -->
    <section class="container">
        <h2 class="section-heading">At a Glance</h2>
        <div class="grid stats-grid">
            <article class="stat-card">
                <p class="stat-number">5</p>
                <p class="stat-label">Models</p>
                <small>Avici, BCNP, DiBS, BayesDAG, Random</small>
            </article>
            <article class="stat-card">
                <p class="stat-number">7</p>
                <p class="stat-label">Mechanism Families</p>
                <small>Linear, MLP, Periodic, GP, PNL, &hellip;</small>
            </article>
            <article class="stat-card">
                <p class="stat-number">8</p>
                <p class="stat-label">Evaluation Metrics</p>
                <small>E-SHD, E-F1, AUC, SID, I-NIL, &hellip;</small>
            </article>
            <article class="stat-card">
                <p class="stat-number">3</p>
                <p class="stat-label">Graph Families</p>
                <small>Erd&#337;s&ndash;R&eacute;nyi, Scale-Free, SBM</small>
            </article>
        </div>
    </section>

    <!-- Method Overview -->
    <section class="container">
        <h2 class="section-heading">Two Inference Paradigms</h2>
        <div class="grid paradigm-grid">
            <article>
                <header>Amortized Inference</header>
                <p>
                    Meta-learned predictors (Avici, BCNP) are pre-trained on
                    diverse synthetic SCM tasks. At test time, they produce
                    posterior graph samples in a single forward pass&mdash;fast
                    but potentially fragile under distribution shift.
                </p>
                <ul>
                    <li>Avici: max-pool encoder + acyclicity constraint</li>
                    <li>BCNP: attention encoder + Sinkhorn permutation DAGs</li>
                </ul>
            </article>
            <article>
                <header>Explicit Inference</header>
                <p>
                    Per-dataset optimization methods (DiBS, BayesDAG) run
                    gradient-based posterior inference on each test instance
                    independently&mdash;slower but potentially more robust to
                    unseen data distributions.
                </p>
                <ul>
                    <li>DiBS: SVGD-based differentiable structure learning</li>
                    <li>BayesDAG: Sinkhorn-based variational inference</li>
                </ul>
            </article>
        </div>
    </section>

    <!-- Pipeline -->
    <section class="container">
        <h2 class="section-heading">Benchmark Pipeline</h2>
        <div class="grid pipeline-steps">
            <article class="step-card">
                <p class="step-num">1</p>
                <h4>Configure</h4>
                <p>Hydra-driven experiment setup: data family, model, trainer, cluster launcher.</p>
            </article>
            <article class="step-card">
                <p class="step-num">2</p>
                <h4>Generate</h4>
                <p>Sample reproducible SCM tasks with disjoint train/val/test splits.</p>
            </article>
            <article class="step-card">
                <p class="step-num">3</p>
                <h4>Infer</h4>
                <p>Run amortized pre-training or explicit per-dataset posterior inference.</p>
            </article>
            <article class="step-card">
                <p class="step-num">4</p>
                <h4>Evaluate</h4>
                <p>Score graph quality, posterior fidelity, and interventional metrics.</p>
            </article>
        </div>
    </section>

    <!-- Navigate -->
    <section class="container navigate-section">
        <h2 class="section-heading">Explore</h2>
        <div class="grid nav-grid">
            <a href="motivation" class="nav-link" role="button" class:outline={true}
                onclick={(e) => { e.preventDefault(); updateState("Motivation"); }}>
                Motivation
            </a>
            <a href="methodology" class="nav-link" role="button" class:outline={true}
                onclick={(e) => { e.preventDefault(); updateState("Methodology"); }}>
                Methodology
            </a>
            <a href="benchmark" class="nav-link" role="button" class:outline={true}
                onclick={(e) => { e.preventDefault(); updateState("Benchmark"); }}>
                Benchmark
            </a>
            <a href="results" class="nav-link" role="button" class:outline={true}
                onclick={(e) => { e.preventDefault(); updateState("Results"); }}>
                Results
            </a>
            <a href="appendix" class="nav-link secondary" role="button" class:outline={true}
                onclick={(e) => { e.preventDefault(); updateState("Appendix"); }}>
                Appendix
            </a>
        </div>
    </section>
</div>

<style>
    .background-particles {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
    }

    .hero-wrapper {
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding-top: clamp(2.5rem, 8vh, 5rem);
        padding-bottom: 2rem;
    }

    .kicker {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--pico-secondary);
        margin-bottom: 0.75rem;
    }

    .hero h1 {
        font-size: clamp(1.8rem, 4vw, 3rem);
        line-height: 1.2;
        margin-bottom: 0.75rem;
    }

    .subtitle {
        max-width: 38rem;
        margin: 0 auto 1.5rem;
        color: var(--pico-secondary);
        font-size: clamp(0.95rem, 1.2vw, 1.1rem);
    }

    .author {
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }

    .hero-actions {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }

    /* Section headings */
    .section-heading {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* RQ Cards */
    .rq-grid {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }

    .rq-card {
        cursor: pointer;
        transition: transform 0.2s ease;
    }

    .rq-card:hover {
        transform: translateY(-2px);
    }

    .rq-card:hover article {
        border-color: var(--pico-primary);
    }

    .rq-card article {
        height: 100%;
        transition: border-color 0.2s ease;
    }

    .rq-card header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--pico-primary);
        font-weight: 700;
    }

    .rq-card h3 {
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }

    .rq-tag {
        display: inline-block;
        margin-top: 0.5rem;
        color: var(--pico-secondary);
        font-style: italic;
    }

    /* Stats */
    .stats-grid {
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        text-align: center;
    }

    .stat-card {
        margin: 0;
    }

    .stat-number {
        font-family: var(--display-font-family);
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0;
        color: var(--pico-primary);
    }

    .stat-label {
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    .stat-card small {
        color: var(--pico-secondary);
    }

    /* Paradigm grid */
    .paradigm-grid {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }

    .paradigm-grid article ul {
        margin-top: 0.5rem;
        padding-left: 1.2rem;
    }

    /* Pipeline steps */
    .pipeline-steps {
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }

    .step-card {
        text-align: center;
        margin: 0;
    }

    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 50%;
        border: 2px solid var(--pico-primary);
        color: var(--pico-primary);
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .step-card h4 {
        margin-bottom: 0.4rem;
    }

    .step-card p:not(.step-num) {
        font-size: 0.92rem;
        margin-bottom: 0;
    }

    /* Navigate section */
    .nav-grid {
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        text-align: center;
    }

    .navigate-section {
        padding-bottom: 2rem;
    }
</style>

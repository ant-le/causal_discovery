<script lang="ts">
    import { onDestroy, onMount } from "svelte";

    interface StepItem {
        title: string;
        summary: string;
    }

    const steps: StepItem[] = [
        {
            title: "Configure",
            summary: "Choose data family, model setup, and runtime controls.",
        },
        {
            title: "Generate Data",
            summary: "Sample train/validation/test tasks from SCM families.",
        },
        {
            title: "Run Inference",
            summary: "Execute amortized pre-training or explicit posterior inference.",
        },
        {
            title: "Evaluate",
            summary: "Score graph quality, posterior quality, and runtime behavior.",
        },
    ];

    let activeStep = $state(0);
    let autoplay = $state(true);
    let intervalHandle: ReturnType<typeof setInterval> | undefined;

    function moveNext(): void {
        activeStep = (activeStep + 1) % steps.length;
    }

    function startAutoplay(): void {
        if (!autoplay || intervalHandle) {
            return;
        }

        intervalHandle = setInterval(moveNext, 2100);
    }

    function stopAutoplay(): void {
        if (intervalHandle) {
            clearInterval(intervalHandle);
            intervalHandle = undefined;
        }
    }

    $effect(() => {
        if (autoplay) {
            startAutoplay();
        } else {
            stopAutoplay();
        }
    });

    onMount(() => {
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            autoplay = false;
        }
    });

    onDestroy(() => {
        stopAutoplay();
    });
</script>

<article>
    <header>
        <strong>Execution Flow</strong>
        <br />
        <small>Animated process map of the benchmark pipeline.</small>
    </header>

    <p>
        Step {activeStep + 1} of {steps.length}: <strong>{steps[activeStep].title}</strong>
    </p>
    <progress max={steps.length - 1} value={activeStep}></progress>

    <div class="grid steps-grid">
        {#each steps as step, index}
            <article class:active={activeStep === index}>
                <header>{index + 1}. {step.title}</header>
                <p>{step.summary}</p>
            </article>
        {/each}
    </div>

    <button
        class="secondary"
        onclick={() => {
            autoplay = !autoplay;
        }}
    >
        {autoplay ? "Pause animation" : "Resume animation"}
    </button>
</article>

<style>
    .steps-grid {
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        margin-bottom: 1rem;
    }

    .steps-grid article {
        margin: 0;
        border: 1px solid var(--pico-muted-border-color);
        transition:
            border-color 0.25s ease,
            transform 0.25s ease;
    }

    .steps-grid article.active {
        border-color: var(--pico-primary);
        transform: translateY(-2px);
    }

    .steps-grid p {
        margin-bottom: 0;
    }
</style>

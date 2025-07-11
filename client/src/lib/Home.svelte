<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { thesisData } from "../data/textData";
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
            console.log(newColor);
            particlesContainer.options.particles.color.value = newColor;
            particlesContainer.options.particles.links.color = newColor;
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
                        onHover: {
                            enable: true,
                            mode: "repulse",
                        },
                        resize: true,
                    },
                    modes: {
                        repulse: {
                            distance: 100,
                            duration: 0.4,
                        },
                    },
                },
                particles: {
                    color: {
                        value: initialColor,
                    },
                    links: {
                        color: initialColor,
                        distance: 150,
                        enable: true,
                        opacity: 0.2,
                        width: 1,
                    },
                    collisions: {
                        enable: true,
                    },
                    move: {
                        direction: "none",
                        enable: true,
                        outModes: {
                            default: "bounce",
                        },
                        random: false,
                        speed: 1,
                        straight: false,
                    },
                    number: {
                        density: {
                            enable: true,
                            area: 800,
                        },
                        value: 80,
                    },
                    opacity: {
                        value: 0.2,
                    },
                    shape: {
                        type: "circle",
                    },
                    size: {
                        value: { min: 1, max: 5 },
                    },
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

<section>
    <article class="round">
        <hgroup>
            <h2>My Master Thesis</h2>
            <cite>{thesisData.title}</cite>
        </hgroup>
        <div role="group">
            <button onclick={() => updateState("motivation")} class="outline"
                >Why this matters</button
            >
            <button onclick={() => updateState("content")} class="outline"
                >Explore My Thesis</button
            >
        </div>
    </article>
</section>

<style>
    .background-particles {
        position: fixed;
        left: 0;
        width: 100%;
        z-index: -2;
    }
    section {
        height: 74vh;
    }
    .round {
        position: relative;
        max-width: 680px;
        margin: 10vh auto;
        border-radius: 50%;
        aspect-ratio: 1 / 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 1.5em;
        justify-content: center;
        border: 2px solid;
        transition: border-color 0.3s ease;
    }

    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    .round:hover::before {
        filter: blur(1.75rem);
        animation-duration: 5s;
    }
    .round::before {
        transition:
            filter 0.3s ease,
            animation-duration 0.3s ease;

        content: "";
        position: absolute;
        top: -5%;
        left: -5%;
        width: 110%;
        height: 110%;
        background: linear-gradient(
            45deg,
            var(--pico-primary),
            var(--pico-secondary)
        );
        border-radius: 50%;
        filter: blur(1.25rem);
        z-index: -1;
        animation: rotate 10s linear infinite;
    }

    div {
        width: 70%;
    }

    div button {
        font-size: 0.8em;
    }
</style>

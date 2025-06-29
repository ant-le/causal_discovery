<script lang="ts">
    import { fade } from "svelte/transition";
    import type { AppStates } from "./types.ts";

    import Navbar from "./lib/Navbar.svelte";
    import Content from "./lib/Content.svelte";
    import Home from "./lib/Home.svelte";
    import Motivation from "./lib/Motivation.svelte";
    import Background from "./lib/Background.svelte";

    let appState: AppStates = $state("home");
    function updateState(newState: AppStates) {
        appState = newState;
    }

    let theme = $state<"light" | "dark">("dark");
    function toggleTheme() {
        theme = theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", theme);
    }
</script>

<header>
    <Navbar {appState} {updateState} {theme} {toggleTheme} />
    <hr />
</header>

{#key appState}
    <main transition:fade={{ duration: 250 }}>
        {#if appState === "motivation"}
            <Motivation />
        {:else if appState === "background"}
            <Background />
        {:else if appState === "content"}
            <Content />
        {:else}
            <Home {updateState} {theme} />
        {/if}
    </main>
    {#if appState !== "home"}
        <footer>
            <hr />
            Code licenced by
        </footer>
    {/if}
{/key}

<style>
    header {
        background-color: var(--pico-background-color);
    }
    main {
        padding: 2rem 1rem;
    }
    footer {
        padding-bottom: 1rem;
    }
</style>

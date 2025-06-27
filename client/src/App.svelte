<script lang="ts">
    import Navbar from "./lib/Navbar.svelte";
    import Content from "./lib/Content.svelte";
    import Home from "./lib/Home.svelte";
    import Motivation from "./lib/Motivation.svelte";
    import { setContext } from "svelte";
    import Background from "./lib/Background.svelte";
    import type { AppStates } from "./types";

    let appState: AppStates = $state({
        name: "home",
    });
    setContext("appState", appState);
</script>

<header>
    <Navbar />
</header>
<main>
    {#if appState.name === "home"}
        <Home />
    {:else if appState.name === "motivation"}
        <Motivation />
    {:else if appState.name === "background"}
        <Background />
    {:else if appState.name === "content"}
        <Content />
    {:else}
        <button class="secondary">Secondary</button>
        <button class="contrast">Contrast</button>
    {/if}
</main>

<style>
    header {
        margin-bottom: 2em;
    }
</style>

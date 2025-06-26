<script lang="ts">
    import "@picocss/pico";
    import Counter from "./lib/Counter.svelte";
    import Navbar from "./lib/Navbar.svelte";
    import Content from "./lib/Content.svelte";
    import { setContext } from "svelte";

    interface AppStates {
        name: "home" | "motivation" | "background" | "content";
    }
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
        <div class="card">
            <Counter />
        </div>
    {:else if appState.name === "motivation"}
        <p>
            Check out <a
                href="https://github.com/sveltejs/kit#readme"
                target="_blank"
                rel="noreferrer">SvelteKit</a
            >, the official Svelte app framework powered by Vite!
        </p>
    {:else if appState.name === "background"}
        <p class="read-the-docs">
            Click on the Vite and Svelte logos to learn more
        </p>
    {:else if appState.name === "content"}
        <Content />
    {:else}
        <button class="secondary">Secondary</button>
        <button class="contrast">Contrast</button>
    {/if}
</main>

<style>
    .read-the-docs {
        color: #888;
    }
</style>

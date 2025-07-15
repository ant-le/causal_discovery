<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import { appMetaData, type AppStates } from "./types.ts";
    import Navbar from "./lib/Navbar.svelte";
    import Content from "./pages/Content.svelte";
    import Home from "./pages/Home.svelte";
    import Footer from "./lib/Footer.svelte";

    // Define App States for navigation
    let appState: AppStates | "Home" = $state("Home");
    function updateState(newState: AppStates | "Home") {
        if (newState !== appState) {
            appState = newState;
        }
    }

    const appStates = Object.keys(appMetaData);

    // Define Color Theme (global)
    let theme = $state<"light" | "dark">("dark");
    function toggleTheme() {
        theme = theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", theme);
    }
</script>

<header>
    <Navbar {appStates} {appState} {updateState} {theme} {toggleTheme} />
</header>
<hr style="margin:0;" />

{#key appState}
    <main
        in:fly={{ y: -30, duration: 300, delay: 300 }}
        out:fade={{ duration: 300 }}
    >
        {#if appState !== "Home"}
            <Content {appState} />
        {:else}
            <Home {updateState} {theme} />
        {/if}
    </main>
    {#if appState !== "Home"}
        <footer>
            <Footer {updateState} />
        </footer>
    {/if}
{/key}

<style>
    header {
        background-color: var(--pico-background-color);
        position: sticky;
        top: 0;
        z-index: 100;
    }
</style>

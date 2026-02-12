<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import { appMetaData, type AppStates } from "./assets/navigation.ts";
    import Navbar from "./lib/Navbar.svelte";
    import Content from "./pages/Content.svelte";
    import Home from "./pages/Home.svelte";
    import Footer from "./lib/Footer.svelte";

    // Define App States for navigation
    let appState: AppStates | "Home" = $state("Home");
    let displayFooter : boolean = $derived(appState !== "Home");
    function updateState(newState: AppStates | "Home") {
        if (newState !== appState) {
            appState = newState;
        }
    }

    const appStates = Object.keys(appMetaData);

    // Define Color Theme (global)
    let theme = $state<"light" | "dark">("dark");

    $effect(() => {
        document.documentElement.setAttribute("data-theme", theme);
    });

    function toggleTheme() {
        theme = theme === "dark" ? "light" : "dark";
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
{/key}
{#if displayFooter}
    <footer>
        <Footer {updateState} />
    </footer>
{/if}

<style>
    header {
        background-color: var(--pico-background-color);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    main {
        flex-grow: 1;
    }
</style>

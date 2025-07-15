<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import {
        appMetaData,
        type AppStates,
        type PageData,
        type PageMetaData,
    } from "../assets/navigation.ts";

    import PageTitle from "../lib/PageTitle.svelte";
    import Sidebar from "../lib/Sidebar.svelte";

    // Define Page States (sections to be rendered)
    let { appState }: { appState: AppStates } = $props();
    let pageMetaData: PageMetaData = $derived(appMetaData[appState]);
    let pageState: string = $state("Introduction");
    let currentPage: PageData = $derived(pageMetaData[pageState]);

    function updatePageState(newState: string) {
        if (newState !== pageState) {
            pageState = newState;
        }
    }
    // 3. Window Management
    let innerWidth: number = $state(0);
    let isMobile: boolean = $derived(innerWidth < 768);
</script>

<svelte:window bind:innerWidth />
<div class="container">
    <Sidebar {pageMetaData} {pageState} {updatePageState} {isMobile} />
    {#key pageState}
        <div
            class="content-wrapper"
            in:fly={{ x: 50, duration: 300, delay: 300 }}
            out:fade={{ duration: 250 }}
        >
            <PageTitle title={pageState} />
            {#each currentPage.sections as section}
                <h3 id={section.title}>{section.title}</h3>
                <section.component />
            {/each}
        </div>
    {/key}
</div>

<style>
    .container {
        display: flex;
        flex-direction: row;
        overflow: hidden;
    }

    .content-wrapper {
        width: 100%;
    }

    /* --- Desktop Styles --- */
    @media (min-width: 769px) {
        .content-wrapper {
            margin-left: 250px;
            padding-right: 4em;
        }
    }

    /* --- Mobile Styles --- */
    /* This handles the layout when the sidebar is not fixed */
    @media (max-width: 768px) {
        .container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 1em;
        }
    }
</style>

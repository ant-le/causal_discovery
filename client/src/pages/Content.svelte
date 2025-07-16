<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import { setContext } from "svelte";
    import {
        appMetaData,
        type AppStates,
        type PageData,
        type PageMetaData,
    } from "../assets/navigation.ts";
    import { type CitationKey } from "../assets/citation.ts";
    import PageTitle from "../lib/PageTitle.svelte";
    import Sidebar from "../lib/Sidebar.svelte";
    import Bibliography from "../lib/Bibliography.svelte";

    // 1. Manage Citations
    let citedKeys: CitationKey[] = $state([]);
    function addCitation(key: CitationKey): number {
        const index = citedKeys.indexOf(key);
        if (index > -1) {
            return index + 1;
        }
        citedKeys.push(key);
        citedKeys = citedKeys;
        return citedKeys.length;
    }
    setContext("addCitation", addCitation);

    // 2. Define Page States (sections to be rendered)
    let { appState }: { appState: AppStates } = $props();
    let pageMetaData: PageMetaData = $derived(appMetaData[appState]);
    let pageState: string = $state("Introduction");
    let currentPage: PageData = $derived(pageMetaData[pageState]);

    function updatePageState(newState: string) {
        if (newState !== pageState) {
            pageState = newState;
            citedKeys = [];
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
                <!-- TODO: Add Section Title here -->
                <section.component />
            {/each}
            <Bibliography {citedKeys} />
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
        }
    }
</style>

<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import type { Component } from "svelte";
    import {
        appMetaData,
        type AppStates,
        type PageMetaData,
    } from "../types.ts";

    import PageTitle from "../lib/PageTitle.svelte";
    import Sidebar from "../lib/Sidebar.svelte";
    import Motivation from "../sections/motivation/Introduction.svelte";
    import AGI from "../sections/motivation/AGI.svelte";
    import Applications from "../sections/motivation/Applications.svelte";

    // TODO: Component Mapping based on appStates
    type CompMapping = Record <string, Component>;
    const components: Record<string, CompMapping> = {
        Motivation: {
            Introduction: Motivation,
            AGI: AGI,
            Applications: Applications,
        },
    };
    // Define Page States (sections to be rendered)
    let { appState }: { appState: AppStates } = $props();
    let pageMetaData: PageMetaData = $derived(appMetaData[appState]);
    let pageState: string = $state("Introduction");
    let currentPage = $derived(pageMetaData[pageState]);

    function updatePageState(newState: string) {
        if (newState !== pageState) {
            pageState = newState;
        }
    }
    // 3. Window Management
    let innerWidth = $state(0);
    let isMobile = $derived(innerWidth < 768);
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
            {#each Object.entries(components[appState]) as [title, sections]}
                <h3>{title}</h3>
            {/each}
            <Motivation />
        </div>
    {/key}
</div>

<style>
    .container {
        display: flex;
        flex-direction: row;
        gap: 2rem;
        padding-left: 0;
        padding-right: 4em;
        overflow: hidden;
    }

    .content-wrapper {
        flex: 1;
    }

    @media (max-width: 768px) {
        .container {
            flex-direction: column;
            padding: 1em;
        }
    }
</style>

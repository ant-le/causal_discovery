<script lang="ts">
    import { fly, fade } from "svelte/transition";
    import { type Component } from "svelte";
    import { setContext } from "svelte";
    import {
        appMetaData,
        type AppStates,
        type SectionsData,
        type PageMetaData,
    } from "../assets/navigation.ts";
    import { type CitationKey } from "../assets/citation.ts";
    import sourceMap from "../assets/data/sourceMap.json" with {
        type: "json",
    };
    import PageTitle from "../lib/PageTitle.svelte";
    import Sidebar from "../lib/Sidebar.svelte";
    import Bibliography from "../lib/Bibliography.svelte";
    import Sources from "../lib/Sources.svelte";

    interface SourceReferences {
        code: string[];
        notes: string[];
        docs: string[];
    }

    type SourceMapData = Record<
        AppStates,
        Record<string, Record<string, SourceReferences>>
    >;

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
    let pageState: string = $state("");
    let currentPage: SectionsData = $derived(pageMetaData[pageState] ?? {});
    let sectionState: string = $state("");
    let CurrentSection: Component | null = $derived(
        currentPage[sectionState] ?? null,
    );
    const sectionSourceMap = sourceMap as SourceMapData;
    let sectionSources: SourceReferences | null = $derived(
        sectionSourceMap[appState]?.[pageState]?.[sectionState] ?? null,
    );

    $effect(() => {
        const pageKeys = Object.keys(pageMetaData);
        if (pageKeys.length === 0) {
            pageState = "";
            sectionState = "";
            return;
        }

        if (!pageKeys.includes(pageState)) {
            pageState = pageKeys[0];
        }

        const sections = pageMetaData[pageState] ?? {};
        const sectionKeys = Object.keys(sections);

        if (sectionKeys.length === 0) {
            sectionState = "";
            return;
        }

        if (!sectionKeys.includes(sectionState)) {
            sectionState = sectionKeys[0];
        }
    });

    function updatePageState(newState: string) {
        if (newState !== pageState && pageMetaData[newState]) {
            pageState = newState;
            const nextSections = pageMetaData[newState] ?? {};
            sectionState = Object.keys(nextSections)[0] ?? "";
            citedKeys = [];
        }
    }

    function updateSectionState(newState: string) {
        if (newState !== sectionState) {
            sectionState = newState;
            citedKeys = [];
        }
    }

    // 3. Window Management
    let innerWidth: number = $state(0);
    let isMobile: boolean = $derived(innerWidth < 768);
</script>

<svelte:window bind:innerWidth />
<div class="container">
    <Sidebar {pageMetaData} {pageState} {updatePageState} {updateSectionState} {isMobile} />
    {#key pageState}
        <div
            class="content-wrapper"
            in:fly={{ x: 50, duration: 300, delay: 300 }}
            out:fade={{ duration: 250 }}
        >
            <PageTitle title={sectionState} />
            {#if CurrentSection}
                <CurrentSection />
            {:else}
                <p>This section is currently being prepared.</p>
            {/if}
            <Sources sources={sectionSources} />
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

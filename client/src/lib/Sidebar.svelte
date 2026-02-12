<script lang="ts">
    import type { PageMetaData, SectionsData } from "../assets/navigation.ts";

    interface Props {
        pageMetaData: PageMetaData;
        pageState: string;
        sectionState: string;
        updatePageState: (newstate: string) => void;
        updateSectionState: (newstate: string) => void;
        isMobile: boolean;
    }
    let {
        pageMetaData,
        pageState,
        sectionState,
        updatePageState,
        updateSectionState,
        isMobile,
    }: Props = $props();

    let sectionMetaData: SectionsData = $derived(pageMetaData[pageState] ?? {});
</script>

<div class="sidebar-panel">
    {#if isMobile}
        <nav aria-label="Section groups" class="group-nav">
            <ul>
                {#each Object.keys(pageMetaData) as pageTitle}
                    <li>
                        <a
                            class="mobile-page-link"
                            class:secondary={pageState !== pageTitle}
                            aria-current={pageState === pageTitle ? "page" : undefined}
                            href={pageTitle}
                            onclick={(e) => {
                                e.preventDefault();
                                updatePageState(pageTitle);
                            }}
                            >{pageTitle}
                        </a>
                    </li>
                {/each}
            </ul>
        </nav>

        <aside class="sections-panel">
            <nav aria-label="Page contents">
                <h3>Contents</h3>
                <ul>
                    {#each Object.keys(sectionMetaData) as sectionTitle}
                        <li>
                            <a
                                class="mobile-section-link"
                                class:secondary={sectionState !== sectionTitle}
                                aria-current={sectionState === sectionTitle
                                    ? "page"
                                    : undefined}
                                href="#{sectionTitle}"
                                onclick={(e) => {
                                    e.preventDefault();
                                    updateSectionState(sectionTitle);
                                }}
                                >{sectionTitle}
                            </a>
                        </li>
                    {/each}
                </ul>
            </nav>
        </aside>
    {:else}
        <aside class="sections-panel">
            <nav aria-label="On this page">
                <h3>On this page</h3>
                {#each Object.entries(pageMetaData) as [pageTitle, pageData]}
                    <details open={pageState === pageTitle}>
                        <summary>
                            <a
                                class:secondary={pageState !== pageTitle}
                                aria-current={pageState === pageTitle
                                    ? "page"
                                    : undefined}
                                href={pageTitle}
                                onclick={(e) => {
                                    e.preventDefault();
                                    updatePageState(pageTitle);
                                }}
                                >{pageTitle}
                            </a>
                        </summary>
                        <ul>
                            {#each Object.keys(pageData) as sectionTitle}
                                <li>
                                    <a
                                        class:secondary={!(pageState ===
                                            pageTitle &&
                                            sectionState === sectionTitle)}
                                        aria-current={pageState === pageTitle &&
                                        sectionState === sectionTitle
                                            ? "page"
                                            : undefined}
                                        href="#{sectionTitle}"
                                        onclick={(e) => {
                                            e.preventDefault();
                                            updateSectionState(sectionTitle);
                                        }}
                                        >{sectionTitle}
                                    </a>
                                </li>
                            {/each}
                        </ul>
                    </details>
                {/each}
            </nav>
        </aside>
    {/if}
</div>

<style>
    .sidebar-panel {
        min-width: 0;
    }

    .sections-panel {
        margin-top: 0;
    }

    .group-nav {
        margin-bottom: 0.75rem;
    }

    @media (min-width: 769px) {
        .sidebar-panel {
            position: sticky;
            top: 5rem;
            align-self: start;
            max-height: calc(100vh - 6rem);
            overflow-y: auto;
            padding-right: 0.25rem;
        }
    }

    @media (max-width: 768px) {
        .sections-panel {
            margin-bottom: 1.25rem;
        }
    }

    .mobile-page-link {
        font-size: 0.88rem;
        font-weight: 600;
    }

    .mobile-section-link {
        font-size: 0.84rem;
    }

    summary::after {
        display: none;
    }

    details[open] > summary {
        margin-bottom: 0.5rem;
    }

    a {
        font-size: 0.95rem;
        text-decoration: none;
    }

    .sections-panel ul {
        margin-left: 0.25rem;
        line-height: 1.3;
    }

    .sections-panel ul > li {
        border-left: 0.1em solid var(--pico-muted-border-color);
        margin: 0.1em;
        overflow-wrap: break-word;
    }

    .sections-panel ul > li:hover {
        border-left: 1px solid var(--pico-primary);
    }

    .sections-panel ul > li a {
        display: block;
        padding: 0.25rem 0;
        color: var(--pico-secondary);
        text-decoration: none;
        transition: color 0.2s ease-in-out;
        margin-left: 0.2em;
    }

    .sections-panel ul > li a:hover {
        filter: brightness(1.1);
    }
</style>

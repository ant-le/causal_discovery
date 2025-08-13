<script lang="ts">
    import type { PageMetaData, SectionsData} from "../assets/navigation.ts";

    interface Props {
        pageMetaData: PageMetaData;
        pageState: string;
        updatePageState: (newstate: string) => void;
        updateSectionState: (newstate: string) => void;
        isMobile: boolean;
    }
    let {
        pageMetaData,
        pageState,
        updatePageState,
        updateSectionState,
        isMobile,
    }: Props = $props();

    let sectionMetaData: SectionsData = $derived(pageMetaData[pageState]);
</script>

{#if isMobile}
    <nav aria-label="breadcrumb">
        <ul>
            {#each Object.keys(pageMetaData) as title}
                <li>
                    <a
                        style="font-size: 13px;"
                        class:secondary={pageState !== title}
                        href={title}
                        onclick={(e) => {
                            e.preventDefault();
                            updatePageState(title);
                        }}
                        >{title}
                    </a>
                </li>
            {/each}
        </ul>
    </nav>
    <aside style="margin-top:0; margin-bottom: 2em;">
        <nav>
            <h3>Contents</h3>
            <ul>
                {#each Object.keys(sectionMetaData) as title}
                    <li>
                        <a
                            style="font-size: 12px;"
                            class="secondary"
                            href="#{title}"
                            onclick={(e) => {
                                e.preventDefault();
                                updateSectionState(title);
                            }}
                            >{title}
                        </a>
                    </li>
                {/each}
            </ul>
        </nav>
    </aside>
{:else}
    <aside>
        <nav>
            <h3>On this page</h3>
            {#each Object.entries(pageMetaData) as [title, pageData]}
                <details open={pageState === title}>
                    <summary>
                        <a
                            class:secondary={pageState !== title}
                            href={title}
                            onclick={(e) => {
                                e.preventDefault();
                                updatePageState(title);
                            }}
                            >{title}
                        </a>
                    </summary>
                    <ul>
                        {#each Object.keys(pageData) as title}
                            <li>
                                <a
                                    class="secondary"
                                    href="#{title}"
                                    onclick={(e) => {
                                        e.preventDefault();
                                        updateSectionState(title);
                                    }}
                                    >{title}
                                </a>
                            </li>
                        {/each}
                    </ul>
                </details>
            {/each}
        </nav>
    </aside>
{/if}

<style>
    @media (min-width: 769px) {
        aside {
            position: fixed;
            width: 250px;
            height: 70vh;
            z-index: 10;
        }
        a {
            font-size: 14px;
        }
    }

    summary {
        padding: 0.25rem 0;
    }

    summary:after {
        display: none;
    }

    details[open] > summary {
        margin-bottom: 0.5rem;
    }

    a {
        font-size: 17px;
        text-decoration: none;
    }

    aside ul {
        margin-left: 0.25rem;
        line-height: 1.5;
    }

    aside ul > li {
        border-left: 1px solid var(--pico-muted-border-color);
        margin: 0.1rem 0;
    }

    aside ul > li:hover {
        border-left: 1px solid var(--pico-primary);
    }

    aside ul > li a {
        display: block;
        padding: 0.25rem 0;
        color: var(--pico-secondary);
        text-decoration: none;
        transition: color 0.2s ease-in-out;
        margin-left: 0.2em;
    }

    aside ul > li a:hover {
        filter: brightness(1.1);
    }
</style>

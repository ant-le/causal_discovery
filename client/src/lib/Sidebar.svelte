<script lang="ts">
    import type { PageMetaData, PageData } from "../assets/navigation.ts";

    interface Props {
        pageMetaData: PageMetaData;
        pageState: string;
        updatePageState: (newState: string) => void;
        isMobile: boolean;
    }
    let { pageMetaData, pageState, updatePageState, isMobile }: Props =
        $props();

    let sectionMetaData: PageData = $derived(pageMetaData[pageState]);
</script>

{#if isMobile}
    <nav aria-label="breadcrumb">
        <ul>
            {#each Object.keys(pageMetaData) as title}
                <li>
                    <a
                        class:secondary={pageState !== title}
                        href="nowthing"
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
    <aside style="margin-top:0">
        <nav>
            <h3>Contents</h3>
            <ul>
                {#each sectionMetaData.sections as section}
                    <li>
                        <a class="secondary" href="#{section.title}"
                            >{section.title}</a
                        >
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
                            href="nothing"
                            onclick={(e) => {
                                e.preventDefault();
                                updatePageState(title);
                            }}
                            >{title}
                        </a>
                    </summary>
                    <ul>
                        {#each pageData.sections as section}
                            <li>
                                <a class="secondary" href="#{section.title}"
                                    >{section.title}</a
                                >
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
            height: 100vh;
            z-index: 10;
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

    aside summary > a {
        text-decoration: none;
    }

    aside ul {
        border-left: 1px solid var(--pico-muted-border-color);
        margin-left: 0.25rem;
        padding-left: 1rem;
        line-height: 1.5;
    }

    aside ul > li {
        margin: 0.1rem 0;
    }

    aside ul > li a {
        display: block;
        padding: 0.25rem 0;
        color: var(--pico-secondary);
        text-decoration: none;
        transition: color 0.2s ease-in-out;
    }

    aside ul > li a:hover {
        color: var(--pico-primary);
    }
</style>

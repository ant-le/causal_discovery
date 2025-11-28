<script lang="ts">
    import { bibliography, type CitationKey } from "../assets/citation.ts";
    let {citedKeys}: {citedKeys: CitationKey[]} = $props();
</script>

{#if citedKeys.length > 0}
    <section id="bibliography">
        <hr />
        <h3>Bibliography</h3>
        <ol>
            {#each citedKeys as key, i}
                {@const entry = bibliography[key]}
                <li>
                    [{i + 1}] {entry.author} ({entry.year}).
                    <i>{entry.title}</i>.
                    {#if entry.url}
                        <a
                            href={entry.url}
                            target="_blank"
                            rel="noopener noreferrer">{entry.url}</a
                        >
                    {/if}
                </li>
            {/each}
        </ol>
    </section>
{/if}

<style>
    #bibliography {
        margin-top: 2em;
        padding: 1em 0;
    }

    #bibliography ol {
        list-style: none;
        padding-left: 0;
    }

    #bibliography li {
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }
</style>

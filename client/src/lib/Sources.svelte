<script lang="ts">
    interface SourceReferences {
        code: string[];
        notes: string[];
        docs: string[];
    }

    let { sources }: { sources: SourceReferences | null } = $props();
</script>

{#if sources}
    <section id="sources">
        <hr />
        <h4>Sources</h4>

        {#if sources.code.length > 0}
            <p><strong>Code</strong></p>
            <ul>
                {#each sources.code as path}
                    <li><code>{path}</code></li>
                {/each}
            </ul>
        {/if}

        {#if sources.notes.length > 0}
            <p><strong>Notes</strong></p>
            <ul>
                {#each sources.notes as path}
                    <li><code>{path}</code></li>
                {/each}
            </ul>
        {/if}

        {#if sources.docs.length > 0}
            <p><strong>Docs</strong></p>
            <ul>
                {#each sources.docs as path}
                    <li><code>{path}</code></li>
                {/each}
            </ul>
        {/if}
    </section>
{/if}

<style>
    #sources ul {
        margin-bottom: 1rem;
    }

    #sources code {
        font-size: 0.85em;
    }
</style>

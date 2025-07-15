<script lang="ts">
    import type { ExplainContent } from "../types.ts";
    let {
        items,
        title = null,
    }: { items: ExplainContent[]; title?: string | null } = $props();
</script>

<article>
    {#if title}
        <header>{title}</header>
    {/if}
    {#each items as item, idx}
        <details name="example">
            <summary><strong>{idx + 1}. {item.title}</strong></summary>
            <p>{@html item.explainText}</p>
        </details>
        {#if idx < items.length - 1}
            <hr />
        {/if}
    {/each}
</article>

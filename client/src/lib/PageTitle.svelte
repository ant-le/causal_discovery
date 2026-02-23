<script lang="ts">
    // Define the shape of your props
    type Props = {
        title: string;
        superscript?: string | null;
        subscript?: string | null;
    };

    let { title, superscript = null, subscript = null }: Props = $props();
</script>

{#if superscript}
    <p class="superscript">{superscript}</p>
{/if}
<div id="title">
    <hgroup>
        <h1>{title}</h1>
        {#if subscript}
            <p>{subscript}</p>
        {/if}
    </hgroup>
</div>

<style>
    hgroup p {
        color: var(--pico-primary);
    }

    .superscript {
        font-size: 0.85rem;
        color: var(--pico-secondary);
        margin-bottom: 0.25rem;
    }
</style>

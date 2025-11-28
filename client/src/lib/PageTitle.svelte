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
    <p style="c">{superscript}</p>
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
</style>

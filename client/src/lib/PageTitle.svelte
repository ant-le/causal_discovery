<script lang="ts">
    // Define the shape of your props
    type Props = {
        title: string;
        superscript?: string | null;
        subscript?: string | null;
    };

    let { title, superscript = null, subscript = null }: Props = $props();
</script>

<section id="title">
    {#if superscript}
        <p style="c">{superscript}</p>
    {/if}
    <hgroup>
        <h1>{title}</h1>
        {#if subscript}
            <p>{subscript}</p>
        {/if}
    </hgroup>
</section>

<style>
    hgroup p {
        color: var(--pico-primary);
    }
</style>

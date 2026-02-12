<script lang="ts">
    import { thesisData } from "../../assets/data/textData.ts";
    import Summary from "../../lib/Summary.svelte";
</script>

<section>
    <p>{thesisData.content}</p>

    {#if thesisData.topics}
        <ul>
            {#each thesisData.topics as topic}
                <li>{@html topic}</li>
            {/each}
        </ul>
    {/if}

    <Summary items={thesisData.explanation} title="Pipeline" />
</section>

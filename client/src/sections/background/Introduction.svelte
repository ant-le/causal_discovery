<script lang="ts">
    import { backgroundData } from "../../assets/data/textData.ts";
</script>

<section>
    <p>
        This project combines causal theory, Bayesian inference, and modern
        meta-learning to benchmark how well different methods recover causal
        structure under realistic dataset shifts.
    </p>

    {#each backgroundData as block, idx}
        <article>
            <header>{idx + 1}. {block.title}</header>
            <p>{block.content}</p>
            {#if block.topics}
                <ul>
                    {#each block.topics as topic}
                        <li>{@html topic}</li>
                    {/each}
                </ul>
            {/if}
        </article>
    {/each}
</section>

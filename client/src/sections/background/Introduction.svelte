<script lang="ts">
    import { backgroundData } from "../../assets/data/textData.ts";
</script>

<section>
    <article>
        <header>Core Concepts</header>
        <p>
            The background keeps one main thread: define causal assumptions,
            infer Bayesian posteriors over plausible SCMs, and evaluate behavior
            under controlled distribution shifts.
        </p>
    </article>

    <div class="grid">
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
    </div>

    <details>
        <summary><strong>Technical Appendix Scope</strong></summary>
        <p>
            The mathematical appendix provides compact reference material for
            calculus, measure theory, probability, and information-theoretic
            quantities used in model design and evaluation.
        </p>
    </details>
</section>

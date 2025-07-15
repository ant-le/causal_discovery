<script lang="ts">
    // TODO: change to causal and DL data
    import { backgroundData } from "../assets/data/textData.ts";
    import PageTitle from "../lib/PageTitle.svelte";
    import type { SideBarData } from "../types.ts";
</script>

<PageTitle
    title="Background"
    subscript="The path to Strong AI"
    desc="
        This Master's thesis is positioned at the convergence of several
        cutting-edge fields in artificial intelligence. To fully appreciate its
        core contributions, a foundational understanding of Causal Inference and
        the challenges of Structural Causal Discovery is essential. Furthermore,
        it requires an examination of the dual role of Large Language Models
        (LLMs) as both powerful knowledge repositories and non-causal reasoners,
        and the pivotal function of Probabilistic Programming as the framework
        for a principled, Bayesian synthesis of knowledge and data.
    "
/>


<section>
    <article>
        {#each backgroundData as section, i}
            <details name="example">
                <summary>{i + 1}. {section.title}</summary>
                <p>{@html section.content}</p>

                {#if section.topics && section.topics.length > 0}
                    <ul>
                        {#each section.topics as topic}
                            <li>{@html topic}</li>
                        {/each}
                    </ul>
                {/if}
            </details>
            {#if i < backgroundData.length - 1}
                <hr />
            {/if}
        {/each}
    </article>
    <p>
        By combining these three pillars—the formalisms of Causal Inference, the
        expert-level knowledge extracted from LLMs, and the robust Bayesian
        framework of Probabilistic Programming—this thesis aims to develop and
        evaluate a novel methodology that pushes the boundaries of automated
        causal discovery, contributing to the development of more intelligent
        and generalizable AI.
    </p>
</section>

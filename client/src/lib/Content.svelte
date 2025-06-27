<script lang="ts">
    import { researchData } from "../data/textData";
    import type { TextData } from "../types";

    // The data for "My Thesis Topic" is directly assigned
    const myThesisTopicData: TextData = researchData["My Thesis Topic"];

    // All Gemini-related state variables removed
</script>

<article>
    <header>
        <h2>My Thesis Proposal: {myThesisTopicData.title}</h2>
    </header>
    <section>
        <p class="thesis-description">
            {myThesisTopicData.description}
        </p>

        {#each myThesisTopicData.topics as item}
            {@html item}
        {/each}

        <div class="thesis-content-note">
            <p>
                This section details the full proposal for my Master's thesis.
                It covers the problem context, the proposed LLM-guided
                probabilistic causal discovery solution, its intersections with
                current AI frontiers, and potential contributions.
            </p>
        </div>
    </section>
</article>

<style>
    /* Component-specific styles for Content.svelte (Pico.css based) */
    article {
        margin-bottom: 4rem;
    }
    h2 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        color: var(--pico-primary);
    }
    h3 {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: var(--pico-secondary-hover);
    }
    h4 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: var(--pico-primary-focus);
    }
    p {
        margin-bottom: 1rem;
        line-height: 1.6;
        color: var(--pico-secondary);
    }
    ul {
        list-style: disc;
        padding-left: 1.5rem;
        margin-bottom: 1rem;
    }
    ul li {
        margin-bottom: 0.5rem;
        color: var(--pico-secondary);
    }
    /* Styles specifically for the content note */
    .thesis-content-note {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--pico-border-color);
        font-style: italic;
        color: var(--pico-secondary);
    }
</style>

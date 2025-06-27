<script lang="ts">
    import { researchData } from "../data/textData";
    import type { TextData } from "../types";

    // The data for "My Thesis Topic" is directly assigned
    const myThesisTopicData: TextData = researchData["My Thesis Topic"];

</script>

<article>
    <header>
        <h2>My Thesis Proposal: {myThesisTopicData.title}</h2>
    </header>
    <section>
        <p class="thesis-description">
            {myThesisTopicData.content}
        </p>

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
    .thesis-content-note {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--pico-border-color);
        font-style: italic;
        color: var(--pico-secondary);
    }
</style>

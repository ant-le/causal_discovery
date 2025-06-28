<script lang="ts">
    import { thesisData } from "../data/textData";

    import type { TextData } from "../types";
</script>

<article>
    <header>
        <h2>My Thesis Proposal: {thesisData.title}</h2>
    </header>
    <section>
        <p class="thesis-description">
            {thesisData.content}
        </p>

        <div class="thesis-content-note">
            <p>
                This section details the full proposal for my Master's thesis.
                It covers the problem context, the proposed LLM-guided
                probabilistic causal discovery solution, its intersections with
                current AI frontiers, and potential contributions.
                <code>2 \cdot 5 = \pi</code>
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

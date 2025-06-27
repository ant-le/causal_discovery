<script lang="ts">
    import { getContext } from "svelte";
    import type { AppStates } from "../types";

    const appState: AppStates = getContext("appState");
</script>

<article>
    <header>
        <hgroup>
            <h1>My Master Thesis</h1>
            <p>
                Agent-enhanced Structural Causal Model discovery within a
                probabilistic programming framework
            </p>
        </hgroup>
    </header>
    This web application presents my Master's thesis, exploring a novel approach
    to build more intelligent and trustworthy AI. This work investigates how Structural
    Causal Discovery (SCD) methods can be enhanced with most recent developments
    in Natural Language Processing (NLP) and Probibalistic Programming to overcome
    challanges presented by Large Language Models (LLMs).
    <footer>
        <div role="group">
            <button
                on:click={() => (appState.name = "motivation")}
                class="outline">Why this matters</button
            >
            <button
                on:click={() => (appState.name = "content")}
                class="outline">Explore My Thesis</button
            >
        </div>
    </footer>
</article>

<style>
    /* Component-specific styles for Home.svelte (Pico.css based) */
    article {
        text-align: center;
        max-width: 880px;
        margin: 15vh auto;
    }

    header h1 {
        font-size: 2rem;
        color: var(--pico-primary);
    }
    div {
        width: 70%;
    }
</style>

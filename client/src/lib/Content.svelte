<script lang="ts">
    import { thesisData } from "../data/textData";
    import Math from "./Math.svelte";
</script>

<header>
    <h1>{thesisData.title}</h1>
</header>
<section>
    <p>{thesisData.content}</p>
    <p>
        This section details the full proposal for my Master's thesis. It covers
        the problem context, the proposed LLM-guided probabilistic causal
        discovery solution, its intersections with current AI frontiers, and
        potential contributions.
        <Math expression="2 \cdot 5 = \pi" inline={true} />
    </p>
</section>

<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <ol>
        <li>
            Use the <strong>BNCP</strong> meta-learning approach for generating
            plausible DAGs <Math expression={"\\mathbb{P}(G\\mid D)"} /> and then
            use Bayesian methods for learning functional dependencies
            <Math expression={"\\mathbb{P}(\\Theta \\mid G,D)"} /> for obtaining
            full posterior <Math expression={"\\mathbb{P}(\\Theta,G\\mid D)"} />
        </li>
        <li>
            <strong>Causal Representation Learning:</strong> Define causal structures
            as an abstraction layer (latent variables) and try to estimate theses
            from obserational data. Use this to make procedure extendable to potentially
            unobserved vars and different data (text, images, audio)
        </li>
        <li>Search for Open Weights</li>
        <li>Tiss: Working Title</li>
        <li>Rudimental Implementation - Problem Formulation</li>
    </ol>
</section>

<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <p>
        From a Bayesian viewpoint, the we want to estimate the underlying
        <strong>SCM</strong> by jointly looking at the edge structure and the functional
        forms:
    </p>
    <Math
        expression={"\\mathbb{P}(G,\\Theta,D)=\\mathbb{P}(D\\mid G,\\Theta)\\cdot\\mathbb{P}(G,\\Theta)"}
        inline={false}
    />
    Now, as we are interested in the SCMs, conditioning on the data gives us:
    <Math
        expression={"\\mathbb{P}(G,\\Theta\\mid D)=\\frac{\\mathbb{P}(D,G,\\Theta)}{\\mathbb{P}(D)}"}
        inline={false}
    />
    Calculating the joint probability numerically is however intracable as the number
    of graphs and functional forms grow exponentially with data.
</section>

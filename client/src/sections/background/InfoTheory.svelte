<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section id="entropy">
    <p>
        When talking about entropy, we operate on probability distributions. So
        we first fix a sample space and two random variables:

        <Math expression={"(\\Omega, \\Alpha, \\mathbb{P})"} inline={false} />
        <Math expression={"X:\\Omega \\rightarrow \\mathbb{R}"} inline={false} />
        <Math expression={"Y:\\Omega \\rightarrow \\mathbb{R}"} inline={false} />
    Next, we define probability distributions on these random variables and 
        fix them:
        <Math expression={"P=\\mathbb{P}_X"} inline={false} />
        <Math expression={"Q=\\mathbb{P}_Y"} inline={false} />
        The probability mass functions of the distributions are given by:
        <Math expression={"(p_x)_{x \\in X(\\Omega)}, \\quad (q_x)_{x \\in X(\\Omega)}"} inline={false} />

    </p>
    <p>
        When we only have a single probability distribution present, the entropy
        the <strong>entropy</strong> of that distribution is given by:
    </p>
    <Math
        expression={"H(P)=-\\sum_{x\\in X(\\Omega)}{p_x \\ln (p_x)}"}
        inline={false}
    />
</section>

<section id="cross-entropy">
    <p>
        Next, if there is a second distribution present which shows the ground
        truth, we can compare it with the <strong>cross-entropy</strong>:
    </p>
    <Math
        expression={"H(P,Q)=-\\sum_{x \\in X(\\Omega)} p_x \\ln q_x"}
        inline={false}
    />
</section>

<section id="kl-divergence">
    <p>When comparing the entropy to the cross-entropy, we arrive at</p>
    <Math
        expression={` \\begin{align*} D_{KL}(P \\mid \\mid Q) &=H(P,Q) - H(P) \\\\ &=\\sum_{x \\in X(\\Omega)} p_x \\ln (p_x)
        - \\sum_{x \\in X(\\Omega)} p_x \\ln (q_x)\\\\ &=\\sum_{x \\in X(\\Omega)} p_x \\ln \\left(\\frac{p_x}{q_x}\\right)
        \\end{align*} `}
        inline={false}
    />
</section>

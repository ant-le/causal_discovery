<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <article>
        <header>Appendix Goal</header>
        <p>
            Information-theoretic quantities summarize uncertainty and mismatch
            between distributions, which is useful for posterior comparison.
        </p>
    </article>

    <article>
        <header>Core Quantities</header>
        <Math expression={"H(P)=-\\sum_x p_x \\log p_x"} inline={false} />
        <Math expression={"H(P,Q)=-\\sum_x p_x \\log q_x"} inline={false} />
        <Math
            expression={"D_{KL}(P\\|Q)=\\sum_x p_x \\log \\left(\\frac{p_x}{q_x}\\right)"}
            inline={false}
        />
    </article>

    <details>
        <summary><strong>Interpretation</strong></summary>
        <ul>
            <li>Entropy measures uncertainty within one distribution.</li>
            <li>Cross-entropy measures coding loss under a reference model.</li>
            <li>
                KL divergence measures directional mismatch between candidate and
                target distributions.
            </li>
        </ul>
    </details>

    <details>
        <summary><strong>Use in This Project</strong></summary>
        <p>
            These quantities motivate posterior-sensitive evaluation: two methods
            can have similar edge recovery while assigning substantially
            different uncertainty mass over plausible causal mechanisms.
        </p>
    </details>
</section>

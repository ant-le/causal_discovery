<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section>
    <article>
        <header>Appendix Goal</header>
        <p>
            Probability theory turns measure-theoretic foundations into practical
            tools for uncertainty quantification, conditioning, and expectation.
        </p>
    </article>

    <article>
        <header>Probability Space</header>
        <p>
            We work with
            <Math expression={"(\Omega, \mathcal{A}, \mathbb{P})"} />,
            where events in <Math expression={"\mathcal{A}"} /> receive
            probabilities from <Math expression={"\mathbb{P}"} />.
        </p>
        <Math
            expression={"\mathbb{P}(A \mid B)=\frac{\mathbb{P}(A\cap B)}{\mathbb{P}(B)}"}
            inline={false}
        />
    </article>

    <details open>
        <summary><strong>Core Definitions</strong></summary>
        <Definition key="probMeasure" />
        <Definition key="conditionalProb" />
        <Definition key="randomVar" />
        <Definition key="expectation" />
    </details>

    <details>
        <summary><strong>Key Consequences for Benchmarking</strong></summary>
        <ul>
            <li>
                Conditional probability formalizes posterior updating from data.
            </li>
            <li>
                Expectations and variances summarize central tendency and
                uncertainty in predictive behavior.
            </li>
            <li>
                Independence assumptions shape both model design and diagnostic
                interpretation.
            </li>
        </ul>
    </details>
</section>

<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
</script>

<section>
    <!-- 1. Problem Statement -->
    <article>
        <header>The Causal Discovery Problem</header>
        <p>
            Causal discovery asks: given observational data
            <Math expression={"D = \\{x^{(1)}, \\ldots, x^{(n)}\\}"} inline />,
            drawn i.i.d. from an unknown structural causal model, can we recover
            the underlying directed acyclic graph (DAG)
            <Math expression={"G"} inline /> that generated the data?
        </p>
        <p>
            Unlike standard supervised learning, the target is not a function
            output but the <em>generative structure</em> itself. The graph
            <Math expression={"G"} inline /> encodes which variables causally
            influence which others, and knowledge of
            <Math expression={"G"} inline /> enables predicting the effects of
            interventions that go beyond the observed distribution
            <Cite key="pearl2009" />.
        </p>
    </article>

    <!-- 2. Identifiability Assumptions -->
    <article>
        <header>Identifiability Assumptions</header>
        <p>
            Recovering <Math expression={"G"} inline /> from observational data
            alone requires structural assumptions. This benchmark assumes:
        </p>

        <details open>
            <summary><strong>Causal Markov Condition</strong></summary>
            <p>
                Each variable is conditionally independent of its non-descendants
                given its parents. This links the graph to the observational
                distribution via the Markov factorisation:
            </p>
            <Math
                expression={"p(X_1, \\ldots, X_d) = \\prod_{i=1}^{d} p(X_i \\mid \\mathrm{PA}_i^G)"}
                inline={false}
            />
        </details>

        <details open>
            <summary><strong>Faithfulness</strong></summary>
            <p>
                Every conditional independence in
                <Math expression={"p(X)"} inline /> corresponds to a
                <em>d</em>-separation statement in <Math expression={"G"} inline />.
                Equivalently, no conditional independence holds
                &ldquo;by accident&rdquo; due to parameter fine-tuning. This
                ensures that the distribution is maximally informative about the
                graph.
            </p>
        </details>

        <details open>
            <summary><strong>Causal Sufficiency</strong></summary>
            <p>
                All common causes of measured variables are themselves measured.
                There are no hidden confounders. This is a strong assumption, but
                standard in the synthetic benchmark setting used here
                <Cite key="peters2017" />.
            </p>
        </details>

        <details>
            <summary><strong>Acyclicity</strong></summary>
            <p>
                The causal graph <Math expression={"G"} inline /> is a DAG
                &mdash; there are no feedback loops. All five models in the
                benchmark enforce or encourage acyclicity, either structurally
                (BCNP, Random) or via soft constraints (Avici, DiBS, BayesDAG).
            </p>
        </details>
    </article>

    <!-- 3. Bayesian Formulation -->
    <article>
        <header>Bayesian Causal Discovery</header>
        <p>
            Rather than committing to a single point-estimate graph, the Bayesian
            approach maintains a <em>posterior distribution</em> over graphs
            <Cite key="causalDiscovery" />. Given data
            <Math expression={"D"} inline />, Bayes&rsquo; rule gives:
        </p>

        <article class="definition">
            <header><strong>Graph Posterior</strong></header>
            <Math
                expression={"p(G \\mid D) = \\frac{p(D \\mid G)\\, p(G)}{\\sum_{G' \\in \\mathcal{G}_d} p(D \\mid G')\\, p(G')}"}
                inline={false}
            />
        </article>

        <p>where:</p>
        <ul>
            <li>
                <Math expression={"p(G)"} inline /> is the <strong>graph prior</strong>,
                encoding beliefs about sparsity and structure before seeing data.
            </li>
            <li>
                <Math expression={"p(D \\mid G)"} inline /> is the
                <strong>marginal likelihood</strong>, obtained by integrating out the
                mechanism parameters
                <Math expression={"\\Theta"} inline />:
            </li>
        </ul>

        <Math
            expression={"p(D \\mid G) = \\int p(D \\mid G, \\Theta)\\, p(\\Theta \\mid G)\\, d\\Theta"}
            inline={false}
        />
    </article>

    <!-- 4. Joint vs Marginal Posterior -->
    <article>
        <header>Joint vs Marginal Posterior</header>
        <p>
            In practice, models may target different posterior objects depending on
            whether mechanism parameters are of interest:
        </p>

        <div class="grid">
            <article>
                <header>Marginal posterior</header>
                <Math
                    expression={"p(G \\mid D)"}
                    inline={false}
                />
                <p>
                    Integrates out <Math expression={"\\Theta"} inline />.
                    Sufficient for structure learning, but discards mechanism
                    information.
                </p>
                <small>Used by: Avici, BCNP, Random</small>
            </article>

            <article>
                <header>Joint posterior</header>
                <Math
                    expression={"p(G, \\Theta \\mid D)"}
                    inline={false}
                />
                <p>
                    Retains both structure and parameters. Required for
                    interventional predictions and the full-SCM approximation
                    (RQ2).
                </p>
                <small>Used by: DiBS (Joint mode), BayesDAG</small>
            </article>
        </div>
    </article>

    <!-- 5. Computational Challenge -->
    <article>
        <header>Why Exact Inference Is Intractable</header>
        <p>
            The number of DAGs on <Math expression={"d"} inline /> nodes grows
            super-exponentially. For <Math expression={"d = 20"} inline /> (our
            benchmark setting), the number of possible DAGs exceeds
            <Math expression={"10^{43}"} inline /> &mdash; far beyond exhaustive
            enumeration <Cite key="pearl2009" />.
        </p>
        <p>
            Even evaluating the marginal likelihood
            <Math expression={"p(D \\mid G)"} inline /> requires integrating over
            all mechanism parameters, which is only tractable in closed form for
            restricted model classes (e.g., linear Gaussian with conjugate
            priors). For nonlinear or non-Gaussian mechanisms, approximate
            inference is essential.
        </p>
        <p>
            The benchmark evaluates two families of approximation:
            <strong>amortized</strong> methods that learn a mapping
            <Math expression={"D \\mapsto q(G \\mid D)"} inline /> shared across
            datasets, and <strong>explicit</strong> methods that optimise a
            variational distribution per dataset instance.
        </p>
    </article>

    <!-- 6. From Theory to Practice -->
    <article>
        <header>Benchmark Formalisation</header>
        <p>
            The benchmark concretises this framework by fixing
            <Math expression={"d = 20"} inline /> observed variables, generating
            data from known SCM families, and measuring how well each model
            recovers <Math expression={"G"} inline /> (and optionally
            <Math expression={"\\Theta"} inline />) under both in-distribution
            and out-of-distribution test conditions. The two research questions
            map directly onto posterior quality:
        </p>
        <ul>
            <li>
                <strong>RQ1 (OOD Robustness):</strong> Does the posterior
                <Math expression={"q(G \\mid D)"} inline /> degrade when
                <Math expression={"D"} inline /> comes from graph or mechanism
                families unseen during training?
            </li>
            <li>
                <strong>RQ2 (Full-SCM Approximation):</strong> Can models that
                target the joint posterior
                <Math expression={"p(G, \\Theta \\mid D)"} inline /> produce
                mechanism estimates accurate enough for interventional reasoning?
            </li>
        </ul>
    </article>
</section>

<style>
    .definition {
        border-left: 4px solid var(--pico-primary);
        border-radius: 1em;
    }

    .definition header {
        font-weight: bold;
    }
</style>

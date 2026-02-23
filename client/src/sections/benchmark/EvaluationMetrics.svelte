<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import Math from "../../lib/Math.svelte";
    import MetricCatalog from "../../lib/MetricCatalog.svelte";
</script>

<section>
    <article>
        <header>Evaluation Metrics</header>
        <ContentStatus status="implemented" text="All metrics implemented" />
        <p>
            Eight metrics evaluate model quality across two dimensions:
            <strong>graph structural accuracy</strong> (how well the inferred
            DAG matches the ground truth) and <strong>causal/interventional
            fidelity</strong> (how useful the inferred structure is for
            predicting the effects of interventions)
            <Cite key="evaluation" />.
        </p>
    </article>

    <!-- ═══ GRAPH STRUCTURE METRICS ═════════════════════ -->
    <article>
        <header>Graph Structure Metrics</header>
        <p>
            These metrics compare the posterior graph samples
            <Math expression={"\\{\\hat{G}_k\\}_{k=1}^K \\sim q(G \\mid D)"} inline />
            against the ground-truth adjacency
            <Math expression={"G^*"} inline />.
        </p>

        <details>
            <summary><strong>E-SHD</strong> &mdash; Expected Structural Hamming Distance</summary>
            <Math
                expression={"\\mathrm{E\\text{-}SHD} = \\frac{1}{K}\\sum_{k=1}^K \\sum_{i,j} |G^*_{ij} - \\hat{G}_{k,ij}|"}
                inline={false}
            />
            <p>
                Counts edge additions, deletions, and reversals needed to
                transform each sampled graph into the ground truth, averaged
                over posterior samples.  Lower is better.
            </p>
        </details>

        <details>
            <summary><strong>E-F1</strong> &mdash; Expected Edge F1 Score</summary>
            <Math
                expression={"\\mathrm{E\\text{-}F1} = \\frac{1}{K}\\sum_{k=1}^K \\frac{2\\,\\mathrm{TP}_k}{2\\,\\mathrm{TP}_k + \\mathrm{FP}_k + \\mathrm{FN}_k}"}
                inline={false}
            />
            <p>
                Precision/recall balance for edge recovery, averaged over
                posterior samples.  Higher is better. A value of 0 means no
                correct edges; 1 means perfect recovery.
            </p>
        </details>

        <details>
            <summary><strong>Ancestor F1</strong></summary>
            <p>
                F1 score computed on the <em>transitive closure</em> (ancestor
                reachability matrix) rather than direct edges. This metric is
                more forgiving of Markov-equivalent structures&mdash;it credits
                models that identify the correct causal ancestry even if they
                misplace individual edges.
            </p>
            <Math
                expression={"\\mathrm{AncF1}_k = \\frac{2\\,|R^* \\cap \\hat{R}_k|}{2\\,|R^* \\cap \\hat{R}_k| + |\\hat{R}_k \\setminus R^*| + |R^* \\setminus \\hat{R}_k|}"}
                inline={false}
            />
            <p>
                where <Math expression={"R^*, \\hat{R}_k"} inline /> are the
                reachability matrices of the true and sampled graphs.
                Transitive closure is computed via boolean matrix squaring in
                <Math expression={"O(\\lceil \\log_2 d \\rceil)"} inline /> iterations.
            </p>
        </details>

        <details>
            <summary><strong>AUC</strong> &mdash; Area Under ROC Curve</summary>
            <p>
                Evaluates how well the <em>posterior mean</em> edge probabilities
                <Math expression={"\\bar{p}_{ij} = \\frac{1}{K}\\sum_k \\hat{G}_{k,ij}"} inline />
                can discriminate true edges from non-edges.  Computed with
                class balancing (1000 random subsamples) to handle sparse DAGs
                where non-edges dominate.
            </p>
            <p>
                AUC = 0.5 corresponds to random guessing; 1.0 to perfect
                ranking.
            </p>
        </details>

        <details>
            <summary><strong>Graph NLL</strong> &mdash; Negative Log-Likelihood</summary>
            <Math
                expression={"\\mathrm{NLL} = -\\sum_{i,j}\\bigl[G^*_{ij}\\log \\bar{p}_{ij} + (1-G^*_{ij})\\log(1-\\bar{p}_{ij})\\bigr]"}
                inline={false}
            />
            <p>
                Measures calibration: how well the posterior mean edge
                probabilities match the ground truth under a Bernoulli model.
                A well-calibrated model assigns high probability to true edges
                and low probability to absent edges.  Lower is better.
            </p>
        </details>

        <details>
            <summary><strong>Edge Entropy</strong></summary>
            <Math
                expression={"H = -\\frac{1}{d^2}\\sum_{i,j}\\bigl[\\bar{p}_{ij}\\log \\bar{p}_{ij} + (1-\\bar{p}_{ij})\\log(1-\\bar{p}_{ij})\\bigr]"}
                inline={false}
            />
            <p>
                Mean binary entropy of the posterior edge probabilities.
                Quantifies model <em>uncertainty</em>&mdash;high entropy means
                the model is unsure about many edges.  Does not require ground
                truth.
            </p>
        </details>
    </article>

    <!-- ═══ CAUSAL / INTERVENTIONAL METRICS ════════════ -->
    <article>
        <header>Causal and Interventional Metrics</header>
        <p>
            These metrics assess whether the inferred structure is useful for
            causal reasoning&mdash;specifically for predicting the effects of
            interventions.
        </p>

        <details>
            <summary><strong>E-SID</strong> &mdash; Expected Structural Intervention Distance</summary>
            <p>
                Counts (intervention-node, affected-node) pairs for which the
                true and estimated graphs disagree on the interventional
                distribution <Cite key="sid" />.  Unlike SHD, SID penalises
                only structural differences that <em>matter</em> for
                interventional predictions.
            </p>
            <p>
                SID uses a Bayes-ball fixed-point algorithm to determine
                d-connection in the mutilated graph.  Values range from 0
                (perfect) to <Math expression={"d(d-1)"} inline /> (worst case).
                Averaged over posterior samples to yield E-SID.
            </p>
        </details>

        <details>
            <summary><strong>I-NIL</strong> &mdash; Interventional Negative Log-Likelihood</summary>
            <p>
                The most causally rigorous metric.  For each posterior graph
                sample, a Linear Gaussian SCM is fitted to observational data
                via OLS, then evaluated on held-out interventional data
                (single-node do-interventions):
            </p>
            <Math
                expression={"\\mathrm{I\\text{-}NIL} = -\\log\\frac{1}{K}\\sum_{k=1}^K \\exp\\bigl(-\\mathrm{NLL}_k\\bigr)"}
                inline={false}
            />
            <p>
                where <Math expression={"\\mathrm{NLL}_k"} inline /> is the
                Gaussian negative log-likelihood of the interventional data
                under the SCM fitted with graph sample
                <Math expression={"\\hat{G}_k"} inline />.  Intervened nodes are
                excluded from scoring (their value is a delta function).  Graph
                deduplication is applied for efficiency.
            </p>
        </details>
    </article>

    <!-- ═══ METRIC CATALOG INTERACTIVE ═══════════════════ -->
    <MetricCatalog />
</section>

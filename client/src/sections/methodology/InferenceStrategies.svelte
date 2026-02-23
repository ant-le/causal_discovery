<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import InferenceComparison from "../../lib/InferenceComparison.svelte";
</script>

<section>
    <!-- 1. The Approximation Problem -->
    <article>
        <header>The Posterior Approximation Problem</header>
        <ContentStatus status="implemented" text="Implemented model families" />
        <p>
            The exact posterior
            <Math expression={"p(G \\mid D)"} inline /> requires summing over all
            possible DAGs &mdash; a space that grows super-exponentially in the
            number of variables. For <Math expression={"d = 20"} inline />, this
            is computationally intractable. All practical methods therefore
            construct an <em>approximate</em> posterior
            <Math expression={"q(G \\mid D)"} inline />.
        </p>
        <p>
            The benchmark organises its five models into two approximation
            paradigms, plus a structural baseline:
        </p>
    </article>

    <!-- 2. Amortized Inference -->
    <article>
        <header>Amortized Inference</header>
        <p>
            Amortized methods train a neural network with parameters
            <Math expression={"\\phi"} inline /> to map <em>any</em> dataset
            directly to a posterior approximation:
        </p>
        <Math
            expression={"q_\\phi(G \\mid D) \\;\\text{shared across all datasets}"}
            inline={false}
        />
        <p>
            The key idea is <strong>meta-learning</strong>: during training, the
            model sees many <em>tasks</em>
            <Math expression={"(D_t, G_t)"} inline /> sampled from a distribution
            of SCMs and learns to generalise. At test time, a single forward pass
            produces posterior edge probabilities &mdash; no per-dataset
            optimisation is needed <Cite key="avici" /><Cite key="bcnp" />.
        </p>

        <details open>
            <summary><strong>BCNP</strong> &mdash; Permutation-Marginalized Likelihood</summary>
            <p>
                BCNP parameterises DAGs structurally via
                <Math expression={"A = P \\cdot L \\cdot P^\\top"} inline />,
                where <Math expression={"L"} inline /> is a strictly
                lower-triangular edge matrix and
                <Math expression={"P"} inline /> is a permutation matrix sampled
                via the Gumbel-Sinkhorn operator <Cite key="gumbel" />. The loss
                marginalises over <Math expression={"K"} inline /> permutation
                samples:
            </p>
            <Math
                expression={"\\mathcal{L}_{\\text{BCNP}} = -\\frac{1}{d^2} \\sum_{i,j} \\log \\frac{1}{K} \\sum_{k=1}^{K} \\text{Bern}(G_{ij}^* \\mid \\hat{A}_{ij}^{(k)})"}
                inline={false}
            />
            <p>
                This structural decomposition <em>guarantees acyclicity by
                construction</em> &mdash; every sample is a valid DAG regardless
                of the learned parameters.
            </p>
        </details>

        <details open>
            <summary><strong>Avici</strong> &mdash; Acyclicity-Regularized Classification</summary>
            <p>
                Avici treats each edge as an independent binary classification
                problem, optimising a per-edge BCE loss. Acyclicity is
                encouraged via an augmented Lagrangian penalty based on the
                matrix exponential constraint <Cite key="zheng2018" />:
            </p>
            <Math
                expression={"h(A) = \\operatorname{tr}\\bigl(\\exp(\\sigma(\\text{logits}))\\bigr) - d"}
                inline={false}
            />
            <Math
                expression={"\\mathcal{L}_{\\text{Avici}} = \\text{BCE}(\\hat{A}, G^*) + \\lambda \\cdot h(\\sigma(\\text{logits}))"}
                inline={false}
            />
            <p>
                The dual variable <Math expression={"\\lambda"} inline /> is
                updated via an EMA schedule every 250 training steps, gradually
                increasing the acyclicity pressure.
            </p>
        </details>
    </article>

    <!-- 3. Explicit Inference -->
    <article>
        <header>Explicit Inference</header>
        <p>
            Explicit methods optimise a <em>separate</em> variational
            distribution for each test dataset. There is no shared
            <Math expression={"\\phi"} inline /> across datasets &mdash;
            each instance requires its own inference run:
        </p>
        <Math
            expression={"q_{\\psi_t}(G \\mid D_t) \\;\\text{optimised per dataset } D_t"}
            inline={false}
        />
        <p>
            This is computationally expensive but makes no assumptions about the
            data-generating process at test time.
        </p>

        <details open>
            <summary><strong>DiBS</strong> &mdash; Stein Variational Gradient Descent</summary>
            <p>
                DiBS maintains a set of particles in a continuous latent space
                that represents graph structures. Particles are updated via
                SVGD <Cite key="svgd" />, which combines a score function with
                a repulsive kernel to approximate the posterior with a set of
                diverse graph samples <Cite key="dibs" />.
            </p>
            <p>
                The soft adjacency matrices are thresholded at 0.5 to obtain
                binary DAGs. DiBS supports both <em>marginal</em> mode
                (optimises <Math expression={"p(G \\mid D)"} inline /> only) and
                <em>joint</em> mode (optimises
                <Math expression={"p(G, \\Theta \\mid D)"} inline />), with
                linear and nonlinear likelihood models.
            </p>
        </details>

        <details open>
            <summary><strong>BayesDAG</strong> &mdash; Sinkhorn-Based Variational Inference</summary>
            <p>
                BayesDAG uses a Sinkhorn-based continuous relaxation of
                permutation matrices to parameterise DAGs within a variational
                inference framework <Cite key="bayesdag" />. The model runs
                multiple parallel VI chains (default: 10) for up to 3000
                iterations per dataset instance.
            </p>
            <p>
                Due to dependency constraints (the <code>causica</code> backend
                requires Python 3.8&ndash;3.9), BayesDAG supports execution in an
                isolated subprocess with serialised data exchange.
            </p>
        </details>
    </article>

    <!-- 4. The Random Baseline -->
    <article>
        <header>Structural Baseline</header>
        <p>
            The <strong>Random</strong> model samples DAGs by drawing edges from
            a Bernoulli distribution over the upper-triangular adjacency matrix,
            then applying a random node permutation. This baseline is
            <em>data-independent</em> &mdash; it ignores the observations entirely
            and serves as a lower bound on what structural priors alone can achieve.
        </p>
        <Math
            expression={"L_{ij} \\sim \\text{Bern}(p_{\\text{edge}}), \\quad i < j, \\qquad A = P \\cdot L \\cdot P^\\top"}
            inline={false}
        />
        <p>
            The edge probability
            <Math expression={"p_{\\text{edge}}"} inline /> is matched to the
            expected sparsity of the training distribution to ensure a fair
            comparison on graph-structural metrics.
        </p>
    </article>

    <!-- 5. Paradigm Comparison -->
    <article>
        <header>Paradigm Comparison</header>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th scope="col">Property</th>
                        <th scope="col">Amortized</th>
                        <th scope="col">Explicit</th>
                        <th scope="col">Baseline</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Models</td>
                        <td>BCNP, Avici</td>
                        <td>DiBS, BayesDAG</td>
                        <td>Random</td>
                    </tr>
                    <tr>
                        <td>Training cost</td>
                        <td>High (pre-training)</td>
                        <td>None</td>
                        <td>None</td>
                    </tr>
                    <tr>
                        <td>Test-time cost (per dataset)</td>
                        <td>Single forward pass</td>
                        <td>Full optimisation loop</td>
                        <td>Sampling only</td>
                    </tr>
                    <tr>
                        <td>DAG guarantee</td>
                        <td>Structural (BCNP) / Soft (Avici)</td>
                        <td>Soft (threshold / VI)</td>
                        <td>Structural</td>
                    </tr>
                    <tr>
                        <td>OOD vulnerability</td>
                        <td>Distribution shift from training</td>
                        <td>Optimisation quality only</td>
                        <td>None (data-independent)</td>
                    </tr>
                    <tr>
                        <td>Posterior type</td>
                        <td>Marginal <Math expression={"p(G \\mid D)"} inline /></td>
                        <td>Joint or Marginal</td>
                        <td>Prior only</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </article>

    <!-- 6. The Core Tradeoff -->
    <article>
        <header>The Core Trade-off</header>
        <p>
            The central tension driving this benchmark is between
            <strong>generalisation speed</strong> and <strong>per-instance
            fidelity</strong>:
        </p>
        <ul>
            <li>
                <strong>Amortized models</strong> can process thousands of
                datasets in seconds after pre-training, but their posterior
                quality is bounded by the diversity and relevance of the training
                distribution. When test data comes from an unseen graph or
                mechanism family (OOD), the learned mapping
                <Math expression={"q_\\phi"} inline /> may produce poorly
                calibrated posteriors.
            </li>
            <li>
                <strong>Explicit models</strong> make no distributional
                assumptions and optimise each instance independently, but at
                orders-of-magnitude higher computational cost. Their quality
                depends on convergence of the optimisation procedure rather than
                training distribution match.
            </li>
        </ul>
        <p>
            This trade-off is exactly what RQ1 (OOD robustness) is designed to
            probe: do amortized models retain their speed advantage when the
            test distribution shifts, or does the explicit approach&rsquo;s
            per-instance optimisation become necessary?
        </p>
    </article>

    <!-- 7. Interactive Explorer -->
    <InferenceComparison />
</section>

<style>
    .table-responsive {
        overflow-x: auto;
    }
</style>

<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import Math from "../../lib/Math.svelte";

    type ModelKey = "bcnp" | "avici" | "dibs" | "bayesdag" | "random";
    let expanded = $state<ModelKey | null>(null);

    function toggle(key: ModelKey) {
        expanded = expanded === key ? null : key;
    }
</script>

<section>
    <article>
        <header>Benchmarked Model Families</header>
        <ContentStatus status="implemented" text="Implemented in model factory" />
        <p>
            Five causal discovery methods are evaluated behind a shared
            <code>BaseModel</code> interface.  Two are <strong>amortized</strong>
            (pre-trained on diverse synthetic SCM tasks, then applied in a single
            forward pass) and two are <strong>explicit</strong> (run full posterior
            inference per dataset instance).  A <strong>random</strong> baseline
            anchors all comparisons.
        </p>
    </article>

    <!-- Overview comparison table -->
    <article>
        <header>Comparison at a Glance</header>
        <div class="overflow-auto">
            <table>
                <thead>
                    <tr>
                        <th scope="col">Model</th>
                        <th scope="col">Paradigm</th>
                        <th scope="col">Encoder</th>
                        <th scope="col">DAG Guarantee</th>
                        <th scope="col">Backend</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>BCNP</strong></td>
                        <td>Amortized</td>
                        <td>Alternating attention + cross-attn summary</td>
                        <td>Structural (Gumbel-Sinkhorn)</td>
                        <td>PyTorch</td>
                    </tr>
                    <tr>
                        <td><strong>Avici</strong></td>
                        <td>Amortized</td>
                        <td>Alternating attention + max-pool summary</td>
                        <td>Soft (acyclicity regulariser)</td>
                        <td>PyTorch</td>
                    </tr>
                    <tr>
                        <td><strong>DiBS</strong></td>
                        <td>Explicit</td>
                        <td>&mdash;</td>
                        <td>Internal (SVGD particles)</td>
                        <td>JAX</td>
                    </tr>
                    <tr>
                        <td><strong>BayesDAG</strong></td>
                        <td>Explicit</td>
                        <td>&mdash;</td>
                        <td>Internal (Sinkhorn VI)</td>
                        <td>PyTorch (causica)</td>
                    </tr>
                    <tr>
                        <td><strong>Random</strong></td>
                        <td>Baseline</td>
                        <td>&mdash;</td>
                        <td>Structural (upper-tri + permute)</td>
                        <td>PyTorch</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </article>

    <!-- ─── BCNP ───────────────────────────────────────────── -->
    <article>
        <header>
            <div class="model-header" role="button" tabindex="0"
                onclick={() => toggle("bcnp")}
                onkeydown={(e) => { if (e.key === 'Enter') toggle("bcnp"); }}
            >
                <div>
                    <strong>BCNP</strong> &mdash; Bayesian Causal Neural Process
                    <br /><small>Amortized &middot; Meta-learned posterior &middot; <Cite key="bcnp" /></small>
                </div>
                <span class="toggle-icon">{expanded === "bcnp" ? "▾" : "▸"}</span>
            </div>
        </header>

        <p>
            BCNP is an amortized causal discovery model that meta-learns a
            mapping from observational data to a posterior distribution over
            DAGs.  It uses a shared <em>CausalTransformerEncoder</em> with
            alternating sample-axis and node-axis attention layers, followed by
            a dual-branch decoder that produces permutation-equivariant DAG
            samples via <strong>Gumbel-Sinkhorn permutation sampling</strong>
            <Cite key="gumbel" />.
        </p>

        {#if expanded === "bcnp"}
            <h5>Encoder</h5>
            <p>
                The encoder operates on 4D tensors of shape
                <Math expression={"(B, S, V, d)"} inline />.
                Even-indexed layers attend over the <strong>sample axis</strong>
                (each variable independently aggregates across observations),
                while odd-indexed layers attend over the
                <strong>node axis</strong> (each observation independently mixes
                across variables).  A learned zero-query token is appended along
                the sample axis and extracted after encoding; multi-head
                cross-attention then compresses the per-variable sample
                representations into a summary <Math expression={"(B, V, d)"} inline />.
            </p>

            <h5>Decoder &amp; DAG Construction</h5>
            <p>
                Two parallel transformer-decoder branches process the encoder
                summary:
            </p>
            <ul>
                <li>
                    <strong>L-branch:</strong> A QK dot-product adjacency
                    predictor produces a symmetrised logit matrix
                    <Math expression={"L = (L + L^\\top)/2"} inline />,
                    yielding edge probabilities
                    <Math expression={"\\sigma(L)"} inline />.
                </li>
                <li>
                    <strong>Q-branch:</strong> An MLP maps per-node
                    representations to scalar logits, combined with an ordinal
                    vector via outer product to produce permutation log-scores.
                </li>
            </ul>

            <p>
                DAGs are constructed via the factorisation:
            </p>
            <Math
                expression={"A = P \\cdot \\mathrm{tril}(\\mathbf{1}) \\cdot P^\\top \\odot \\sigma(L)"}
                inline={false}
            />
            <p>
                where <Math expression={"P"} inline /> is a permutation matrix
                sampled via the Gumbel-Sinkhorn procedure (with straight-through
                Hungarian matching for hard assignments), and
                <Math expression={"\\mathrm{tril}"} inline /> is a strictly
                lower-triangular mask ensuring acyclicity by construction.
            </p>

            <h5>Loss Function</h5>
            <p>
                Bernoulli log-likelihood marginalised over
                <Math expression={"K"} inline /> permutation samples:
            </p>
            <Math
                expression={"\\mathcal{L} = -\\frac{1}{|E|}\\sum_e \\left[\\log\\frac{1}{K}\\sum_{k=1}^{K} p(A^*_e \\mid \\sigma(L_e) \\cdot M_{k,e})\\right]"}
                inline={false}
            />
            <p>
                where <Math expression={"M_k = P_k \\cdot \\mathrm{tril} \\cdot P_k^\\top"} inline />
                is the acyclicity mask for permutation sample <Math expression={"k"} inline />.
            </p>

            <h5>Key Hyperparameters</h5>
            <div class="overflow-auto">
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>d_model</code></td><td>64</td><td>Embedding dimension</td></tr>
                        <tr><td><code>nhead</code></td><td>4</td><td>Attention heads</td></tr>
                        <tr><td><code>num_layers</code></td><td>2</td><td>Encoder layers (must be even)</td></tr>
                        <tr><td><code>n_perm_samples</code></td><td>10</td><td>Gumbel-Sinkhorn permutation samples</td></tr>
                        <tr><td><code>sinkhorn_iter</code></td><td>20</td><td>Sinkhorn normalisation iterations</td></tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </article>

    <!-- ─── Avici ──────────────────────────────────────────── -->
    <article>
        <header>
            <div class="model-header" role="button" tabindex="0"
                onclick={() => toggle("avici")}
                onkeydown={(e) => { if (e.key === 'Enter') toggle("avici"); }}
            >
                <div>
                    <strong>Avici</strong> &mdash; Amortized Variational Causal Inference
                    <br /><small>Amortized &middot; Fast test-time inference &middot; <Cite key="avici" /></small>
                </div>
                <span class="toggle-icon">{expanded === "avici" ? "▾" : "▸"}</span>
            </div>
        </header>

        <p>
            Avici shares the same alternating-attention encoder as BCNP, but
            replaces the cross-attention summary with <strong>max-pooling</strong>
            over the sample dimension and uses a simpler adjacency predictor with
            a <strong>differentiable acyclicity constraint</strong> instead of
            structural permutation sampling.
        </p>

        {#if expanded === "avici"}
            <h5>Encoder</h5>
            <p>
                Identical <em>CausalTransformerEncoder</em> with alternating
                sample/node attention.  The summary step applies
                <strong>max-pooling</strong> along the sample axis, yielding a
                fixed-size node representation
                <Math expression={"(B, V, d)"} inline /> regardless of the number
                of observations.  This is simpler and cheaper than BCNP's
                cross-attention summary.
            </p>

            <h5>Adjacency Predictor</h5>
            <p>
                A single-head QK dot-product attention mechanism computes edge
                logits:
            </p>
            <Math
                expression={"\\hat{A}_{ij} = \\frac{q_i^\\top k_j}{\\sqrt{d}} \\quad \\text{with } \\hat{A}_{ii} = 0"}
                inline={false}
            />
            <p>
                Self-loops are zeroed out by multiplying with
                <Math expression={"(\\mathbf{1} - I)"} inline />.
                There is no decoder transformer&mdash;the encoder output feeds
                directly into the predictor.
            </p>

            <h5>Loss Function</h5>
            <p>
                Binary cross-entropy plus an augmented-Lagrangian acyclicity
                regulariser <Cite key="zheng2018" />:
            </p>
            <Math
                expression={"\\mathcal{L} = \\mathrm{BCE}(\\hat{A}, A^*) + \\lambda \\cdot h\\bigl(\\sigma(\\hat{A})\\bigr)"}
                inline={false}
            />
            <p>
                where <Math expression={"h(A) = \\mathrm{tr}(e^A) - d"} inline />
                equals zero if and only if
                <Math expression={"A"} inline /> is acyclic.
                The dual variable <Math expression={"\\lambda"} inline /> is updated
                every 250 steps via exponential-moving-average tracking of
                <Math expression={"h"} inline />.
            </p>

            <h5>Key Hyperparameters</h5>
            <div class="overflow-auto">
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>d_model</code></td><td>64</td><td>Embedding dimension</td></tr>
                        <tr><td><code>nhead</code></td><td>4</td><td>Encoder heads (predictor uses 1)</td></tr>
                        <tr><td><code>num_layers</code></td><td>2</td><td>Encoder layers</td></tr>
                        <tr><td><code>regulariser_lr</code></td><td>1e-4</td><td>Dual-variable learning rate</td></tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </article>

    <!-- ─── DiBS ───────────────────────────────────────────── -->
    <article>
        <header>
            <div class="model-header" role="button" tabindex="0"
                onclick={() => toggle("dibs")}
                onkeydown={(e) => { if (e.key === 'Enter') toggle("dibs"); }}
            >
                <div>
                    <strong>DiBS</strong> &mdash; Differentiable Bayesian Structure Learning
                    <br /><small>Explicit &middot; SVGD particle inference &middot; <Cite key="dibs" /></small>
                </div>
                <span class="toggle-icon">{expanded === "dibs" ? "▾" : "▸"}</span>
            </div>
        </header>

        <p>
            DiBS uses <strong>Stein Variational Gradient Descent</strong>
            (SVGD) <Cite key="svgd" /> to approximate the posterior over DAG
            structures.  It runs full posterior inference from scratch for each
            dataset&mdash;no pre-training required.  The implementation wraps
            the external <code>dibs-lib</code> JAX package.
        </p>

        {#if expanded === "dibs"}
            <h5>Inference Procedure</h5>
            <p>
                A set of <em>particles</em> in a continuous latent space are
                evolved via SVGD, which combines gradient ascent on the log
                posterior with a repulsive kernel term for diversity.
                Two modes are available:
            </p>
            <ul>
                <li>
                    <strong>Joint DiBS:</strong> Jointly infers graph structure
                    <em>and</em> SCM parameters (weights/functions).
                </li>
                <li>
                    <strong>Marginal DiBS:</strong> Marginalises out SCM
                    parameters analytically, inferring only graph structure.
                </li>
            </ul>
            <p>
                Two likelihood models are supported:
                <code>linear</code> (Linear Gaussian SCM) and
                <code>nonlinear</code> (Nonlinear Gaussian SCM).
                After SVGD converges, particle outputs are thresholded at 0.5 to
                produce binary adjacency samples.
            </p>

            <h5>Key Hyperparameters</h5>
            <div class="overflow-auto">
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>mode</code></td><td>nonlinear</td><td>SCM type: linear or nonlinear</td></tr>
                        <tr><td><code>steps</code></td><td>1000</td><td>SVGD iterations</td></tr>
                        <tr><td><code>use_marginal</code></td><td>false</td><td>MarginalDiBS vs JointDiBS</td></tr>
                        <tr><td><code>n_particles</code></td><td>num_samples</td><td>Number of SVGD particles</td></tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </article>

    <!-- ─── BayesDAG ───────────────────────────────────────── -->
    <article>
        <header>
            <div class="model-header" role="button" tabindex="0"
                onclick={() => toggle("bayesdag")}
                onkeydown={(e) => { if (e.key === 'Enter') toggle("bayesdag"); }}
            >
                <div>
                    <strong>BayesDAG</strong> &mdash; Gradient-Based Posterior Inference
                    <br /><small>Explicit &middot; Sinkhorn variational inference &middot; <Cite key="bayesdag" /></small>
                </div>
                <span class="toggle-icon">{expanded === "bayesdag" ? "▾" : "▸"}</span>
            </div>
        </header>

        <p>
            BayesDAG uses <strong>Sinkhorn-based variational inference</strong>
            to approximate the posterior over DAGs.  Like DiBS, it performs
            per-dataset optimisation.  The implementation wraps Microsoft's
            <code>causica</code> library and supports both in-process and
            subprocess execution for dependency isolation.
        </p>

        {#if expanded === "bayesdag"}
            <h5>Inference Procedure</h5>
            <p>
                DAGs are parameterised via Sinkhorn permutation matrices: a
                permutation defines a topological ordering, and a
                lower-triangular matrix defines the edges within that ordering.
                Multiple VI chains run in parallel, and the variational
                distribution is trained with sparsity regularisation.
            </p>
            <p>
                The framework supports both <code>linear</code> and
                <code>nonlinear</code> SCM variants.  The nonlinear variant adds
                optional normalisation layers and residual connections.
            </p>

            <h5>Subprocess Execution</h5>
            <p>
                Since <code>causica</code> requires Python&nbsp;3.8&ndash;3.9,
                inference can be dispatched to an external subprocess with a
                separate Python interpreter.  Data and configs are serialised to
                temporary files, and results are loaded back after completion.
            </p>

            <h5>Key Hyperparameters</h5>
            <div class="overflow-auto">
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>variant</code></td><td>nonlinear</td><td>SCM type</td></tr>
                        <tr><td><code>sinkhorn_n_iter</code></td><td>3000</td><td>Sinkhorn iterations</td></tr>
                        <tr><td><code>num_chains</code></td><td>10</td><td>Parallel VI chains</td></tr>
                        <tr><td><code>lambda_sparse</code></td><td>1.0</td><td>Sparsity weight</td></tr>
                        <tr><td><code>max_epochs</code></td><td>100</td><td>Training epochs per instance</td></tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </article>

    <!-- ─── Random ─────────────────────────────────────────── -->
    <article>
        <header>
            <div class="model-header" role="button" tabindex="0"
                onclick={() => toggle("random")}
                onkeydown={(e) => { if (e.key === 'Enter') toggle("random"); }}
            >
                <div>
                    <strong>Random</strong> &mdash; Sparsity-Matched Baseline
                    <br /><small>Baseline &middot; No inference &middot; No pre-training</small>
                </div>
                <span class="toggle-icon">{expanded === "random" ? "▾" : "▸"}</span>
            </div>
        </header>

        <p>
            A non-learning baseline that generates random DAGs with
            <strong>auto-matched sparsity</strong>.  It provides a calibration
            anchor: any learned method that does not outperform the random
            baseline on a test family is adding no value.
        </p>

        {#if expanded === "random"}
            <h5>Sampling Procedure</h5>
            <p>
                An upper-triangular binary matrix is sampled via
                <Math expression={"\\mathrm{Bernoulli}(p_{\\mathrm{edge}})"} inline />,
                guaranteeing acyclicity.  A random permutation is then applied to
                rows and columns, producing a uniformly random DAG over the given
                sparsity level.  This is equivalent to choosing a random
                topological ordering and independently including each consistent
                edge with probability <Math expression={"p_{\\mathrm{edge}}"} inline />.
            </p>

            <h5>Key Hyperparameters</h5>
            <div class="overflow-auto">
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>p_edge</code></td><td>(required)</td><td>Bernoulli edge probability</td></tr>
                        <tr><td><code>randomize_topological_order</code></td><td>true</td><td>Apply random node permutation</td></tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </article>
</section>

<style>
    .overflow-auto {
        overflow-x: auto;
    }

    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        gap: 1rem;
    }

    .toggle-icon {
        font-size: 1.2rem;
        flex-shrink: 0;
        color: var(--pico-primary);
    }

    h5 {
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--pico-muted-border-color);
        padding-bottom: 0.25rem;
    }
</style>

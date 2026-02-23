<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import Math from "../../lib/Math.svelte";
    import DataFamilyGallery from "../../lib/DataFamilyGallery.svelte";
</script>

<section>
    <article>
        <header>Configurable Data Families</header>
        <ContentStatus status="implemented" text="Implemented generator families" />
        <p>
            Synthetic tasks are generated from composable
            <strong>graph generators</strong> and <strong>mechanism factories</strong>,
            combined into <code>SCMFamily</code> instances.  Each call to
            <code>sample_task(seed)</code> produces a fresh SCM with a random
            graph and per-node mechanisms, then generates data via forward
            ancestral sampling in topological order <Cite key="peters2017" />.
        </p>
    </article>

    <!-- ═══ GRAPH GENERATORS ═══════════════════════════════ -->
    <article>
        <header>Graph Generators</header>
        <p>
            All generators return a binary adjacency matrix of shape
            <Math expression={"(d, d)"} inline /> representing a DAG.
            Acyclicity is guaranteed either structurally (upper-triangular
            sampling) or by orienting edges along a random permutation.
        </p>

        <details>
            <summary><strong>Erd&#337;s&ndash;R&eacute;nyi (ER)</strong></summary>
            <p>
                Each potential edge <Math expression={"i \\to j"} inline /> (with
                <Math expression={"i < j"} inline /> in canonical ordering) is
                independently included with probability
                <Math expression={"p_{\\mathrm{edge}}"} inline />.
                Only the upper triangle is sampled, guaranteeing acyclicity.
            </p>
            <Math
                expression={"P(i \\to j) = p_{\\mathrm{edge}} \\quad \\forall\\; i < j"}
                inline={false}
            />
            <p>
                <strong>Key parameter:</strong> <code>edge_prob</code>
                (aliased <code>sparsity</code>).
                The benchmark uses three sparsity levels corresponding to
                expected in-degree &asymp; 1, 2, and 3 for
                <Math expression={"d = 20"} inline /> nodes.
            </p>
        </details>

        <details>
            <summary><strong>Scale-Free (Barab&aacute;si&ndash;Albert)</strong></summary>
            <p>
                A preferential-attachment process <Cite key="barabasiAlbert" />
                generates an undirected graph where each new node attaches
                <Math expression={"m"} inline /> edges to existing nodes.
                Edges are oriented along a random permutation of nodes
                (lower position &rarr; higher position), producing a DAG with
                heavy-tailed degree distribution.
            </p>
            <p>
                <strong>Key parameter:</strong> <code>m</code> (default 2) &mdash;
                edges per new node.
            </p>
        </details>

        <details>
            <summary><strong>Stochastic Block Model (SBM)</strong></summary>
            <p>
                Nodes are partitioned into <Math expression={"B"} inline /> blocks.
                Edges within a block appear with probability
                <Math expression={"p_{\\mathrm{intra}}"} inline /> and
                across blocks with
                <Math expression={"p_{\\mathrm{inter}}"} inline />.
                The undirected graph is oriented via a random permutation.
            </p>
            <p>
                <strong>Key parameters:</strong>
                <code>n_blocks</code>, <code>p_intra</code>, <code>p_inter</code>.
                The benchmark default uses 4 blocks with
                <Math expression={"p_{\\mathrm{intra}} = 0.6"} inline />,
                <Math expression={"p_{\\mathrm{inter}} = 0.01"} inline />.
            </p>
        </details>

        <details>
            <summary><strong>Mixture</strong></summary>
            <p>
                A weighted mixture of the above generators.  On each call, one
                generator is sampled according to the mixture weights and
                produces the full graph.  Different SCM tasks may therefore
                use different graph topologies.
            </p>
        </details>
    </article>

    <!-- ═══ MECHANISM FAMILIES ════════════════════════════ -->
    <article>
        <header>Mechanism Families</header>
        <p>
            Each node <Math expression={"X_i"} inline /> in the SCM computes its
            value from its parents and exogenous noise via a mechanism
            <Math expression={"f_i"} inline />.  All mechanisms implement the
            interface <code>forward(parents, noise)</code>.
        </p>

        <details>
            <summary><strong>Linear</strong></summary>
            <Math
                expression={"X_i = \\mathbf{w}^\\top \\mathrm{pa}(X_i) + \\sigma_i \\cdot \\varepsilon_i, \\quad \\varepsilon_i \\sim \\mathcal{N}(0,1)"}
                inline={false}
            />
            <p>
                Weights <Math expression={"\\mathbf{w} \\sim \\mathcal{N}(0, s^2 I)"} inline />,
                noise scale <Math expression={"\\sigma_i \\sim \\mathrm{Gamma}(\\alpha, \\beta)"} inline />.
            </p>
            <p>
                <strong>Defaults:</strong> <code>weight_scale=1.0</code>,
                <code>noise_concentration=2.0</code>, <code>noise_rate=2.0</code>.
            </p>
        </details>

        <details>
            <summary><strong>MLP</strong></summary>
            <Math
                expression={"X_i = \\mathrm{MLP}\\bigl([\\mathrm{pa}(X_i);\\, \\varepsilon_i]\\bigr)"}
                inline={false}
            />
            <p>
                A 2-layer network:
                <Math expression={"\\mathrm{Linear}(d_{\\mathrm{pa}}+1, h) \\to \\mathrm{LeakyReLU} \\to \\mathrm{Linear}(h, 1)"} inline />.
                Noise is concatenated as an extra input, so the MLP learns an
                implicit noise model.
            </p>
            <p>
                <strong>Default:</strong> <code>hidden_dim=32</code>.
            </p>
        </details>

        <details>
            <summary><strong>Square</strong></summary>
            <Math
                expression={"X_i = \\bigl(\\mathbf{w}^\\top \\mathrm{pa}(X_i)\\bigr)^2 + \\sigma \\cdot \\varepsilon_i"}
                inline={false}
            />
            <p>
                A nonlinear mechanism that squares the linear combination.
                <strong>Defaults:</strong> <code>weight_scale=1.0</code>,
                <code>noise_scale=0.1</code>.
            </p>
        </details>

        <details>
            <summary><strong>Periodic</strong></summary>
            <Math
                expression={"X_i = \\sin\\bigl(4\\pi\\, \\mathbf{w}^\\top \\mathrm{pa}(X_i)\\bigr) + \\sigma \\cdot \\varepsilon_i"}
                inline={false}
            />
            <p>
                Tests robustness under oscillatory functional forms.
                <strong>Defaults:</strong> <code>weight_scale=1.0</code>,
                <code>noise_scale=0.1</code>.
            </p>
        </details>

        <details>
            <summary><strong>Logistic Map</strong></summary>
            <Math
                expression={"X_i = 4\\,\\sigma(z)\\,(1 - \\sigma(z)), \\quad z = \\mathbf{w}^\\top \\mathrm{pa}(X_i)"}
                inline={false}
            />
            <p>
                A deterministic chaotic map (no additive noise).  The logistic
                sigmoid <Math expression={"\\sigma"} inline /> maps the linear
                combination to <Math expression={"(0,1)"} inline />, and the
                quadratic form produces complex dynamics.
            </p>
        </details>

        <details>
            <summary><strong>GP (Gaussian Process)</strong></summary>
            <p>Two modes are available:</p>
            <ul>
                <li>
                    <strong>Approximate (RFF):</strong> Random Fourier Features
                    <Cite key="rff" /> approximate a mixture of kernels.
                    Each mechanism samples random lengthscales, variances,
                    projection weights, and phases, then computes:
                    <Math
                        expression={"f(x) \\approx \\sqrt{\\tfrac{2}{D}} \\sum_k v_k \\cos(\\mathbf{W}_k x + b_k)"}
                        inline={false}
                    />
                    <strong>Defaults:</strong> <code>rff_dim=512</code>,
                    <code>num_kernels=4</code>.
                </li>
                <li>
                    <strong>Exact:</strong> Draws from a sum-kernel GP prior
                    using Cholesky decomposition. The kernel is a sum of
                    Rational-Quadratic and Exponential-Gamma components with
                    ARD lengthscales.
                    <strong>Defaults:</strong> <code>num_kernel_pairs=2</code>.
                </li>
            </ul>
        </details>

        <details>
            <summary><strong>PNL (Post-Nonlinear)</strong></summary>
            <Math
                expression={"X_i = g\\bigl(f(\\mathrm{pa}(X_i)) + \\sigma \\cdot \\varepsilon_i\\bigr)"}
                inline={false}
            />
            <p>
                Wraps an inner mechanism <Math expression={"f"} inline />
                (default: Linear) with a fixed outer nonlinearity
                <Math expression={"g"} inline />.  Available choices:
                <code>cube</code> (<Math expression={"y^3"} inline />),
                <code>sigmoid</code>, <code>tanh</code>.
            </p>
        </details>

        <details>
            <summary><strong>Mixture</strong></summary>
            <p>
                For each node independently, one mechanism factory is
                sampled from a weighted set.  Different nodes in the
                <em>same graph</em> can have different mechanism types,
                producing heterogeneous SCMs.
            </p>
        </details>
    </article>

    <!-- ═══ BENCHMARK SPLITS ═════════════════════════════ -->
    <article>
        <header>Benchmark Data Configuration</header>
        <p>
            The full benchmark uses <Math expression={"d = 20"} inline /> nodes,
            1000 samples per task, and a rich mixture for training:
        </p>
        <ul>
            <li>
                <strong>Training:</strong> Mixture of ER graphs at 3 sparsity
                levels &times; mixture of Linear, MLP, and GP mechanisms.
            </li>
            <li>
                <strong>ID test:</strong> 9 families (3 mechanisms &times; 3
                sparsities) matching the training distribution.
            </li>
            <li>
                <strong>OOD test:</strong> 7 families including held-out
                mechanisms (Square, Periodic, Logistic Map, PNL) and held-out
                graph topologies (SBM, Scale-Free).
            </li>
        </ul>
        <p>
            Train/val/test splits are disjoint by seed, with SHA-256 hash
            collision checking at setup time.
        </p>
    </article>

    <!-- ═══ INTERACTIVE GALLERY ══════════════════════════ -->
    <DataFamilyGallery />
</section>

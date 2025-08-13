<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section id="definitons">
    Generally, we cosider the functions
    <Math expression={"f:\\mathbb{R}^n \\rightarrow \\mathbb{R}^m"} />,
    <Math expression={"g:\\mathbb{R}^n \\rightarrow \\mathbb{R}^m"} />
    <Math expression={"h:\\mathbb{R}^m \\rightarrow \\mathbb{R}^o"} />
    and some point
    <Math expression={"x \\in \\mathbb{R}^n"} /> for this section.
</section>

<section id="convergence">
    <p>
        The assessment of continuity requires a measure of distance. For
        generalising the notion of distance to higher dimensions, I will use the <b
            >euclidean distance</b
        > and we start with defining the convergence of a sequence to some point:
    </p>
    <Definition key="convergence" />
    <p>
        Note that convergence in a muli-dimensional setting can look much more
        complicated compared to the one-dimensional case.
    </p>
</section>

<section id="continuity">
    <p>
        With the notion of convergence we can now turn to defining continuity,
        which is a <b>point-wise</b> property.
    </p>
    <Definition key="continuity" />
</section>

<section id="derivatives">
    <p>
        Next, we look at derivates of continous functions. We will start with
        <b>partial</b> derivates. Notation-wise, we will represent a function as <Math
            expression={"f(\\vec{x})=f(x_1,x_2,\\dots,x_n)"}
        />. In the case of partial derivates, we fix all variables to some
        value. Hence, we can say that we look at the functions behavior in a
        single direction wehere the partial derivates is consitites the function
        explaining that behavior.
    </p>
    <Definition key="partial_derivatives" />
    <p>
        Now, we want to now if a function is differentiable. That is, we want to
        investigate how/if a function differs at any two infinitally close
        points (e.g. differentiate it). We do this by finding a linear
        approximation of the slope at any point and the check the quality of
        that approximation at that point. Again, this definition is defined
        point-wisely.
    </p>
    <Definition key="differentiable" />
    <p>
        Hence, the derivate at some point is a linear map in the multidiemsional
        case, which can also be represented in matrix representation with the
        <b>Jacobian Matrix</b> which can be calculated with the partial
        derivates if we know that a function is totally differentiable:
        <Math
            expression={`J_f(\\tilde{x})h=\\left( \\frac{\\partial f}{\\partial
            x_1}(\\tilde{x}),\\dots,\\frac{\\partial f}{\\partial x_n}(\\tilde{x}) \\right)`}
        />.
    </p>
</section>

<section id="chain-rule">
    <p>
        The total differentiation is linear, meaning that for two functions
        <Math expression="f,g" /> differentiable at
        <Math expression={"\\tilde{x}\\in\\mathbb{R}^n"} />
        holds that:
        <!-- TODO: Add component for Lemmas -->
    </p>
    <ul>
        <li>
            <b>Sum Rule</b>: <Math
                expression={"f+g:\\mathbb{R}^n\\rightarrow \\mathbb{R}^m"}
            /> is totally differentiable at
            <Math expression={"\\tilde{x}"} /> and
            <Math
                expression={"J_{f+g}(\\tilde{x})=J_f(\\tilde{x})+J_g(\\tilde{x})"}
            />
        </li>
        <li>
            <b>Factor Rule</b>: <Math
                expression={`\\forall\\lambda\\in\\mathbb{R},\\lambda
                f:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m`}
                inline={true}
            /> is totally differentiable at
            <Math expression={"\\tilde{x}"} /> and
            <Math
                expression={"J_{\\lambda f}(\\tilde{x})=\\lambda J_f(\\tilde{x})"}
            />
        </li>
        <li>
            <b>Chain Rule</b>:
            <Math
                expression={"(f \\circ h):\\mathbb{R}^n\\rightarrow \\mathbb{R}^o"}
            /> is totally differentiable at
            <Math expression={"\\tilde{x},f(\\tilde{x})"} /> and
            <Math
                expression={"J_{f\\circ h}(\\tilde{x})=J_f(g(\\tilde{x}))+J_g(\\tilde{x})"}
            />. In this case, we now can use matrix multiplication of the linear
            maps.
        </li>
    </ul>
</section>

<section id="gradient">
    <p>
        Next, we could also be interested in the rate of change of a function in
        any direction besides the axis (e.g. parial derivates). For that we need
        to introduce an additional vector. For this, we only consider function
        mapping to a scalar range
        <Math expression={"g:\\mathbb{R}^n\\rightarrow\\mathbb{R}"} />.
    </p>
    <Definition key="directional_derivatives" />
    <p>
        As we can see, partial derivatives are here only a special case where
        <Math expression="v" /> is the respective basis vector.
    </p>
    <p>
        Now, we can look at the gradient of a function. Gradients are hereby the
        maximal directional derivate of a function
        <Math expression="q" />. Note that this is one of the reasons, why the
        loss in Neural Networks is always a scalar value. For every input vector
        in the domain, the gradient is a vector pointing in some direction.
    </p>
    <Definition key="gradient" />
    <p>
        Gradient shows with what change of values of the functions would change
        the most. That's why it is important for finding a stationary state of a
        function. However, since the gradient is defined point-wisely, it is a <b
            >local property</b
        > and hence does not guarantee finding global extrema.
    </p>
</section>

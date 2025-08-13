<script lang="ts">
    //TODO: Definitions are prbably needed up to MCMC and KL Divergence
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section id="prob-measures">
    <p>
        A probability measure can be formulated as a measure known from <strong
            >measure theory</strong
        >. Herefore, we define a
        <Math expression={"\\text{sample space }\\Omega"} inline={false} />
        <Math
            expression={"\\sigma\\text{-algebra}:\\Alpha \\subseteq \\Omega"}
            inline={false}
        />on which we operate. We call elements of the algebra
        <Math expression={"\\text{events}: A \\in \\Alpha"} inline={false} />
    </p>
    <Definition key="probMeasure" />
    <p>
        We can distinguish between two case, when defining a probability
        measure:
    </p>
    <div role="group">
        <div>
            <h6>Discrete Case</h6>
            <p>
                For a probability mass function <Math
                    expression={"(p_w)_{w\\in\\Omega}"}
                />, we define the probability measure as:
                <Math
                    expression={"\\mathbb{P}(A) :=\\sum_{w\\in\\Omega}p_w"}
                    inline={false}
                />
            </p>
        </div>
        <div>
            <h6>Continous Case</h6>
            <p>
                For a probability density function <Math
                    expression={"f:\\Omega\\rightarrow \\mathbb{R}"}
                />, we define the probability measure as:
                <Math
                    expression={"\\mathbb{P}(A) :=\\int_A f(x)dx"}
                    inline={false}
                />
            </p>
        </div>
    </div>
</section>

<section id="product probability spaces">
    <p>
        We can define a <strong>probability space</strong> as a triple of
        already defined components:
        <Math expression={"(\\Omega, \\Alpha, \\mathbb{P})"} inline={false} />
        From this, there arises the question what to do with products of such spaces,
        defined as:
        <Math
            expression={"(\\Omega_n, \\Alpha_n, \\mathbb{P}_n), \\quad n\\in\\{1,2,\\dots\\}"}
            inline={false}
        />. From measure theory, we know that this results in a new probability
        space defined as:
        <Math
            expression={`(\\prod_{j\\in\\mathbb{N}}\\Omega_j, \\sigma(\\text{"cylinder sets"}), \\mathbb{P})`}
            inline={false}
        />where <Math
            expression={`\\mathbb{P}(A_1 \\times \\dots \\times A_m \\times \\Omega_{m+1}
            \\times \\dots)=\\mathbb{P}_1(A_1) \\times \\dots \\times \\mathbb{P}_m(A_m)`}
        />
    </p>
</section>

<section id="conditional-prob">
    <p>
        We can start with any probability space where we consider a subset
        <Math
            expression={"B \\in \\Alpha \\text{ with }\\mathbb{P}(B)\\neq 0"}
        />. With that, we get a new probability space
        <Math
            expression={"(B, \\tilde{\\Alpha}, \\tilde{\\mathbb{P}})"}
            inline={false}
        /> where we limit the algebra to
        <Math expression={"A \\in \\Alpha \\text{ where }A\\subseteq B"} />
        and adjust the probability measure to <Math
            expression={"\\mathbb{\\tilde{P}}(A)=\\frac{\\mathbb{P}(A)}{\\mathbb{P}(B)}"}
        />, such that <Math expression={"\\tilde{\\mathbb{P}}(B)=1"} /> and all other
        measures are correctly defined.
    </p>
    <p>
        It might also be the case that there are subsets <Math
            expression={"A \\in \\Alpha"}
        /> that lie in B and A. This gives us the notion of a conditional probability.
    </p>
    <Definition key="conditionalProb" />
</section>

<section id="Bayes">
    <p>
        With conditional probabilities, we can directly arrive to
        <strong>Bayes Theorem</strong>
        <Math
            expression={"\\mathbb{P}(A\\mid B)=\\frac{\\mathbb{P}(B\\mid A)\\cdot \\mathbb{P}(A)}{\\mathbb{P}(B)}"}
            inline={false}
        />
    </p>
</section>

<section id="totalProb">
    <p>
        Next, we look at an event under countably many disjoint sets
        <Math
            expression={`B_i \\in \\Alpha, i \\in I \\subseteq \\mathbb{N} \\text{ with } \\bigcup_{i\\in
            I}B_i=\\Omega`}
        />. In this case, we can write the probability of the event with the
        <strong>law of total probabilities</strong>:
        <Math
            expression={`\\mathbb{P}(A)=\\mathbb{P}\\left(\\bigcup_{i\\in I}(A\\cap B_i)\\right)=\\sum_{i\\in
            I}\\mathbb{P}(A\\mid B_i)\\cdot \\mathbb{P}(B_i)`}
            inline={false}
        />This law is important for obtaining the denominator from
        <strong>bayes theorem</strong>
        <Math expression={"\\mathbb{P}(B)"} /> where we need to sum up all (disjoint)
        events of B.
    </p>
</section>

<section id="idependence">
    <p>
        Next, we want to define the independence of events, what means that
        <Math
            expression={`\\mathbb{P}(A \\mid B)=\\mathbb{P}(A) \\land \\mathbb{P}(B \\mid A)=\\mathbb{P}(B)`}
            inline={false}
        />This means that knowledge about one event should not effect the
        outcome of another event.
        <Definition key="independence" />
    </p>
</section>

<section id="random-vars">
    <p>
        <strong>Random variables</strong> are an abstraction of a random
        experiment. For the definition, we need two measurable spaces (event
        spaces)
        <Math expression={`(\\Omega,\\Alpha)`} inline={false} />
        <Math
            expression={`(\\tilde{\\Omega},\\tilde{\\Alpha)}`}
            inline={false}
        />Here, no probability measure is fixed yet. In measure theory, a random
        variable would hereby be a measurable map. This is e.g. useful when we
        want to map events to the number line with mathematical operations.
    </p>
    <Definition key="randomVar" />
    <p>
        If we now have two event spaces, a random variable on the two spaces and
        a probability measure
        <Math expression={"\\mathbb{P}:\\Alpha \\rightarrow [0,1]"} /> on the first
        event space, we can define a probability of an event with a shorter notion:
        <Math
            expression={`\\mathbb{P}\\left(X\\in
            \\tilde{\\Alpha}\\right):=\\mathbb{P}\\left(X^{-1}(\\tilde{\\Alpha})\\right)=\\mathbb{P}\\left(\\{w\\in
            \\Omega \\mid X(w)\\in \\tilde{\\Alpha} \\}\\right)`}
            inline={false}
        />
    </p>
</section>

<section id="distribution">
    <p>
        We will now mainly look at random variables, where the second event
        space sits on the real number line:
        <Math expression={`(\\Omega,\\Alpha, \\mathbb{P})`} inline={false} />
        <Math
            expression={`(\\tilde{\\Omega},\\tilde{\\Alpha},\\tilde{\\mathbb{P}})=\\left( \\mathbb{R}, B(\\mathbb{R}),
            \\mathbb{P}_X \\right)`}
            inline={false}
        />
    </p>
    <Definition key="distribution" />
    <p>
        Without proof, it holds that the probability distribution is a
        probability measure. It is often also written as
        <Math expression={"X \\sim \\tilde{P}"} />. Generally, the acutal
        experiment conducted is often hidden in a random variable instead of
        defined in the abstract probability space.
    </p>
</section>

<section id="cdf">
    <p>
        By integrating over the probability mass of a random variable, we can
        get a new function.
    </p>
    <Definition key="cdf" />
    <p>This definition has some useful properties:</p>
    <ul>
        <li><Math expression={"F_X(x)\\xrightarrow{x\\to -\\infty}0"} /></li>
        <li><Math expression={"F_X(x)\\xrightarrow{x\\to \\infty}1"} /></li>
        <li>
            <Math
                expression={"x_1 < x_2 \\Rightarrow F_X(x_1) \\leq F_X(x_2)"}
            /> (monotonically increasing)
        </li>
        <li>
            <Math expression={"\\lim_{x \\downarrow x_o}F_X(x)=F_X(x_0)"} /> (right-continous)
        </li>
    </ul>
</section>

<section id="rv-independence">
    <p>
        Not only events, but also random variables can be independent. For that,
        we consider two random variables
        <Math expression={`X:\\Omega\\rightarrow \\mathbb{R}`} inline={false} />
        <Math expression={`Y:\\Omega\\rightarrow \\mathbb{R}`} inline={false} />
        We will do this by looking at the <strong>preimage</strong> of the
        random variables (e.g. <Math
            expression={`X^{-1}\\left( (-\\infty, x] \\right)`}
        />) , which is defined as a set of events.
    </p>
    <Definition key="randomVarIndependence" />
    <p>
        With former definitions, we can rewrite this independence statement for
        the case of two random variables
        <Math
            expression={`\\begin{align*} & X^{-1}\\left((-\\infty, x]\\right) \\land Y^{-1} \\left( (-\\infty,y]
            \\right) \\text{ are independent events}\\\\ \\iff & \\mathbb{P}\\left( X^{-1}((-\\infty,x]) \\cap
            Y^{-1}((-\\infty,y])\\right)=\\mathbb{P}\\left( X^{-1}((-\\infty,x]) \\right) \\cdot \\mathbb{P} \\left(
            Y^{-1}((-\\infty,y])\\right) \\\\ \\iff & \\mathbb{P}(X\\leq x, Y \\leq y)=F_X(x) \\cdot F_Y(y)
            \\end{align*}`}
            inline={false}
        />Often, we define
        <Math expression={"F_{(X,Y)}(x,y):=\\mathbb{P}(X\\leq x, Y\\leq y) "} />
        as the joint cdf of the random variables. With this, we can also specify
        a more general definition of independence for a familty of events:
        <Math
            expression={`\\mathbb{P}((X_j\\leq x_j))_{j\\in J}=\\prod_{j\\in J} \\mathbb{P}(X_j\\leq x_j), \\quad \\forall J \\subseteq I`}
            inline={false}
        />
    </p>
</section>

<section id="expectation">
    <p>As random variables are often described with functions, we can ask the 
    question what values are most likely.</p>
    <Definition key="expectation"/>
</section>

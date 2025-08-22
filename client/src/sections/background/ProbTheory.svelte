<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section id="prob-measures">
    <p>
        A probability measure can be formulated as a measure known from <strong
            >measure theory</strong
        >
        as:
        <Math expression={"\\text{sample space }\\Omega"} inline={false} />
        <Math
            expression={"\\sigma\\text{-algebra}:\\Alpha \\subseteq P(\\Omega)"}
            inline={false}
        />We call elements of the algebra we assign probabilities to:
        <Math expression={"\\text{events}: A \\in \\Alpha"} inline={false} />
        Now, a measure in probability theory has a specfic name and some additional
        restriction which make it usable for probabilities:
    </p>
    <Definition key="probMeasure" />
    <p>
        We can define a <strong>probability space</strong> as a triple:
        <Math expression={"(\\Omega, \\Alpha, \\mathbb{P})"} inline={false} />
        Now, <strong>probability functions</strong> are measurable maps which
        assign probabilities to individual outcomes <Math
            expression={"\\omega\\in\\Omega"}
        />. We use them to construct correct probability measures for all events <Math
            expression={"A\\in\\Alpha"}
        />. Amongst other, the most frequent cases are:
    </p>
    <div role="group">
        <div>
            <h6>1) Discrete Case</h6>
            <p>
                Probabilities for discrete outcomes are given by a probability
                mass function <Math expression={"(p_w)_{w\\in\\Omega}"} /> and we
                define the probability measure as:
                <Math
                    expression={"\\mathbb{P}(A) :=\\sum_{w\\in A}p_w"}
                    inline={false}
                />
            </p>
        </div>
        <div>
            <h6>2) Continous Case</h6>
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
        Now, we can ask the question what to do with products of probability
        spaces, defined as:
        <Math
            expression={"(\\Omega_n, \\Alpha_n, \\mathbb{P}_n), \\quad n\\in\\{1,2,\\dots\\}"}
            inline={false}
        />From measure theory, we know that this results in a new probability
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
        If we only consider a subset <Math
            expression={"B \\in \\Alpha \\text{ with }\\mathbb{P}(B)\\neq 0"}
        /> of our original probability space, the result can be formulated as a new
        probability space with adjusted event space and normalised probability measure:
        <Math
            expression={"(B, \\tilde{\\Alpha}, \\tilde{\\mathbb{P}})"}
            inline={false}
        />
        <Math
            expression={"\\tilde{\\Alpha}=\\{A \\cap B \\mid A \\in \\Alpha \\}"}
            inline={false}
        />
        <Math
            expression={"\\tilde{\\mathbb{P}}(A)=\\frac{\\mathbb{P}(A)}{\\mathbb{P}(B)}, \\quad A\\subseteq B"}
            inline={false}
        />
        If we now have subsets <Math expression={"A \\in \\Alpha"} /> that lie in
        B and A and we want to look at the probability of one event occuring after
        the other in our original probability space, we can use the normalised probability:
    </p>
    <Definition key="conditionalProb" />
    <p>
        Conditional probabilities directly lead to
        <strong>Bayes Theorem</strong>
        <Math
            expression={` \\begin{align*} & \\mathbb{P}(A\\mid B)=\\frac{\\mathbb{P}(B\\mid A)\\cdot
            \\mathbb{P}(A)}{\\mathbb{P}(B)}\\\\ \\iff& \\mathbb{P}(A\\mid B)\\cdot\\mathbb{P}(B)=\\mathbb{P}(B\\mid
            A)\\cdot \\mathbb{P}(A) \\end{align*}`}
            inline={false}
        />
    </p>
    <p>
        Now, the question arises how to obtain the denominator
        <Math expression={"\\mathbb{P}(B)"} />. For this, we look at countably
        disjoint events:
        <Math
            expression={`\\left\\{A_i \\in \\Alpha, i \\in I \\subseteq \\mathbb{N}\\mid\\bigcup_{i\\in
            I}A_i=\\Omega\\land A_i \\cap A_j=\\emptyset, \\forall i\\neq j \\right\\}`}
            inline={false}
        />We need to sum over all possible events to get the total probability
        of a, as stated in the
        <strong>law of total probabilities</strong>:
        <Math
            expression={`\\mathbb{P}(B)=\\mathbb{P}\\left(\\bigcup_{i\\in I}(A_i\\cap B)\\right)=\\sum_{i\\in
            I}\\mathbb{P}(B\\mid A_i)\\cdot \\mathbb{P}(A_i)`}
            inline={false}
        />
    </p>
</section>

<section id="idependence">
    <p>
        When looking at two events, a notion of <strong>independence</strong> is
        imporntant, thus a fromal definiton for the fact, that knowledge about
        event A does not affect the oucome of event B. Formally, this can be
        written as:
        <Math
            expression={`\\mathbb{P}(A \\mid B)=\\mathbb{P}(A) \\land \\mathbb{P}(B \\mid A)=\\mathbb{P}(B)`}
            inline={false}
        />
        In a more general formulation, we want the total probabiliy the be just the
        same of the individual event probabilities:
        <Definition key="independence" />
    </p>
</section>

<section id="random-vars">
    <p>
        <strong>Random variables</strong> are an abstraction of a random
        experiment. For the definition, we introduce a second measurable space
        <Math
            expression={`(\\tilde{\\Omega},\\tilde{\\Alpha)}`}
            inline={false}
        />
    </p>
    <Definition key="randomVar" />
    <p>
        Often, event probabilities are stated with a shorter notion:
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
        We will now look at random variables, where the second event space sits
        on the real number line. This is often done in probability theory to
        define the outcome of an experiment:
        <Math
            expression={`\\left( \\mathbb{R}, B(\\mathbb{R}), \\mathbb{P}_X \\right)`}
            inline={false}
        />
        <Math expression={`X:\\Omega\\rightarrow\\mathbb{R}`} inline={false} />
        Now, the probabiliy measure of the second event space
        <Math expression={"\\mathbb{P}_X"} />
        has a special name, as it directly relates to the image of the random variable:
    </p>
    <Definition key="distribution" />
    <p>
        Without proof, it holds that the probability distribution is a
        probability measure. It is often also written as
        <Math expression={"X \\sim \\tilde{P}"} />. Generally, the acutal
        experiment conducted is often hidden in a random variable instead of
        defined in the abstract probability space. By integrating over the
        probability mass of a random variable, we get a new function:
    </p>
    <Definition key="cdf" />
    <p>This definition has some useful properties:</p>
    <ol>
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
    </ol>
</section>

<section id="rv-independence">
    <p>
        Like events, independence also applies to random variables. For that, we
        consider two random variables:
        <Math expression={`X:\\Omega\\rightarrow \\mathbb{R}`} inline={false} />
        <Math expression={`Y:\\Omega\\rightarrow \\mathbb{R}`} inline={false} />
        When looking at the <strong>pre-images</strong> of the random variables,
        we can use the event independence established earlier:
    </p>
    <Definition key="randomVarIndependence" />
    <p>
        In a more general definition of independence, a familty of events is
        called independent if:
        <Math
            expression={`\\mathbb{P}\\left((X_j\\leq x_j)_{j\\in J}\\right)=\\prod_{j\\in J} \\mathbb{P}(X_j\\leq
            x_j), \\quad \\forall J \\subseteq I`}
            inline={false}
        />
        Often, we define
        <Math
            expression={"F_{(X,Y)}(x,y):=\\mathbb{P}(X\\leq x, Y\\leq y) "}
            inline={false}
        />
        as the <strong>joint CDF</strong> of the random variables. In the case
        of two events, we can rewrite this independence statement as:
        <Math
            expression={`\\begin{align*} & X^{-1}\\left((-\\infty, x]\\right) \\land Y^{-1} \\left( (-\\infty,y]
            \\right) \\text{ are independent events}\\\\ \\iff & \\mathbb{P}(X\\leq x, Y \\leq y)=F_X(x) \\cdot F_Y(y)
            \\\\ \\iff & F_{(X,Y)}(x,y)=F_X(x) \\cdot F_Y(y)
\\end{align*}`}
            inline={false}
        />
    </p>
</section>

<section id=" expectation">
    <p>
        Since we only consider random variables, where the second probabiliy
        space sits on the real number line and thus has a well-defined
        probabiliy distribution, we may ask about the average event:
    </p>
    <Definition key="expectation" />
    <p>
        So far, we are dealing with an abstract integral from measure theory. In
        order to arrive at actual formulas we can calculate with, we need the <strong
            >change of variables</strong
        >
        formula. We define a random variables as a chain
        <Math expression={"Y \\circ X=Y(X)"} /> with:
        <Math expression={"X:\\Omega\\rightarrow \\mathbb{R}"} inline={false} />
        <Math
            expression={"Y:\\mathbb{R} \\rightarrow \\mathbb{R}"}
            inline={false}
        />

        The abstract integral for <Math expression={"\\omega\\in\\Omega"} /> of this
        chain is given by
        <Math
            expression={`\\int_\\Omega Y(X(\\omega )) d\\mathbb{P(\\omega )}`}
            inline={false}
        /> Now, we define <Math
            expression={"x:=X(w) \\text{ with } X^{-1}(x)=w"}
        />. This allows us to reformulate the expresion as:
        <Math
            expression={`\\int_{X(\\Omega)} Y(x) d(\\mathbb{P}\\left( X^{-1}(x)\\right)=\\int_{X(\\Omega)}
                    Y(x)d\\mathbb{P}_X(x)`}
            inline={false}
        /> For discrete and continous case, we can now formulate the expectation
        of a random variable with sums or integrals:
        <Math
            expression={`\\int_{X(\\Omega)} Y(x)d\\mathbb{P}_X(x)=\\begin{cases} \\int_{X(\\Omega)} Y(x)
                    f_X(x) dx \\quad\\text{continous} \\\\ \\sum_{x \\in X(\\Omega)} Y(x)\\cdot p_x
                    \\quad\\text{discrete} \\end{cases} `}
            inline={false}
        />
    </p>
    <p>Lastly, we want to define some properties of the expectation:</p>
    <ol>
        <li>
            <Math
                expression={`\\mathbb{E}(aX+bY)=a \\cdot \\mathbb{E}(X) + b \\cdot
                        \\mathbb{E}(Y)\\quad\\forall a,b\\in\\mathbb{R}`}
            />
        </li>
        <li>
            X,Y are independent <Math
                expression={`\\Rightarrow
                        \\mathbb{E}(XY)=\\mathbb{E}(X)\\cdot\\mathbb{E}(Y)`}
            />
        </li>
        <li>
            <Math
                expression={"\\mathbb{P}_X=\\mathbb{P}_Y \\Rightarrow \\mathbb{E}(X)=\\mathbb{E}(Y)"}
            />
        </li>
        <li>
            <Math
                expression={`\\mathbb{P}\\left( \\{ \\omega \\in \\Omega \\mid X(\\omega)\\leq Y(\\omega) \\}
                        \\right)=1 \\Rightarrow \\mathbb{E}(X)\\leq \\mathbb{E}(Y)`}
            />
        </li>
    </ol>
</section>

<section id="spread">
    <p>
        Wile the expectation provides information about the average value, we
        also want a notion of how elements fluctuate around the expectation.
        Here, we need to assume that <Math
            expression={"\\mathbb{E}(X^2)=\\int_\\Omega X^2d\\mathbb{P}"}
        /> exists.
    </p>
    <Definition key="variance" />
    <p>Similarily, we can define the standard deviation:</p>
    <Definition key="standardDeviation" />
    <p>
        For calculating the variance, we can again use the change-of-variables
        formula to get proper sums or integrals. Both, variance and standard
        deviation have the following properties:
    </p>
    <ol>
        <li>
            X,Y are independent <Math
                expression={"\\Rightarrow Var(X+Y)=Var(X)+Var(Y)"}
            />
        </li>
        <li>
            <Math
                expression={`Var(\\lambda \\cdot Y)=\\lambda^2 \\cdot Var(Y) \\quad \\forall \\lambda \\in
                \\mathbb{R}`}
            />
        </li>
        <li>
            <Math
                expression={`\\sigma(\\lambda \\cdot Y)=|\\lambda| \\cdot\\sigma(Y) \\quad \\forall \\lambda \\in
                \\mathbb{R}`}
            />
        </li>
    </ol>
</section>

<section id="correlation">
    <p>
        When investigating two random variables, we can look at how both random
        variables influence each other:
    </p>
    <Definition key="covariance" />
    <p>
        Note that X,Y are independent <Math
            expression={"\\Rightarrow Cov(X,Y)=0"}
        />. As we cannot infer independence from covariance, we call X and Y
        <strong>uncorrelated</strong>. For a meaningful interpretation of the
        covariance, we need some normalisation factor for which we will use the
        <strong>cauchy-schwarz inequality</strong>:
        <Math expression={"Cov(X,Y)^2 \\leq Var(X) \\cdot Var(Y)"} />.
    </p>
    <Definition key="correlation" />
</section>

<section id="marginal-distribution">
    <p>
        When talking about <strong>marginal distributions</strong>, we have a
        collection of random variables represented as a vector. We are
        interested in a single random variable, which projects the original
        space to a one-dimensional probability space:
        <Math
            expression={"X_1:\\Omega \\rightarrow \\mathbb{R}^n"}
            inline={false}
        />
        <Math
            expression={`X_2: \\mathbb{R}^n \\rightarrow \\mathbb{R}`}
            inline={false}
        />
        Setting <Math expression={"X=X_2\\circ X_1"} />, we now have our map of
        interest
        <Math
            expression={"X:\\Omega \\rightarrow \\mathbb{R}"}
            inline={false}
        />where the distribution (probability measure) on the codomain
        <Math expression={"(\\mathbb{R},B(\\mathbb{R}))"} /> is called the marginal
        distribution:
    </p>
    <Definition key="marginalDistribution" />
    <p>
        For calculating the marginal distribution, we again can specifically
        look at discrete (marginal PMF) and continous (marginal PDF) cases:
    </p>
    <ol>
        <li>
            Continous: <Math
                expression={`f_{X_i}(t)=\\int_{-\\infty}^\\infty\\dots\\int_{-\\infty}^\\infty
                f_X(x_1,\\dots,t,\\dots,x_n)dx_1,\\dots,x_{i-1},x_{i+1},x_n`}
            />
        </li>
        <li>
            Discrete: <Math
                expression={`p_{X_i}(t)=\\sum_{x_1}\\dots\\sum_{x_{i-1}}\\sum_{x_{i+1}}\\dots\\sum_{x_n}
                p_X(x_1,\\dots,t,\\dots,x_n)`}
            />
        </li>
    </ol>
    <p>
        Depending on the dimensionality, this calculation can be very costly
        what often makes exact calculations not tracable!
    </p>
</section>

<section id="conditional-expectation">
    <p>
        When observing an event <Math
            expression={"B\\in\\Alpha, \\mathbb{P}(B)>0"}
        />, the expectation logically changes. While the probability measure
        remains very similar, we need to account for the fact that some other
        events will not occur at all:
        <Math
            expression={"(\\Omega, \\Alpha, \\mathbb{P}(\\cdot\\mid B))"}
            inline={false}
        />
        We now can formulate a definition based on our knowledge about conditional
        probabilities:
    </p>
    <Definition key="conditionalExpectation" />
    <p>
        We can of course also define conditional expectations based on other
        random variables. For instance, a second random variable can be defined
        as:
        <Math
            expression={"Y:\\Omega, \\rightarrow \\mathbb{R}, B=\\{Y=y \\}"}
            inline={false}
        />Now, we have a composition of random variables where Y maps from the
        sample space to the real number line and the probability measure turns
        these values into probabilities. Hence, the resulting conditional
        expectation is a <strong>random variable</strong> stating the expectations
        for all possible outcomes of Y.
    </p>
    <Definition key="conditionalDiscreteExpectation" />
    <Definition key="conditionalContinousExpectation" />
    <p>These new random variables have the following properties:</p>
    <ol>
        <li>
            X,Y independent <Math
                expression={"\\Rightarrow \\mathbb{E}(X\\mid Y)=\\mathbb{E}(X)"}
            />
        </li>
        <li><Math expression={"\\mathbb{E}(X\\mid X)=X"} /></li>
        <li>
            <Math
                expression={"\\mathbb{E}(\\mathbb{E}(X\\mid Y))=\\mathbb{E}(X)"}
            />
        </li>
    </ol>
</section>

<section id="stochastic-processes">
    <p>
        <strong>Stochastic Processes</strong> define many random variables in a
        row or alternatively a random experiment evolving over time. Therefore
        we define a set, which is often defined as
        <Math
            expression={"T \\text{ set } (T=\\mathbb{N}, T=\\mathbb{Z}, T=\\mathbb{R})"}
            inline={false}
        />
    </p>
    <Definition key="stochasticProcess" />
    <p>
        Then, for <Math expression={"\\omega \\in \\Omega"} />, the map
        <Math
            expression={`\\begin{align*}T &\\rightarrow \\mathbb{R} \\\\t &\\rightarrow X_t(\\omega) \\end{align*}`}
            inline={false}
        />
        t is called a path.
    </p>
</section>

<section id="markov-chain">
    <p>
        While a stochastic process is a very general notion, <strong
            >markov chains</strong
        >
        are much more specific. Here, we need an order
        <Math
            expression={`T \\subseteq \\mathbb{Z} \\quad\\text{(discrete-time)}`}
            inline={false}
        />
        <Math
            expression={`T \\subseteq \\mathbb{R} \\quad\\text{(continous-time)}`}
            inline={false}
        />
    </p>
    <Definition key="markovChain" />
    <p>
        For discrete timesteps, markov chains can nicely be visualised with a
        graph, as we have a smallest possible timestep. If a chain does not
        depend on the exact timestamp, we call it <strong
            >time-homogenous</strong
        >
    </p>
    <p>
        Mathematically, the transition states of a discrete markov chain with n
        states can be formulated with a so-called <strong
            >transition matrix</strong
        >
        <Math expression={`P^{n\\times n}`} />.
    </p>
</section>

<section id="stationary-distribution">
    <p>
        With the trainsition matrix, we can calculate the probability
        distribution for any state <Math expression="q^i=q^oP" />. For getting
        to <strong>stationary distributions</strong>, we only consider
        time-homogenous, discrete markov chains where the probability can be
        formulated by a conditional probability:
        <Math
            expression={"p_{x,y} :=P(X_{k+1}=y \\mid X_k=x)"}
            inline={false}
        />
    </p>
    <p>
        The Markov process <Math
            expression={"X_l:\\Omega \\rightarrow \\{1,2,\\dots,N \\}"}
        />
        starting at <Math expression="k=0" /> with measure
        <Math expression={"\\mathbb{P}_{X_0}"} /> the has a
        <strong>probability mass function</strong> given by a row vector
        <Math
            expression={`\\begin{align*} (q^0)_m &=\\mathbb{P}(X_o=m), \\quad q^0 \\in \\mathbb{R}^{1\\times N} \\\\
            (q^1)_m &=\\mathbb{P}(X_1=m) \\\\ &=\\sum_{i=1}^n \\mathbb{P}(X_1=m \\mid B_i)\\cdot \\mathbb{P}(B_i),
            \\quad B_i=\\{X_0=i\\} \\\\ &=\\sum_{i=1}^n \\mathbb{P}(X_0=i) \\cdot \\mathbb{P}(X_1=m \\mid X_0=i)\\\\
            &=(q^0 P)_m \\end{align*}`}
            inline={false}
        />By induction, we get <Math
            expression={"q^k=q^0 P^k"}
            inline={false}
        />
    </p>
    <Definition key="stationaryDistribution" />
    <p>
        The stationary distribution can be easily computed with eigenvector and
        a contraint on the scaling to fit probabilities.
    </p>
</section>

<section id="central-limit">
    <p>
        When talking about the <strong>central limit theorem</strong>, we must
        first introduce two very generral inequalities:
    </p>
    <p>
        First, <strong>markovs inequality</strong> states for any random
        variable,
        <Math expression={"|X|:\\Omega \\rightarrow [0,\\infty)"} /> satisfies

        <Math
            expression={`\\mathbb{P}(|X|\\geq
            \\epsilon)\\leq\\frac{\\mathbb{E}(|X|^p)}{\\epsilon^p}\\quad\\forall\\epsilon>0,p>0`}
            inline={false}
        />
    </p>
    <p>
        Second, we can use the former inequality to state \
        <strong>Chebyshev's inequality</strong> which can be used to estiamte
        the deviation from the expected value for a given random variable:
        <Math expression={"X:\\Omega \\rightarrow \\mathbb{R}"} /> satisfies

        <Math
            expression={`\\mathbb{E}(|X|)<\\infty \\Rightarrow
            \\mathbb{P}\\left(|X-\\mathbb{E}(X)|\\geq\\epsilon\\right) \\leq \\frac{Var(X)}{\\epsilon^2},
            \\quad\\forall\\epsilon> 0`}
            inline={false}
        />
    </p>
    <p>
        Next, we need <Math expression={"k\\sigma"} />-intervals for
        constructing the important statistical theorems. Herefore, the variance
        and standard deviation need to exist! - Given a probability
        distribution, these intervals try to estimate the area of a PDF within
        the interval using Chebyshev's inequality:

        <Math
            expression={`\\begin{align*} \\mathbb{P}\\left(X\\in [\\mu-k\\sigma, \\mu+k\\sigma] \\right)
            &=\\mathbb{P}\\left(|X-\\mu|\\leq k\\sigma\\right) \\\\ &\\geq \\mathbb{P}(|X-\\mu|<k\\sigma) \\\\ &=1 -
            \\mathbb{P}(|X-\\mu|\\geq k\\sigma) \\\\ &\\geq 1 - \\frac{Var(X)}{k^2\\sigma^2}\\\\ &=1 - \\frac{1}{k^2},
            \\quad k \\in\\mathbb{N}^{>1}
            \\end{align*}`}
            inline={false}
        />
        This gives us very general information about the number of events, for instance
        75% for k=2 and 88.8% for k=3.
    </p>
    <p>
        Now, when measuring event spaces we often need to distinguish between
        theoretical (<Math expression={"\\mathbb{P}(X)"} />) and empirical (<Math
            expression={`\\frac{\\text{number of
            outcomes in A}}{\\text{total number}}`}
        />) event probabilities. Hereby, the law of large number states the
        connection between both. Starting with the <strong>weak law</strong>, we
        first look at a definition in terms of probability:
    </p>
    <Definition key="weakLLN" />
    <p>
        A famous application of this law is <strong
            >Monte Carlo integration</strong
        >, which is especially useful in multi-dimensional settings. As the
        above law tells us, we can approximate a theoretical expectation with
        enough repititions. Hence, we can approximate integrals by assuming the
        expectation of random variables to be the area of interest. We start by
        defining an integrable function
        <Math
            expression={"g:\\Omega\\rightarrow [-c,c], \\quad c>0"}
            inline={false}
        />
        If we now define a random variable <Math expression={"X_1"} /> as unifromly
        picking a point from <Math expression={"\\Omega"} /> and define
        <Math expression={"Y_1:=g(X_1)"} />, we can formulate the expecation as:
        <Math
            expression={`\\begin{align*}\\mathbb{E}(Y_1) &=\\mathbb{E}(g(X_1))\\\\ &=\\int_\\Omega
                g(x)f_{X_1}(x)dx \\\\ &=\\int_\\Omega g(x)dx \\\\ &\\approx \\frac{1}{n}\\sum_{k=1}^n g(X_k)
                \\quad\\text{(l.l.n)}\\end{align*}`}
            inline={false}
        />
        Hence, the procedure will be to take many random points as defined in the
        random variable and then approximate the expectation with a sum.
    </p>

    <p>
        Next, we introduce the <strong>strong law</strong> which builds upon the
        weak law. Here, we want to establish
        <strong>point-wise convergence</strong>, as the weak law only gives
        information about the collection in the limit., not about a single
        sample.
    </p>
    <Definition key="strongLLN" />
</section>

<section id="central-limit">
    <p>
        In this section, we highlight the importance of the Normal distribution
        formulated by the central limit theorem.
    </p>
    <Definition key="centralLimit" />
</section>

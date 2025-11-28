<script lang="ts">
    import Math from "../../lib/Math.svelte";
    import Definition from "../../lib/Definition.svelte";
</script>

<section id="introduction">
    <p>
        Measure theory is a fundamental branch of mathematical analysis that
        generalizes the concepts of length, area, and volume. It provides a
        rigorous framework for integration, probability, and other areas of
        mathematics. While calculus deals with integrals over continuous
        functions on well-behaved domains, measure theory extends this to a much
        broader class of sets and functions, laying the groundwork for
        <b>Probability Theory</b>,<b>Functional Analysis</b>, and
        <b>Deep Learning</b>.
    </p>
</section>

<section id="Sigma Algebra">
    <p>
        We start with a basic subset of some universal set <strong>X</strong>:
        <Math expression={"\\Alpha \\subseteq{P(X)}"} inline={false} /> Next, we
        want to define a collection of subsets, for which we want to estblish a meaningful
        notion of measurement:
    </p>
    <Definition key="sigma_algebra" />
    <p>
        We call any subset <Math expression={"A \\in \\Alpha"} />
        an <strong><Math expression={"\\Alpha"} />-measurable set</strong>.
        Based on the properties of the algebra, we can define how to construct
        such a set based on some arbitrary subet of the power set:
    </p>
    <Definition key="smallest_sigma_algebra" />
    <p>
        Since we are using intersections, we naturally will happen to find the
        smallest sigma algebra constructed by M. For uncountable (open) sets,
        the notion is formulated more generally as:
    </p>
    <Definition key="borel_sigma_algebra" />
    <p>
        We needed to define the borel-algebra, since the notion of measurements
        requires some rules which cannot be fulfilled by the power set. All
        subsets included here can be constructed by set theory and can be
        defined some kind of measure on.
    </p>
</section>

<section id="Measure">
    <p>
        We now call a set
        <Math expression={"(X,\\Alpha)"} inline={false} />
        a <strong>measurable space</strong>. For getting actual measures, we now
        define a map, that assigns a value to each of ours subsets of interest.
    </p>
    <Definition key="measure" />
</section>

<section id="measure-problem">
    <p>
        For the real number line, the most general measure that provides us with
        our intuitive notion of length would be a measure
        <Math expression={"\\mu \\text{ on }P(\\mathbb{R^n})"} />
        that satisfies:
    </p>
    <ol>
        <li><Math expression={"\\mu \\left([0,1]^n\\right)=1"} /></li>
        <li>
            <Math
                expression={"\\mu(x+A)=\\mu(A), \\quad \\forall x \\in \\mathbb{R}^n"}
            />
        </li>
        <li><Math expression={"\\mu([a,b])=b-a, \\quad b>a"} /></li>
    </ol>
    <p>
        However, such a measure does not exist for <Math
            expression={"P(\\mathbb{R})"}
        />. This is called the
        <strong>measure problem</strong> and forces us to be more restrictive on
        the algebra. Hence, we will use Borel sigma-algebra, as this one gives
        us a valid measure. With that, we get a triple which we call
        <strong>measure space</strong>:
        <Math expression={"(X,\\Alpha,\\mu)"} inline={false} />
    </p>
</section>

<section id="measurable">
    <p>
        Now that we have established a way to measure the size of any set <Math
            expression={"A\\in\\Alpha"}
        />, we want to look at functions that map from our original measure
        space to any measurable space:
        <Math expression={"(\\Omega,\\Alpha_2)"} inline={false} />
        If want to define a mapping from one set to another, we need to make sure
        that the mapping is compatible with the measure. Therefore, we introduce
        the concept of <strong>measurable maps</strong>:
    </p>
    <Definition key="measurable" />
    <p>
        With that, we make sure that all elements in the range of the function
        map to a corresponding domain in the measure space and thus, have a
        well-defined measure. For example, the
        <strong>characteristic function</strong>:
        <Math
            expression={`\\chi_A: X \\rightarrow \\mathbb{R}, \\quad \\chi_A(x):=\\begin{cases} 1, \\quad x \\in A
            \\\\ 0,\\quad x \\notin A \\end{cases}, \\quad\\forall A \\in\\Alpha`}
            inline={false}
        />
        is a measurable map, because for the case <Math
            expression={"x=\\{0\\}"}
        />, the pre-image <Math expression={"\\chi_A^{-1}(x)=A^C"} /> is always by
        definition in the sigma algebra.
    </p>
</section>

<section id="lebesque-integral">
    <p>
        Note that by using the characteristic function, we get a very naive
        notion of integration where the height is fixed to the only possible
        outcome 1 and the measure provides information about the size of the
        set, since it indicates wheather an element <Math
            expression={"x\\in X"}
        />
        belongs to some set <Math expression={"A\\in\\Alpha"} />:
        <Math
            expression={`I(\\chi_A):=1\\cdot \\mu(A)=\\mu(A)`}
            inline={false}
        />
        Now, the measure of the pre-image is already defined with precision while
        the height is very simplistically the constant 1. By extending the characteristic
        function to a simple function, we can get a more nuanced picture by approximating
        the area with a weighted sum (step function) over disjoint <Math
            expression={`A_1,\\dots,A_n\\in\\Alpha \\text{ and
            }c_1,\\dots,c_n\\in\\mathbb{R}^+`}
        />:
        <Math
            expression={`\\phi(x):=\\sum_{i=1}^nc_i \\cdot \\chi_{A_i}(x)`}
            inline={false}
        />
        <Math
            expression={`I(\\phi):=\\sum_{i=1}^nc_i \\cdot \\mu(A_i)`}
            inline={false}
        />
        Since measurable maps are additive, we know that simple functions are also
        measurable maps. We put our knowledge up to now into a final definition
        where we introduce a set of simple functions with enforced 
        positivity before defining the integral:
        <Math
            expression={`S^+ :=\\{ \\phi:X\\rightarrow \\mathbb{R} \\mid f \\text{ simple function }, f \\geq 0\\}`}
            inline={false}
        />
    </p>
    <Definition key="integral" />
    <p>
        The definition of the integral has some nice properties for
        <Math expression={"\\alpha, \\beta \\geq 0"} /> and is hence
        <strong>almost linear</strong> and monotonic:
    </p>
    <ol>
        <li>
            <Math
                expression={"I(\\alpha f + \\beta g)=\\alpha I(f) + \\beta I(g)"}
            />
        </li>
        <li><Math expression={"f\\leq g \\Rightarrow I(f) \\leq I(g)"} /></li>
    </ol>
    <p>
        This notion of integration only needs to be extended to more complex
        functions. We do this by approximating the full integrals with simple
        functions taking the lower bound (to ensure we are under the actual
        graph). Of this newly bounded set of simple functions, we just need to
        take the supremum for getting an adequate approximation of the integral:
    </p>
    <Definition key="lebeque" />
    <p>
        The lebeque integral has some nice properties, but since we are limited
        to positive functions, such properties are defined to hold
        <Math expression={"\\mu"} />-almost everywhere (<strong
            ><Math expression={"\\mu"} />-a.e.</strong
        >) for two measurable maps <Math
            expression={"f,g: X \\rightarrow [0,\\infty)"}
        />:
    </p>
    <ol>
        <li>
            <Math
                expression={"f=g \\xRightarrow{\\mu-a.e.} \\int_{x}fd \\mu=\\int_{x}gd\\mu"}
            />
        </li>
        <li>
            <Math
                expression={`f \\leq g \\xRightarrow{\\mu-a.e.} \\int_{x}fd \\mu \\leq \\int_{x}gd\\mu`}
            />
        </li>
        <li><Math expression={"f=0 \\xLeftrightarrow{\\mu-a.e.} \\int_{x}fd\\mu=0"} /></li>
    </ol>
</section>

<section id="monotome-convergence">
    <p>
        Next, we look at convergence properties of the lebesque integral,
        starting with the <strong>monotome convergence theorem</strong>. We
        introduce measurable functions:
    </p>
    <Math
        expression={"f_n:X \\rightarrow [0,\\infty), \\quad \\forall n \\in \\mathbb{N}"}
        inline={false}
    />
    <Math expression={"f:X\\rightarrow [0,\\infty) "} inline={false} />
    <p>
        with the following properties holding <Math
            expression={" \\mu"}
        />-a.e.:
    </p>
    <ol>
        <li><Math expression={"f_1 \\leq f_2 \\leq \\dots"} /></li>
        <li>
            <Math
                expression={"\\lim_{n\\to \\infty} f_n(x)=f(x), \\quad x \\in X"}
            />
        </li>
    </ol>
    <p>
        it follows that the limit can be pushed inside of the integral. Hereby,
        the properties state that the sequence (of functions) need to be
        monotomically increasing and point-wisely converge to some function
    </p>
    <Math
        expression={`\\Rightarrow \\lim_{n\\to\\infty} \\int_X f_n d\\mu=\\int_X fd\\mu`}
        inline={false}
    />
    <p>
        Such convergence properties are the main advantage of the lebesque
        integral compared to the riemand integral, since it can also operate on
        highly discontinous measures.
    </p>
</section>
<section id="fatou-lemma">
    <p>
        Next, we get a very general statement, where we only need a few
        components we already defined previously. It can be used to get
        information about the lower bound of a limit:
    </p>
    <Definition key="fatou" />
    <p>
        Fatou's Lemma sets a bound for the integral of the limit, which is often
        difficult to obtain.
    </p>
</section>

<section id="lebesque-convergence">
    <p>
        For the next convergence theorem, we introduce a new set of
        lebesque-integrable functions of power 1:
    </p>
    <Math
        expression={`L^1(\\mu) :=\\left\\{ f:X \\rightarrow \\mathbb{R}, \\text{ measurable} \\mid \\int_X |f|^1 d
        \\mu < \\infty \\right\\}`}
        inline={false}
    />
    <p>
        For integration, wee need to look at positive and nevative parts of
        functions <Math expression={"f\\in L^1(\\mu)"} /> seperately:
    </p>
    <Math expression={"f=f^+-f^-, \\quad f^+,f^- \\geq 0"} inline={false} />
    <p>Now, we can define the integral as:</p>
    <Math
        expression={"\\int_X f d \\mu :=\\int_X f^+ d \\mu - \\int_X f^- d \\mu"}
        inline={false}
    />
    <Definition key="lebesque2" />
    <p>
        This means, one must only find a suitable majorant for being able to
        tell a lot about integration and convergence.
    </p>
</section>

<section id="caratheodory-extension">
    <p>
        In this case, we define a new subset of the powet set and its
        pre-measure:
    </p>
    <Math
        expression={"\\hat{\\Alpha}\\subseteq P(X) \\quad \\text{(semiring)}"}
        inline={false}
    />
    <Math
        expression={"\\hat{\\mu}: \\hat{\\Alpha} \\rightarrow [0,\\infty] \\quad \\text{(pre-measure)}"}
        inline={false}
    />
    <p>we get the properties:</p>
    <ol>
        <li>
            We have an extension on <Math
                expression={`\\hat{\\mu}: \\tilde{\\mu}: \\sigma(\\hat{\\Alpha}) \\rightarrow
                [0,\\infty]`}
            />
        </li>
        <li>
            If there is a sequence <Math expression={`(S_j)`} /> with:
            <Math
                expression={`S_j \\in \\hat{\\Alpha}, \\bigcup_{j=1}^\\infty S_j=X, \\hat{\\mu}(S_j)<\\infty
                \\Rightarrow \\tilde{\\mu} \\text{ is unique}`}
                inline={false}
            />
        </li>
    </ol>
    <p>
        We use semirings as lighter structures, from which we can construct
        borel algebras. Hereby, the most famous example of a semiring is the
        open interval <Math
            expression={"C:=\\left\\{ [a,b) \\mid a,b \\in \\mathbb{R}, a \\leq b \\right\\}"}
        />, which is not a <Math expression={"\\sigma"} />-algebra (<Math
            expression={"\\mathbb{R}\\notin C"}
        />), but that it generates the borel-sigma algebra:
        <Math expression={"\\sigma(C)=B(\\mathbb{R})"} />. Hence, it is enough
        to use these specific semirings for constructing a borel algebra on
        <Math expression={"\\mathbb{R}"} />.
    </p>
    <p>
        In the end, with this extension theorem we are able to construct the
        borel set of the real numbers <Math expression={"B(\\mathbb{R}"} />
        and the measure of this borel algebra is unique. This extension is referred
        to as the <strong>Lebesque measure</strong>.
    </p>
</section>

<section id="lebesque-stieltjes">
    <p>
        For <strong>Lebesque-Stieltjes measures</strong>, we only need
        monotomically increasing functions:
    </p>
    <Math expression={"F:\\mathbb{R}\\rightarrow\\mathbb{R}"} inline={false} />
    <p>
        Hence, the functions do not need to be continous and we want to look at
        lenghts of intervals:
    </p>
    <Math
        expression={"\\mu_F\\left([a,b)\\right):=F(b^-)-F(a^-)"}
        inline={false}
    />
    <p>In case of jumps, border points must be defined with limits:</p>
    <Math
        expression={"F(a^-):=\\lim_{\\epsilon\\to0^+}F(a+\\epsilon)"}
        inline={false}
    />
    <p>
        Given this interval, we get a semiring
        <Math
            expression={"A=\\left\\{ [a,b): a,b \\in \\mathbb{R}, a \\leq b \\right\\}"}
        />, for which the extension theory tells us that there exits only one
        measure <Math
            expression={"\\mu_F:B(\\mathbb{R})\\rightarrow[0,\\infty]"}
        />.
    </p>
</section>

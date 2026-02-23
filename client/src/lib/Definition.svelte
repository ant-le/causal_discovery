<script lang="ts">
    //TODO: Add intuition graphs for definitions
    import Math from "./Math.svelte";
    import katex from "katex";
    let { key }: { key: string } = $props();

    interface Definition {
        title: string;
        text: string;
        formula: string;
    }
    type Definitions = Record<string, Definition>;
    const definitions: Definitions = {
        convergence: {
            title: "Convergence",
            text:
                "A sequence " +
                katex.renderToString(
                    "(x_k)_{k\\in \\mathbb{N}}, x_k\\in \\mathbb{R}^n,",
                ) +
                " is called convergent to " +
                katex.renderToString("x") +
                " if:",
            formula:
                "\\forall\\epsilon>0, \\exists K \\in \\mathbb{N}, \\forall k\\geq K: d(x_k,x)<\\epsilon",
        },
        continuity: {
            title: "Continuity",
            text:
                "A function " +
                katex.renderToString(
                    "\\mathbb{R}^n \\rightarrow \\mathbb{R}^m",
                ) +
                " is called continuous at " +
                katex.renderToString("x") +
                " if for all sequences " +
                katex.renderToString("(x_k)_{k\\in \\mathbb{N}}") +
                " we have: ",
            formula: "x_k \\rightarrow x \\Rightarrow f(x_k) \\rightarrow f(x)",
        },
        partial_derivatives: {
            title: "Partial Derivatives",
            text:
                "A function " +
                katex.renderToString(
                    "\\mathbb{R}^n \\rightarrow \\mathbb{R}^m",
                ) +
                " is called partially differentiable wrt. " +
                katex.renderToString("x_1") +
                " at " +
                katex.renderToString(
                    "\\tilde{x}=\\begin{pmatrix} \\tilde{x}_1 \\\\ \\dots \\\\ \\tilde{x}_n \\end{pmatrix}",
                ) +
                " if there exists:",
            formula:
                "\\frac{\\partial f}{\\partial x_1}(\\tilde{x}) = \\lim_{h\\to0} \\frac{f(\\tilde{x}_1+h,\\tilde{x}_2, \\dots,\\tilde{x}_n)-f(\\tilde{x}_1,\\tilde{x}_2, \\dots,\\tilde{x}_n)}{h}",
        },
        directional_derivatives: {
            title: "Directional Derivatives",
            text:
                "For any unit vector " +
                katex.renderToString("v\\in \\mathbb{R}^n") +
                ", the directional derivative of " +
                katex.renderToString("f") +
                "along the function at point " +
                katex.renderToString("\\tilde{x}") +
                " is defined as:",
            formula: "\\frac{f(\\tilde{x}+hv) - f{(\\tilde{x})}}{h}",
        },

        differentiable: {
            title: "Differentiable",
            text:
                "A function is totally differentiable, if there is a linear map " +
                katex.renderToString(
                    "l:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m",
                ) +
                " and a map " +
                katex.renderToString(
                    "\\phi:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m",
                ) +
                " with " +
                katex.renderToString(
                    "\\frac{\\phi(h)}{\\|h\\|}\\xrightarrow{h\\to0}0",
                ) +
                " s.t. for all " +
                katex.renderToString("h\\in\\mathbb{R}^n") +
                ":",
            formula: "f(\\tilde{x}+h)=f(\\tilde{x})+l(h)+\\phi(h)",
        },
        gradient: {
            title: "Gradient",
            text:
                "For totally differentiable functions " +
                katex.renderToString("q:\\mathbb{R}^n\\rightarrow\\mathbb{R}") +
                " mapping to a scalar output, the gradient is defined as:",
            formula:
                "\\nabla f(\\tilde{x})=\\begin{pmatrix} \\frac{\\partial f}{\\partial x_1}(\\tilde{x}) \\\\ \\dots \\\\ \\frac{\\partial f}{\\partial x_n}(\\tilde{x}) \\end{pmatrix} \\in \\mathbb{R}^n",
        },
        sigma_algebra: {
            title: "Sigma Algebra",
            text:
                "A subset " +
                katex.renderToString("\\Alpha") +
                " is called a " +
                katex.renderToString("\\sigma") +
                "-algebra if:",
            formula: `\\begin{align*} \\emptyset,X &\\in \\Alpha \\\\ A &\\in \\Alpha \\Rightarrow A^C := X \\setminus A \\in \\Alpha \\\\ A_i &\\in \\Alpha, i \\in \\mathbb{N} \\Rightarrow \\bigcup_{i =1}^\\infty A_i \\in \\Alpha \\end{align*}`,
        },
        smallest_sigma_algebra: {
            title: "Smallest Sigma Algebra",
            text:
                "For some subset " +
                katex.renderToString("M \\subseteq P(X)") +
                ", there is a smallest " +
                katex.renderToString("\\sigma") +
                "-algebra that contains " +
                katex.renderToString("M:"),
            formula:
                "\\sigma(M) := \\bigcap_{\\Alpha \\supseteq M, \\Alpha := \\sigma\\text{ -algebras}} \\Alpha",
        },
        borel_sigma_algebra: {
            title: "Borel Sigma Algebra",
            text:
                "For an open set X, the " +
                katex.renderToString("\\sigma") +
                "-algebra generated is called a borel " +
                katex.renderToString("\\sigma") +
                "-algebra on X:",
            formula: "B(X) := \\sigma(X)",
        },
        measure: {
            title: "Measure",
            text:
                "A map " +
                katex.renderToString(
                    "\\mu: \\Alpha \\rightarrow [0,\\infty ]",
                ) +
                " is called a measure if:",
            formula: `\\begin{align} \\mu(\\emptyset)&=0 \\\\ \\mu(\\bigcup_{i=1}^\\infty A_i)&=\\sum_{i=1}^\\infty \\mu(A_i), \\quad \\forall A_i \\in \\Alpha  \\end{align}`,
        },
        measurable: {
            title: "Measurable",
            text:
                "A map " +
                katex.renderToString("f:X \\rightarrow \\omega") +
                " is called measurable if:",
            formula:
                "f^{-1}(A_2) \\in \\Alpha_1, \\quad \\forall A_2 \\in \\Alpha_2",
        },
        integral: {
            title: "Integral",
            text:
                "For a measure space " +
                katex.renderToString("(X,\\Alpha,\\mu)") +
                ", a function " +
                katex.renderToString(`f\\in S^+`) +
                "and constants " +
                katex.renderToString(`c_1,\\dots,c_n\\in\\mathbb{R}^+`) +
                ", the integral of f w.r.t. " +
                katex.renderToString("\\mu") +
                " is defined as:",
            formula:
                "\\int_X f d\\mu = I(f) = \\sum_{i=1}^n c_i \\cdot \\mu(A_i) \\in [0,\\infty]",
        },
        lebeque: {
            title: "Lebesgue Integral",
            text:
                "For a measurable map " +
                katex.renderToString("f:X\\rightarrow [0,\\infty)") +
                ", the Lebesgue integral is defined as:",
            formula:
                "\\int_{x}f d\\mu := \\sup\\left\\{ I(h) \\mid h \\in S^+, h \\leq f \\right\\}",
        },
        fatou: {
            title: "Fatou's Lemma",
            text:
                "For a measure space " +
                katex.renderToString("(X,\\Alpha, \\mu)") +
                " and a measure " +
                katex.renderToString("f_n") +
                ", it holds that:",
            formula:
                "\\int_{x} \\lim_{n\\to\\infty} \\inf\\{f_n d\\mu\\} \\leq \\lim_{n\\to\\infty} \\inf\\{\\int_X f_n d\\mu\\}",
        },
        lebesque2: {
            title: "Lebesgue's Dominated Convergence Theorem",
            text:
                "For a sequence of functions:" +
                katex.renderToString(
                    "f_n:X \\rightarrow \\mathbb{R}, \\quad \\text{ measurable } \\forall n \\in \\mathbb{N}",
                    {
                        displayMode: true,
                    },
                ) +
                "with a point-wise limit: " +
                katex.renderToString("f:X \\rightarrow \\mathbb{R}") +
                katex.renderToString(
                    "f_n(x) \\rightarrow^{n \\to \\infty} f(x), \\quad \\forall x \\in X (\\mu \\text{-a.e.})",
                    { displayMode: true },
                ) +
                "and an integrable majorant g:" +
                katex.renderToString(
                    "|f_n| \\leq g, \\quad g \\in L^1(\\mu) \\forall n \\in \\mathbb{N}",
                    { displayMode: true },
                ) +
                " it follows that:",
            formula: `\\begin{align*} f_1,\\dots ,f_n,f \\in L^1(\\mu) \\\\ \\lim_{n \\to \\infty} f_n d \\mu = \\int_X f d \\mu \\end{align*}`,
        },
        probMeasure: {
            title: "Probability Measure",
            text:
                "A map " +
                katex.renderToString(
                    "\\mathbb{P}:\\Alpha \\rightarrow \\mathbb{R}",
                ) +
                " is called a <strong>probability measure </strong> if " +
                katex.renderToString(
                    "\\forall i \\neq j: A_i \\cap A_j=\\emptyset:",
                ),
            formula: `
            \\begin{align*}
                \\mathbb{P}(A) &\\in [0,1] \\\\
                \\mathbb{P}(\\emptyset)&=0 \\\\
                \\mathbb{P}(\\Omega)&=1\\\\ 
                \\mathbb{P}(\\bigcup_{j=1}^\\infty A_j)&=\\sum_{j=1}^\\infty \\mathbb{P}(A_j)
            \\end{align*}`,
        },
        conditionalProb: {
            title: "Conditional Probability",
            text:
                "A probability measure based on an event " +
                katex.renderToString("B\\in\\Alpha") +
                " with " +
                katex.renderToString("\\mathbb{P}(B)\\neq 0") +
                "is defined as:",
            formula:
                "\\mathbb{P}(A\\mid B):=\\frac{\\mathbb{P}(A\\cap B)}{\\mathbb{P}(B)}",
        },
        independence: {
            title: "Independence of Events",
            text:
                "Given some probability space, a family of events " +
                katex.renderToString("(A_i)_{i\\in I}") +
                " with " +
                katex.renderToString("A_i \\in \\Alpha") +
                " are called independent if:",
            formula: `\\mathbb{P}\\left(\\bigcap_{j\\in  J}A_j \\right)=\\prod_{j \\in J} \\mathbb{P}(A_j)`,
        },
        randomVar: {
            title: "Random Variable",
            text:
                "A map " +
                katex.renderToString(
                    "X:\\Omega \\rightarrow \\tilde{\\Omega}",
                ) +
                " is called a random variable if:",
            formula: `X^{-1}(\\tilde{A})\\in \\Alpha, \\quad \\forall \\tilde{A}\\in \\tilde{\\Alpha}`,
        },
        distribution: {
            title: "Distribution of Random Variable",
            text:
                "A map " +
                katex.renderToString(
                    "\\mathbb{P}_X:B(\\mathbb{R}) \\rightarrow [0,1]",
                ) +
                ` on a probability space with a random variable is called 
                probability distribution of X:`,
            formula: `\\mathbb{P}_X(B):= \\mathbb{P}\\left(X^{-1}(B)\\right) = \\mathbb{P}(X\\in B)`,
        },
        cdf: {
            title: "Cumulative Distribution Function (CDF)",
            text: `A CDF given a probability space and a random variable is 
                defined as:`,
            formula: `F_X:\\mathbb{R} \\rightarrow [0,1], \\quad F_X(x):= \\mathbb{P}_X \\left( (-\\infty, x] \\right)=\\mathbb{P}(X\\leq x)`,
        },
        randomVarIndependence: {
            title: "Independence of Events",
            text:
                "Given some probability space, two random variables are called independent if " +
                katex.renderToString("\\forall x,y \\in \\mathbb{R}:"),
            formula: `X^{-1}\\left((-\\infty, x]\\right) \\land Y^{-1} \\left( (-\\infty,y] \\right) \\text{ are independent events}`,
        },
        expectation: {
            title: "Expectation of a Random Variable",
            text: `Given some probability space, the expectation of a random 
            variable can be defined with its lebesque integral:`,
            formula: "\\mathbb{E}(X):=\\int_\\Omega Xd\\mathbb{P}",
        },
        variance: {
            title: "Variance of a Random Variable",
            text: `Given some probability space, the variance of a random 
            variable can be defined as:`,
            formula: "Var(X):=\\mathbb{E}\\left( (X-\\mathbb{E}(X))^2 \\right)",
        },
        standardDeviation: {
            title: "Standard Deviation of a Random Variable",
            text: `Given some probability space, the standard deviation of a random 
            variable can be defined as:`,
            formula: "\\sigma(X):=\\sqrt{Var(X)}",
        },
        covariance: {
            title: "Covariance of Random Variables",
            text: `Given some probability space, the covariance of two random 
            variables can be defined as:`,
            formula: `\\begin{align*}
                Cov(X,Y) :&=\\mathbb{E}\\left( (X-\\mathbb{E}(X)) \\cdot  (Y-\\mathbb{E}(Y)) \\right)\\\\
                          &=\\mathbb{E}(XY) - \\mathbb{E}(X)\\mathbb{E}(Y)
                    \\end{align*}`,
        },
        correlation: {
            title: "Correlation of Random Variable",
            text: `Given some probability space, the correlation of two random 
            variables can be defined as:`,
            formula: `\\rho_{X,Y} := \\frac{Cov(X,Y)}{\\sigma(X)\\cdot \\sigma(Y)}\\in [-1,1]`,
        },
        marginalDistribution: {
            title: "Marginal Distribution of Random Variable",
            text:
                "The marginal distribution of random variables " +
                katex.renderToString("X: \\Omega \\rightarrow \\mathbb{R}^n") +
                ` on a probability space is given by ` +
                katex.renderToString(`\\mathbb{P}_{X_i}, 1\\leq i\\leq n`) +
                " which has a marginal CDF defined as:",
            formula: `
                \\begin{align*}
                    F_{X_i}(t)  &= \\mathbb{P}_{X_i}\\left((-\\infty,t]\\right) \\\\
                                &= \\mathbb{P}_X\\left( \\mathbb{R}\\times \\dots\\times(-\\infty,t] \\times \\dots \\times \\mathbb{R} \\right) \\\\
                                &= \\mathbb{P}(X_1\\in\\mathbb{R}, \\dots , X_i \\leq t,\\dots,X_n\\in\\mathbb{R}) 
                \\end{align*}
            `,
        },
        conditionalExpectation: {
            title: "Conditional Expectation (event)",
            text: `Given some probability space and an event B, the conditional expectation 
            of a random variable:`,
            formula: `\\begin{align*}
                \\mathbb{E}(X\\mid B):  &=\\int_\\Omega Xd\\mathbb{P}(\\cdot\\mid B) \\\\
                                        &=\\frac{1}{\\mathbb{P}(B)}\\int_\\Omega X \\mathbb{1}_B d \\mathbb{P} \\\\
                                        &=\\frac{1}{\\mathbb{P}(B)}\\mathbb{E}(\\mathbb{1}_B X)
            \\end{align*}`,
        },
        conditionalDiscreteExpectation: {
            title: "Conditional Expectation (discrete RV)",
            text: `Given some probability space and two random variables, 
            the conditional expectation of a random variable is given by:`,
            formula: `g(y):=\\mathbb{E}(X\\mid Y=y)=\\sum_x x \\frac{\\mathbb{P}(X=x \\land Y=y)}{\\mathbb{P}(Y=y)}`,
        },
        conditionalContinousExpectation: {
            title: "Conditional Expectation (continuous RV)",
            text: `Given some probability space and two random variables, 
            the conditional expectation of a random variable is given by:`,
            formula: `g(y):=\\mathbb{E}(X\\mid Y=y)=\\int_\\mathbb{R} x \\frac{f_{(X,Y)}(x,y)}{f_Y(y)}dx`,
        },
        stochasticProcess: {
            title: "Stochastic Process",
            text:
                "Given a set T," +
                katex.renderToString("\\forall t \\in T") +
                " we define " +
                katex.renderToString("X_t:\\Omega \\rightarrow \\mathbb{R}") +
                "and define a stochastic process as:",
            formula: `(X_t)_{t\\in T}`,
        },
        markovChain: {
            title: "Markov Chains",
            text:
                "Given a stochastic process, we call " +
                katex.renderToString("(X_t)_{t\\in T}") +
                " a Markov process or Markov Chain if:",
            formula: `\\forall n \\in \\mathbb{N}, t_1,\\dots,t_n,t\\in T, t_1<\\dots<t_n<t,\\\\
                 x_1,\\dots,x_n,x\\in \\mathbb{R}: \\\\
                \\mathbb{P}(X_t=x \\mid X_{t_1} = x_1, \\dots, X_{t_n}=x_n)=\\mathbb{P}(X_t=x \\mid X_{t_n}=x_n)`,
        },
        stationaryDistribution: {
            title: "Stationary Distribution",
            text:
                katex.renderToString("q\\in\\mathbb{R}^{1\\times N}") +
                " is called a stationary distribution for a markov chain if:",
            formula: `qP=q`,
        },
        weakLLN: {
            title: "Weak Law of Large Numbers",
            text:
                "Let random variables " +
                katex.renderToString("(X_k)_{k\\in\\mathbb{N}}") +
                " be independent and identically distributed (i.i.d.) and " +
                katex.renderToString("\\mathbb{E}(|X_1|<\\infty)") +
                ", then for " +
                katex.renderToString("\\epsilon>0:"),
            formula: `\\mathbb{P}\\left(\\left| \\frac{1}{n}\\sum_{k=1}^n X_k - \\mathbb{E}(X_1)\\right|\\geq\\epsilon\\right) \\xrightarrow{n\\to\\infty}0`,
        },
        strongLLN: {
            title: "Strong Law of Large Numbers",
            text:
                "Let random variables " +
                katex.renderToString("(X_k)_{k\\in\\mathbb{N}}") +
                " be i.i.d. and " +
                katex.renderToString("\\mathbb{E}(|X_1|<\\infty)") +
                ", then for " +
                katex.renderToString("\\omega\\in\\Omega)") +
                " almost surely:",
            formula: `\\frac{1}{n}\\sum_{k=1}^nX_k(\\omega)=: \\frac{1}{n}\\sum_{k=1}^n X_k (\\omega) \\xrightarrow{n\\to\\infty} \\mathbb{E}(X_1)`,
        },
        centralLimit: {
            title: "Central Limit Theorem",
            text:
                "Let random variables " +
                katex.renderToString("(X_k)_{k\\in\\mathbb{N}}") +
                " be i.i.d. with " +
                katex.renderToString("\\sigma:=\\sqrt{Var(X_1)}<\\infty:"),
            formula: `Y_n:= \\left(\\frac{1}{n}\\sum_{k=1}^n X_k -\\mathbb{E}(X_1)\\right)\\cdot\\left(\\frac{\\sigma}{\\sqrt{n}}\\right)^{-1}\\\\ \\Rightarrow \\mathbb{P}(Y_n\\leq x)\\xrightarrow{n\\to\\infty} \\phi(x),\\quad\\forall x\\in\\mathbb{R}`,
        },
    };

    let definition = $derived(definitions[key]);
</script>

<article>
    <header>Definition: {definition.title}</header>
    <p>{@html definition.text}</p>
    <Math expression={definition.formula} inline={false} />
</article>

<style>
    article {
        border-radius: 1em;
    }

    article > header {
        font-weight: bold;
    }
</style>

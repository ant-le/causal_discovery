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
                " is called continous at " +
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
            formula:
                "\\begin{align} \\emptyset,X &\\in \\Alpha \\\\ A &\\in \\Alpha \\Rightarrow A^C := X \\setminus A \\in \\Alpha \\\\ A_i &\\in \\Alpha, i \\in \\mathbb{N} \\Rightarrow \\bigcup_{i =1}^\\infty A_i \\in \\Alpha \\end{align}",
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
            formula: `\\begin{align} \\mu(\\emptyset)&=0 \\\\ \\mu(\\bigcup_{i=1}^n A_i)&=\\sum_{i=1}^n \\mu(A_i), \\quad \\forall A_i \\in \\Alpha  \\end{align}`,
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
                " and a function " +
                katex.renderToString(`f\\in S^+`) +
                ", the integral of f w.r.t. " +
                katex.renderToString("\\mu") +
                " is defined as:",
            formula:
                "\\int_{x}f d\\mu = I(f) = \\sum_{i=1}^n c_i \\cdot \\mu(A_i) \\in [0,\\infty]",
        },
        lebeque: {
            title: "Lebeque Integral",
            text:
                "For a measurable map " +
                katex.renderToString("f:X\\rightarrow [0,\\infty)") +
                ", the lebesque integral is defined as:",
            formula:
                "\\int_{x}f d\\mu := sup \\{ I(h) \\mid h \\in S^+, h \\leq f \\}",
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
            title: "Lebesque's dominated convergence theorem",
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
            \\begin{align}
                \\mathbb{P}(A) &\\in [0,1] \\\\
                \\mathbb{P}(\\emptyset)&=0 \\\\
                \\mathbb{P}(\\Omega)&=1\\\\ 
                \\mathbb{P}(\\bigcup_{j=1}^\\infty A_j)&=\\sum_{j=1}^\\infty \\mathbb{P}(A_j)
            \\end{align}`,
        },
        conditionalProb: {
            title: "Conditional robability",
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
                " are called a random variable if:",
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
            title: "Comulative Distribution Function (CDF)",
            text: `A CDF given a probability space and a random variable is 
                defined as:`,
            formula: `F_X:\\mathbb{R} \\rightarrow [0,1], \\quad F_X(x):= \\mathbb{P}_X \\left( (-\\infty, x] \\right)=\\mathbb{P}(X\\leq x)`,
        },
        randomVarIndependence: {
            title: "Independence of Events",
            text: "Given some probability space, two random variables are called independent if " +
            katex.renderToString("\\forall x,y \\in \\mathbb{R}:"),
            formula: `X^{-1}\\left((-\\infty, x]\\right) \\land Y^{-1} \\left( (-\\infty,y] \\right) \\text{ are independent events}`,
        },
        expectation: {
            title: "Expectation of a Random Variable",
            text: `Given some probability space, the expectation of a random 
            variable can be defined with its lebesque integral:`,
            formula: "\\mathbb{E}(X):=\\int_\\Omega Xd\\mathbb{P}"
        },
    };

    let definition = $derived(definitions[key]);
</script>

<article>
    <header>Defintion: {definition.title}</header>
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

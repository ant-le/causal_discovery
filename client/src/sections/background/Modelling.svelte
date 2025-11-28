<script lang="ts">
    import Math from "../../lib/Math.svelte";
</script>

<section id="intro">
    <p>
        Causality lies between differential equations know from phisics which
        need to be carefully designed and statistical learning techniches used
        in current ML which are mostly learned from observational data
    </p>
</section>

<section id="inputs">
    <p>We can divide data across two dimensions:</p>
    <ol>
        <li>Observational vs. Interventional</li>
        <li>Hand-engineered vs. raw, unstructured (perceptual)</li>
    </ol>
</section>

<section id="models">
    <article>
        <header>Statistical Models</header>
        Current ML which heavily relies on a large amount of in-distribution data.
    </article>
    <article>
        <header>Causal Graphical Models</header>
        Here, dependencies between phenomena are modelled with a graph. Hereby, these
        models can have many different representations for data which would always
        be equal for models based on statistical learning.
    </article>
    <article>
        <header>Structural Causal Models</header>
        Considers set of observables associated with vertices of a DAG.
        <Math
            expression={"X_i:=f_i(PA_i,U_i), \\quad (i=1,\\dots,n)"}
            inline={false}
        /> using a deterministic function f depending on a set of parents PA
        and some unexplained random variable U.
    </article>
</section>

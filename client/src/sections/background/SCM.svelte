<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
    import Math from "../../lib/Math.svelte";
</script>

<section id="formulation">
    <p>
        <strong>Structural Causal Models (SCM)</strong> allow us to specify a
        DAG in terms of causal dependencies instead of statistical associations
        (e.g. in Bayesian Networks). In this, a node ca be described by its
        parents <Math expression="PA_i" /> and some noise <Math
            expression="U_i"
        />:

        <Math
            expression={`X_i :=f_i(PA_i,U_i)=\\mathbb{P}(X_i\\mid PA_i)=f_i(PA_i)+U_i, \\quad U_i \\sim
            N(0,\\sigma^2_i)`}
            inline={false}
        />
        If the noise terms are pair-wise independent, we can reformulate our functional
        form with probabilities:
        <Math
            expression={`f_i(PA_i,U_i)=\\mathbb{P}(X_i\\mid PA_i)=f_i(PA_i)+U_i, \\quad U_i \\sim N(0,\\sigma^2_i)`}
            inline={false}
        />
        The full (joint) causal model can be also formulated with probabilities:
        <Math
            expression={"\\mathbb{P}(X_1,\\dots,X_n):=\\prod_{i=1}^n\\mathbb{P}(X_i\\mid PA_i)"}
            inline={false}
        />
        Hence, we the specification of an SCM requires two main components:
    </p>
    <ol>
        <li>Learning the structure <Math expression={"PA_i"} /></li>
        <li>Learning the functional forms of <Math expression="f_i" /></li>
    </ol>

    <Cite key="causalDiscovery" />
</section>

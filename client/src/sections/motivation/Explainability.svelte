<script lang="ts">
    import Cite from "../../lib/Cite.svelte";
</script>

<section>
    <p>
        <strong>RQ2</strong> focuses on moving from graph-only predictions toward
        richer posterior objects that better reflect uncertainty about both
        structure and mechanisms.
    </p>

    <ul>
        <li>
            Keep the model output Bayesian: calibrated uncertainty is as
            important as mean performance.
        </li>
        <li>
            Evaluate how runtime and memory costs grow when approximating richer
            posterior families <Cite key="bcnp" />.
        </li>
        <li>
            Preserve graph-level quality while extending inference objectives to
            better support downstream intervention analysis <Cite
                key="evaluation"
            />.
        </li>
    </ul>

    <p>
        This framing follows the theory and evaluation concerns summarized in
        the project notes under <code>paper/markdown/keep_both</code> and
        <code>paper/markdown/keep_rq2</code>.
    </p>
</section>

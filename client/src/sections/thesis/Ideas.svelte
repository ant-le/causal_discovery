<script lang="ts">
    import PipelineFlow from "../../lib/PipelineFlow.svelte";
</script>

<section>
    <article>
        <header>Architecture View</header>
        <p>
            The repository is organized as a modular benchmark architecture that
            maps directly to the thesis workflow.
        </p>
    </article>

    <article>
        <header>Layered Design</header>
        <ol>
            <li>
                <strong>Configuration Layer:</strong> Hydra groups define data
                families, model variants, training, inference, and logging.
            </li>
            <li>
                <strong>Dataset Layer:</strong> modules sample SCM families and
                expose train/evaluation datasets.
            </li>
            <li>
                <strong>Model Layer:</strong> amortized and explicit inference
                methods share a common interface.
            </li>
            <li>
                <strong>Runner Layer:</strong> task-oriented runners separate
                pre-training, inference, and evaluation.
            </li>
            <li>
                <strong>Metrics Layer:</strong> graph and SCM metrics provide
                comparable outputs across methods.
            </li>
        </ol>
    </article>

    <article>
        <header>Flow to Method Section</header>
        <p>
            The architecture explains <em>where</em> components live; the next
            section explains <em>how</em> these components execute an end-to-end
            Bayesian discovery run.
        </p>
    </article>

    <PipelineFlow />
</section>

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
        <header>How to Read This</header>
        <p>
            This view explains <em>where</em> benchmark components live. The
            methodology and results pages then show <em>how</em> they are used
            and <em>what</em> metrics are produced.
        </p>
    </article>

    <PipelineFlow />
</section>

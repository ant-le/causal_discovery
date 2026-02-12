<section>
    <p>
        The repository is organized around a modular benchmark architecture that
        maps directly to the thesis workflow.
    </p>

    <ol>
        <li>
            <strong>Configuration Layer:</strong> Hydra configs in
            <code>src/causal_meta/configs/</code> define data families, model
            variants, training, inference, and logging.
        </li>
        <li>
            <strong>Dataset Layer:</strong> <code>datasets/</code> samples SCM
            families and exposes iterable/fixed PyTorch datasets for training and
            evaluation.
        </li>
        <li>
            <strong>Model Layer:</strong> <code>models/</code> keeps amortized and
            explicit inference methods behind a shared base interface.
        </li>
        <li>
            <strong>Runner Layer:</strong> <code>runners/tasks/</code> separates
            pre-training, inference, and evaluation to keep experiments
            reproducible.
        </li>
        <li>
            <strong>Metrics Layer:</strong> graph- and SCM-focused metrics provide
            comparable benchmark outputs across model classes.
        </li>
    </ol>
</section>

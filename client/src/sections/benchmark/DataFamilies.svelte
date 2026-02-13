<script lang="ts">
    import ContentStatus from "../../lib/ContentStatus.svelte";
    import DataFamilyGallery from "../../lib/DataFamilyGallery.svelte";
</script>

<section>
    <article>
        <header>Configurable Data Families</header>
        <ContentStatus status="implemented" text="Implemented generator families" />
        <p>
            Synthetic tasks are generated from configurable graph and mechanism
            families, including Erdos-Renyi, scale-free, SBM, and mixture graph
            settings plus mechanism families such as linear, periodic,
            logistic-map, GP, and PNL.
        </p>
    </article>

    <article>
        <header>Benchmark Use</header>
        <ul>
            <li>Train/validation/test splits are sampled reproducibly.</li>
            <li>Shift studies alter mechanism, graph, and noise factors.</li>
            <li>Family choices are configured through Hydra-driven loaders.</li>
        </ul>
    </article>

    <DataFamilyGallery />
</section>

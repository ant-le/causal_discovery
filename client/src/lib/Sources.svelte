<script lang="ts">
    interface SourceReferences {
        code: string[];
        notes: string[];
        docs: string[];
    }

    let { sources }: { sources: SourceReferences | null } = $props();

    type TopicRule = readonly [RegExp, string];

    const CODE_TOPICS: TopicRule[] = [
        [/main\.py$/, "Pipeline orchestration, task routing, and distributed execution controls."],
        [/models\/base\.py$/, "Shared model contract used by amortized and explicit inference methods."],
        [/models\/factory\.py$/, "Factory-based model construction for benchmarked method families."],
        [/models\/bcnp\/model\.py$/, "BCNP-specific posterior modeling and inference behavior."],
        [/models\/utils\/nn\.py$/, "Neural building blocks and helper utilities for model components."],
        [/datasets\/data_module\.py$/, "Dataset module wiring for reproducible train/validation/test splits."],
        [/datasets\/scm\.py$/, "SCM sampling and mechanism definitions used in synthetic data generation."],
        [/datasets\/generators\/factory\.py$/, "Generator dispatch for configurable graph and mechanism families."],
        [/runners\/tasks\/pre_training\.py$/, "Amortized pre-training stage of the benchmark pipeline."],
        [/runners\/tasks\/inference\.py$/, "Explicit inference stage for posterior approximation."],
        [/runners\/tasks\/evaluation\.py$/, "Evaluation stage for shift robustness and posterior quality analysis."],
        [/runners\/metrics\/graph\.py$/, "Graph-structure metrics such as edge recovery and ranking quality."],
        [/runners\/metrics\/scm\.py$/, "Mechanism and intervention-focused metrics for SCM fidelity."],
        [/runners\/utils\/distributed\.py$/, "Distributed runtime utilities for device placement and process setup."],
        [/runners\/utils\/scoring\.py$/, "Scoring utilities used by probabilistic and benchmark metrics."],
    ];

    const NOTE_TOPICS: TopicRule[] = [
        [/keep_background/, "Background literature on causality, SCM assumptions, and Bayesian foundations."],
        [/keep_rq1/, "RQ1 notes on OOD robustness, meta-learning, and shift-aware evaluation."],
        [/keep_rq2/, "RQ2 notes on richer posterior approximations and short-chain trade-offs."],
        [/keep_both/, "Shared evaluation notes used by both research questions."],
        [/optional_rq2/, "Optional acceleration notes on GPU-enabled MCMC workflows."],
        [/drop_background/, "Archived appendix notes for mathematical background context."],
        [/meta/, "Meta literature curation used for scope and motivation framing."],
    ];

    const DOC_TOPICS: TopicRule[] = [
        [/DESIGN\.md$/, "Architecture rationale and component interaction design notes."],
        [/RUNBOOK\.md$/, "Runbook guidance for reproducible experiment execution."],
        [/CLASS_STRUCTURE\.md$/, "Module and class organization overview."],
        [/PROFILING\.md$/, "Profiling guidance for throughput and acceleration analysis."],
        [/proposal\.md$/, "Early-stage proposal context and initial project framing."],
        [/README\.md$/, "Project overview and high-level benchmark objectives."],
    ];

    function summarizeTopics(
        entries: string[],
        rules: TopicRule[],
        fallback: string,
    ): string[] {
        const topics = new Set<string>();

        for (const entry of entries) {
            let matched = false;
            for (const [pattern, topic] of rules) {
                if (pattern.test(entry)) {
                    topics.add(topic);
                    matched = true;
                }
            }

            if (!matched) {
                topics.add(fallback);
            }
        }

        return Array.from(topics);
    }

    let codeTopics: string[] = $derived(
        sources
            ? summarizeTopics(
                  sources.code,
                  CODE_TOPICS,
                  "Implementation details from the benchmark codebase.",
              )
            : [],
    );

    let noteTopics: string[] = $derived(
        sources
            ? summarizeTopics(
                  sources.notes,
                  NOTE_TOPICS,
                  "Curated thesis notes connected to this section.",
              )
            : [],
    );

    let docTopics: string[] = $derived(
        sources
            ? summarizeTopics(
                  sources.docs,
                  DOC_TOPICS,
                  "Supporting project documentation for this section.",
              )
            : [],
    );
</script>

{#if sources}
    <section id="sources">
        <hr />
        <h4>Content Grounding</h4>

        <p class="sources-intro">
            This section is backed by implementation, literature notes, and
            project documentation covering the topics below.
        </p>

        {#if codeTopics.length > 0}
            <p><strong>Implementation</strong></p>
            <ul>
                {#each codeTopics as topic}
                    <li>{topic}</li>
                {/each}
            </ul>
        {/if}

        {#if noteTopics.length > 0}
            <p><strong>Theory and Literature</strong></p>
            <ul>
                {#each noteTopics as topic}
                    <li>{topic}</li>
                {/each}
            </ul>
        {/if}

        {#if docTopics.length > 0}
            <p><strong>Project Documentation</strong></p>
            <ul>
                {#each docTopics as topic}
                    <li>{topic}</li>
                {/each}
            </ul>
        {/if}
    </section>
{/if}

<style>
    #sources ul {
        margin-bottom: 1rem;
    }

    .sources-intro {
        margin-bottom: 0.75rem;
    }
</style>

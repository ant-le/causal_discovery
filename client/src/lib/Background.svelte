<script lang="ts">
  import { backgroundData } from "../data/textData";
</script>

<article>
  <header>
    <h1>
      Background: Pillars of LLM-Guided Probabilistic Causal Discovery
    </h1>
  </header>

  <section>
    <p>
      This Master's thesis is positioned at the convergence of several
      cutting-edge fields in artificial intelligence. To fully appreciate its
      core contributions, a foundational understanding of Causal Inference and the
      challenges of Structural Causal Discovery is essential. Furthermore, it requires
      an examination of the dual role of Large Language Models (LLMs) as both
      powerful knowledge repositories and non-causal reasoners, and the pivotal
      function of Probabilistic Programming as the framework for a principled,
      Bayesian synthesis of knowledge and data.
    </p>

    {#each backgroundData as section, i}
      <section>
        <h3>{i + 1}. {section.title}</h3>
        <p>{@html section.content}</p>

        {#if section.topics && section.topics.length > 0}
          <ul>
            {#each section.topics as topic}
              <li>{@html topic}</li>
            {/each}
          </ul>
        {/if}
      </section>
      {#if i < backgroundData.length - 1}
        <hr />
      {/if}
    {/each}

     <hr />

    <p>
      By combining these three pillars—the formalisms of Causal Inference, the
      expert-level knowledge extracted from LLMs, and the robust Bayesian
      framework of Probabilistic Programming—this thesis aims to develop and
      evaluate a novel methodology that pushes the boundaries of automated
      causal discovery, contributing to the development of more intelligent and
      generalizable AI.
    </p>
  </section>
</article>

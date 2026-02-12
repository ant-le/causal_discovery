<script lang="ts">
    import { motivationData } from "../../assets/data/textData.ts";
</script>

<section>
    <article>
        <header>Where This Matters</header>
        <p>
            The benchmark is motivated by domains where incorrect causal
            assumptions can create costly or unsafe decisions.
        </p>
    </article>

    <div class="grid cards">
        {#each motivationData.practical as item}
            <article>
                <header>{item.icon} {item.category}</header>
                <h5>{item.title}</h5>
                <p>{item.content}</p>
            </article>
        {/each}
    </div>

    <article>
        <header>Decision Stakes</header>
        <table>
            <thead>
                <tr>
                    <th scope="col">Risk if Causality Is Wrong</th>
                    <th scope="col">Value of Bayesian Posteriors</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Overconfident interventions under data shift</td>
                    <td>Uncertainty-aware policy choices</td>
                </tr>
                <tr>
                    <td>Mis-prioritized experiments and resources</td>
                    <td>Ranked plausible mechanisms for follow-up</td>
                </tr>
                <tr>
                    <td>Poor transfer across institutions or cohorts</td>
                    <td>Explicit robustness tracking across shifts</td>
                </tr>
            </tbody>
        </table>
    </article>
</section>

<style>
    .cards {
        gap: 1rem;
        margin-bottom: 1rem;
    }

    h5 {
        margin-bottom: 0.5rem;
    }
</style>

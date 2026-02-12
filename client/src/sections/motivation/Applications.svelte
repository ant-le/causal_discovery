<script lang="ts">
    import { motivationData } from "../../assets/data/textData.ts";
</script>

<section>
    <p>
        The benchmark is motivated by domains where wrong causal assumptions can
        produce costly decisions.
    </p>

    <div class="cards">
        {#each motivationData.practical as item}
            <article>
                <header>{item.icon} {item.category}</header>
                <h5>{item.title}</h5>
                <p>{item.content}</p>
            </article>
        {/each}
    </div>
</section>

<style>
    .cards {
        display: grid;
        gap: 1rem;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }

    h5 {
        margin-bottom: 0.5rem;
    }
</style>

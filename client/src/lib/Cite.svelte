<script lang="ts">
    import { citations } from "..data/citations.ts";

    let { id }: { id: string } = $props();

    citations.add(id);

    // Get the ordered list of all citations to find the index of this one.
    const bibliographyOrder = citations.bibliographyOrder;
    const citationNumber = $derived(bibliographyOrder().indexOf(id) + 1);
</script>

<a href="#citation-{id}" class="citation-link">
    <sup>[{citationNumber}]</sup>
</a>

<style>
    .citation-link {
        color: var(--pico-primary);
        text-decoration: none;
        font-weight: bold;
    }
    .citation-link:hover {
        text-decoration: underline;
    }
</style>

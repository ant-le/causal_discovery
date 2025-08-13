<script lang="ts">
    import { getContext } from "svelte";
    import { type CitationKey } from "../assets/citation.ts";

    type AddCitationFn = (key: CitationKey) => number;
    const addCitation = getContext<AddCitationFn>("addCitation");
    let { key }: { key: CitationKey } = $props();
    const citationNumber = addCitation(key);
</script>

<sup class="citation">[{citationNumber}]</sup>

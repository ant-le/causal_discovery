<script lang="ts">
    import katex from "katex";
    import "katex/dist/katex.min.css";

    let {
        expression,
        inline = true,
    }: { expression: string; inline?: boolean } = $props();
    let renderedhtml = $derived(
        katex.renderToString(expression, {
            displayMode: !inline,
            throwOnError:false,
        }),
    );
</script>

{@html renderedhtml}

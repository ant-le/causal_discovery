<script lang="ts">
    import katex from "katex";
    import "katex/dist/katex.min.css";

    export let expression: string;
    export let inline: boolean = false;

    let renderedHtml: string = "";

    $: {
        try {
            renderedHtml = katex.renderToString(expression, {
                displayMode: !inline,
                throwOnError: false, // Prevents crashing on invalid LaTeX
            });
        } catch (error) {
            console.error("KaTeX rendering error:", error);
            renderedHtml = `<span class="error" style="color: red;">${error}</span>`;
        }
    }
</script>

<code>{@html renderedHtml}</code>

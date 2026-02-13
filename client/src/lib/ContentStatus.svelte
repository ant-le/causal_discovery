<script lang="ts">
    export type ContentStatusKind =
        | "implemented"
        | "measured"
        | "illustrative"
        | "planned";

    const statusLabels: Record<ContentStatusKind, string> = {
        implemented: "Implemented",
        measured: "Measured",
        illustrative: "Illustrative",
        planned: "Planned",
    };

    let {
        status,
        text = null,
    }: {
        status: ContentStatusKind;
        text?: string | null;
    } = $props();

    let label = $derived(text ?? statusLabels[status]);
</script>

<small class={`content-status ${status}`}>{label}</small>

<style>
    .content-status {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.12rem 0.55rem;
        border-radius: 999px;
        border: 1px solid var(--pico-muted-border-color);
        font-size: 0.73rem;
        letter-spacing: 0.02em;
        font-weight: 600;
        text-transform: uppercase;
    }

    .content-status.implemented {
        border-color: color-mix(in oklab, var(--pico-primary) 45%, var(--pico-muted-border-color) 55%);
    }

    .content-status.measured {
        border-color: color-mix(in oklab, var(--pico-contrast) 55%, var(--pico-muted-border-color) 45%);
    }

    .content-status.illustrative {
        border-color: color-mix(in oklab, var(--pico-secondary) 50%, var(--pico-muted-border-color) 50%);
    }

    .content-status.planned {
        border-color: color-mix(in oklab, var(--pico-secondary) 55%, var(--pico-muted-border-color) 45%);
    }
</style>

<script>
    import "@picocss/pico";
    import { getContext } from "svelte";
    const appState = getContext("appState");
</script>

<nav>
    <ul>
        <li>
            <a
                href="prevented"
                class="contrast"
                on:click|preventDefault={() => (appState.name = "home")}
                ><strong>Anton Lechuga</strong>
            </a>
        </li>
    </ul>
    <ul>
        <li>
            <a
                href="prevented"
                class={appState.name === "motivation" ? undefined : "secondary"}
                on:click|preventDefault={() => (appState.name = "motivation")}
                aria-current={appState.name === "motivation"
                    ? "page"
                    : undefined}>Motivation</a
            >
        </li>
        <li>
            <a
                href="prevented"
                class={appState.name === "background" ? undefined : "secondary"}
                on:click|preventDefault={() => (appState.name = "background")}
                aria-current={appState.name === "motivation"
                    ? "page"
                    : undefined}>Background</a
            >
        </li>
        <li>
            <a
                href="prevented"
                class={appState.name === "content" ? undefined : "secondary"}
                on:click|preventDefault={() => (appState.name = "content")}
                aria-current={appState.name === "motivation"
                    ? "page"
                    : undefined}>Content</a
            >
        </li>
    </ul>
</nav>

<script lang="ts">
    let { appStates, appState, updateState, theme, toggleTheme } = $props();

    let iconSize = 20;
</script>

<nav>
    <ul>
        <li>
            <a
                href="home"
                class="contrast"
                onclick={(e) => {
                    e.preventDefault();
                    updateState("Home");
                }}
                aria-label="Home Button"
                ><svg
                    xmlns="http://www.w3.org/2000/svg"
                    width={iconSize}
                    height={iconSize}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    style="vertical-align: middle;"
                >
                    <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
                    ></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
            </a>
        </li>
    </ul>
    <ul>
        {#each appStates as state}
            <li>
                <a
                    href={state}
                    class:secondary={appState !== state}
                    onclick={(e) => {
                        e.preventDefault();
                        updateState(state);
                    }}
                >
                    {state}
                </a>
            </li>
        {/each}
        <li>|</li>
        <li class="hide-on-mobile">
            <a
                href="mailto:anton.luis.lechuga@proton.me"
                class="secondary"
                aria-label="Email"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width={iconSize}
                    height={iconSize}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path
                        d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"
                    ></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                </svg>
            </a>
        </li>
        <li class="hide-on-mobile">
            <a
                href="https://github.com/ant-le/causal_discovery"
                target="_blank"
                rel="noopener noreferrer"
                class="secondary"
                aria-label="GitHub"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width={iconSize}
                    height={iconSize}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path
                        d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
                    >
                    </path>
                </svg>
            </a>
        </li>
        <li>
            <a
                href="toggle"
                onclick={(e) => {
                    e.preventDefault();
                    toggleTheme();
                }}
                class="secondary"
                aria-label="Toggle color scheme"
            >
                {#if theme === "light"}
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width={iconSize}
                        height={iconSize}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <path
                            d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
                        ></path>
                    </svg>
                {:else}
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width={iconSize}
                        height={iconSize}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"
                        ></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                    </svg>
                {/if}
            </a>
        </li>
    </ul>
</nav>

<style>
    @media (max-width: 576px) {
        .hide-on-mobile {
            display: none;
        }
    }

    a:hover {
        text-decoration: none;
    }

    nav {
        min-height: 3.4rem;
    }

    nav a {
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    @media (max-width: 700px) {
        nav ul {
            gap: 0.1rem;
        }

        nav a {
            padding-inline: 0.45rem;
            font-size: 0.88rem;
        }
    }
</style>

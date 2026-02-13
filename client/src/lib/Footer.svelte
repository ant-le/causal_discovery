<script lang="ts">
    let { updateState } = $props();
    const currentYear = new Date().getFullYear();
    const licenseType = "MIT";
</script>

<div class="mobile-footer container">
    <div class="social-icons">
        <a
            href="mailto:anton.luis.lechuga@proton.me"
            class="secondary"
            aria-label="Email"
        >
            <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                ><path
                    d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"
                ></path><polyline points="22,6 12,13 2,6"></polyline></svg
            >
        </a>
        <a
            href="https://github.com/ant-le/causal_discovery"
            target="_blank"
            rel="noopener noreferrer"
            class="secondary"
            aria-label="GitHub"
        >
            <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                ><path
                    d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
                ></path></svg
            >
        </a>
    </div>
    <small>© {currentYear} Anton Lechuga | {licenseType} License</small>
</div>

<div class="desktop-footer container">
    <div class="grid">
        <hgroup>
            <h5>Anton Lechuga</h5>
            <p>Thesis on Causal Discovery</p>
        </hgroup>
        <div class="grid">
            <hgroup>
                <h6>Links</h6>
                <p>
                    <a
                        href="prevented"
                        onclick={(e) => {
                            e.preventDefault();
                            updateState("Motivation");
                        }}>Motivation</a
                    >
                    |
                    <a
                        href="prevented"
                        onclick={(e) => {
                            e.preventDefault();
                            updateState("Methodology");
                        }}>Methodology</a
                    >
                    |
                    <a
                        href="prevented"
                        onclick={(e) => {
                            e.preventDefault();
                            updateState("Results");
                        }}>Results</a
                    >
                </p>
            </hgroup>
        </div>

        <div class="grid">
            <hgroup>
                <h6>Connect</h6>
                <p>
                    <a href="mailto:anton.luis.lechuga@proton.me">Email</a> |
                    <a
                        href="https://github.com/ant-le/causal_discovery"
                        target="_blank"
                        rel="noopener noreferrer">GitHub</a
                    >
                </p>
            </hgroup>
        </div>
    </div>
    <hr />
    <small class="license-text">
        © {currentYear} Anton Lechuga. Code licensed under the {licenseType} License.
        Built with
        <a href="https://svelte.dev/" target="_blank" rel="noopener noreferrer"
            >Svelte</a
        >
        &
        <a href="https://picocss.com" target="_blank" rel="noopener noreferrer"
            >Pico.css</a
        >.
    </small>
</div>

<style>
    .desktop-footer {
        display: none;
    }
    .mobile-footer {
        display: block;
        text-align: center;
    }
    .social-icons a {
        margin: 0 0.5rem;
    }
    .social-icons {
        margin-bottom: 1rem;
    }
    p > a {
        font-size: 0.92em;
    }

    .license-text {
        font-size: 0.88em;
    }

    @media (min-width: 768px) {
        .mobile-footer {
            display: none;
        }
        .desktop-footer {
            display: block;
            text-align: left;
        }
        .desktop-footer small {
            display: block;
            text-align: center;
        }
    }
</style>

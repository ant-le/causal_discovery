import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  base: "/causal_discovery/",
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          katex: ["katex"],
          particles: ["@tsparticles/engine", "@tsparticles/slim"],
        },
      },
    },
  },
});

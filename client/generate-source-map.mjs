import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sourceMapPath = path.join(__dirname, "src/assets/data/sourceMap.json");
const outputPath = path.join(__dirname, "SOURCE_MAP.md");

const raw = await readFile(sourceMapPath, "utf8");
const sourceMap = JSON.parse(raw);

const lines = [
    "# Source Map",
    "",
    "Auto-generated from `src/assets/data/sourceMap.json`.",
    `Generated: ${new Date().toISOString()}`,
    "",
];

for (const [appState, pages] of Object.entries(sourceMap)) {
    lines.push(`## ${appState}`);
    lines.push("");

    for (const [pageName, sections] of Object.entries(pages)) {
        lines.push(`### ${pageName}`);
        lines.push("");

        for (const [sectionName, refs] of Object.entries(sections)) {
            lines.push(`- **${sectionName}**`);
            lines.push(`  - code: ${refs.code.join(", ")}`);
            lines.push(`  - notes: ${refs.notes.join(", ")}`);
            lines.push(`  - docs: ${refs.docs.join(", ")}`);
        }

        lines.push("");
    }
}

await writeFile(outputPath, `${lines.join("\n")}\n`, "utf8");

console.log(`Wrote ${path.relative(__dirname, outputPath)}`);

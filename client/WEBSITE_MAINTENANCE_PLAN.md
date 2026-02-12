# Website Maintenance Plan

## Goals

1. Keep the website build-stable (`npm run check`, `npm run build`).
2. Keep content synchronized with:
   - `src/causal_meta/` implementation details.
   - `paper/markdown/` theory and literature notes.
3. Improve clarity for both code and thesis navigation.

## Phase 1: Stability (Done)

- Fix invalid default page/section state logic in `src/pages/Content.svelte`.
- Remove empty navigation branches and guarantee renderable sections in
  `src/assets/navigation.ts`.
- Repair invalid TypeScript assets and broken import paths.

## Phase 2: Content Refresh (Done)

- Replace stale or empty sections in:
  - `src/sections/motivation/*`
  - `src/sections/background/{Introduction,Modelling,SCM,POF}.svelte`
  - `src/sections/thesis/{Ideas,CausalDiscovery,GPUMCMC}.svelte`
- Refresh shared text content in `src/assets/data/textData.ts`.

## Phase 3: Ongoing Sync Workflow (Next)

For each website update:

1. Trace claim -> source mapping.
2. Update section content.
3. Run `npm run check` and `npm run build`.
4. Record any unresolved gaps.

## Traceability (Done)

- Added machine-readable mapping: `src/assets/data/sourceMap.json`.
- Added generated report: `SOURCE_MAP.md`.
- Added generator command: `npm run sources:map`.
- Added in-page source visibility via `src/lib/Sources.svelte`.

## Backlog

- Add automated content freshness checks against selected `src/` and
  `paper/markdown/` anchors.
- Expand bibliography entries to align with final thesis citations.

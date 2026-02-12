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

## Phase 3: Content + Visualization Roadmap (In Progress)

### Milestone M1 (Done): Motivation Narrative + First Interactive

Scope:

1. Reorganize Motivation sections into a clear sequence:
   - Problem
   - Research questions
   - Decision impact
2. Keep claims content-focused with minimal file-path references.
3. Add one typed interactive visualization for RQ1 shift robustness.
4. Validate mobile/desktop readability and section spacing.

Acceptance:

- `npm run check` passes.
- `npm run build` passes.
- Motivation page has at least one explanatory interactive module.

### Milestone M2 (Done): Background Structure Cleanup + Appendix Containment

Scope:

1. Split Background into:
   - Core concepts (SCM, Bayesian discovery, interventions, evaluation)
   - Technical appendix (extended math foundations)
2. Convert long theory blocks into progressive `article` + `details` layouts.
3. Fix wording quality and mathematical terminology consistency.

Acceptance:

- Core Background flow is readable in one pass.
- Appendix content remains accessible without dominating the main flow.

### Milestone M3 (Done): Thesis Flow + Process Visuals

Scope:

1. Present architecture, method, and acceleration as one connected pipeline.
2. Add animated pipeline/process visual and compact outputs panel.
3. Keep wording tied to actual implementation capabilities and evaluation setup.

Acceptance:

- Thesis page communicates end-to-end process without section jumps.
- Visual modules remain lightweight and typed.

## Phase 4: Ongoing Sync Workflow (Continuous)

Current focus:

1. Keep section claims synchronized with implementation and notes during iterative edits.
2. Expand visual modules only when they clarify benchmark behavior.
3. Maintain Pico.css-first layouts and avoid custom layout regressions.

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
- Add reusable visualization primitives (typed scales, legends, tooltip helpers)
  so new interactives stay consistent.

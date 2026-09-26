---
# gstack: design-md-format=spec
name: cachemax
description: A warm bone-and-graphite instrument face; amber reserved for measured cache reuse; numbers set in ink.
colors:
  background: "#e9e4db"
  surface: "#e0dad0"
  surface-raised: "#f2eee7"
  text: "#1d1c18"
  text-muted: "#5c5749"
  text-faint: "#6a6557"
  accent: "#c47a12"
  accent-resent: "#8a5a3c"
  line: "#cbc5b8"
  line-strong: "#8f8a7d"
  on-accent: "#1d1c18"
typography:
  display:
    fontFamily: Archivo
    fontWeight: 600
    fontSize: 1.375rem
    letterSpacing: -0.01em
  body:
    fontFamily: Source Sans 3
    fontSize: 1rem
    lineHeight: 1.5
  label:
    fontFamily: JetBrains Mono
    fontSize: 0.6875rem
    letterSpacing: 0.08em
  mono:
    fontFamily: JetBrains Mono
    fontSize: 0.84375rem
    fontFeature: tnum
rounded:
  sm: 2px
  md: 2px
  lg: 2px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
motion:
  micro: 80ms
  short: 180ms
  medium: 280ms
components:
  status-strip:
    borderColor: "{colors.line-strong}"
    textColor: "{colors.text}"
  hero-value:
    fontFamily: "{typography.mono.fontFamily}"
    textColor: "{colors.text}"
  provenance-tag:
    borderColor: "{colors.line-strong}"
    textColor: "{colors.text-muted}"
  data-table-row:
    borderColor: "{colors.line}"
    textColor: "{colors.text}"
  tape-cell:
    fontFamily: "{typography.mono.fontFamily}"
    textColor: "{colors.text}"
  control:
    borderColor: "{colors.line-strong}"
    textColor: "{colors.text}"
---

# cachemax

## Overview

**Creative North Star:** A calibrated bench instrument: warm faceplate, hairline seams, numbers you can trust, color that means something.
**Product context:** A local, single-user OpenAI-compatible proxy that measures prompt-cache reuse in front of a cloud LLM (typical) or a local engine (minority), and reports the cost or speed consequence. Peers: oMLX, Grafana, llama.cpp dashboards.
**Mode per surface:** Dashboard = Operate. README and docs = Read.
**Reference sites:** omlx.ai (adjacent local-LLM dashboard; broad strokes only, not the palette).
**Key characteristics:**
- One continuous faceplate divided by hairline seams. No cards.
- Amber is a signal, not a brand. It marks verified cache-served data, and the instrument's own live/active states, and nothing else.
- Numbers are the hero: mono, tabular, oversized.
- The cold-to-warm (cloud: billed-to-cached) transition is always in the hero.
- `—` is visually distinct from `0`, so unexposed never reads as zero.

## Colors

**Strategy:** Restrained. One amber accent plus warm neutrals; color is rare and meaningful.
**Light or dark:** Both, following the OS preference. The use scene is a developer watching a live session on a laptop, often next to a terminal, at any hour, so the chosen palette must hold up in both. Light is warm bone (`#e9e4db`), not white. Dark is charcoal (`#16181a`), not black.

Dark-mode values (parallel to the front-matter tokens):
- background `#16181a`, surface `#1e2124`, surface-raised `#24282b`
- text `#ece8de`, text-muted `#9a968b`, text-faint `#8a8578`
- accent `#e0a23c`, accent-resent `#cf8f5f`
- line `#33373a`, line-strong `#4a4f53`

Amber (`accent`) is the one signal color, in two roles only: it marks verified cache-served data (the hit rate and the tape's hit cells), and it marks the instrument's own live/active states (the status-strip live dot, focus-visible outlines). It is never a button fill, never a logo, never the hero number. Cost and money are facts, so they stay in `text` ink. `accent-resent` marks resent prefix. `text-faint` marks cold, absent, and past values. Neutrals derive from the warm bone hue, so dark mode dims and slightly desaturates rather than inverting.

## Typography

Display and body draw from a working-sans world; data draws from notation.
- **Display: Archivo 700 / 600.** Used for the product name (700) and section headings (600) only. A grotesque built for signage, which reads as machined rather than editorial.
- **Body/UI: Source Sans 3 400.** Labels, sentences, and controls. On the Operate surface this is the sanctioned readable UI face, so no legibility tax is paid for novelty.
- **Mono: JetBrains Mono 400/500/700** for every number, table cell, tape glyph, and `—`. `font-variant-numeric: tabular-nums` always, so columns never shift.

All three load from Google Fonts and subset to WOFF2 for offline use in the embedded binary. The scale is small and deliberate: page display 22px, body 16px, table/mono 13.5px (0.84375rem), label 11px. The hero value is the one oversized exception at 72px (4.5rem) mono 700, tabular. Levels differ by more than a weight.

## Layout

Composition-first, one screen, no navigation. 1240px max content, 32px outer padding (24px at 1024px). Structure:

1. **Status strip** (top): product name, connection state, tape mode, incomplete count, export.
2. **Hero band**: the dominant consequence (cost for cloud, TTFT collapse for local), with the cold-to-warm readout always beside it.
3. **Working canvas**: session table (42%) and prefix tape (58%), separated by 24px, turn rows aligned horizontally.

Grid is disciplined. No sidebar, no enclosing card, no dashboard mosaic. Below 1024px the table and tape stack, the hero stays full width, and the tape scrolls horizontally rather than compressing its marks.

## Elevation & Depth

Depth is tonal, not blurred. Recessed wells use `surface`; the hero uses `surface-raised`. Structure comes from 1px `line` seams and the heavier `line-strong` zone boundaries. No drop shadows, no translucent panels, no glow.

## Shapes

Near-square throughout. Radius is 2px on every surface, input, and tag. Instruments are not rounded. The only full-round element is the small live-state dot in the status strip.

## Components

- **status-strip:** uppercase mono labels in `text-muted`; the live dot in `accent`. The labels and dot are not interactive (no hover, no focus). The embedded `control`s (tape-mode toggle, export) own their own hover and focus-visible treatment.
- **hero-value:** mono 700, tabular, `text` color for facts; `text-faint` for the past value in a transition. Never in `accent` or any signal color.
- **provenance-tag:** 1px `line-strong` border, mono 11px `text-muted`, uppercase. Used for `provider_reported`, `engine_measured`, and `no_cache_truth`.
- **data-table-row:** bottom border `line`; cold rows in `text-faint`; the cumulative row gets a `line-strong` top border and weight 700. Hover raises the row to `surface`. Focus-visible adds an `accent` outline.
- **tape-cell:** monospace glyph, color by state (hit `accent`, resent `accent-resent`, cold `text-faint`, miss `text`, break `text-muted`, incomplete `text-faint`), always distinguishable with color removed by glyph shape (incomplete is marked by its own glyph, never conflated with cold).
- **control** (tape-mode toggle, export): body face 14px, `text` color, no fill; 1px `line-strong` underline or border only. Hover darkens the label to `text`. Focus-visible draws an `accent` outline at 2px offset. Minimum 44x44px target even when the label is smaller. Never amber-filled.

## Do's and Don'ts

- Do: reserve amber for verified cache-served data and live/focus states only.
- Do: render every unexposed value as `—`, styled to look unlike a digit.
- Do: use tabular mono for all figures and keep units attached.
- Do: keep the cold-to-warm transition in the hero at all times.
- Do: separate zones with 1px seams, not cards or shadows.
- Don't: use amber or green on a button, link, logo, or the hero number.
- Don't: stack panels as cards, or nest a card in a card.
- Don't: use gradients, glows, blurs, or decorative dividers.
- Don't: imply precision the data lacks, or print a zero where data is missing.
- Don't: render a disclaimer about what a provider figure is not. Label it once (`provider_reported`) and move on.

## Motion

- **Approach:** minimal-functional. Motion only marks incoming evidence.
- **Easing:** enter ease-out, exit ease-in, move ease-in-out.
- **Duration:** micro 80ms, short 180ms, medium 280ms.
- **The one authored moment:** a new session row and its tape cell arrive with a single 180ms highlight, so the eye catches new evidence without the page pulsing. Respect `prefers-reduced-motion`; never steal focus or scroll position.

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-26 | Initial design system created | Created by /design-consultation from the design review's deferred visual-language decision. Direction A chosen from three HTML previews. Memorable thing: trust, honest numbers. |

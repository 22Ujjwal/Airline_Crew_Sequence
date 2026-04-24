# Aethera Design System

## Overview

**Aethera®** is a brand built around cinematic restraint — design as atmosphere, not interface. The system is used to produce **slide decks, landing pages, and visual narratives** that feel quiet, precise, and timeless.

> Design not as interface, but as atmosphere. Every frame should feel like a pause between noise.

### Sources

No external codebase or Figma was provided. This system was derived entirely from the brand brief. If you have access to a codebase or Figma file, attach it and the system can be updated.

---

## CONTENT FUNDAMENTALS

### Voice & Tone

- **Restrained, poetic, and declarative** — never explanatory
- Write as if carving into stone: every word must earn its place
- Prefer implication over explanation
- Short sentences. Often fragments.
- First-person brand voice ("we build", "we stay") — never instructional
- No second-person unless used as an invitation ("Begin Journey")

### Casing

- Headlines: **Sentence case** — not title case, never all-caps
- Brand name always rendered: **Aethera®** (with registered mark)
- No shouting. No exclamation marks.

### Punctuation & Structure

- Em dashes used for weight, not parenthetical asides
- Commas used sparingly — pauses should feel intentional
- Line breaks are meaningful; a new line is a new breath
- Italics introduce tonal contrast: *silence*, *eternal*, *those who stay*

### Examples

```
Beyond silence, we build the eternal.

We build
for those
who stay.

Past the noise.
Past the obvious.
```

### What to Avoid

- No paragraphs longer than 2 lines
- No more than one core idea per frame
- No bullet points in hero or statement contexts
- No emoji
- No jargon or tech-speak
- No urgency language ("Act now", "Limited time")

---

## VISUAL FOUNDATIONS

### Color System

| Name | Hex | Role |
|---|---|---|
| Absolute White | `#FFFFFF` | Space, silence, background |
| Absolute Black | `#000000` | Intent, structure, primary text |
| Soft Ash | `#6F6F6F` | Thought, distance, secondary text |

No additional colors unless they serve a deliberate narrative break.

### Typography

**Display / Voice:** Instrument Serif (Google Fonts)
- Used for headlines, brand name, poetic text
- Tight tracking: `-0.04em` to `-0.06em`
- Line height: `0.95`–`1.05`
- Italics used sparingly for tonal contrast

**Body / Utility:** Inter (Google Fonts)
- Used for supporting copy, captions, labels
- Tracking: `-0.01em` to `0`
- Line height: `1.5`–`1.6`

### Spacing

- Content lives in a **calm center field**: 60–70% of the viewport width
- Generous padding — minimum 80px vertical, 120px horizontal on desktop
- Negative space carries meaning; do not fill it
- Grid: implicit horizontal center; no complex grid overlays

### Backgrounds

- **Absolute White** or **Absolute Black** as default
- Cinematic still photography: desaturated, soft, non-distracting
- Optional texture: subtle film grain (low opacity, 2–4%)
- Optional effect: depth blur on background imagery
- Optional: slow imperceptible zoom on background (Ken Burns, ~0.02 scale over 10s)
- Gradient overlays: gentle white or black fade at top/bottom edges to protect text

### Animation

- Fades and subtle upward rises only
- Duration: `0.6s`–`0.8s`
- Easing: `cubic-bezier(0.16, 1, 0.3, 1)` — ease out quint
- No bounces, no spring, no aggressive entrance
- If motion is noticeable, it's too much

### Hover & Interaction States

- Opacity reduction: `0.6`–`0.7` on hover for interactive elements
- No color changes on hover; opacity is the signal
- Press state: subtle scale down `0.97` with opacity `0.5`
- No box shadows on hover
- Cursor: default except on explicit CTAs

### Borders & Radius

- No decorative borders
- No rounded corners on primary elements (cards, panels)
- Sharp edges only — zero border radius as default
- Single-pixel rules (`1px solid`) used rarely, at `#6F6F6F` or very low opacity

### Cards & Containers

- No card metaphor by default — content floats in space
- If a container is needed: flat, no shadow, no border, white or near-black fill
- Separation via spacing only, not visual containers

### Imagery

- **Color vibe:** Desaturated, muted, cool-to-neutral tones
- Black and white or near-monochrome preferred
- Grain: yes, subtle
- High contrast between subject and background preferred
- No stock-looking imagery; cinematic, editorial quality

### Iconography

→ See ICONOGRAPHY section below

### Shadows

- No drop shadows on text or content elements
- No box shadows
- Depth achieved via opacity, scale, and spacing alone

---

## ICONOGRAPHY

No icon system is used. Aethera® deliberately avoids iconographic shorthand — no icon fonts, no SVG icon libraries, no emoji.

Where navigation or action is needed, text labels are used exclusively. The brand relies on typography and spacing to create visual hierarchy.

**Assets available in `assets/`:**
- `logo.svg` — Aethera® wordmark

---

## FILE INDEX

```
README.md                  ← this file
SKILL.md                   ← agent skill definition
colors_and_type.css        ← CSS design tokens (colors, type, spacing)
assets/
  logo.svg                 ← Aethera® wordmark
preview/
  colors-primary.html      ← Color palette card
  colors-semantic.html     ← Semantic color roles card
  type-display.html        ← Instrument Serif specimens
  type-body.html           ← Inter specimens
  type-scale.html          ← Full type scale
  spacing-tokens.html      ← Spacing scale tokens
  spacing-in-use.html      ← Spacing composition example
  motion-tokens.html       ← Motion/easing tokens
  component-cta.html       ← CTA button states
  component-text-block.html← Text block archetypes
slides/
  index.html               ← Slide deck showcase
  OriginSlide.jsx
  StatementSlide.jsx
  ExpansionSlide.jsx
  FragmentSlide.jsx
  TransitionSlide.jsx
  ClosureSlide.jsx
```

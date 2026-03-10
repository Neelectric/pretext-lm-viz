# pretext

Text measurement for the browser. Predicts text block heights without triggering layout reflow on the resize hot path.

## Problem

Measuring text in the browser requires DOM reads (`getBoundingClientRect`, `offsetHeight`), which trigger synchronous layout reflow. When UI components independently measure text — e.g. a virtual scrolling list sizing 500 comments — each measurement forces the browser to recompute layout for the entire document. This creates read/write interleaving that can cost 30ms+ per frame.

## Solution

Two-phase measurement centered around canvas `measureText()`:

```js
import { prepare, layout } from './src/layout.ts'

// Phase 1: measure word widths (once, when text appears)
const block = prepare(commentText, '16px Inter')

// Phase 2: compute height at any width (pure arithmetic, on every resize)
const { height, lineCount } = layout(block, containerWidth, 19)
```

`prepare()` does a one-time text analysis pass (whitespace normalization, segmentation, punctuation/CJK fixes), then measures the resulting segments via canvas and caches the widths. On browsers that need emoji correction, it also does one cached DOM calibration read per font. `layout()` walks the cached widths to count lines and multiplies by the caller-provided `lineHeight` — no canvas, no DOM, no string operations. Each `layout()` call is ~0.0002ms.

`prepare()` intentionally returns an opaque handle for the hot path. If you need the richer segment-level structure for diagnostics or custom line rendering, use `prepareWithSegments()` and treat that result as the experimental escape hatch.

## Practical uses

- Virtualized feeds and comment lists: predict row heights before mount so scrolling stays stable without DOM measurement passes.
- Masonry or card grids: size text-heavy cards up front before placing them into columns.
- Chat or messaging UIs: recompute bubble heights on every width change without touching the DOM layout engine.
- Loading skeletons and cumulative layout shift reduction: reserve the right amount of vertical space before the final text renders.
- Responsive card/layout decisions: switch between compact and expanded variants based on predicted text height.
- Canvas or custom renderers: use `layoutWithLines()` to get browser-like wrapping plus per-line segment/grapheme cursors, without relying on DOM text nodes.

## Performance

Measured on a MacBook Pro (Apple M2 Pro, 12 CPU cores, 16 GB RAM, macOS 26.3, arm64).

500 comments, resize to a new width (the hot path):

| Browser | prepare() | layout() | DOM batch | DOM interleaved |
|---|---|---|---|---|
| Chrome | 29.80ms | **0.03ms** | 4.00ms | 41.80ms |
| Firefox | 15.00ms | **0.06ms** | 11.00ms | 79.00ms |
| Safari | 16.00ms | **0.04ms** | 80.00ms | 142.50ms |

`prepare()` is the one-time setup cost when text first appears. `layout()` is the resize hot path and stays comfortably sub-0.1ms per 500-text batch on this workload. See [pages/benchmark-results.txt](pages/benchmark-results.txt) for the checked-in snapshot and [pages/benchmark.ts](pages/benchmark.ts) for the live benchmark harness.

## Accuracy

Tested across 4 fonts × 8 sizes × 8 widths × 30 i18n texts (7680 tests):

| Browser | Match rate | Tests | Remaining mismatches |
|---|---|---|---|
| Chrome | 100.00% | 7680 | None on the current browser sweep |
| Safari | 100.00% | 7680 | None on the current browser sweep |
| Firefox | 100.00% | 7680 | None on the current browser sweep |
Tested across 4 fonts (Helvetica Neue, Georgia, Verdana, Courier New) × 8 sizes × 8 widths × 30 i18n texts. The browser sweep is now clean on fresh runs in Chrome, Safari, and Firefox. See [RESEARCH.md](RESEARCH.md) for the exploration log and the remaining Gatsby canary edge cases.

## i18n

- **Line breaking**: `Intl.Segmenter` with `granularity: 'word'` handles CJK (per-character breaks), Thai, Arabic, and all scripts the browser supports.
- **Bidi**: Unicode Bidirectional Algorithm (UAX #9) for mixed LTR/RTL text. Pure LTR text fast-paths with zero overhead.
- **Shaping**: canvas `measureText()` uses the browser's font engine, so ligatures, kerning, and contextual forms (Arabic connected letters) are handled correctly.
- **Emoji**: auto-corrected. Chrome/Firefox canvas inflates emoji widths at small font sizes on macOS; the library detects and compensates automatically.

## Known limitations

- **CSS config**: targets a common app-text configuration (`white-space: normal`, `word-break: normal`, `overflow-wrap: break-word`, `line-break: auto`). Source newlines are treated as collapsible whitespace, not explicit `<br>`/paragraph breaks. Other configurations (`break-all`, `keep-all`, `strict`, `loose`, `anywhere`) are untested.
- **`line-height`**: the library does not infer CSS line height. Pass the exact value you render with into `layout()` / `layoutWithLines()`. `line-height: normal` differs across fonts and browsers.
- **`system-ui` font**: canvas and DOM resolve this CSS keyword to different font variants at certain sizes on macOS. Use a named font (Inter, Helvetica, Arial, etc.) for guaranteed accuracy. See [RESEARCH.md](RESEARCH.md#discovery-system-ui-font-resolution-mismatch).
- **Server-side**: not zero-config today. The current entry expects browser DOM/canvas APIs; server use would need a canvas-backed path. We still keep a HarfBuzz (WASM) backend around for headless probes and research.

## How it works

1. **Text analysis**: normalize collapsible whitespace, segment with `Intl.Segmenter('word')`, merge punctuation, and carry opening punctuation forward so browser break opportunities are modeled more closely.
2. **CJK splitting + kinsoku**: CJK word segments are re-split into individual graphemes, since CSS allows line breaks between any CJK characters. Kinsoku shori rules keep CJK punctuation (，。「」 etc.) attached to their adjacent characters so they can't be separated across line breaks.
3. **Measurement + caching**: each final segment is measured via canvas `measureText()` and cached in a per-font segment-metrics cache. Common words across texts share not just widths but lazily-derived segment data such as grapheme widths for breakable words. The cache has no eviction — it grows monotonically per font string. For a typical single-font comment feed this is a few KB; `clearCache()` exists for manual eviction if needed.
4. **Emoji correction**: canvas `measureText` inflates emoji widths on Chrome/Firefox at font sizes <24px on macOS. Auto-detected by measuring a reference emoji; correction subtracted per emoji grapheme. Safari is unaffected (correction = 0).
5. **Bidi classification**: characters are classified into bidi types and embedding levels are computed. Pure LTR text skips this entirely.
6. **Layout** (per resize): walk the cached widths, accumulate per line, break when exceeding `maxWidth`. Trailing whitespace hangs past the edge (CSS behavior). Non-space overflow (words, emoji, punctuation) triggers a line break. Segments wider than `maxWidth` are broken at grapheme boundaries.

## Research

See [RESEARCH.md](RESEARCH.md) for the full exploration log: every approach we tried, benchmarks, the system-ui font discovery, punctuation accumulation error analysis, emoji width tables, HarfBuzz RTL bug, server-side engine comparison, and what Sebastian already knew.

## Credits

Based on [Sebastian Markbage's text-layout](https://github.com/chenglou/text-layout) research prototype (2016). Sebastian's design — canvas `measureText` for shaping, bidi algorithm from pdf.js, streaming line breaking — informed the architecture. We added: two-phase caching (making resize O(n) arithmetic), `Intl.Segmenter` (replacing the `linebreak` npm dependency and non-standard `Intl.v8BreakIterator`), punctuation merging, CJK grapheme splitting, overflow-wrap support, and trailing whitespace handling.

## Development

```bash
bun install
bun start        # http://localhost:3000 — demo pages (clears stale :3000 listeners first)
bun run check    # typecheck + lint
bun test         # lightweight invariants against the shipped implementation
bun run accuracy-check         # Chrome browser sweep
bun run accuracy-check:safari  # Safari browser sweep
bun run accuracy-check:firefox # Firefox browser sweep
bun run benchmark-check        # Chrome benchmark snapshot (short corpus + long-form corpora)
```

Pages:
- `/demo.html` — visual demo placeholder (`TODO`)
- `/accuracy.html` — sweep across fonts, sizes, widths, i18n texts
- `/benchmark.html` — performance comparison
- `/bubbles.html` — bubble shrinkwrap demo
- `/emoji-test.html` — canvas vs DOM emoji width sweep

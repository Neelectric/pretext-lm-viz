# Corpus Status

Current sweep snapshot for the checked-in canaries.

This is the compact status page. Historical reasoning, failed experiments, and
why the numbers moved live in `RESEARCH.md`.

Conventions:
- "anchors" means `300 / 600 / 800` unless noted otherwise
- "sampled" usually means `--samples=9`
- "step=10" means `300..900`
- values are the last recorded results on this machine, not a claim of universal permanence

## Browser Regression Gate

| Sweep | Status |
|---|---|
| Official browser corpus (Chrome) | `7680/7680` |
| Official browser corpus (Safari) | `7680/7680` |
| Official browser corpus (Firefox) | `7680/7680` |

## Product-Shaped Canary

| Corpus | Chrome anchors | Chrome step=10 | Notes |
|---|---:|---:|---|
| `mixed-app-text` | exact at `300 / 600 / 800` | `60/61 exact` | remaining `710px` miss is SHY / extractor-sensitive |

## Long-Form Corpora

| Corpus | Language | Chrome anchors | Safari anchors | Chrome sampled | Chrome step=10 | Notes |
|---|---|---:|---:|---:|---:|---|
| `ko-unsu-joh-eun-nal` | Korean | exact | not recently rerun | n/a | `61/61 exact` | Korean coarse corpus is clean |
| `th-nithan-vetal-story-1` | Thai | exact at key sentinels after fixes | not recently rerun | n/a | `59/61 exact` | two remaining coarse one-line misses |
| `th-nithan-vetal-story-7` | Thai | exact | exact | `9/9 exact` | not fully rerun | second Thai canary stays healthy |
| `km-prachum-reuang-preng-khmer-volume-7-stories-1-10` | Khmer | exact | exact | `9/9 exact` | not fully rerun | full `step=10` is slower; sampled check is the preferred first pass |
| `my-cunning-heron-teacher` | Myanmar | exact | exact at anchors | `9/9 exact` | `56/61 exact` | real residual Myanmar canary; quote/follower and phrase-break classes remain |
| `my-bad-deeds-return-to-you-teacher` | Myanmar | exact | exact | `8/9 exact` | `57/61 exact` | healthier than the first Myanmar text, but still shows the same broad quote+follower class in Chrome |
| `hi-eidgah` | Hindi | exact | not recently rerun | n/a | `61/61 exact` | Hindi coarse corpus is clean |
| `ar-risalat-al-ghufran-part-1` | Arabic | exact | exact at key sentinels | n/a | `61/61 exact` | Arabic coarse corpus is clean; fine sweep still has a small positive one-line field |
| `ar-al-bukhala` | Arabic | exact | exact at anchors | not recently rerun | not fully rerun | large second Arabic canary; anchors are clean |
| `he-masaot-binyamin-metudela` | Hebrew | exact | exact | n/a | `61/61 exact` | Hebrew coarse corpus is clean |

## Fine-Sweep Notes

These are the main "harder than coarse step=10" canaries worth remembering.

| Corpus | Result | Notes |
|---|---:|---|
| `ar-risalat-al-ghufran-part-1` | `594/601 exact` | remaining misses are one-line positive edge-fit cases |
| `my-cunning-heron-teacher` | not fully mapped at `step=1` | current useful sentinels are the shared `350` and `690` classes |

## Font Matrix Notes

These are sampled, not exhaustive.

| Corpus | Status | Notes |
|---|---|---|
| `ko-unsu-joh-eun-nal` | clean on sampled matrix | `Apple SD Gothic Neo`, `AppleMyungjo` |
| `th-nithan-vetal-story-1` | clean on sampled matrix | `Thonburi`, `Ayuthaya` |
| `th-nithan-vetal-story-7` | clean on sampled matrix | `Thonburi`, `Ayuthaya` |
| `km-prachum-reuang-preng-khmer-volume-7-stories-1-10` | clean on sampled matrix | `Khmer Sangam MN`, `Khmer MN` |
| `hi-eidgah` | clean on sampled matrix | `Kohinoor Devanagari`, `Devanagari Sangam MN`, `ITF Devanagari` |
| `ar-risalat-al-ghufran-part-1` | clean on sampled matrix | `Geeza Pro`, `SF Arabic`, `Arial` |
| `he-masaot-binyamin-metudela` | clean on sampled matrix | `Times New Roman`, `SF Hebrew` |
| `my-cunning-heron-teacher` | clean on sampled matrix | `Myanmar MN`, `Myanmar Sangam MN`, `Noto Sans Myanmar` |
| `my-bad-deeds-return-to-you-teacher` | one sampled miss | `Myanmar Sangam MN` had `-32px` at `300px`; `Myanmar MN` and `Noto Sans Myanmar` stayed exact |

## Recompute

Useful commands:

```sh
bun run corpus-check --id=ko-unsu-joh-eun-nal 300 600 800
bun run corpus-sweep --id=my-cunning-heron-teacher --start=300 --end=900 --step=10
bun run corpus-sweep --id=my-bad-deeds-return-to-you-teacher --samples=9
bun run corpus-font-matrix --id=my-bad-deeds-return-to-you-teacher --samples=5
```

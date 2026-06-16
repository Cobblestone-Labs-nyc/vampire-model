# EVAL — Vampire Restyle Student vs. Live Teacher

> Fill from `python src/eval.py` output. Run on the GPU box against a held-out face set
> (`data/holdout/`) that was NOT used in training. Compare the new model against the
> current live Fal `ip-adapter-face-id` teacher.

## Acceptance criteria (config.yaml → acceptance)

| Metric | Target | Stage-1 (hosted) | Stage-2 (student) | Live Fal teacher |
|---|---|---|---|---|
| ArcFace cosine (input↔output identity) | ≥ 0.65 | _TBD_ | _TBD_ | _TBD (ref)_ |
| Landmark RMSE (px, input↔output) | ≤ 6.0 | _TBD_ | _TBD_ | _TBD (ref)_ |
| Latency p95 (s) | ≤ 1.0 (student) | _TBD_ | _TBD_ | ~11–90 |
| Model size (MB) | ≤ 15 (hard cap 30) | n/a (hosted) | _TBD_ | multi-GB |
| Style "looks vampiric" (human, /5) | ≥ 4.0 | _TBD_ | _TBD_ | _TBD (ref)_ |
| Failure rate (mouth-melt / wrong-person / artifact) | report | _TBD_ | _TBD_ | _TBD_ |

## Latency table (p50 / p95)

| Runtime | Stage-1 hosted | Stage-2 student |
|---|---|---|
| Target GPU (CUDA) | _TBD_ | _TBD_ |
| CPU | _TBD_ | _TBD_ |
| WebGPU / ORT-Web (in-browser) | n/a | _TBD_ |

## Before/after samples

20-image holdout triplets (input | new model | live Fal) written to `out/triplets/`.
Embed/attach 10–20 here for review. Flag any identity-broken or mouth-melt cases.

## Decisions log

- Deploy: **Both** (Stage-1 hosted ships first; Stage-2 student exported for browser).
- Architecture: **Option B** SD-Turbo + LoRA + InstantID (Stage-1) → distilled tiny U-Net (Stage-2).
- Scope: **single cathedral aristocratic vampire** look.
- Identity loss computed vs **INPUT** (not teacher target) — the lever for identity retention.
- Teacher target img2img strength: **0.45** (low, geometry-faithful).

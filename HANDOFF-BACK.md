# Handoff BACK to Pebble — Vampire Restyle Implementation Spec

**From:** Claude (with Christopher Strawley) · **Date:** 2026-06-15
**For:** Pebble (owns the live site `https://christopher.cobblestonelabs.ai/vampire/`)

This is the drop-in spec the original handoff §8 asked for. It covers the six required
items. The trigger → align → hold → fade → cooldown flow is **untouched**; only the
generator changes.

> Status: the training repo (`~/Documents/vampire model`) is complete and runnable, but
> **training has not been executed** (needs CUDA GPU + face dataset + Fal budget). Numbers
> marked _TBD_ get filled by running `src/eval.py`. Wire Stage-1 first; Stage-2 is the
> later in-browser export.

---

## 1. The model artifact + how to run it

### Stage 1 — hosted (ship this first) — Deploy B
- **Artifact:** SDXL-Turbo + trained vampire **LoRA** (`checkpoints/stage1_lora/`, a few MB)
  + **InstantID** identity lock, served by `src/infer_server.py` (FastAPI/uvicorn).
- **Runtime:** one CUDA GPU (the LoRA + InstantID stack is the "trimmed" few-step model,
  ~4 inference steps vs the current 30 — much faster/cheaper than full SDXL, identity held).
- **Request/response schema** (identical contract to today):
  ```
  POST /api/restyle
    { "image": "<data URL or http(s) URL>", "strength": 0.30..0.85 }
  200
    { "image": { "b64": "data:image/png;base64,..." },   // or { "url": ... }
      "engine": "vampire-turbo-instantid", "identity": true, "ms": <int> }
  500  { "error": "...", "ms": <int> }   // caller falls back, page re-arms
  ```
- **Expected latency:** few-step turbo target p95 ≤ ~2s on GPU (vs 11–90s live). Confirm via eval.
- **Slots into the ENGINES array** as PRIMARY — see `engines/engine_entry.js`
  (name/url/buildBody/readImage/logFields ready to paste). Set `STAGE1_URL` in `.env`.

### Stage 2 — in-browser (later) — Deploy A
- **Artifact:** `web/model/vampire_student_int8.onnx` (target ≤ 15 MB) — a tiny U-Net
  distilled from Stage 1, one forward pass.
- **Pre-processing:** RGB, [0,1], NCHW, square @ 512 (or 384). **Post:** sigmoid [0,1] → canvas.
- **JS inference snippet:** `web/infer.js` — `createVampireRestyler(modelUrl, res)` returns a
  `restyle(snapshot, strength01)` that **replaces** the `fetch('/api/restyle')` call. Runs on
  ONNX Runtime Web (WebGPU, WASM fallback), on-device, no API hop, no per-gen cost.

---

## 2. Exact I/O contract
- **Input:** one webcam still — data URL or image URL. Expected ~512×512 (square; page
  already downscales/compresses before POST). Aspect: square center; the page's existing
  align step uses the MediaPipe face box, unchanged.
- **Output:** one restyled still (PNG; same resolution as input request).
- **Anything beyond the snapshot?** No. Stage-1 InstantID derives the face embedding from the
  snapshot itself. The MediaPipe face box is **not** required by the generator (the page still
  uses it for overlay alignment as today). If you'd rather pass a precomputed embedding to save
  time, the server can accept it — not needed for correctness.

## 3. Identity/quality knobs
- The page's **0.30–0.85 "subtle ↔ dramatic" slider** maps straight through as `strength`.
  The server converts it to turbo's lower ranges (`config.yaml: stage1.slider_to_strength`
  [0.20–0.45], `slider_to_guidance` [1.0–2.5]). Identity is held by InstantID regardless of
  the knob (same principle as today's embedding-held identity).
- **Safe range:** keep the slider within 0.30–0.85 as today. No mouth-melt cliff like SDXL's
  >8.5 guidance, but cap exposed range at the configured ends.

## 4. Performance + size numbers
- Fill from `src/eval.py` / `src/export_onnx.py` and `EVAL.md`. Required: model size (MB),
  measured p50/p95 latency (GPU/CPU/WebGPU), ArcFace cosine identity retention (accept ≥ 0.65),
  landmark RMSE (≤ 6 px), and failure rate vs the live teacher on a 20-image holdout.

## 5. Rollout instructions
- **Ship as PRIMARY with `ip-adapter-face-id` as FALLBACK.** Keep the existing error-path
  reset so the page never locks in "generating" (Stage-1 returns 500 → fall through to Fal).
- **Log to `restyle.log`** (existing JSON-lines): `engine` (`vampire-turbo-instantid`),
  `identity` (true), `ms`, plus the existing strength/guidance/imgBytes fields.
- **Infra changes:**
  - `.env`: add `STAGE1_URL` (the GPU endpoint). Secrets stay in `.env`.
  - `server.js`: append the engine entry, `pm2 restart christopher-checklist && pm2 save`.
  - nginx: existing `proxy_read_timeout 150s` / `client_max_body_size 15m` already cover it;
    turbo is faster, so no increase needed. `nginx -t` before any reload; don't disturb other
    sites on the shared droplet.
  - **Back up** `server.js` to a timestamped `.bak` before editing (existing convention).
- **Stage-2 (browser):** serve `web/model/*.onnx` + `web/infer.js` statically under
  `/vampire/`; swap the `fetch('/api/restyle')` for the local `restyle()` call behind a
  WebGPU-capability check, with `/api/restyle` (Stage-1) as the automatic fallback.

## 6. File manifest (what Pebble places + where)

| File (this repo) | Goes to | Purpose |
|---|---|---|
| `src/infer_server.py` + `checkpoints/stage1_lora/` + InstantID weights | GPU box | Stage-1 endpoint |
| `engines/engine_entry.js` (merge into `ENGINES`) | `/var/www/christopher/server.js` | wire PRIMARY/FALLBACK |
| `.env` key `STAGE1_URL` | `/var/www/christopher/.env` | endpoint URL |
| `web/model/vampire_student_int8.onnx` | `/var/www/christopher/public/vampire/model/` | Stage-2 in-browser model |
| `web/infer.js` | `/var/www/christopher/public/vampire/` | Stage-2 local inference |
| `EVAL.md` | with the PR | results / sign-off |

**New dependencies:** Stage-1 server box — `torch`, `diffusers`, `peft`, `insightface`,
InstantID weights (see `requirements.txt`). Front-end Stage-2 — `onnxruntime-web` via CDN
(no build step). Live `server.js` needs no new npm deps (just an HTTP call to `STAGE1_URL`).

---

### One honest caveat to carry forward
"Both, one artifact" is a **two-stage lineage**, not a single file: Option B (SDXL-Turbo +
LoRA + InstantID) is a diffusion model and cannot itself be a ≤15 MB browser bundle. Stage 1
is the real, shippable speed/cost win now; the ≤15 MB on-device version is the Stage-2 student
distilled from it. Ship Stage 1, validate identity/latency, then export Stage 2.

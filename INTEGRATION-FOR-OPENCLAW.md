# Integration Checklist — Vampire Model → Attention-Trigger Site

**For:** openclaw (implementing agent on the live site)
**From:** Claude (with Christopher Strawley) · **Date:** 2026-06-16
**Live page:** `https://christopher.cobblestonelabs.ai/vampire/`
**Repo this references:** `~/Documents/vampire model/` (read `HANDOFF-BACK.md` for the full spec)

This is the operator's checklist for wiring the trimmed vampire generator into the existing
attention-trigger experience. The trigger → snapshot → align → hold → fade → cooldown flow
stays **untouched**; you are only swapping the *generator* behind `/api/restyle`.

---

## ⚠️ READ FIRST — current status (what is and isn't real today)

- **There is NO trained model artifact yet.** The repo is a complete, runnable training
  pipeline, but training has **not been executed** (needs a CUDA GPU + a licensed face
  dataset + Fal budget for teacher targets). So there are **no weights to deploy right now.**
- Therefore "implement" splits into two tracks:
  - **Track A — do now (safe, no model needed):** stand up the plumbing so the new engine
    can go live the instant weights exist, *without changing current behavior*. Because it
    ships behind the existing fallback, the site keeps working exactly as today.
  - **Track B — gated on training:** run the GPU training, validate identity/latency, then
    flip the new engine to PRIMARY.
- **Do not tell Chris the new model is live until Track B is validated.** Until then the page
  is still served by the current `ip-adapter-face-id` Fal call.

---

## The contract (this is the whole reason it drops in cleanly)

Keep this exact I/O and nothing else changes:

```
POST /api/restyle   { "image": <data URL | image URL>, "strength": 0.30..0.85 }
200                 { "image": { "b64" | "url" }, "engine": "...", "identity": true, "ms": <int> }
500                 { "error": "...", "ms": <int> }   // page falls back + re-arms cleanly
```

Input = one webcam still (page already downscales/compresses). Output = one restyled still.
The generator needs **nothing beyond the snapshot** — InstantID derives the face embedding
from it. The MediaPipe face box stays a front-end-only concern (overlay alignment), unchanged.

---

## Two-stage plan (why "both, one artifact" = a pipeline, not a file)

Per Chris's decisions: architecture **B (SDXL-Turbo + LoRA + InstantID)**, **single cathedral
look**, deploy to **both** hosted and browser. Option B is a diffusion model, so it **cannot**
be a ≤15 MB in-browser file. So:

- **Stage 1 (hosted, ships first):** SDXL-Turbo + vampire LoRA + InstantID behind a GPU
  endpoint → new entry in the `ENGINES` array. The real speed/cost win over full SDXL.
- **Stage 2 (browser, later):** distill Stage 1 into a tiny U-Net student → ONNX/WebGPU in
  `index.html`, replacing the `fetch('/api/restyle')` with on-device inference.

Ship Stage 1, validate, then export Stage 2. Don't block Stage 1 on Stage 2.

---

## Live-site facts (respect these — shared droplet)

| Thing | Value |
|---|---|
| Droplet | `<DROPLET_IP>` — SSH access via the team's key (ask the maintainer; not committed) |
| PM2 process | `christopher-checklist` (runs as root) |
| Server file | `/var/www/christopher/server.js` — route `POST /api/restyle` (`express.json limit 15mb`) |
| Front-end | `/var/www/christopher/public/vampire/index.html` (**static** — edits = browser reload, no restart) |
| Secrets | `/var/www/christopher/.env` (`FAL_KEY`/`FAL_API_KEY`, `OPENAI_API_KEY`) — never hardcode |
| Log | `/var/www/christopher/data/restyle.log` (JSON lines: outcome, engine, ms, strength, guidance, identity, imgBytes) |
| nginx | `proxy_read_timeout 150s`, `client_max_body_size 15m` (christopher block only) |

**Rules:** back up any edited file to a timestamped `.bak` first. `server.js` edits →
`pm2 restart christopher-checklist && pm2 save`. `nginx -t` before any `systemctl reload
nginx`. **Never disrupt the other sites on this droplet.** Identity is the headline feature —
any size/speed win that loses identity lock is a regression.

---

## Track A — wire the plumbing now (no model required)

1. **Add the new engine as a NON-primary entry first.** Copy `engines/engine_entry.js` from
   the repo into the `ENGINES` array in `server.js`. Keep `ip-adapter-face-id` as PRIMARY for
   now; add `vampire-turbo-instantid` as a **disabled/secondary** entry so the live path is
   unchanged. Back up `server.js` → `.bak` first.
2. **Add `STAGE1_URL` to `.env`** (the future GPU endpoint; can be a placeholder until the
   endpoint exists). Secrets stay in `.env`.
3. **Confirm the error/fallback path is intact:** a 500 from the new engine must fall through
   to `ip-adapter-face-id` and the page must re-arm (never lock in "generating"). Test by
   pointing `STAGE1_URL` at an unreachable host and confirming the page still works.
4. **Logging:** ensure `restyle.log` will capture `engine`, `identity`, `ms` for the new
   engine (the entry's `logFields` already returns these).
5. `pm2 restart christopher-checklist && pm2 save`. **Behavior is unchanged** — this is just
   pre-wiring. Nothing user-visible happens yet.

## Track B — once the model is trained (the actual swap)

1. **Train + validate** (on the GPU box, per repo README §Quickstart): `teacher_generate.py`
   → `train_lora.py` → `infer_server.py`. Then `eval.py` must **PASS** the acceptance bars in
   `config.yaml`: ArcFace cosine ≥ 0.65, landmark RMSE ≤ 6 px, latency p95 ≤ target. Fill
   `EVAL.md` with the before/after vs the current Fal teacher.
2. **Stand up the Stage-1 endpoint** (`src/infer_server.py` on the GPU box) and point
   `STAGE1_URL` at it. Hit `/health` and a test `/api/restyle` from the droplet.
3. **Flip to PRIMARY:** reorder `ENGINES` so `vampire-turbo-instantid` is first and
   `ip-adapter-face-id` is the FALLBACK. `pm2 restart christopher-checklist && pm2 save`.
4. **Watch `restyle.log`** for `engine=vampire-turbo-instantid`, `identity=true`, and `ms`.
   If failure rate or identity regresses, reorder back to Fal-primary (instant rollback) and
   report.
5. **Stage 2 (browser), later:** serve `web/model/*.onnx` + `web/infer.js` under `/vampire/`,
   gate on a WebGPU capability check, replace the `fetch('/api/restyle')` with the local
   `restyle()` call, and keep `/api/restyle` (Stage 1) as the automatic fallback for browsers
   without WebGPU.

---

## Knobs to preserve
- The page's **0.30–0.85 "subtle ↔ dramatic" slider** maps straight through as `strength`;
  the Stage-1 server converts it to turbo's lower ranges (`config.yaml`). Identity is held by
  InstantID regardless of the slider. Keep the exposed range at 0.30–0.85.

## File manifest (what goes where) — see `HANDOFF-BACK.md §6` for the full table
- `engines/engine_entry.js` → merge into `server.js` `ENGINES`
- `STAGE1_URL` → `/var/www/christopher/.env`
- `src/infer_server.py` + `checkpoints/stage1_lora/` + InstantID weights → GPU box
- (Stage 2) `web/model/*.onnx` + `web/infer.js` → `public/vampire/`

## Definition of done
- Track A: new engine wired behind fallback, site behavior unchanged, rollback trivial.
- Track B: `eval.py` passes acceptance bars; new engine PRIMARY; `restyle.log` shows it
  serving with `identity=true`; one-step rollback to Fal verified; `EVAL.md` filled.

> If anything here conflicts with what you find live on the droplet, **stop and surface it** —
> don't force the swap. The current Fal pipeline is the safe default at all times.

# Vampire Restyle — Fast Identity-Preserving Model

Trimmed, **vampire-only** generator that drops into the existing attention-trigger site
(`https://christopher.cobblestonelabs.ai/vampire/`) in place of the full hosted SDXL +
IP-Adapter-Face-ID call, keeping the exact **one snapshot in → one restyled still out**
contract.

> Built from `vampire-attention-handoff-for-claude.md` + `vampire-model-training-spec.md`.
> The final output of this repo is an **implementation spec + artifacts for Pebble** to
> wire onto the live site — see `HANDOFF-BACK.md`.

---

## Decisions locked (Chris, 2026-06-15)

| Question | Choice | Consequence |
|---|---|---|
| Deploy target | **Both, one artifact** | Two-stage: ship hosted first, export to browser second (see below). |
| Architecture | **B: SD-Turbo + LoRA + InstantID** | Few-step diffusion, identity locked by InstantID. The Stage-1 teacher/server. |
| Scope | **Single cathedral look** | One prompt, one LoRA. Smallest, fastest to train. |
| Train env | **Remote GPU / cloud** | Scripts are CUDA-first. This Mac (no CUDA) is for editing/orchestration only. |

### The "Both, one artifact" reality (read this)

Option B is a **diffusion model**. It is fast and cheap *relative to full SDXL*, but it is
**not** a 10–30 MB thing you can run in a browser. You cannot have a single SD-Turbo file
that is also an on-device WebGPU bundle. So "Both" is delivered as a **two-stage pipeline
with one lineage**:

```
                 current Fal teacher (ip-adapter-face-id, full SDXL)
                              │  teacher_generate.py
                              ▼
        Stage 1:  SDXL-Turbo + vampire LoRA + InstantID   ───► HOSTED endpoint (Deploy B)
                  (train_lora.py → infer_server.py)             slots into ENGINES array
                              │  it becomes the new teacher
                              │  teacher_generate.py --teacher local
                              ▼
        Stage 2:  tiny U-Net student (~10–15 MB, distilled) ─► ONNX/WebGPU (Deploy A)
                  (train_student.py → export_onnx.py)           runs in index.html, no API
```

- **Stage 1 is the near-term shippable.** It's the honest "trim the fat" win: a few-step,
  vampire-only, identity-locked endpoint that replaces full 30-step SDXL — much faster +
  cheaper, identity held by InstantID. Ships behind the existing fallback ladder.
- **Stage 2 is the dramatic version.** Distill Stage 1 into a tiny one-forward-pass student
  for true on-device, sub-second, zero-API generation. Higher risk; export, don't block
  Stage 1 on it.

You can stop after Stage 1 and already have a big speed/cost win on the live site.

---

## Repo layout

```
vampire model/
├── README.md                 ← you are here
├── HANDOFF-BACK.md           ← the 6-item spec to hand to Pebble (the actual deliverable)
├── EVAL.md                   ← results template (fill from eval.py output)
├── requirements.txt
├── .env.example              ← copy to .env, add FAL_KEY (never commit .env)
├── config.yaml               ← all knobs: prompt, sizes, loss weights, paths
├── src/
│   ├── prompts.py            ← exact vampire prompt + negative (from live config)
│   ├── teacher_generate.py   ← build (face → vampire) pairs from a teacher (Fal or local)
│   ├── train_lora.py         ← Stage 1: SDXL-Turbo + vampire LoRA training
│   ├── losses.py             ← ArcFace identity + landmark + LPIPS + L1
│   ├── train_student.py      ← Stage 2: distill Stage-1 model → tiny U-Net student
│   ├── export_onnx.py        ← export student to ONNX + INT8/FP16 quantize + size report
│   ├── infer_server.py       ← /api/restyle-compatible FastAPI server (Stage-1 hosted)
│   └── eval.py               ← ArcFace cosine, landmark RMSE, latency p50/p95, size
├── engines/
│   └── engine_entry.js       ← drop-in entry for the ENGINES array in /api/restyle
├── web/
│   └── infer.js              ← ONNX Runtime Web / WebGPU snippet (Stage-2 in-browser)
└── data/                     ← (gitignored) faces/, targets/, pairs.jsonl
```

---

## Quickstart (on the GPU box)

```bash
# 0. env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit: FAL_KEY=...

# 1. Build the distillation dataset from the CURRENT live teacher (incurs Fal cost).
#    Put source faces in data/faces/ first (see "Data" below).
python src/teacher_generate.py --faces data/faces --out data/targets --pairs data/pairs.jsonl

# 2. Stage 1 — train the vampire LoRA on SDXL-Turbo (CUDA).
python src/train_lora.py --config config.yaml

# 3. Serve Stage 1 as a /api/restyle-compatible endpoint.
python src/infer_server.py --config config.yaml --port 8008
#    -> POST /api/restyle  { image, strength }  -> { image: { url|b64 } }

# 4. (Stage 2, optional) distill Stage 1 into a tiny browser student, then export.
python src/train_student.py  --config config.yaml --teacher stage1
python src/export_onnx.py    --config config.yaml --quantize int8
#    -> web/model/vampire_student.onnx  (+ size + self-bench printed)

# 5. Evaluate either model vs the live Fal teacher on a holdout set.
python src/eval.py --config config.yaml --model stage1   # or --model student
```

## Data

- **Source faces (X):** a few thousand diverse, permissively-licensed, webcam-style
  head-and-shoulders portraits (varied gender/age/skin tone/pose/framing; include glasses,
  hats, off-center). FFHQ-style or CC-licensed portrait sets. **Verify licensing** and any
  consent constraint before generating production weights (`config.yaml: data.license_note`).
- **Targets (Y):** generated by `teacher_generate.py` — each X run through the teacher with
  the exact live prompt at low img2img strength so the *teacher's* outputs already preserve
  identity. Bad/identity-broken pairs are auto-flagged (ArcFace cosine < threshold) for cull.
- Start ~2–5k pairs. Augment: flips, mild color jitter, varied crops/zoom.

## Honest status / what is NOT done here

- **No training has been run.** This repo is the complete, runnable pipeline. Actual
  distillation needs a CUDA GPU, the face dataset, and Fal budget for teacher targets —
  run it on the cloud box per Quickstart.
- Identity is the headline feature. The acceptance bar (`config.yaml`) is **ArcFace cosine
  ≥ 0.65** input↔output and a landmark RMSE tolerance — do not ship a smaller/faster model
  that fails these. `eval.py` enforces them.
- Secrets live in `.env` only. Never hardcode `FAL_KEY` / `OPENAI_API_KEY`.

See `HANDOFF-BACK.md` for exactly what to give Pebble.

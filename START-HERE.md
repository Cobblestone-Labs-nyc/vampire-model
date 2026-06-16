# START HERE — Vampire Model Training Handoff

**For:** openclaw · **From:** Claude (with Christopher Strawley) · **Date:** 2026-06-16

This package trains the **vampire restyle model** for the attention-trigger site
(`https://christopher.cobblestonelabs.ai/vampire/`). **Your job in this handoff is to TRAIN
the model.** Deployment happens after, and is documented separately (see below).

---

## ⚠️ Read these two things first

1. **No trained weights exist yet** — this is a complete, runnable pipeline, not a finished
   model. Training has not been run. That's the task.
2. **LICENSE GUARDRAIL:** the default dataset is **FFHQ, which is NON-COMMERCIAL / research
   only.** Weights trained on it are a **PROOF-OF-CONCEPT** to validate the pipeline + quality
   — they **MUST NOT ship on the live client site.** Production weights must be re-trained on
   a **cleared/consented** face set (`src/fetch_ffhq.py --from-dir`). This is non-negotiable
   for client work. The Colab notebook repeats this at the start and end.

---

## What to do (the path)

1. **Train on Colab** — open **`train_vampire_colab.ipynb`** in Google Colab (GPU runtime).
   It walks you through: get repo → `FAL_KEY` → fetch FFHQ → **smoke test** → full run → eval
   → download artifacts. See **`COLAB.md`** for the quickstart.
2. **Smoke test is mandatory before the full run.** `src/smoke_test.py` exercises the entire
   pipeline on ~10 images for a couple dollars of Fal spend and reports PASS/FAIL per stage.
   Do not launch the full 2–5k run until it ends in ✅ — it's the cheap insurance against
   burning GPU/Fal budget on a misconfig.
3. **Validate** against the acceptance bars in `config.yaml` (ArcFace cosine ≥ 0.65, landmark
   RMSE ≤ 6px, latency, size). Fill **`EVAL.md`** with results vs the current Fal teacher.
4. **For production:** re-run the pipeline with `--from-dir <cleared_faces>` and re-validate.

## Three inputs you must supply
- **GPU:** Colab GPU runtime (T4 ok for smoke test; A100/L4 for full run).
- **`FAL_KEY`:** to call the teacher (`ip-adapter-face-id`) for target generation. Incurs Fal
  cost (order tens–low-hundreds of dollars for the full dataset; verify current pricing).
- **Dataset:** FFHQ for POC (built in); a cleared/consented set for production.

## Known gaps the smoke test will surface (not silent failures)
- **InstantID** identity-lock weights — `infer_server.py` boots without them so the HTTP
  contract works, but identity quality needs them wired (README §InstantID).
- **ArcFace + landmark torch backbones** for `losses.py` / `eval.py` (README §losses). The
  smoke test fails step 4 cleanly with instructions if these are missing.

> If you want these auto-downloaded so the smoke test passes end-to-end without manual setup,
> ask Chris — Claude offered to add a weights-fetch cell + helper.

---

## File map

| File | Purpose |
|---|---|
| **`train_vampire_colab.ipynb`** | the turnkey training notebook — start here |
| **`COLAB.md`** | Colab quickstart |
| `README.md` | full repo overview, decisions, two-stage plan, repro commands |
| `config.yaml` | all knobs: prompt, sizes, loss weights, acceptance bars |
| `src/fetch_ffhq.py` | dataset fetch (`--hf-repo` POC / `--from-dir` production) |
| `src/smoke_test.py` | end-to-end plumbing test on ~10 images |
| `src/teacher_generate.py` | build (face → vampire) pairs from the Fal teacher |
| `src/train_lora.py` | Stage-1: SDXL-Turbo + vampire LoRA training |
| `src/train_student.py` + `src/losses.py` | Stage-2: distill to tiny browser U-Net |
| `src/export_onnx.py` | export Stage-2 student to ONNX + quantize |
| `src/infer_server.py` | `/api/restyle`-compatible serving endpoint |
| `src/eval.py` | acceptance metrics vs the live teacher |
| `engines/engine_entry.js`, `web/infer.js` | deployment wiring (used later, not for training) |
| **`INTEGRATION-FOR-OPENCLAW.md`** | **deployment** checklist — do this AFTER training passes |
| `HANDOFF-BACK.md` | full implementation spec (the six deliverables for the live site) |
| `.env.example` | copy to `.env`, add `FAL_KEY` (never commit `.env`) |

## Definition of done (training)
Smoke test ✅ → full run complete → `eval.py` PASSES the acceptance bars → `EVAL.md` filled
with before/after vs the current Fal teacher → artifacts (LoRA + eval) saved. Then move to
`INTEGRATION-FOR-OPENCLAW.md` for the live-site swap (Track B), keeping `ip-adapter-face-id`
as the fallback throughout.

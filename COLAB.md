# Colab Training — Quickstart (Proof-of-Concept)

Runs the Stage-1 vampire LoRA training on a Colab GPU, distilled from the live Fal teacher.

> ## ⚠️ FFHQ = NON-COMMERCIAL. Proof-of-concept only.
> The weights this produces validate the pipeline + quality but **cannot ship on the live
> client site**. Re-train production weights on a cleared/consented set (`--from-dir`) first.

## Steps
1. Open **`train_vampire_colab.ipynb`** in Colab (upload it, or File → Open → GitHub).
2. Runtime → Change runtime type → **GPU**.
3. Get the repo into Colab — set `REPO_URL` (push `~/Documents/vampire model` to git) or use
   the upload-zip path in the notebook.
4. Paste your `FAL_KEY` when prompted (needed for teacher target generation; incurs Fal cost).
5. **Run the smoke test (step 5).** It exercises the whole pipeline on ~10 images for a couple
   dollars. Must end in ✅ before you proceed.
6. Only then run the full dataset + train + eval (step 6).
7. Download the artifacts zip (step 7).

## What needs verifying
- **HF dataset repo** (`HF_REPO`): IDs move on Hugging Face — confirm it exists, swap if it
  404s. Use a 256px+ mirror for the full run; 64px is fine only for the smoke test.
- **InstantID weights:** `infer_server.py` boots without them (HTTP contract works for the
  smoke test), but identity-lock quality needs them wired — see repo README §losses / InstantID.
- **ArcFace/landmark backbones** for `losses.py` / `eval.py` — see README; the smoke test will
  FAIL step 4 cleanly if they're missing, telling you what to add.

## Then hand to openclaw
Once you have production weights that pass the acceptance bars, follow
`INTEGRATION-FOR-OPENCLAW.md` (Track B) to deploy: stand up `src/infer_server.py`, point
`STAGE1_URL` at it, and flip the `ENGINES` entry to PRIMARY with `ip-adapter-face-id` as fallback.

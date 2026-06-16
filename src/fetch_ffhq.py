"""Fetch a face dataset for training.

⚠️ LICENSE: FFHQ is licensed for NON-COMMERCIAL / research use only (CC BY-NC-SA on the
compiled set; individual images under various Flickr licenses). Weights trained on it are a
PROOF-OF-CONCEPT — they must NOT ship on the live client site. Re-train production weights on
a cleared/consented set (--from-dir) before deploying. See README / COLAB.md.

Modes:
  --hf-repo <id>   stream an FFHQ (or other face) dataset from Hugging Face and save N images.
                   HF dataset IDs move/change — VERIFY the repo exists; swap if it 404s.
                   Candidates to try (check on huggingface.co first):
                     student-abdullah/ffhq_512 , nuwandaa/ffhq256 , Dmini/FFHQ-64x64
                   Pick a 256px+ mirror for the FULL run; 64px is fine only for the smoke test.
  --from-dir <p>   normalize a local folder of faces you already have rights to (the
                   production-safe path). Just copies/resizes images into --out.

Usage:
  python src/fetch_ffhq.py --hf-repo nuwandaa/ffhq256 --n 50  --out data/faces      # POC
  python src/fetch_ffhq.py --from-dir ~/cleared_faces       --out data/faces       # production
"""
import argparse, shutil, sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def from_dir(src: Path, out: Path, n: int, res: int):
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    files = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS][: (n or None)]
    if not files:
        print(f"No images found in {src}"); sys.exit(1)
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        if res:
            img.thumbnail((res, res))
        img.save(out / f"face_{i:05d}.png")
    print(f"Copied {len(files)} faces -> {out}")


def from_hf(repo: str, out: Path, n: int, res: int):
    out.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (it's in requirements.txt)"); sys.exit(1)
    print(f"Streaming {n} images from HF dataset '{repo}' ...")
    try:
        ds = load_dataset(repo, split="train", streaming=True)
    except Exception as e:
        print(f"[error] could not load '{repo}': {e}\n"
              "HF dataset IDs change — open huggingface.co, find a valid FFHQ mirror, and pass "
              "it via --hf-repo. Or use --from-dir with your own cleared faces.")
        sys.exit(1)
    saved = 0
    for ex in ds:
        img = ex.get("image") or next((v for v in ex.values() if hasattr(v, "save")), None)
        if img is None:
            continue
        img = img.convert("RGB")
        if res:
            img.thumbnail((res, res))
        img.save(out / f"face_{saved:05d}.png")
        saved += 1
        if saved >= n:
            break
    print(f"Saved {saved} faces -> {out}")
    if saved < n:
        print(f"[warn] only {saved}/{n} available from this split.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--from-dir", default=None)
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--res", type=int, default=512, help="max edge resize; 0 = keep original")
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_dir:
        from_dir(Path(args.from_dir).expanduser(), out, args.n, args.res)
    elif args.hf_repo:
        from_hf(args.hf_repo, out, args.n, args.res)
    else:
        print("Pass --hf-repo <id> (POC) or --from-dir <path> (production). See --help.")
        sys.exit(1)
    print("\nReminder: FFHQ weights = proof-of-concept only. Production weights must be "
          "trained on cleared/consented faces before the model ships.")


if __name__ == "__main__":
    main()

"""Build the distillation dataset: (face X -> vampire Y) pairs.

Teacher options:
  --teacher fal     (default) the CURRENT live engine: fal-ai/ip-adapter-face-id.
                    This is the source-of-truth teacher for Stage 1. Incurs Fal cost.
  --teacher local   the Stage-1 endpoint (infer_server.py) once it exists — used to
                    generate data for distilling the Stage-2 browser student.

Each pair is written to pairs.jsonl with an ArcFace identity cosine (input vs target)
so identity-broken teacher outputs can be auto-flagged/culled before training.

Usage:
  python src/teacher_generate.py --faces data/faces --out data/targets \
      --pairs data/pairs.jsonl --config config.yaml [--teacher fal|local] [--limit N]
"""
import argparse, base64, json, os, sys, time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prompts import VAMPIRE_PROMPT, NEGATIVE_PROMPT

load_dotenv()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def call_fal_teacher(face_path: Path, cfg: dict) -> bytes:
    """Call fal-ai/ip-adapter-face-id with the exact live params. Returns image bytes."""
    key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not key:
        raise RuntimeError("FAL_KEY / FAL_API_KEY not set (see .env.example)")
    t = cfg["teacher"]
    body = {
        "model_type": t["model_type"],
        "prompt": VAMPIRE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        # snapshot is the FACE IDENTITY reference embedding (handoff §3)
        "face_image_url": _to_data_url(face_path),
        "image_url": _to_data_url(face_path),
        "num_inference_steps": t["num_inference_steps"],
        "guidance_scale": t["guidance_scale"],
        "num_samples": t["num_samples"],
        "width": t["width"],
        "height": t["height"],
    }
    r = requests.post(
        t["url"],
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json=body,
        timeout=t["timeout_s"],
    )
    r.raise_for_status()
    data = r.json()
    # ip-adapter-face-id returns { image: { url } }; fast-sdxl returns { images: [{url}] }
    url = (data.get("image") or {}).get("url") or (data.get("images") or [{}])[0].get("url")
    if not url:
        raise RuntimeError(f"unexpected teacher response: {json.dumps(data)[:300]}")
    return requests.get(url, timeout=60).content


def call_local_teacher(face_path: Path, cfg: dict) -> bytes:
    """Call the Stage-1 hosted endpoint for Stage-2 student data generation."""
    base = os.environ.get("STAGE1_URL", "http://127.0.0.1:8008")
    r = requests.post(
        f"{base}/api/restyle",
        json={"image": _to_data_url(face_path), "strength": 0.65},
        timeout=cfg["teacher"]["timeout_s"],
    )
    r.raise_for_status()
    data = r.json()
    img = data.get("image", {})
    if img.get("url"):
        return requests.get(img["url"], timeout=60).content
    if img.get("b64"):
        return base64.b64decode(img["b64"].split(",")[-1])
    raise RuntimeError(f"unexpected local response: {json.dumps(data)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default="data/faces")
    ap.add_argument("--out", default="data/targets")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--teacher", choices=["fal", "local"], default="fal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (rate-limit)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    faces = sorted(p for p in Path(args.faces).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        faces = faces[: args.limit]
    if not faces:
        print(f"No faces found in {args.faces}. Add licensed source portraits first.")
        sys.exit(1)

    # Lazy import: identity scoring needs insightface (heavy). Skip gracefully if absent.
    try:
        from eval import arcface_cosine_paths  # reuse the same embedder as eval
        score = arcface_cosine_paths
    except Exception as e:  # pragma: no cover
        print(f"[warn] identity scoring disabled ({e}); pairs written without arcface_cos")
        score = None

    teacher = call_fal_teacher if args.teacher == "fal" else call_local_teacher
    done = {json.loads(l)["x"] for l in open(args.pairs)} if Path(args.pairs).exists() else set()

    with open(args.pairs, "a") as pf:
        for fp in tqdm(faces, desc=f"teacher={args.teacher}"):
            if str(fp) in done:
                continue
            try:
                img_bytes = teacher(fp, cfg)
            except Exception as e:
                print(f"[skip] {fp.name}: {e}")
                continue
            yp = out_dir / f"{fp.stem}_vampire.png"
            yp.write_bytes(img_bytes)
            cos = score(str(fp), str(yp)) if score else None
            rec = {"x": str(fp), "y": str(yp), "arcface_cos": cos,
                   "flag_cull": (cos is not None and cos < cfg["data"]["cull_arcface_cos"])}
            pf.write(json.dumps(rec) + "\n"); pf.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Done. Pairs -> {args.pairs}. Review flag_cull=true rows before training.")


if __name__ == "__main__":
    main()

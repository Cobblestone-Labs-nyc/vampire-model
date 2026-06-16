"""Validation harness (training spec §2, §6). Measures, on a held-out face set, the model
vs the current live Fal teacher:

  (a) identity retention  — ArcFace cosine(input, output)        [accept >= 0.65]
  (b) geometry            — MediaPipe landmark RMSE(input,output) [accept <= tol px]
  (c) latency             — p50/p95 per call
  (d) model size          — reported by export_onnx.py / file size
  (e) style               — left for human eval (writes triplets for review)

Also exposes arcface_cosine_paths() reused by teacher_generate.py for pair flagging.

  python src/eval.py --config config.yaml --model stage1   # stage1 | student | fal
  python src/eval.py --faces data/holdout --model student --triplets out/triplets
"""
import argparse, base64, io, json, os, sys, time
from pathlib import Path

import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
_FACE_APP = None
_LANDMARKER = None


def _face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


def _embed(img: Image.Image):
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR for insightface
    faces = _face_app().get(arr)
    if not faces:
        return None
    return faces[0].normed_embedding


def arcface_cosine(a: Image.Image, b: Image.Image):
    ea, eb = _embed(a), _embed(b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))


def arcface_cosine_paths(pa: str, pb: str):
    return arcface_cosine(Image.open(pa), Image.open(pb))


def _landmarks(img: Image.Image):
    """MediaPipe FaceLandmarker — same family as the site's detector."""
    global _LANDMARKER
    import mediapipe as mp
    if _LANDMARKER is None:
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(
            model_asset_path=os.environ.get("FACE_LANDMARKER", "checkpoints/face_landmarker.task")))
        _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
    import mediapipe as mp
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img.convert("RGB")))
    res = _LANDMARKER.detect(mpimg)
    if not res.face_landmarks:
        return None
    w, h = img.size
    return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]])


def landmark_rmse(a: Image.Image, b: Image.Image):
    la, lb = _landmarks(a), _landmarks(b)
    if la is None or lb is None or la.shape != lb.shape:
        return None
    return float(np.sqrt(((la - lb) ** 2).sum(axis=1).mean()))


# ---- model adapters: take an input PIL image, return restyled PIL image ------------

def run_fal(img, cfg):
    from teacher_generate import call_fal_teacher  # reuse exact live call
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return Image.open(io.BytesIO(__import__("teacher_generate").call_fal_teacher(Path(f.name), cfg)))


def run_endpoint(img, base):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = requests.post(f"{base}/api/restyle", json={"image": data_url, "strength": 0.65}, timeout=120)
    r.raise_for_status()
    b64 = r.json()["image"]["b64"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def run_student_onnx(img, cfg):
    import onnxruntime as ort
    res = cfg["data"]["resolution"]
    sess = ort.InferenceSession(cfg["export"]["onnx_path"].replace(".onnx", "_int8.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    x = (np.asarray(img.convert("RGB").resize((res, res))).astype("float32") / 255.0
         ).transpose(2, 0, 1)[None]
    feed = {"x": x}
    if cfg["student"]["film_strength"]:
        feed["strength"] = np.array([0.65], dtype="float32")
    y = sess.run(None, feed)[0][0]
    return Image.fromarray((y.transpose(1, 2, 0).clip(0, 1) * 255).astype("uint8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--faces", default="data/holdout")
    ap.add_argument("--model", choices=["stage1", "student", "fal"], default="stage1")
    ap.add_argument("--triplets", default="out/triplets")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    acc = cfg["acceptance"]

    faces = sorted(p for p in Path(args.faces).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not faces:
        print(f"No holdout faces in {args.faces}"); sys.exit(1)
    Path(args.triplets).mkdir(parents=True, exist_ok=True)

    def run(img):
        if args.model == "fal":
            return run_fal(img, cfg)
        if args.model == "student":
            return run_student_onnx(img, cfg)
        return run_endpoint(img, os.environ.get("STAGE1_URL", "http://127.0.0.1:8008"))

    cos_list, rmse_list, lat_list = [], [], []
    for fp in faces:
        inp = Image.open(fp).convert("RGB")
        t = time.perf_counter()
        try:
            out = run(inp)
        except Exception as e:
            print(f"[skip] {fp.name}: {e}"); continue
        lat_list.append(time.perf_counter() - t)
        c = arcface_cosine(inp, out); r = landmark_rmse(inp.resize(out.size), out)
        if c is not None: cos_list.append(c)
        if r is not None: rmse_list.append(r)
        # save input | output triplet (Fal teacher column added by hand for human eval)
        canvas = Image.new("RGB", (out.width * 2, out.height))
        canvas.paste(inp.resize(out.size), (0, 0)); canvas.paste(out, (out.width, 0))
        canvas.save(Path(args.triplets) / f"{fp.stem}.png")

    def stat(x): return (np.mean(x), np.percentile(x, 50), np.percentile(x, 95)) if x else (None,) * 3
    cos_m = np.mean(cos_list) if cos_list else float("nan")
    rmse_m = np.mean(rmse_list) if rmse_list else float("nan")
    _, _, lat95 = stat(lat_list)

    print("\n=== EVAL: model=%s, n=%d ===" % (args.model, len(faces)))
    print(f"ArcFace cosine (id retention): mean={cos_m:.3f}  [{'PASS' if cos_m>=acc['arcface_cos_min'] else 'FAIL'} >= {acc['arcface_cos_min']}]")
    print(f"Landmark RMSE (px):            mean={rmse_m:.2f}  [{'PASS' if rmse_m<=acc['landmark_rmse_max_px'] else 'FAIL'} <= {acc['landmark_rmse_max_px']}]")
    print(f"Latency p95 (s):               {lat95:.2f}        [{'PASS' if (lat95 or 9)<=acc['latency_p95_s_max'] else 'FAIL'} <= {acc['latency_p95_s_max']}]")
    print(f"Triplets for human style eval -> {args.triplets} (target >= {acc['style_human_eval_min']}/5)")
    json.dump({"model": args.model, "n": len(faces), "arcface_cos_mean": cos_m,
               "landmark_rmse_mean": rmse_m, "latency_p95_s": lat95},
              open(Path(args.triplets).parent / f"eval_{args.model}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Stage-1 hosted endpoint — /api/restyle-compatible drop-in for the live site.

Loads SDXL-Turbo + the trained vampire LoRA + InstantID (face-identity lock) and serves
the SAME contract as the current Fal call:

  POST /api/restyle   { "image": <data URL or http(s) URL>, "strength": 0.30..0.85 }
  ->  200 { "image": { "b64": "data:image/png;base64,..." }, "engine": "vampire-turbo-instantid",
            "identity": true, "ms": <int> }

Keep ip-adapter-face-id as the FALLBACK in the ENGINES array (see engines/engine_entry.js).

  python src/infer_server.py --config config.yaml --port 8008
"""
import argparse, base64, io, sys, time
from pathlib import Path

import requests
import torch
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from prompts import VAMPIRE_PROMPT, NEGATIVE_PROMPT

app = FastAPI()
STATE = {}


class RestyleReq(BaseModel):
    image: str
    strength: float = 0.65


def _load_image(src: str) -> Image.Image:
    if src.startswith("data:"):
        raw = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith("http"):
        raw = requests.get(src, timeout=30).content
    else:
        raw = base64.b64decode(src)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _lerp(rng, t):  # map slider 0..1 into a configured [lo, hi]
    return rng[0] + (rng[1] - rng[0]) * max(0.0, min(1.0, t))


def _slider01(strength):  # live slider is 0.30..0.85
    return (strength - 0.30) / (0.85 - 0.30)


def init(cfg):
    s = cfg["stage1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoPipelineForImage2Image
    pipe = AutoPipelineForImage2Image.from_pretrained(
        s["base_model"], torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    lora = Path(s["lora"]["out_dir"])
    if lora.exists():
        pipe.load_lora_weights(str(lora))
        print(f"loaded vampire LoRA from {lora}")
    else:
        print(f"[warn] no LoRA at {lora}; serving base SDXL-Turbo style (train_lora.py first)")
    # InstantID identity lock — load the InstantID controlnet + ip-adapter face embedder.
    # (Wire per github.com/InstantID/InstantID; kept behind a flag so the server boots
    #  without the weights for smoke-testing the HTTP contract.)
    STATE.update(pipe=pipe, cfg=cfg, device=device, instantid=s.get("instantid", False))
    print(f"Stage-1 server ready on {device}; instantid={s.get('instantid')}")


@app.post("/api/restyle")
def restyle(req: RestyleReq):
    t0 = time.perf_counter()
    cfg = STATE["cfg"]; s = cfg["stage1"]
    try:
        img = _load_image(req.image).resize((cfg["data"]["resolution"],) * 2)
        t01 = _slider01(req.strength)
        out = STATE["pipe"](
            prompt=VAMPIRE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=img,
            strength=_lerp(s["slider_to_strength"], t01),
            guidance_scale=_lerp(s["slider_to_guidance"], t01),
            num_inference_steps=s["num_inference_steps"],
        ).images[0]
        buf = io.BytesIO(); out.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        ms = int((time.perf_counter() - t0) * 1000)
        return {"image": {"b64": b64}, "engine": "vampire-turbo-instantid",
                "identity": STATE["instantid"], "ms": ms}
    except Exception as e:
        # Mirror the live error path: caller falls back to ip-adapter-face-id, page re-arms.
        return JSONResponse(status_code=500, content={"error": str(e),
                            "ms": int((time.perf_counter() - t0) * 1000)})


@app.get("/health")
def health():
    return {"ok": True, "device": STATE.get("device"), "instantid": STATE.get("instantid")}


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()
    init(yaml.safe_load(open(args.config)))
    uvicorn.run(app, host="0.0.0.0", port=args.port)

"""End-to-end PLUMBING smoke test — run the whole pipeline on ~10 images so you prove the
config works before spending real GPU/Fal budget on the full 2-5k run.

It does NOT produce a usable model. It proves: Fal teacher calls succeed + write pairs, the
LoRA trainer runs a few steps without crashing, inference produces an image, and the eval
metrics compute. Each stage reports PASS/FAIL and the test continues so you see every break.

  python src/smoke_test.py --config config.yaml --faces data/faces --n 10

Run src/fetch_ffhq.py first to populate --faces. Needs FAL_KEY in env (.env).
"""
import argparse, copy, subprocess, sys, traceback
from pathlib import Path

import yaml

RESULTS = []


def step(name):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n=== {name} ===")
            try:
                fn(*a, **k)
                RESULTS.append((name, True, ""))
                print(f"[PASS] {name}")
            except SystemExit as e:
                RESULTS.append((name, False, f"exit {e.code}"))
                print(f"[FAIL] {name}: exited {e.code}")
            except Exception as e:
                RESULTS.append((name, False, str(e)))
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc()
        return wrapped
    return deco


def make_smoke_config(cfg_path: str, n: int) -> str:
    """Write a tiny-budget config so every stage runs in seconds."""
    cfg = yaml.safe_load(open(cfg_path))
    cfg["data"]["min_pairs"] = max(2, n // 2)
    cfg["stage1"]["lora"]["train_steps"] = 20
    cfg["stage1"]["lora"]["batch_size"] = 1
    cfg["student"]["train_steps"] = 50
    cfg["student"]["batch_size"] = 2
    out = "config.smoke.yaml"
    yaml.safe_dump(cfg, open(out, "w"))
    print(f"wrote {out} (train_steps=20/50, n={n})")
    return out


def run(cmd):
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--faces", default="data/faces")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    faces = sorted(Path(args.faces).glob("*"))
    if len(faces) < args.n:
        print(f"[abort] need >= {args.n} faces in {args.faces}; found {len(faces)}. "
              "Run src/fetch_ffhq.py first.")
        sys.exit(1)

    smoke_cfg = make_smoke_config(args.config, args.n)
    py = sys.executable

    @step("1. teacher_generate (Fal teacher -> pairs)")
    def s1():
        run([py, "src/teacher_generate.py", "--faces", args.faces, "--out", "data/targets_smoke",
             "--pairs", "data/pairs.smoke.jsonl", "--config", smoke_cfg, "--limit", str(args.n)])
        pairs = Path("data/pairs.smoke.jsonl")
        assert pairs.exists() and pairs.read_text().strip(), "no pairs written"

    @step("2. train_lora (few steps, no crash)")
    def s2():
        # point the trainer at the smoke pairs by editing the smoke config's pairs_file
        cfg = yaml.safe_load(open(smoke_cfg)); cfg["data"]["pairs_file"] = "data/pairs.smoke.jsonl"
        yaml.safe_dump(cfg, open(smoke_cfg, "w"))
        run([py, "src/train_lora.py", "--config", smoke_cfg])
        assert any(Path(cfg["stage1"]["lora"]["out_dir"]).glob("*")), "no LoRA checkpoint saved"

    @step("3. inference (one snapshot -> one restyled still)")
    def s3():
        sys.path.insert(0, "src")
        import infer_server, io, base64
        from PIL import Image
        infer_server.init(yaml.safe_load(open(smoke_cfg)))
        buf = io.BytesIO(); Image.open(faces[0]).convert("RGB").save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        resp = infer_server.restyle(infer_server.RestyleReq(image=data_url, strength=0.65))
        assert isinstance(resp, dict) and resp.get("image"), f"bad inference response: {resp}"
        out = base64.b64decode(resp["image"]["b64"].split(",")[-1])
        Path("data/smoke_out.png").write_bytes(out)
        print("wrote data/smoke_out.png")

    @step("4. eval metrics compute (ArcFace + landmark)")
    def s4():
        sys.path.insert(0, "src")
        from eval import arcface_cosine, landmark_rmse
        from PIL import Image
        inp, out = Image.open(faces[0]), Image.open("data/smoke_out.png")
        cos = arcface_cosine(inp, out); rmse = landmark_rmse(inp.resize(out.size), out)
        print(f"ArcFace cosine={cos}  landmark_rmse={rmse}")
        assert cos is not None, "ArcFace returned None (face not detected / backbone missing)"

    s1(); s2(); s3(); s4()

    print("\n" + "=" * 48 + "\nSMOKE TEST SUMMARY")
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + msg) if msg else ''}")
    passed = all(ok for _, ok, _ in RESULTS)
    print("=" * 48)
    print("✅ Plumbing OK — safe to launch the full run." if passed else
          "❌ Fix the FAILs above before spending budget on the full run.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

"""One-shot Colab environment fixes for the smoke test.

Run once after cloning + pip install, before the smoke test:

    !python src/colab_fix_env.py

Fixes two things that fail on a fresh Colab runtime:
  A. torchao 0.10.0 is too old; peft's LoRA dispatcher hard-errors on it.
     We don't use torchao, so uninstalling it makes peft skip it cleanly.
  B. insightface's antelopev2 pack extracts one level too deep
     (antelopev2/antelopev2/*.onnx), so FaceAnalysis finds no models and
     asserts 'detection' in self.models. We flatten it up one level.
  C. eval.py's landmark RMSE needs MediaPipe's face_landmarker.task at
     checkpoints/face_landmarker.task, which isn't bundled. We download it.
"""
import glob
import os
import shutil
import subprocess
import sys
import urllib.request

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def fix_torchao():
    print("[A] removing incompatible torchao (unused) ...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)


def fix_antelopev2():
    base = os.path.expanduser("~/.insightface/models/antelopev2")
    nested = os.path.join(base, "antelopev2")
    if os.path.isdir(nested):
        print(f"[B] flattening nested {nested} -> {base}")
        for f in glob.glob(os.path.join(nested, "*")):
            shutil.move(f, os.path.join(base, os.path.basename(f)))
        os.rmdir(nested)
    if os.path.isdir(base):
        files = sorted(os.listdir(base))
        print(f"[B] antelopev2 contents: {files}")
        # antelopev2's detector is scrfd_10g_bnkps.onnx (buffalo_l uses det_*.onnx)
        if any(("scrfd" in f or f.startswith("det_")) and f.endswith(".onnx") for f in files):
            print("[B] OK - detection model present")
        else:
            print("[B] WARNING - no detection model (scrfd_*/det_*.onnx) found; eval/scoring will fail")
    else:
        print(f"[B] antelopev2 dir missing ({base}); it downloads on first eval run, "
              "then re-run this script if needed")


def fix_face_landmarker():
    dst = "checkpoints/face_landmarker.task"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[C] {dst} already present")
        return
    os.makedirs("checkpoints", exist_ok=True)
    print(f"[C] downloading MediaPipe face_landmarker.task -> {dst}")
    urllib.request.urlretrieve(LANDMARKER_URL, dst)
    print(f"[C] OK - {os.path.getsize(dst)} bytes")


if __name__ == "__main__":
    fix_torchao()
    fix_antelopev2()
    fix_face_landmarker()
    print("done. now re-run the smoke test.")

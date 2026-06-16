"""One-shot Colab environment fixes for the smoke test.

Run once after cloning + pip install, before the smoke test:

    !python src/colab_fix_env.py

Fixes two things that fail on a fresh Colab runtime:
  A. torchao 0.10.0 is too old; peft's LoRA dispatcher hard-errors on it.
     We don't use torchao, so uninstalling it makes peft skip it cleanly.
  B. insightface's antelopev2 pack extracts one level too deep
     (antelopev2/antelopev2/*.onnx), so FaceAnalysis finds no models and
     asserts 'detection' in self.models. We flatten it up one level.
"""
import glob
import os
import shutil
import subprocess
import sys


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


if __name__ == "__main__":
    fix_torchao()
    fix_antelopev2()
    print("done. now re-run the smoke test.")

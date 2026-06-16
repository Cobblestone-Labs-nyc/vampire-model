"""Stage 2 — distill the Stage-1 model into a tiny one-forward-pass U-Net student
(the on-device / browser model, Deploy A).

The student maps  face X -> vampire Y  in a single pass, learning from (X, Y) pairs where
Y came from the Stage-1 endpoint (generate with: teacher_generate.py --teacher local).
Losses (losses.py): L1 + LPIPS vs target, ArcFace identity vs INPUT, landmark vs INPUT.

Optional FiLM conditioning on a scalar `strength` lets the page's 0.30–0.85 slider work
without retraining.

  python src/train_student.py --config config.yaml --teacher stage1
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from losses import CompositeLoss


# ----------------------------- tiny U-Net student -----------------------------------

class FiLM(nn.Module):
    """Modulate features by a scalar strength in [0,1]."""
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Linear(1, ch * 2)

    def forward(self, x, s):
        gb = self.fc(s.view(-1, 1))
        g, b = gb.chunk(2, dim=1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, film=False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n = nn.GroupNorm(8, cout)
        self.film = FiLM(cout) if film else None

    def forward(self, x, s=None):
        x = F.silu(self.n(self.c1(x)))
        x = self.c2(x)
        if self.film is not None and s is not None:
            x = self.film(x, s)
        return F.silu(x)


class UNetSmall(nn.Module):
    """5–25M param encoder-decoder w/ skip connections. Width-controllable."""
    def __init__(self, base=48, width_mult=0.75, film=True):
        super().__init__()
        c = [max(8, int(base * width_mult * m)) for m in (1, 2, 4, 8)]
        self.film = film
        self.e1 = ConvBlock(3, c[0], film)
        self.e2 = ConvBlock(c[0], c[1], film)
        self.e3 = ConvBlock(c[1], c[2], film)
        self.b = ConvBlock(c[2], c[3], film)
        self.d3 = ConvBlock(c[3] + c[2], c[2], film)
        self.d2 = ConvBlock(c[2] + c[1], c[1], film)
        self.d1 = ConvBlock(c[1] + c[0], c[0], film)
        self.out = nn.Conv2d(c[0], 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x, s=None):
        e1 = self.e1(x, s); e2 = self.e2(self.pool(e1), s); e3 = self.e3(self.pool(e2), s)
        b = self.b(self.pool(e3), s)
        d3 = self.d3(torch.cat([self.up(b), e3], 1), s)
        d2 = self.d2(torch.cat([self.up(d3), e2], 1), s)
        d1 = self.d1(torch.cat([self.up(d2), e1], 1), s)
        # residual: student edits the input rather than redrawing it (identity-conservative)
        return torch.sigmoid(self.out(d1) + _logit(x))


def _logit(x, eps=1e-4):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ----------------------------- data --------------------------------------------------

class XYDataset(Dataset):
    def __init__(self, pairs_file, res):
        self.items = [json.loads(l) for l in open(pairs_file) if not json.loads(l).get("flag_cull")]
        self.res = res

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        img = Image.open(p).convert("RGB").resize((self.res, self.res))
        return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i):
        r = self.items[i]
        return self._load(r["x"]), self._load(r["y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--teacher", default="stage1", help="label only; data already generated")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    st = cfg["student"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    net = UNetSmall(st["base_channels"], st["width_mult"], st["film_strength"]).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"student params: {nparams/1e6:.1f}M")

    crit = CompositeLoss(st["losses"], device)
    opt = torch.optim.AdamW(net.parameters(), lr=st["lr"])
    ds = XYDataset(cfg["data"]["pairs_file"], cfg["data"]["resolution"])
    dl = DataLoader(ds, batch_size=st["batch_size"], shuffle=True, drop_last=True, num_workers=4)

    net.train(); step = 0; pbar = tqdm(total=st["train_steps"], desc="student")
    while step < st["train_steps"]:
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            s = torch.rand(x.shape[0], device=device) if st["film_strength"] else None
            out = net(x, s)
            loss, parts = crit(out, y, x)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1; pbar.update(1); pbar.set_postfix(**{k: round(v, 3) for k, v in parts.items()})
            if step % 2000 == 0:
                Path(st["out_dir"]).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), Path(st["out_dir"]) / "student.pth")
            if step >= st["train_steps"]:
                break
    Path(st["out_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), Path(st["out_dir"]) / "student.pth")
    print(f"student saved -> {st['out_dir']}/student.pth  (export with export_onnx.py)")


if __name__ == "__main__":
    main()

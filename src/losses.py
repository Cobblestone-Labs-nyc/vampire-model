"""Losses for the Stage-2 student (training spec §4).

Identity is the whole point: ArcFace identity loss is computed between the OUTPUT and the
INPUT (not the teacher target) — this is the lever that forces identity preservation.
Landmark loss penalizes geometry drift (the mouth-melt / head-reshape failure mode).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Recon(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class LPIPSPerceptual(nn.Module):
    """Perceptual loss via the lpips package (AlexNet/VGG backbone)."""
    def __init__(self, net="vgg"):
        super().__init__()
        import lpips
        self.fn = lpips.LPIPS(net=net)
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def forward(self, pred, target):
        # lpips expects inputs in [-1, 1]
        return self.fn(pred * 2 - 1, target * 2 - 1).mean()


class ArcFaceIdentity(nn.Module):
    """Cosine identity loss between OUTPUT and INPUT faces.

    Uses an ArcFace recognition backbone (InsightFace antelopev2). The backbone is frozen.
    Faces are aligned/cropped upstream (student trains on aligned crops), so we feed the
    full 112x112-resized tensor. 1 - cos(emb_out, emb_in) -> 0 means identity preserved.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        from insightface.app import FaceAnalysis  # noqa
        # We use the recognition model directly for a differentiable-friendly embed.
        # NOTE: InsightFace ONNX recognizer is not differentiable; for training we wrap a
        # torch ArcFace (e.g. a converted r50). See README for the weights drop-in.
        import torch
        self.device = device
        self.backbone = _load_torch_arcface(device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def _embed(self, x):  # x in [0,1], NCHW
        x = F.interpolate(x, size=112, mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return F.normalize(self.backbone(x), dim=1)

    def forward(self, output, input_face):
        with torch.no_grad():
            e_in = self._embed(input_face)
        e_out = self._embed(output)
        return (1 - (e_out * e_in).sum(dim=1)).mean()


class LandmarkLoss(nn.Module):
    """Keep output facial landmarks aligned to input landmarks.

    MediaPipe is not differentiable, so during training we approximate geometry with a
    lightweight torch landmark regressor (or a fixed-point spatial-consistency term). The
    eval path (eval.py) uses the real MediaPipe FaceLandmarker for the reported RMSE.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.regressor = _load_torch_landmarker(device)
        for p in self.regressor.parameters():
            p.requires_grad_(False)

    def forward(self, output, input_face):
        with torch.no_grad():
            lm_in = self.regressor(input_face)
        lm_out = self.regressor(output)
        return F.mse_loss(lm_out, lm_in)


class CompositeLoss(nn.Module):
    """Weighted sum per config.student.losses."""
    def __init__(self, weights: dict, device="cuda"):
        super().__init__()
        self.w = weights
        self.l1 = L1Recon()
        self.lpips = LPIPSPerceptual().to(device)
        self.identity = ArcFaceIdentity(device) if weights.get("identity_arcface", 0) else None
        self.landmark = LandmarkLoss(device) if weights.get("landmark", 0) else None

    def forward(self, output, target, input_face):
        terms = {}
        terms["l1"] = self.w["l1"] * self.l1(output, target)
        terms["lpips"] = self.w["lpips"] * self.lpips(output, target)
        if self.identity is not None:
            terms["identity_arcface"] = self.w["identity_arcface"] * self.identity(output, input_face)
        if self.landmark is not None:
            terms["landmark"] = self.w["landmark"] * self.landmark(output, input_face)
        total = sum(terms.values())
        return total, {k: float(v) for k, v in terms.items()}


# --- weight loaders (drop in converted torch backbones; see README) -----------------

def _load_torch_arcface(device):
    """Load a torch ArcFace recognition backbone (e.g. IR-SE50 / r50).

    Place weights at checkpoints/arcface_r50.pth. We use a standard IResNet50 def from
    insightface's torch zoo or `facenet-pytorch`-style. Kept as a function so the import
    is lazy and the rest of the repo runs without the weights present.
    """
    try:
        from backbones.iresnet import iresnet50  # optional local copy
    except Exception:
        raise RuntimeError(
            "ArcFace torch backbone not found. Drop an IResNet50 def at src/backbones/iresnet.py "
            "and weights at checkpoints/arcface_r50.pth (see README §losses)."
        )
    import torch
    net = iresnet50()
    net.load_state_dict(torch.load("checkpoints/arcface_r50.pth", map_location="cpu"))
    return net.to(device)


def _load_torch_landmarker(device):
    try:
        from backbones.landmarker import TinyLandmarker
    except Exception:
        raise RuntimeError(
            "Torch landmarker not found. Provide src/backbones/landmarker.py (a small "
            "68-pt regressor) + weights, or set config.student.losses.landmark: 0 to disable."
        )
    import torch
    net = TinyLandmarker()
    net.load_state_dict(torch.load("checkpoints/landmarker.pth", map_location="cpu"))
    return net.to(device)

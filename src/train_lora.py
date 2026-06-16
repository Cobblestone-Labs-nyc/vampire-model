"""Stage 1 — train a vampire LoRA on SDXL-Turbo (the hosted, Deploy-B model).

This fine-tunes a small LoRA adapter so SDXL-Turbo produces the single 'cathedral
aristocratic vampire' look in 4 steps. Identity at inference time is locked by InstantID
(see infer_server.py); the LoRA only carries the *style*, which keeps it tiny.

Trains on the (X, Y) pairs from teacher_generate.py: the LoRA learns to reproduce the
teacher's vampire styling. Run on CUDA.

  python src/train_lora.py --config config.yaml
"""
import argparse, json, sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prompts import VAMPIRE_PROMPT


class PairDataset(Dataset):
    """Returns teacher vampire targets Y (LoRA learns the style distribution).
    Identity preservation is handled by InstantID at inference, not the LoRA."""
    def __init__(self, pairs_file, res, cull_below):
        self.items = []
        for line in open(pairs_file):
            r = json.loads(line)
            if r.get("flag_cull"):
                continue
            self.items.append(r)
        self.res = res

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(self.items[i]["y"]).convert("RGB").resize((self.res, self.res))
        t = torch.from_numpy(__import__("numpy").asarray(img)).permute(2, 0, 1).float() / 127.5 - 1
        return {"pixel_values": t, "prompt": VAMPIRE_PROMPT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    s = cfg["stage1"]; lo = s["lora"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warn] no CUDA — SDXL-Turbo LoRA training on CPU/MPS is impractical. "
              "Run this on the GPU box.")

    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from peft import LoraConfig

    pipe = StableDiffusionXLPipeline.from_pretrained(
        s["base_model"], torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)

    # Attach LoRA to the UNet attention layers only.
    pipe.unet.add_adapter(LoraConfig(
        r=lo["rank"], lora_alpha=lo["alpha"], init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lo["lr"])

    ds = PairDataset(cfg["data"]["pairs_file"], cfg["data"]["resolution"], cfg["data"]["cull_arcface_cos"])
    dl = DataLoader(ds, batch_size=lo["batch_size"], shuffle=True, drop_last=True)

    def encode_prompt(prompts):
        return pipe.encode_prompt(prompt=prompts, device=device, num_images_per_prompt=1,
                                  do_classifier_free_guidance=False)

    pipe.unet.train()
    step, pbar = 0, tqdm(total=lo["train_steps"], desc="LoRA")
    while step < lo["train_steps"]:
        for batch in dl:
            px = batch["pixel_values"].to(device, dtype=pipe.vae.dtype)
            with torch.no_grad():
                latents = pipe.vae.encode(px).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy = noise_sched.add_noise(latents, noise, ts)
            pe, _, pooled, _ = encode_prompt(list(batch["prompt"]))
            add_time_ids = torch.tensor([[cfg["data"]["resolution"]] * 2 + [0, 0] +
                                         [cfg["data"]["resolution"]] * 2], device=device).repeat(latents.shape[0], 1)
            model_pred = pipe.unet(noisy, ts, encoder_hidden_states=pe,
                                   added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1; pbar.update(1); pbar.set_postfix(loss=float(loss))
            if step % 500 == 0:
                out = Path(lo["out_dir"]); out.mkdir(parents=True, exist_ok=True)
                pipe.unet.save_attn_procs(out)  # saves only the LoRA weights (small)
            if step >= lo["train_steps"]:
                break
    out = Path(lo["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_attn_procs(out)
    print(f"Stage-1 LoRA saved -> {out} (load alongside InstantID in infer_server.py)")


if __name__ == "__main__":
    main()

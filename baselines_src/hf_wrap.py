"""CelebA-HQ 256 pixel diffusion model (Ho et al. DDPM, public checkpoint google/ddpm-celebahq-256) as a drop-in noise-prediction
network for the DPS and DAPS code bases, so that they use a prior trained on the same dataset as the consistency model."""
import torch, torch.nn as nn
import os
PATH = os.environ.get("HF_DM_PATH", "/usr/scratch/akheirandish3/PNP/baselines_src/ckpt_celeba/ddpm-celebahq-256")   # HF_DM_PATH=.../ckpt_bedroom/ddpm-ema-bedroom-256 for LSUN bedroom
class HFEps(nn.Module):
    def __init__(self):
        super().__init__()
        from diffusers import UNet2DModel
        self.unet = UNet2DModel.from_pretrained(PATH)
    def forward(self, x, t, **kw):
        return self.unet(x, t).sample

"""Run PnP-CM's JPEG / nonlinear-deblur baselines unmodified, supplying two missing pieces:

1. bkse imports its own packages as top-level `models...`, so bkse/ must be on sys.path.
2. The repo calls `DiffJPEGCoding(jpeg_qf, device)` and `.A(x)` on [-1,1] images, but the
   upstream Diff-JPEG class (which the README says to clone) has neither.  The authors'
   modified class is not in the release, so this adapter is OUR reconstruction of it:
   map [-1,1] -> [0,255], run upstream differentiable JPEG at the given quality, map back.

Run from the PnP-CM repo root:  python ../mode_probe/run_nl.py --deg=jpeg ... (main.py args)
"""
import os, sys
import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("bkse"))

import functions.diff_jpeg as dj


class DiffJPEGAdapter:
    def __init__(self, jpeg_qf, device):
        self.q = torch.tensor([float(jpeg_qf)], device=device)

    def A(self, x, **kwargs):
        out = dj.diff_jpeg_coding(image_rgb=(x + 1) / 2 * 255, jpeg_quality=self.q.expand(x.shape[0]))
        return out / 255 * 2 - 1


dj.DiffJPEGCoding = DiffJPEGAdapter

import main as M
M.main()

#!/bin/bash
# One-time setup on a new server. Run from the repository root:  ./setup.sh
# Layout expected by every script (relative paths ../mode_probe, ../pnpcm_env are used throughout):
#   <root>/mode_probe      our code (this repository)
#   <root>/PnP-CM          upstream PnP-CM, unmodified, pinned commit
#   <root>/pnpcm_env       python virtual environment
#   <root>/baselines_src   DPS / DAPS with our patches (optional, only for the baselines)
set -e; ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
[ -d PnP-CM ] || { git clone https://github.com/MerveGulle/PnP-CM.git && git -C PnP-CM checkout c1bfb46; }
[ -d PnP-CM/bkse ] || git clone https://github.com/VinAIResearch/blur-kernel-space-exploring PnP-CM/bkse
if [ ! -d PnP-CM/functions/diff_jpeg ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/necla-ml/Diff-JPEG.git PnP-CM/functions/Diff-JPEG
  (cd PnP-CM/functions/Diff-JPEG && git sparse-checkout set diff_jpeg && mv diff_jpeg ../) && rm -rf PnP-CM/functions/Diff-JPEG
fi
if [ ! -d pnpcm_env ]; then
  python3.11 -m venv pnpcm_env || python3 -m venv pnpcm_env
  pnpcm_env/bin/pip install --upgrade pip
  # torch build must match the server's CUDA; change the index URL if needed
  pnpcm_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pnpcm_env/bin/pip install numpy scipy lpips pillow pyyaml tqdm scikit-image kornia sigpy einops==0.8.0
fi
mkdir -p mode_probe/logs_x PnP-CM/exp/logs/celeba_hq
./relocate.sh
echo
echo "Still needed (not in git, too large): the consistency-model checkpoint and a few data files."
echo "  Fastest: ./sync_large.sh <user>@sfk-srv1.ece.gatech.edu     (copies them from the original server)"
echo "  Or by hand: CelebA-HQ CM checkpoint (link in PnP-CM/README.md) -> PnP-CM/exp/logs/celeba_hq/"
ls PnP-CM/exp/logs/celeba_hq/*.pt >/dev/null 2>&1 && echo "checkpoint: found" || echo "checkpoint: MISSING"

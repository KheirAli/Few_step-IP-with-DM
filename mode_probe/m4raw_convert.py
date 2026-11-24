"""Convert M4Raw (public 0.3 T brain k-space, Zenodo 8056074, 4 coils, 256x256) to the .mat format of PnP-CM's MRI pipeline (kspace + coil maps, 320x320).
Repetitions of one scan are averaged in k-space (reference with better SNR); the image is zero-padded 256 -> 320 (the consistency model is 320x320);
coil sensitivity maps by ESPIRiT (sigpy).   usage: PYTHONPATH=pylibs_m4 python3 m4raw_convert.py <dir with .h5> <out dir> <contrast T1|T2|FLAIR> <n subjects>"""
import sys, glob, os, re, numpy as np, h5py, scipy.io as sio, sigpy as sp, sigpy.mri as mr
src, out, con, nsub = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "**", f"*_{con}*.h5"), recursive=True)); subj = {}
for f in files: subj.setdefault(re.split(r"_", os.path.basename(f))[0], []).append(f)
n = 0
for s, fs in list(subj.items())[:nsub]:
    ks = np.mean([h5py.File(f, "r")["kspace"][()] for f in fs], 0)                 # [slices, coils, 256, 256], averaged over repetitions
    for sl in range(ks.shape[0] // 2 - 2, ks.shape[0] // 2 + 3):                   # 5 central slices
        img = sp.ifft(ks[sl], axes=(-1, -2)); img = np.pad(img, ((0, 0), (32, 32), (32, 32))); k320 = sp.fft(img, axes=(-1, -2)).astype(np.complex64)
        maps = mr.app.EspiritCalib(k320, calib_width=24, thresh=0.02, crop=0.9, show_pbar=False).run().astype(np.complex64)
        sio.savemat(os.path.join(out, f"{s}_{con}_{sl:02d}.mat"), {"kspace": k320, "coils": maps}); n += 1
print("converted", n, "slices from", min(nsub, len(subj)), "subjects,", len(files), "files seen; k-space shape", ks.shape)

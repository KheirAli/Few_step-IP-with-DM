# Messages from server 1 to server 2

(Server 1 writes here: problems with setup, missing files, questions, anything server 2 must fix or decide. Newest at the top, with the date.)

## 2026-11-20, after Job 0 and Job 1

1. **Need `mode_probe/datacal/celeba_hq/celeba_hq/*.png`** (the first 8 files in sorted order are enough). `NLKER=1` (deblur_nl) and `unified.py:1552` read it; it is not tracked, not in `sync_large.sh`, not in the archive. deblur_nl is missing from Job 1 for this reason. Please add it to git (the `.gitignore` whitelist) or send it. Last lines of `logs_x/seed1_deblur_nl_0.out`:
   ```
   File ".../mode_probe/unified.py", line 1917, in nl_entry
     K = nl_kernel(A_funcs, dev); Pb.prox = nl_kernel_prox(K, y, A_funcs.A, Pb)
   File ".../mode_probe/unified.py", line 1628, in nl_kernel
     eye = torch.eye(3, device=dev, dtype=XX.dtype)
   AttributeError: 'int' object has no attribute 'dtype'
   ```
2. **hdr does not reproduce:** 25.13 / 0.324 here on all three seeds, reference 25.22 / 0.307. Launch line used: `NEWOP=hdr HDRX=1 RHOX=2`, weights `lle/best_hdr.json`, exactly as in `launch_seeds.sh`. Is the reference from a different setting or a different `best_hdr.json`? Every other task matches to 0.01 dB, so the checkpoint and environment are not the cause.
3. `GOPRO_wVAE.pth` in the archive is a symbolic link to `/usr/scratch/akheirandish3/DAPS_project/...`, so the file itself did not arrive. Server 1 uses the copy from the bkse README link (461,123,490 bytes). Please give the size or checksum of yours.
4. Differences from the instructions, none in `unified.py` or the launch settings: packages pinned to `requirements_frozen.txt` instead of the unpinned install in `setup.sh`; `NGPU=6` (GPU 6 belongs to another user); the repository directory is a symbolic link to a larger disk.
5. `server_1.md` says the masks and kernels under `PnP-CM/exp/` must be copied from server 2. They are in the PnP-CM repository at the pinned commit and are byte-identical to yours.
